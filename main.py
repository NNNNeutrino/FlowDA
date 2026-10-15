import os
import json
import logging
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import XCO2FlowDatasetPT
from model import FlowUNet
from config import (
    DATA_SPLIT_CSV,
    LOG_DIR,
    PLOT_DIR,
    UNCERTAINTY_OUT_DIR,
    get_region_paths,
)


# Auxiliary fields: u10, v10, sp, flux_apri
ERA5_USED_IDXS = [6, 7, 9]
AUX_USED_NAMES = ["u10", "v10", "sp", "flux_apri"]
AUX_USED_CHANNELS = len(AUX_USED_NAMES)

VAL_ENSEMBLE_SIZE = 3
TEST_ENSEMBLE_SIZE = 10
INFERENCE_STEPS = 50

TRAIN_SEED = 42
VAL_SEED = 2026
TEST_SEED = 2027
UNCERTAINTY_SEED = 2028
TCCON_SEED = 2029

UNCERTAINTY_ENSEMBLE_SIZE = 10
UNCERTAINTY_BATCH_SIZE = 32
UNCERTAINTY_MODES = ["background"]
OBS_PERTURB_SIGMA_PPM = None
RUN_TRAINING = True
RUN_UNCERTAINTY = True


def set_global_seed(seed):
    """Set deterministic random seeds for Python, NumPy, PyTorch, and CUDA."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Seed Python and NumPy for each DataLoader worker."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_torch_generator(device, seed):
    """Create an independent generator for reproducible evaluation sampling."""
    device_type = "cuda" if torch.device(device).type == "cuda" else "cpu"
    generator = torch.Generator(device=device_type)
    generator.manual_seed(seed)
    return generator


def build_aux_fields(batch, device):
    """Build auxiliary inputs [u10, v10, sp, flux_apri] with shape [B, 4, H, W]."""
    era5 = batch["era5"].to(device).float()
    if era5.shape[1] <= max(ERA5_USED_IDXS):
        raise ValueError(f"ERA5 has insufficient channels: shape={era5.shape}, required indices={ERA5_USED_IDXS}")

    era5_selected = era5[:, ERA5_USED_IDXS, :, :]

    if "surface_flux" not in batch:
        raise KeyError("Batch is missing 'surface_flux'; ensure the dataset returns sample['surface_flux']")

    surface_flux = batch["surface_flux"].to(device).float()
    if surface_flux.ndim != 4:
        raise ValueError(f"surface_flux must have shape [B, C, H, W], got {surface_flux.shape}")

    if surface_flux.shape[-2:] != era5_selected.shape[-2:]:
        surface_flux = F.interpolate(surface_flux, size=era5_selected.shape[-2:], mode="bilinear", align_corners=False)

    aux = torch.cat([era5_selected, surface_flux], dim=1)

    if aux.shape[1] != AUX_USED_CHANNELS:
        raise ValueError(f"Expected {AUX_USED_CHANNELS} auxiliary channels, got shape={aux.shape}")

    return aux


class EarlyStopping:
    def __init__(self, patience=7, delta=0.0001, monitor="R2"):
        self.patience = patience
        self.delta = delta
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return True

        if self.monitor == "R2":
            if current_score < self.best_score + self.delta:
                self.counter += 1
                is_improved = False
            else:
                self.best_score = current_score
                self.counter = 0
                is_improved = True
        else:
            if current_score > self.best_score - self.delta:
                self.counter += 1
                is_improved = False
            else:
                self.best_score = current_score
                self.counter = 0
                is_improved = True

        if self.counter >= self.patience:
            self.early_stop = True

        return is_improved


class FlowMatchingTrainer:
    def __init__(self, model, stats_path, lr=2e-4, sigma=0.05, plot_dir="plots"):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.sigma = sigma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.plot_dir = plot_dir

        with open(stats_path, "r") as f:
            stats = json.load(f)

        self.mean = stats["mean_gb"]
        self.std = stats["std_gb"]

        os.makedirs(plot_dir, exist_ok=True)

    def denorm(self, x, base_mean):
        base_mean = base_mean.view(-1, 1, 1, 1)
        return x * self.std + base_mean

    def make_ct_3x2_noise(self, x_ct, generator=None):
        """
        Generate Gaussian perturbations on the native CT 3° × 2° scale.

        The 1° target grid shares one perturbation across each 2-latitude ×
        3-longitude block.
        """
        batch_size, channels, height, width = x_ct.shape
        lat_block, lon_block = 2, 3
        coarse_height = (height + lat_block - 1) // lat_block
        coarse_width = (width + lon_block - 1) // lon_block

        coarse_noise = torch.randn(
            batch_size, channels, coarse_height, coarse_width,
            device=x_ct.device, dtype=x_ct.dtype, generator=generator
        ) * self.sigma

        noise = coarse_noise.repeat_interleave(lat_block, dim=-2)
        noise = noise.repeat_interleave(lon_block, dim=-1)
        return noise[:, :, :height, :width]

    def perturb_ct_background(self, x_ct, generator=None):
        """Add spatially correlated 3° × 2° perturbations to the CT background."""
        return x_ct + self.make_ct_3x2_noise(x_ct, generator=generator)

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch} Train"):
            x_ct = batch["x_ct"].to(self.device)
            x1 = batch["z_cams"].to(self.device)
            y_oco = batch["y_oco"].to(self.device)
            mask = batch["mask"].to(self.device)
            aux = build_aux_fields(batch, self.device)

            doy = batch["doy"].to(self.device)
            tod = batch["tod"].to(self.device)

            x0 = self.perturb_ct_background(x_ct)
            t = torch.rand(x_ct.shape[0], device=self.device)
            xt = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * x1

            vt_pred = self.model(xt, x_ct, y_oco, mask, t, doy, tod, aux=aux)
            loss = F.mse_loss(vt_pred, x1 - x0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def sample(self, x_ct, y_oco, mask, aux, doy, tod, steps=50):
        self.model.eval()
        xt = self.perturb_ct_background(x_ct)
        dt = 1.0 / steps

        for i in range(steps):
            t = torch.full((x_ct.shape[0],), i / steps, device=self.device)
            vt = self.model(xt, x_ct, y_oco, mask, t, doy, tod, aux=aux)
            xt = xt + vt * dt

        return xt

    @torch.no_grad()
    def ensemble_sample(self, x_ct, y_oco, mask, aux, doy, tod, num_samples=10, steps=50, generator=None):
        self.model.eval()
        samples = []

        for _ in range(num_samples):
            xt = self.perturb_ct_background(x_ct, generator=generator)
            dt = 1.0 / steps

            for i in range(steps):
                t = torch.full((x_ct.shape[0],), i / steps, device=self.device)
                vt = self.model(xt, x_ct, y_oco, mask, t, doy, tod, aux=aux)
                xt = xt + vt * dt

            samples.append(xt)

        all_samples = torch.stack(samples)
        return all_samples.mean(dim=0), all_samples.std(dim=0)

    @torch.no_grad()
    def evaluate_uncertainty(self, loader, num_samples=10, seed=UNCERTAINTY_SEED):
        self.model.eval()
        std_mask_list = []
        std_nonmask_list = []
        generator = make_torch_generator(self.device, seed)

        for batch in tqdm(loader, desc="Uncertainty Analysis"):
            x_ct = batch["x_ct"].to(self.device)
            y_oco = batch["y_oco"].to(self.device)
            mask = batch["mask"].to(self.device)
            aux = build_aux_fields(batch, self.device)
            doy = batch["doy"].to(self.device)
            tod = batch["tod"].to(self.device)

            _, std_pred = self.ensemble_sample(
                x_ct, y_oco, mask, aux, doy, tod, num_samples=num_samples,
                steps=INFERENCE_STEPS, generator=generator
            )
            std_pred = std_pred * self.std
            mask_bool = mask > 0.5

            if mask_bool.sum() > 0:
                std_mask_list.append(std_pred[mask_bool].cpu())
            if (~mask_bool).sum() > 0:
                std_nonmask_list.append(std_pred[~mask_bool].cpu())

        if len(std_mask_list) + len(std_nonmask_list) > 0:
            all_std = torch.cat(std_mask_list + std_nonmask_list)
            global_std = all_std.mean().item()
        else:
            global_std = 0.0

        return {
            "std_mask": torch.cat(std_mask_list).mean().item() if std_mask_list else 0.0,
            "std_nonmask": torch.cat(std_nonmask_list).mean().item() if std_nonmask_list else 0.0,
            "std_global": global_std
        }

    def perturb_observation(self, y_oco, mask, obs_sigma_ppm, generator=None):
        """Add Gaussian perturbations only at valid observation grids."""
        if obs_sigma_ppm is None:
            raise ValueError("obs_sigma_ppm must be set for observation/joint uncertainty experiments")

        sigma_norm = float(obs_sigma_ppm) / float(self.std)
        noise = torch.randn(y_oco.shape, device=y_oco.device, dtype=y_oco.dtype, generator=generator) * sigma_norm
        return y_oco + noise * mask

    @torch.no_grad()
    def ensemble_sample_uncertainty(
        self, x_ct, y_oco, mask, aux, doy, tod, mode="background", obs_sigma_ppm=None,
        num_samples=10, steps=50, bg_generator=None, obs_generator=None
    ):
        """
        Test-time uncertainty experiment.

        background:
            Perturb only the CT initial state; keep observations fixed.
        observation:
            Keep the CT initial state fixed; perturb valid observation grids only.
        joint:
            Perturb both the CT initial state and valid observation grids.

        This diagnostic does not modify trained model parameters.
        """
        if mode not in {"background", "observation", "joint"}:
            raise ValueError(f"Unknown uncertainty mode: {mode}")
        if mode in {"observation", "joint"} and obs_sigma_ppm is None:
            raise ValueError(f"obs_sigma_ppm must be set for uncertainty mode: {mode}")

        self.model.eval()
        samples = []

        for _ in range(num_samples):
            xt = self.perturb_ct_background(x_ct, generator=bg_generator) if mode in {"background", "joint"} else x_ct.clone()
            y_member = self.perturb_observation(y_oco, mask, obs_sigma_ppm, generator=obs_generator) if mode in {"observation", "joint"} else y_oco

            dt = 1.0 / steps
            for i in range(steps):
                t = torch.full((x_ct.shape[0],), i / steps, device=self.device)
                vt = self.model(xt, x_ct, y_member, mask, t, doy, tod, aux=aux)
                xt = xt + vt * dt

            samples.append(xt)

        all_samples = torch.stack(samples, dim=0)
        return all_samples.mean(dim=0), all_samples.std(dim=0)

    @staticmethod
    def _safe_corr(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]

        if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return np.nan

        return float(np.corrcoef(x, y)[0, 1])

    @torch.no_grad()
    def evaluate_uncertainty_source(
        self, loader, region, mode="background", obs_sigma_ppm=None,
        num_samples=UNCERTAINTY_ENSEMBLE_SIZE, seed=UNCERTAINTY_SEED
    ):
        """
        Evaluate ensemble characteristics for one uncertainty source on the 2024 simulated test set.

        Reports domain-wide, observed, and non-observed spread; RMS spread;
        ensemble-mean RMSE; spread-skill ratio; scene-level spread-RMSE
        correlation; and empirical 1σ/2σ diagnostic coverage.
        """
        if mode not in {"background", "observation", "joint"}:
            raise ValueError(f"Unknown uncertainty mode: {mode}")
        if mode in {"observation", "joint"} and obs_sigma_ppm is None:
            raise ValueError(f"obs_sigma_ppm must be set for uncertainty mode: {mode}")

        self.model.eval()
        bg_generator = make_torch_generator(self.device, seed)
        obs_generator = make_torch_generator(self.device, seed + 10000)

        spread_sum = spread_sq_sum = error_sq_sum = 0.0
        spread_obs_sum = spread_nonobs_sum = 0.0
        n_all = n_obs = n_nonobs = 0
        cover1 = cover2 = 0
        cover1_obs = cover2_obs = 0
        cover1_nonobs = cover2_nonobs = 0
        scene_rows = []

        for batch in tqdm(loader, desc=f"{region.upper()} | uncertainty={mode}"):
            x_ct = batch["x_ct"].to(self.device).float()
            x_ref = batch["z_cams"].to(self.device).float()
            y_oco = batch["y_oco"].to(self.device).float()
            mask = batch["mask"].to(self.device).float()
            aux = build_aux_fields(batch, self.device)
            doy = batch["doy"].to(self.device).float()
            tod = batch["tod"].to(self.device).float()
            base_mean = batch["base_mean"].to(self.device).float()

            x_mean, x_std = self.ensemble_sample_uncertainty(
                x_ct=x_ct, y_oco=y_oco, mask=mask, aux=aux, doy=doy, tod=tod,
                mode=mode, obs_sigma_ppm=obs_sigma_ppm, num_samples=num_samples,
                steps=INFERENCE_STEPS, bg_generator=bg_generator, obs_generator=obs_generator
            )

            mean_ppm = self.denorm(x_mean, base_mean)
            ref_ppm = self.denorm(x_ref, base_mean)
            spread_ppm = x_std * float(self.std)
            error_ppm = mean_ppm - ref_ppm
            abs_error = torch.abs(error_ppm)

            mask_bool = mask > 0.5
            nonmask_bool = ~mask_bool

            spread_sum += spread_ppm.sum().item()
            spread_sq_sum += torch.sum(spread_ppm ** 2).item()
            error_sq_sum += torch.sum(error_ppm ** 2).item()
            n_all += spread_ppm.numel()

            cover1 += int((abs_error <= spread_ppm).sum().item())
            cover2 += int((abs_error <= 2.0 * spread_ppm).sum().item())

            if mask_bool.any():
                spread_obs_sum += spread_ppm[mask_bool].sum().item()
                n_obs += int(mask_bool.sum().item())
                cover1_obs += int((abs_error[mask_bool] <= spread_ppm[mask_bool]).sum().item())
                cover2_obs += int((abs_error[mask_bool] <= 2.0 * spread_ppm[mask_bool]).sum().item())

            if nonmask_bool.any():
                spread_nonobs_sum += spread_ppm[nonmask_bool].sum().item()
                n_nonobs += int(nonmask_bool.sum().item())
                cover1_nonobs += int((abs_error[nonmask_bool] <= spread_ppm[nonmask_bool]).sum().item())
                cover2_nonobs += int((abs_error[nonmask_bool] <= 2.0 * spread_ppm[nonmask_bool]).sum().item())

            timestamps = batch["timestamp"] if "timestamp" in batch else [None] * x_ct.shape[0]

            for b in range(x_ct.shape[0]):
                spread_b = spread_ppm[b, 0]
                error_b = error_ppm[b, 0]
                mask_b = mask_bool[b, 0]
                nonmask_b = ~mask_b

                scene_rows.append({
                    "Region": region.upper(),
                    "Mode": mode,
                    "Timestamp": str(timestamps[b]),
                    "Mean_spread_global_ppm": spread_b.mean().item(),
                    "RMSE_global_ppm": torch.sqrt(torch.mean(error_b ** 2)).item(),
                    "Mean_spread_observed_ppm": spread_b[mask_b].mean().item() if mask_b.any() else np.nan,
                    "RMSE_observed_ppm": torch.sqrt(torch.mean(error_b[mask_b] ** 2)).item() if mask_b.any() else np.nan,
                    "Mean_spread_nonobserved_ppm": spread_b[nonmask_b].mean().item() if nonmask_b.any() else np.nan,
                    "RMSE_nonobserved_ppm": torch.sqrt(torch.mean(error_b[nonmask_b] ** 2)).item() if nonmask_b.any() else np.nan,
                })

        scene_df = pd.DataFrame(scene_rows)
        mean_spread = spread_sum / n_all
        mean_spread_obs = spread_obs_sum / n_obs if n_obs > 0 else np.nan
        mean_spread_nonobs = spread_nonobs_sum / n_nonobs if n_nonobs > 0 else np.nan
        rms_spread = float(np.sqrt(spread_sq_sum / n_all))
        analysis_rmse = float(np.sqrt(error_sq_sum / n_all))

        results = {
            "Region": region.upper(),
            "Mode": mode,
            "Ensemble_size": int(num_samples),
            "CT_sigma_norm": float(self.sigma) if mode in {"background", "joint"} else 0.0,
            "CT_sigma_ppm": float(self.sigma * self.std) if mode in {"background", "joint"} else 0.0,
            "Obs_sigma_ppm": float(obs_sigma_ppm) if mode in {"observation", "joint"} else np.nan,
            "Mean_spread_global_ppm": mean_spread,
            "Mean_spread_observed_ppm": mean_spread_obs,
            "Mean_spread_nonobserved_ppm": mean_spread_nonobs,
            "Observed_spread_reduction_pct": (
                (mean_spread_nonobs - mean_spread_obs) / mean_spread_nonobs * 100
                if np.isfinite(mean_spread_nonobs) and mean_spread_nonobs > 1e-12 else np.nan
            ),
            "RMS_spread_global_ppm": rms_spread,
            "Analysis_RMSE_ppm": analysis_rmse,
            "Spread_skill_ratio": rms_spread / analysis_rmse if analysis_rmse > 1e-12 else np.nan,
            "Scene_spread_RMSE_R": self._safe_corr(scene_df["Mean_spread_global_ppm"], scene_df["RMSE_global_ppm"]),
            "Scene_spread_RMSE_R_observed": self._safe_corr(scene_df["Mean_spread_observed_ppm"], scene_df["RMSE_observed_ppm"]),
            "Scene_spread_RMSE_R_nonobserved": self._safe_corr(scene_df["Mean_spread_nonobserved_ppm"], scene_df["RMSE_nonobserved_ppm"]),
            "Coverage_1sigma_pct": cover1 / n_all * 100,
            "Coverage_2sigma_pct": cover2 / n_all * 100,
            "Coverage_1sigma_observed_pct": cover1_obs / n_obs * 100 if n_obs > 0 else np.nan,
            "Coverage_2sigma_observed_pct": cover2_obs / n_obs * 100 if n_obs > 0 else np.nan,
            "Coverage_1sigma_nonobserved_pct": cover1_nonobs / n_nonobs * 100 if n_nonobs > 0 else np.nan,
            "Coverage_2sigma_nonobserved_pct": cover2_nonobs / n_nonobs * 100 if n_nonobs > 0 else np.nan,
            "N_pixels": int(n_all),
            "N_observed_pixels": int(n_obs),
            "N_nonobserved_pixels": int(n_nonobs),
        }

        return results, scene_df

    @torch.no_grad()
    def evaluate(self, loader, epoch, mode="Val", num_plots=0):
        self.model.eval()

        sum_sq_error, sum_abs_error, total_pixels = 0.0, 0.0, 0
        sum_t, sum_t_sq = 0.0, 0.0
        sum_sq_error_oco, sum_abs_error_oco, total_pixels_oco = 0.0, 0.0, 0
        sum_t_oco, sum_t_sq_oco = 0.0, 0.0
        is_test = mode.lower() == "test"
        eval_seed = TEST_SEED if is_test else VAL_SEED
        generator = make_torch_generator(self.device, eval_seed)

        for i, batch in enumerate(tqdm(loader, desc=f"Evaluating {mode}")):
            x_ct = batch["x_ct"].to(self.device)
            x1 = batch["z_cams"].to(self.device)
            y_oco = batch["y_oco"].to(self.device)
            mask = batch["mask"].to(self.device)
            aux = build_aux_fields(batch, self.device)

            doy = batch["doy"].to(self.device)
            tod = batch["tod"].to(self.device)
            base_mean = batch["base_mean"].to(self.device)

            ensemble_size = TEST_ENSEMBLE_SIZE if mode.lower() == "test" else VAL_ENSEMBLE_SIZE
            
            x_gen, x_gen_std = self.ensemble_sample(
                x_ct=x_ct, y_oco=y_oco, mask=mask, aux=aux, doy=doy, tod=tod,
                num_samples=ensemble_size, steps=INFERENCE_STEPS, generator=generator
            )

            p_ppm = self.denorm(x_gen, base_mean).cpu().numpy()
            t_ppm = self.denorm(x1, base_mean).cpu().numpy()
            b_ppm = self.denorm(x_ct, base_mean).cpu().numpy()
            m_np = mask.cpu().numpy()

            y_ppm_raw = self.denorm(y_oco, base_mean).cpu().numpy()
            y_ppm = np.where(m_np > 0.5, y_ppm_raw, 0.0)

            p_flat = p_ppm.flatten()
            t_flat = t_ppm.flatten()
            m_flat = m_np.flatten()

            error = p_flat - t_flat
            sum_sq_error += np.sum(error ** 2)
            sum_abs_error += np.sum(np.abs(error))
            total_pixels += p_flat.size
            sum_t += np.sum(t_flat)
            sum_t_sq += np.sum(t_flat ** 2)

            oco_idx = m_flat > 0.5

            if np.sum(oco_idx) > 0:
                p_pts = p_flat[oco_idx]
                t_pts = t_flat[oco_idx]
                error_oco = p_pts - t_pts
                sum_sq_error_oco += np.sum(error_oco ** 2)
                sum_abs_error_oco += np.sum(np.abs(error_oco))
                total_pixels_oco += p_pts.size
                sum_t_oco += np.sum(t_pts)
                sum_t_sq_oco += np.sum(t_pts ** 2)

            if is_test and i < num_plots:
                std_map = (x_gen_std[0, 0] * self.std).cpu().numpy()

                p_plot = self.denorm(x_gen[0:1], base_mean[0:1], )[0, 0].cpu().numpy()

                b_plot = b_ppm[0, 0]
                t_plot = t_ppm[0, 0]
                y_plot = y_ppm[0, 0]
                m_plot = m_np[0, 0]

                self.plot_results(b_plot, t_plot, p_plot, y_plot, m_plot, std_map, epoch, mode, i)

        global_rmse = np.sqrt(sum_sq_error / total_pixels)
        global_mae = sum_abs_error / total_pixels
        global_mean_t = sum_t / total_pixels
        global_ss_tot = sum_t_sq - 2 * global_mean_t * sum_t + total_pixels * (global_mean_t ** 2)
        global_r2 = 1.0 - (sum_sq_error / (global_ss_tot + 1e-8))

        results = {
            "Global_RMSE": global_rmse,
            "Global_MAE": global_mae,
            "Global_R2": global_r2,
        }

        if total_pixels_oco > 0:
            oco_rmse = np.sqrt(sum_sq_error_oco / total_pixels_oco)
            oco_mae = sum_abs_error_oco / total_pixels_oco
            oco_mean_t = sum_t_oco / total_pixels_oco
            oco_ss_tot = sum_t_sq_oco - 2 * oco_mean_t * sum_t_oco + total_pixels_oco * (oco_mean_t ** 2)
            oco_r2 = 1.0 - (sum_sq_error_oco / (oco_ss_tot + 1e-8))

            results.update({
                "OCO_RMSE": oco_rmse,
                "OCO_MAE": oco_mae,
                "OCO_R2": oco_r2,
            })

        return results

    def plot_results(self, bg, gt, pred, oco, mask, std_map, epoch, mode, idx):
        out_dir = os.path.join(self.plot_dir, mode.lower())
        os.makedirs(out_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        bg, gt, pred, oco, mask = map(lambda d: d.detach().cpu().numpy() if torch.is_tensor(d) else d, [bg, gt, pred, oco, mask])

        vmin = min(bg.min(), gt.min(), pred.min())
        vmax = max(bg.max(), gt.max(), pred.max())

        im0 = axes[0, 0].imshow(bg, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0, 0].set_title("CT Background", fontsize=12)
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        im1 = axes[0, 1].imshow(gt, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0, 1].set_title("CAMS Target (GT)", fontsize=12)
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im2 = axes[0, 2].imshow(pred, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0, 2].set_title("FM Prediction", fontsize=12)
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

        oco_masked = np.ma.masked_where(mask == 0, oco)
        axes[1, 0].imshow(bg, cmap="Greys", alpha=0.3)
        im3 = axes[1, 0].imshow(oco_masked, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1, 0].set_title("Stacked OCO-2 Observations", fontsize=12)
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

        residual = pred - gt
        max_err = max(float(np.abs(residual).max()), 0.1)

        im4 = axes[1, 1].imshow(residual, cmap="RdBu_r", vmin=-max_err, vmax=max_err)
        axes[1, 1].set_title("Residual (Pred - GT)", fontsize=12)
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

        im5 = axes[1, 2].imshow(std_map, cmap="magma")
        axes[1, 2].set_title("Uncertainty Map (Std Dev)", fontsize=12)
        plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        save_path = os.path.join(out_dir, f"ep{epoch}_idx{idx}.png")
        plt.savefig(save_path, dpi=120)

        fig.clf()
        plt.close(fig)
        plt.close("all")


def setup_region_logger(region, window_size, aux_lag_hours, timestamp):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)

    for h in logger.handlers[:]:
        logger.removeHandler(h)
        h.close()

    logger.setLevel(logging.INFO)

    log_path = LOG_DIR / (
        f"run_window_rd_sim_{region}_{window_size}_"
        f"innovation_obs_era5_flux_aux_lag{aux_lag_hours}h_"
        f"{timestamp}.log"
    )

    file_handler = logging.FileHandler(log_path)
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def run_one_region(REGION):
    set_global_seed(TRAIN_SEED)
    WINDOW_SIZE = 0
    AUX_LAG_HOURS = 3

    paths = get_region_paths(
        REGION,
        aux_lag_hours=AUX_LAG_HOURS,
        window_size=WINDOW_SIZE,
        seed=TRAIN_SEED,
    )
    CACHE_DIR = paths["CACHE_DIR"]
    STATS_JSON = paths["STATS_JSON"]
    ERA5_PRESSURE_DIR = paths["ERA5_PRESSURE_DIR"]
    ERA5_SINGLE_DIR = paths["ERA5_SINGLE_DIR"]
    ERA5_STATS_JSON = paths["ERA5_STATS_JSON"]
    SURFACE_FLUX_DIR = paths["SURFACE_FLUX_DIR"]
    SURFACE_FLUX_STATS_JSON = paths["SURFACE_FLUX_STATS_JSON"]
    MODEL_PATH = paths["MODEL_PATH"]
    SURFACE_FLUX_VAR = "flux_apri"

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_region_logger(REGION, WINDOW_SIZE, AUX_LAG_HOURS, timestamp)

    shift_x = 0
    shift_y = 0
    rotate_k = 0

    logger.info("=" * 100)
    logger.info(f"Starting region: {REGION}")
    logger.info(f"WINDOW_SIZE: {WINDOW_SIZE}")
    logger.info(f"AUX_LAG_HOURS: {AUX_LAG_HOURS}")
    logger.info(f"CACHE_DIR: {CACHE_DIR}")
    logger.info(f"STATS_JSON: {STATS_JSON}")
    logger.info(f"ERA5_PRESSURE_DIR: {ERA5_PRESSURE_DIR}")
    logger.info(f"ERA5_SINGLE_DIR: {ERA5_SINGLE_DIR}")
    logger.info(f"ERA5_STATS_JSON: {ERA5_STATS_JSON}")
    logger.info(f"SURFACE_FLUX_DIR: {SURFACE_FLUX_DIR}")
    logger.info(f"SURFACE_FLUX_STATS_JSON: {SURFACE_FLUX_STATS_JSON}")
    logger.info(f"SURFACE_FLUX_VAR: {SURFACE_FLUX_VAR}")
    logger.info(f"MODEL_PATH: {MODEL_PATH}")
    logger.info(f"TRAIN_SEED: {TRAIN_SEED}, VAL_SEED: {VAL_SEED}, TEST_SEED: {TEST_SEED}")
    logger.info("=" * 100)

    train_ds, val_ds, test_ds = None, None, None

    try:
        train_ds = XCO2FlowDatasetPT(
            CACHE_DIR, STATS_JSON, "train",
            filter_residual=False, window_size=WINDOW_SIZE, shift_x=shift_x, shift_y=shift_y, rotate_k=rotate_k,
            use_era5=True, era5_pressure_dir=ERA5_PRESSURE_DIR, era5_single_dir=ERA5_SINGLE_DIR, era5_region=REGION, era5_stats_path=ERA5_STATS_JSON,
            use_surface_flux=True, surface_flux_dir=SURFACE_FLUX_DIR, surface_flux_region=REGION, surface_flux_var=SURFACE_FLUX_VAR, surface_flux_stats_path=SURFACE_FLUX_STATS_JSON,
            aux_lag_hours=AUX_LAG_HOURS,obs_source="sim",split_csv=DATA_SPLIT_CSV,region=REGION
        )

        val_ds = XCO2FlowDatasetPT(
            CACHE_DIR, STATS_JSON, "val",
            filter_residual=False, window_size=WINDOW_SIZE, shift_x=shift_x, shift_y=shift_y, rotate_k=rotate_k,
            use_era5=True, era5_pressure_dir=ERA5_PRESSURE_DIR, era5_single_dir=ERA5_SINGLE_DIR, era5_region=REGION, era5_stats_path=ERA5_STATS_JSON,
            use_surface_flux=True, surface_flux_dir=SURFACE_FLUX_DIR, surface_flux_region=REGION, surface_flux_var=SURFACE_FLUX_VAR, surface_flux_stats_path=SURFACE_FLUX_STATS_JSON,
            aux_lag_hours=AUX_LAG_HOURS,obs_source="sim",split_csv=DATA_SPLIT_CSV,region=REGION
        )

        test_ds = XCO2FlowDatasetPT(
            CACHE_DIR, STATS_JSON, "test",
            filter_residual=False, window_size=WINDOW_SIZE, shift_x=shift_x, shift_y=shift_y, rotate_k=rotate_k,
            use_era5=True, era5_pressure_dir=ERA5_PRESSURE_DIR, era5_single_dir=ERA5_SINGLE_DIR, era5_region=REGION, era5_stats_path=ERA5_STATS_JSON,
            use_surface_flux=True, surface_flux_dir=SURFACE_FLUX_DIR, surface_flux_region=REGION, surface_flux_var=SURFACE_FLUX_VAR, surface_flux_stats_path=SURFACE_FLUX_STATS_JSON,
            aux_lag_hours=AUX_LAG_HOURS,obs_source="sim",split_csv=DATA_SPLIT_CSV,region=REGION
        )

        train_generator = torch.Generator().manual_seed(TRAIN_SEED)
        train_loader = DataLoader(
            train_ds, batch_size=32, shuffle=True, num_workers=8, pin_memory=True,
            worker_init_fn=seed_worker, generator=train_generator
        )
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
        logger.info(f"CSV dataset sizes | Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

        model = FlowUNet(
            state_channels=1,
            obs_channels=3,
            aux_channels=AUX_USED_CHANNELS,
            state_embed_channels=32,
            obs_embed_channels=32,
            aux_embed_channels=16,
            embed_dim=256,
        )

        trainer = FlowMatchingTrainer(
            model,
            STATS_JSON,
            plot_dir=PLOT_DIR / (
                f"plots_{REGION}_rd_attn_window_{WINDOW_SIZE}_"
                f"innovation_obs_era5_flux_aux_lag{AUX_LAG_HOURS}h_"
                f"{shift_x}_{shift_y}_{rotate_k}_seed{TRAIN_SEED}"
            ),
        )

        early_stopping = EarlyStopping(patience=6, delta=0.0001, monitor="R2")

        logger.info("Starting FlowDA training")
        logger.info(f"AUX_USED_NAMES: {AUX_USED_NAMES}")
        logger.info(f"AUX_USED_CHANNELS: {AUX_USED_CHANNELS}")

        logger.info("OBS_INPUT: [innovation=(y_oco-x_ct)*mask, y_oco*mask, mask]")

        for epoch in range(1, 101):
            loss = trainer.train_epoch(train_loader, epoch)
            logger.info(f"Epoch {epoch} | Train Loss: {loss:.6f}")

            if epoch % 5 == 0 or epoch == 1:
                metrics = trainer.evaluate(val_loader, epoch, mode="Val")
                logger.info(f">> [Val Evaluation] Epoch {epoch}")
                logger.info(
                    f"  [Global] R2: {metrics['Global_R2']:.4f}, "
                    f"RMSE: {metrics['Global_RMSE']:.4f}, MAE: {metrics['Global_MAE']:.4f}"
                )

                if "OCO_R2" in metrics:
                    logger.info(
                        f"  [Observed grids] R2: {metrics['OCO_R2']:.4f}, "
                        f"RMSE: {metrics['OCO_RMSE']:.4f}, MAE: {metrics['OCO_MAE']:.4f}"
                    )

                is_improved = early_stopping(metrics["Global_R2"])

                if is_improved:
                    torch.save(model.state_dict(), MODEL_PATH)
                    logger.info(f"*** Best Model Updated! Current Best Global R2: {early_stopping.best_score:.4f} ***")
                else:
                    logger.info(f"==> No improvement. EarlyStopping Counter: {early_stopping.counter}/{early_stopping.patience}")

                if early_stopping.early_stop:
                    logger.info("Early stopping triggered.")
                    break

        if os.path.exists(MODEL_PATH):
            logger.info("Loading Best Model checkpoint for Final Inference Test...")
            model.load_state_dict(torch.load(MODEL_PATH, map_location=trainer.device))

            test_metrics = trainer.evaluate(test_loader, epoch=0, mode="Test", num_plots=20)

            logger.info("\n================ Final Test Set Performance ================")
            logger.info(
                f"Global | R2: {test_metrics['Global_R2']:.4f}, "
                f"RMSE: {test_metrics['Global_RMSE']:.4f}, MAE: {test_metrics['Global_MAE']:.4f}"
            )

            if "OCO_R2" in test_metrics:
                logger.info(
                    f"Observed grids | R2: {test_metrics['OCO_R2']:.4f}, "
                    f"RMSE: {test_metrics['OCO_RMSE']:.4f}, MAE: {test_metrics['OCO_MAE']:.4f}"
                )

            logger.info("Calculating Spatial Uncertainty...")
            uncertainty_metrics = trainer.evaluate_uncertainty(test_loader)
            logger.info("Uncertainty Results:")
            logger.info(f"  Mean Std (Global): {uncertainty_metrics['std_global']:.4f} ppm")
            logger.info(f"  Mean Std (Masked Area): {uncertainty_metrics['std_mask']:.4f} ppm")
            logger.info(f"  Mean Std (Non-Masked Area): {uncertainty_metrics['std_nonmask']:.4f} ppm")
            logger.info("===============================================================")

        logger.info(f"[DONE] Region {REGION} finished.")

    except Exception as e:
        logger.exception(f"[FAILED] Region {REGION} failed: {repr(e)}")

    finally:
        if train_ds is not None and hasattr(train_ds, "close"):
            train_ds.close()
        if val_ds is not None and hasattr(val_ds, "close"):
            val_ds.close()
        if test_ds is not None and hasattr(test_ds, "close"):
            test_ds.close()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def make_uncertainty_test_dataset(region):
    """Build the temporally independent 2024 simulated test dataset."""
    paths = get_region_paths(region, aux_lag_hours=3, window_size=0, seed=TRAIN_SEED)

    return XCO2FlowDatasetPT(
        paths["CACHE_DIR"], paths["STATS_JSON"], "test",
        filter_residual=False, window_size=0, shift_x=0, shift_y=0, rotate_k=0,
        use_era5=True,
        era5_pressure_dir=paths["ERA5_PRESSURE_DIR"],
        era5_single_dir=paths["ERA5_SINGLE_DIR"],
        era5_region=region,
        era5_stats_path=paths["ERA5_STATS_JSON"],
        use_surface_flux=True,
        surface_flux_dir=paths["SURFACE_FLUX_DIR"],
        surface_flux_region=region,
        surface_flux_var="flux_apri",
        surface_flux_stats_path=paths["SURFACE_FLUX_STATS_JSON"],
        aux_lag_hours=3, obs_source="sim", split_csv=DATA_SPLIT_CSV, region=region
    )


def load_full_model_for_uncertainty(region):
    """Load an existing full-model checkpoint without training."""
    paths = get_region_paths(region, aux_lag_hours=3, window_size=0, seed=TRAIN_SEED)
    stats_json = paths["STATS_JSON"]
    model_path = paths["MODEL_PATH"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Full-model checkpoint not found: {model_path}")

    model = FlowUNet(
        state_channels=1, obs_channels=3, aux_channels=AUX_USED_CHANNELS,
        state_embed_channels=32, obs_embed_channels=32, aux_embed_channels=16,
        embed_dim=256
    )

    trainer = FlowMatchingTrainer(model, stats_json, sigma=0.05, plot_dir=UNCERTAINTY_OUT_DIR)
    state = torch.load(model_path, map_location=trainer.device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    trainer.model.load_state_dict(state)
    trainer.model.eval()
    print(f"[OK] loaded uncertainty model: {model_path}")
    return trainer


def run_uncertainty_region(region, modes=None, obs_sigma_ppm=OBS_PERTURB_SIGMA_PPM):
    """
    Run uncertainty-characteristics diagnostics for one region.
    Observation and joint modes are skipped when obs_sigma_ppm is not specified.
    """
    modes = UNCERTAINTY_MODES if modes is None else modes
    UNCERTAINTY_OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_global_seed(UNCERTAINTY_SEED)

    ds = make_uncertainty_test_dataset(region)
    loader = DataLoader(ds, batch_size=UNCERTAINTY_BATCH_SIZE, shuffle=False, num_workers=0)
    trainer = load_full_model_for_uncertainty(region)

    summary_rows, scene_tables = [], []

    print("=" * 110)
    print(f"[UNCERTAINTY] region={region.upper()} | N_test={len(ds)} | modes={modes}")
    print(f"[UNCERTAINTY] CT sigma={trainer.sigma:.4f} normalized = {trainer.sigma * trainer.std:.4f} ppm")
    print(f"[UNCERTAINTY] observation sigma={obs_sigma_ppm} ppm")
    print("=" * 110)

    for mode in modes:
        if mode in {"observation", "joint"} and obs_sigma_ppm is None:
            print(f"[SKIP] {region.upper()} | {mode}: OBS_PERTURB_SIGMA_PPM is not set.")
            continue

        metrics, scene_df = trainer.evaluate_uncertainty_source(
            loader=loader, region=region, mode=mode, obs_sigma_ppm=obs_sigma_ppm,
            num_samples=UNCERTAINTY_ENSEMBLE_SIZE, seed=UNCERTAINTY_SEED
        )

        summary_rows.append(metrics)
        scene_tables.append(scene_df)

        print("\n" + "-" * 110)
        for k, v in metrics.items():
            print(f"{k}: {v}")
        print("-" * 110)

    summary_df = pd.DataFrame(summary_rows)
    scene_df = pd.concat(scene_tables, ignore_index=True) if scene_tables else pd.DataFrame()

    summary_path = os.path.join(UNCERTAINTY_OUT_DIR, f"uncertainty_summary_{region}.csv")
    scene_path = os.path.join(UNCERTAINTY_OUT_DIR, f"uncertainty_scene_{region}.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    scene_df.to_csv(scene_path, index=False, encoding="utf-8-sig")

    print(f"[SAVED] {summary_path}")
    print(f"[SAVED] {scene_path}")

    if hasattr(ds, "close"):
        ds.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary_df, scene_df


def run_all_uncertainty():
    """Run uncertainty diagnostics for EA, EU, and NA and save merged tables."""
    all_summary, all_scene = [], []

    for region in ["ea", "eu", "na"]:
        summary_df, scene_df = run_uncertainty_region(region)
        if not summary_df.empty:
            all_summary.append(summary_df)
        if not scene_df.empty:
            all_scene.append(scene_df)

    summary_all = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    scene_all = pd.concat(all_scene, ignore_index=True) if all_scene else pd.DataFrame()

    UNCERTAINTY_OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = os.path.join(UNCERTAINTY_OUT_DIR, "uncertainty_summary_all_regions.csv")
    scene_path = os.path.join(UNCERTAINTY_OUT_DIR, "uncertainty_scene_all_regions.csv")
    summary_all.to_csv(summary_path, index=False, encoding="utf-8-sig")
    scene_all.to_csv(scene_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 110)
    print("UNCERTAINTY SUMMARY")
    print("=" * 110)
    print(summary_all.to_string(index=False) if not summary_all.empty else "No uncertainty results.")
    print(f"\n[SAVED] {summary_path}")
    print(f"[SAVED] {scene_path}")

    return summary_all, scene_all
if __name__ == "__main__":
    if RUN_TRAINING:
        for region in ["ea", "eu", "na"]:
            run_one_region(region)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if RUN_UNCERTAINTY:
        uncertainty_summary, uncertainty_scene = run_all_uncertainty()
