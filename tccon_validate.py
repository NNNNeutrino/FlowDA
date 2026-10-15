import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data._utils.collate import default_collate

from dataset import XCO2FlowDatasetPT
from model import FlowUNet
from config import (
    DATA_SPLIT_CSV,
    get_region_paths,
    get_tccon_match_table,
    get_tccon_output_dir,
    GOSAT2_SPLIT_CSV,
    GOSAT2_TCCON_MATCH_TABLE,
    GOSAT2_TCCON_OUT_DIR,
    get_gosat2_region_paths,
)
from main import (
    AUX_USED_CHANNELS,
    TRAIN_SEED,
    TCCON_SEED,
    FlowMatchingTrainer,
    build_aux_fields,
    make_torch_generator,
    set_global_seed,
)


OBS_SOURCE = "oco2"  # "oco2", "oco3", or "gosat2"
TCCON_MATCH_TABLE = get_tccon_match_table(OBS_SOURCE)
TCCON_VALIDATE_OUT_DIR = get_tccon_output_dir(OBS_SOURCE)
TCCON_VALIDATE_OUT_DIR.mkdir(parents=True, exist_ok=True)

REGIONS_TO_VALIDATE = ["ea", "eu", "na"]
TCCON_WINDOW_SIZE = 0
TCCON_AUX_LAG_HOURS = 3
TCCON_BATCH_SIZE = 16
TCCON_SAMPLE_STEPS = 50
MAX_FIELD_PLOTS_PER_REGION = 20
TCCON_ENSEMBLE_SIZE = 10
TCCON_EXTRACT_METHOD = "nearest"
GOSAT2_TCCON_SEED = 2033  # Independent test-time generator seed


def get_tccon_region_paths(region):
    if OBS_SOURCE == "gosat2":
        return get_gosat2_region_paths(region)
    return get_region_paths(
        region,
        aux_lag_hours=TCCON_AUX_LAG_HOURS,
        window_size=TCCON_WINDOW_SIZE,
        seed=TRAIN_SEED,
    )


def make_tccon_dataset(region, split):
    p = get_tccon_region_paths(region)

    return XCO2FlowDatasetPT(
        p["CACHE_DIR"], p["STATS_JSON"], split,
        filter_residual=False, window_size=TCCON_WINDOW_SIZE,
        shift_x=0, shift_y=0, rotate_k=0,
        use_era5=True, era5_pressure_dir=p["ERA5_PRESSURE_DIR"], era5_single_dir=p["ERA5_SINGLE_DIR"],
        era5_region=region, era5_stats_path=p["ERA5_STATS_JSON"],
        use_surface_flux=True, surface_flux_dir=p["SURFACE_FLUX_DIR"],
        surface_flux_region=region, surface_flux_var="flux_apri",
        surface_flux_stats_path=p["SURFACE_FLUX_STATS_JSON"],
        aux_lag_hours=TCCON_AUX_LAG_HOURS, obs_source=OBS_SOURCE,
        split_csv=GOSAT2_SPLIT_CSV if OBS_SOURCE == "gosat2" else DATA_SPLIT_CSV, region=region
    )


def make_tccon_trainer(region):
    p = get_tccon_region_paths(region)

    if not os.path.exists(p["MODEL_PATH"]):
        raise FileNotFoundError(f"Model checkpoint not found: {p['MODEL_PATH']}")

    model = FlowUNet(
        state_channels=1,
        obs_channels=3,
        aux_channels=AUX_USED_CHANNELS,
        state_embed_channels=32,
        obs_embed_channels=32,
        aux_embed_channels=16,
        embed_dim=256,
    )

    trainer = FlowMatchingTrainer(model, p["STATS_JSON"], plot_dir=TCCON_VALIDATE_OUT_DIR,)

    state = torch.load(p["MODEL_PATH"], map_location=trainer.device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    trainer.model.load_state_dict(state)
    trainer.model.eval()

    print(f"[OK] loaded model: {p['MODEL_PATH']}")

    return trainer


def infer_split_from_cache_path(path):
    # Normalize Windows and Unix separators before checking the split name.
    path = str(path).replace("\\", "/")

    if "/train/" in path:
        return "train"
    if "/val/" in path:
        return "val"
    if "/test/" in path:
        return "test"

    return None


def normalize_tccon_match_table(df):
    """Normalize independent TCCON collocations for the chosen satellite.

    GOSAT-2 uses its OWN split CSV, never the OCO-2/3 split or recorded
    machine-specific cache paths. The GOSAT-only/no-OCO2 selection is
    represented by the dedicated cache and checked against optional counts.
    """
    df = df.copy()
    required = {"region", "station", "cache_time", "cache_path",
                "nearest_tccon_time", "nearest_tccon_xco2", "grid_iy", "grid_ix"}
    if OBS_SOURCE != "gosat2":
        required.update({"region_name", "oco_group"})
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"TCCON match table missing columns: {sorted(missing)}")

    df["region"] = df["region"].astype(str).str.strip().str.lower()
    df["station"] = df["station"].astype(str).str.strip()
    df["cache_path"] = df["cache_path"].astype(str).str.strip()
    df["filename"] = df["cache_path"].str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[-1]
    df["region_name"] = df["region"].map({"ea": "EA", "eu": "EU", "na": "NA"})
    df["cache_time"] = pd.to_datetime(df["cache_time"], errors="coerce")
    df["nearest_tccon_time"] = pd.to_datetime(df["nearest_tccon_time"], errors="coerce")
    for col in ("nearest_tccon_xco2", "grid_iy", "grid_ix"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    valid = (df["cache_time"].notna() & df["nearest_tccon_time"].notna()
             & np.isfinite(df["nearest_tccon_xco2"])
             & np.isfinite(df["grid_iy"]) & np.isfinite(df["grid_ix"])
             & df["region"].isin(REGIONS_TO_VALIDATE))
    df = df.loc[valid].copy()
    df[["grid_iy", "grid_ix"]] = df[["grid_iy", "grid_ix"]].astype(int)
    if ((df["grid_iy"] < 0) | (df["grid_iy"] >= 60) |
        (df["grid_ix"] < 0) | (df["grid_ix"] >= 60)).any():
        raise ValueError("TCCON match table has grid indices outside the 60x60 region")

    if OBS_SOURCE == "gosat2":
        split = pd.read_csv(GOSAT2_SPLIT_CSV, dtype=str, keep_default_na=False)
        required_split = {"region", "split", "filename"}
        if required_split - set(split.columns):
            raise KeyError(f"GOSAT-2 split CSV missing: {sorted(required_split - set(split.columns))}")
        split["region"] = split["region"].str.strip().str.lower()
        split["filename"] = split["filename"].str.strip()
        split["split"] = split["split"].str.strip().str.lower()
        if split.duplicated(["region", "filename"]).any():
            raise ValueError("Duplicate region/filename entries in GOSAT-2 split CSV")
        df = df.drop(columns="split", errors="ignore").merge(
            split[["region", "filename", "split"]], on=["region", "filename"], how="left", validate="many_to_one")
        if df["split"].isna().any() or not df["split"].isin(("train", "val", "test")).all():
            raise ValueError("Some GOSAT-2 TCCON samples are missing from the dedicated split CSV")
        # Only use GOSAT-2-present / OCO-2-absent collocations.
        if "gosat2_obs_count" in df and not (pd.to_numeric(df["gosat2_obs_count"], errors="coerce") > 0).all():
            raise ValueError("GOSAT-2 match table contains samples without GOSAT-2 observations")
        if "oco2_obs_count" in df and not (pd.to_numeric(df["oco2_obs_count"], errors="coerce") == 0).all():
            raise ValueError("GOSAT-2 match table contains samples with OCO-2 observations")
    else:
        if "split" not in df:
            df["split"] = df["cache_path"].apply(infer_split_from_cache_path)
        else:
            df["split"] = df["split"].fillna(df["cache_path"].apply(infer_split_from_cache_path))
        df["split"] = df["split"].astype(str).str.lower()
        df["oco_group"] = df["oco_group"].astype(str).str.lower().str.strip()
        df = df[df["split"].isin(("train", "val", "test"))].copy()

    return df.sort_values(["region", "split", "cache_time", "station"]).reset_index(drop=True)


def classify_observation_constraint(mask, iy, ix):
    """Direct / neighboring (3x3) / no-local category from the source mask."""
    if mask[iy, ix] > 0.5:
        return "direct"
    h, w = mask.shape
    return ("neighboring" if np.any(mask[max(0, iy-1):min(h, iy+2),
                                          max(0, ix-1):min(w, ix+2)] > 0.5)
            else "no_local")


def extract_tccon_station_value(field_2d, iy, ix):
    field_2d = np.asarray(field_2d).astype(float)

    h, w = field_2d.shape
    iy = int(np.clip(int(iy), 0, h - 1))
    ix = int(np.clip(int(ix), 0, w - 1))

    if TCCON_EXTRACT_METHOD == "nearest":
        return float(field_2d[iy, ix])

    if TCCON_EXTRACT_METHOD == "grid3_mean":
        y0 = max(0, iy - 1)
        y1 = min(h, iy + 2)
        x0 = max(0, ix - 1)
        x1 = min(w, ix + 2)
        return float(np.nanmean(field_2d[y0:y1, x0:x1]))

    raise ValueError(f"Unknown TCCON_EXTRACT_METHOD: {TCCON_EXTRACT_METHOD}")


def denormalize_aux_field(field, stats, key):
    """Denormalize an auxiliary field using training-set statistics."""
    if key not in stats:
        raise KeyError(
            f"Auxiliary statistics do not contain {key}. Available: {list(stats.keys())}"
        )

    mean = float(stats[key]["mean"])
    std = float(stats[key]["std"])

    return np.asarray(field, dtype=float) * std + mean


def denormalize_surface_flux(field, stats):
    """Denormalize surface flux with either supported statistics key."""
    if "flux_apri" in stats:
        key = "flux_apri"
    elif "surface_flux" in stats:
        key = "surface_flux"
    else:
        raise KeyError(
            "Surface-flux statistics contain neither flux_apri nor surface_flux. "
            f"Available: {list(stats.keys())}"
        )

    return denormalize_aux_field(field, stats, key)


def save_tccon_field_plot(
    ct, cams, gen, oco, mask, era5_norm, flux_norm, era5_stats, flux_stats, region, target_time, aux_time,
    obs_source, plot_index):
    """
    Save spatial fields for real OCO-2/OCO-3 input.

    ERA5 channel order: u850, v850, u700, v700, u500, v500,
    u10, v10, blh, sp.
    """

    out_dir = os.path.join(TCCON_VALIDATE_OUT_DIR, "field_plots", region)
    os.makedirs(out_dir, exist_ok=True)

    # =====================================================
    # =====================================================
    u10 = denormalize_aux_field(era5_norm[6],era5_stats, "u10")

    v10 = denormalize_aux_field(era5_norm[7], era5_stats, "v10")

    surface_pressure = denormalize_aux_field(era5_norm[9], era5_stats, "sp")

    surface_pressure_hpa = surface_pressure / 100.0

    surface_flux = denormalize_surface_flux(flux_norm[0], flux_stats,)

    # =====================================================
    # =====================================================
    valid_oco = oco[mask > 0.5]

    co2_arrays = [ct, cams, gen]

    if valid_oco.size > 0:
        co2_arrays.append(valid_oco)

    co2_vmin = min(float(np.nanmin(arr)) for arr in co2_arrays)
    co2_vmax = max(float(np.nanmax(arr)) for arr in co2_arrays)

    increment = gen - ct
    gen_minus_cams = gen - cams

    increment_limit = max(float(np.nanmax(np.abs(increment))), 0.1)

    cams_difference_limit = max(float(np.nanmax(np.abs(gen_minus_cams))), 0.1)

    flux_limit = max(float(np.nanmax(np.abs(surface_flux))), 1e-20)

    # =====================================================
    # =====================================================
    fig, axes = plt.subplots(3, 4, figsize=(20, 14),)
    fig.patch.set_alpha(0.0)
    axes = axes.ravel()
    im0 = axes[0].imshow(ct, cmap="viridis", vmin=co2_vmin, vmax=co2_vmax)
    axes[0].set_title("CT2025 Background")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="XCO$_2$ (ppm)")

    im1 = axes[1].imshow(cams, cmap="viridis", vmin=co2_vmin, vmax=co2_vmax)
    axes[1].set_title("CAMS Reference")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="XCO$_2$ (ppm)")

    im2 = axes[2].imshow(gen, cmap="viridis", vmin=co2_vmin, vmax=co2_vmax)
    axes[2].set_title("Generated XCO$_2$ Analysis")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="XCO$_2$ (ppm)")

    oco_masked = np.ma.masked_where(mask <= 0.5, oco)
    axes[3].imshow(ct, cmap="Greys", alpha=0.25)
    im3 = axes[3].imshow(oco_masked, cmap="viridis", vmin=co2_vmin, vmax=co2_vmax)
    axes[3].set_title(f"Real {obs_source.upper()} Observations")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04, label="XCO$_2$ (ppm)")

    im4 = axes[4].imshow(increment, cmap="RdBu_r", vmin=-increment_limit, vmax=increment_limit)
    axes[4].set_title("Analysis Increment (Gen $-$ CT)")
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04, label="Difference (ppm)")

    im5 = axes[5].imshow(gen_minus_cams, cmap="RdBu_r",
                        vmin=-cams_difference_limit, vmax=cams_difference_limit)
    axes[5].set_title("Difference (Gen $-$ CAMS)")
    plt.colorbar(im5, ax=axes[5], fraction=0.046, pad=0.04, label="Difference (ppm)")

    im6 = axes[6].imshow(u10, cmap="RdBu_r")
    axes[6].set_title("ERA5 10-m Zonal Wind ($u_{10}$)")
    plt.colorbar(im6, ax=axes[6], fraction=0.046, pad=0.04, label="m s$^{-1}$")

    im7 = axes[7].imshow(v10, cmap="RdBu_r")
    axes[7].set_title("ERA5 10-m Meridional Wind ($v_{10}$)")
    plt.colorbar(im7, ax=axes[7], fraction=0.046, pad=0.04, label="m s$^{-1}$")

    im8 = axes[8].imshow(surface_pressure_hpa, cmap="viridis")
    axes[8].set_title("ERA5 Surface Pressure")
    plt.colorbar(im8, ax=axes[8], fraction=0.046, pad=0.04, label="hPa")

    im9 = axes[9].imshow(surface_flux, cmap="RdBu_r", vmin=-flux_limit, vmax=flux_limit)
    axes[9].set_title("CAMS Prior Surface CO$_2$ Flux")
    plt.colorbar(im9, ax=axes[9], fraction=0.046, pad=0.04, label="Native flux unit")

    im10 = axes[10].imshow(mask, cmap="binary", vmin=0, vmax=1)
    axes[10].set_title(f"{obs_source.upper()} Observation Mask")
    plt.colorbar(im10, ax=axes[10], fraction=0.046, pad=0.04)

    axes[11].axis("off")

    for ax in axes[:11]:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{region.upper()} | {obs_source.upper()} | target: {target_time} | auxiliary: {aux_time}", fontsize=15)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    safe_target_time = str(target_time).replace(":", "").replace(" ", "_")
    save_path = os.path.join(out_dir, f"{obs_source}_{region}_{safe_target_time}_{plot_index:03d}.png")

    plt.savefig(save_path, dpi=200, bbox_inches="tight",transparent=True)
    
    plt.close(fig)

    print(f"[FIGURE SAVED] {save_path}")


def run_tccon_region_split(region, split, df_sub, plot_count_start=0):
    print("=" * 100)
    print(f"[TCCON RUN] region={region}, split={split}, rows={len(df_sub)}, unique_pt={df_sub['cache_path'].nunique()}")
    print("=" * 100)

    ds = make_tccon_dataset(region, split)
    trainer = make_tccon_trainer(region)
    device = trainer.device

    file_to_idx = {fn: i for i, fn in enumerate(ds.files)}

    unique_paths = sorted(df_sub["cache_path"].astype(str).unique().tolist())
    path_to_rows = {p: df_sub[df_sub["cache_path"].astype(str) == p].copy() for p in unique_paths}

    valid_items = []
    failed_rows = []

    for p in unique_paths:
        basename = os.path.basename(p.replace("\\", "/"))

        if basename not in file_to_idx:
            failed_rows.append({
                "region": region,
                "split": split,
                "cache_path": p,
                "reason": f"{basename} not found in Dataset files",
            })
            continue

        valid_items.append((p, file_to_idx[basename]))

    print(f"[INFO] valid_unique_pt={len(valid_items)}, failed_unique_pt={len(failed_rows)}")

    out_rows = []
    plot_count = plot_count_start
    seed = TCCON_SEED
    if OBS_SOURCE == "gosat2":
        seed = (GOSAT2_TCCON_SEED + {"ea": 0, "eu": 100000, "na": 200000}[region]
                + {"train": 0, "val": 10000, "test": 20000}[split])
    tccon_generator = make_torch_generator(device, seed)

    for start in range(0, len(valid_items), TCCON_BATCH_SIZE):
        chunk = valid_items[start:start + TCCON_BATCH_SIZE]

        samples = []
        chunk_paths = []

        for p, idx in chunk:
            try:
                sample = ds[idx]
                if OBS_SOURCE == "gosat2" and not bool((sample["mask"] > 0.5).any()):
                    raise ValueError("GOSAT-2 sample has no valid observations")
                samples.append(sample)
                chunk_paths.append(p)
            except Exception as e:
                failed_rows.append({
                    "region": region,
                    "split": split,
                    "cache_path": p,
                    "reason": f"Dataset __getitem__ failed: {repr(e)}",
                })

        if not samples:
            continue

        batch = default_collate(samples)

        x_ct = batch["x_ct"].to(device).float()
        x1 = batch["z_cams"].to(device).float()
        y_oco = batch["y_oco"].to(device).float()
        mask = batch["mask"].to(device).float()
        aux = build_aux_fields(batch, device)

        doy = batch["doy"].to(device).float()
        tod = batch["tod"].to(device).float()
        base_mean = batch["base_mean"].to(device).float()

        x_gen, x_gen_std = trainer.ensemble_sample(
            x_ct=x_ct, y_oco=y_oco, mask=mask, aux=aux, doy=doy, tod=tod,
            num_samples=TCCON_ENSEMBLE_SIZE, steps=TCCON_SAMPLE_STEPS, generator=tccon_generator
        )
        gen_ppm = trainer.denorm(x_gen, base_mean).detach().cpu().numpy()
        ct_ppm = trainer.denorm(x_ct, base_mean).detach().cpu().numpy()
        cams_ppm = trainer.denorm(x1, base_mean).detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()

        oco_ppm_raw = trainer.denorm(y_oco, base_mean).detach().cpu().numpy()

        oco_ppm = np.where(mask_np > 0.5, oco_ppm_raw, np.nan)

        era5_norm_np = batch["era5"].detach().cpu().numpy()
        flux_norm_np = batch["surface_flux"].detach().cpu().numpy()

        for b, p in enumerate(chunk_paths):
            gen_2d = gen_ppm[b, 0]
            ct_2d = ct_ppm[b, 0]
            cams_2d = cams_ppm[b, 0]

            oco_2d = oco_ppm[b, 0]
            mask_2d = mask_np[b, 0]

            target_time = batch["timestamp"][b]
            aux_time = batch["aux_timestamp"][b]

            station_rows = path_to_rows[p]

            if OBS_SOURCE != "gosat2" and plot_count < MAX_FIELD_PLOTS_PER_REGION:
                save_tccon_field_plot(
                    ct=ct_2d, cams=cams_2d, gen=gen_2d, oco=oco_2d, mask=mask_2d,
                    era5_norm=era5_norm_np[b], flux_norm=flux_norm_np[b],
                    era5_stats=ds.era5_stats, flux_stats=ds.surface_flux_stats,
                    region=region, target_time=target_time, aux_time=aux_time,
                    obs_source=ds.obs_source, plot_index=plot_count
                )

                plot_count += 1

            for _, r in station_rows.iterrows():
                iy = int(r["grid_iy"])
                ix = int(r["grid_ix"])

                gen_val = extract_tccon_station_value(gen_2d, iy, ix)
                ct_val = extract_tccon_station_value(ct_2d, iy, ix)
                cams_val = extract_tccon_station_value(cams_2d, iy, ix)
                tccon_val = float(r["nearest_tccon_xco2"])

                new_row = r.to_dict()

                new_row["gen_xco2_at_station"] = gen_val
                new_row["ct_xco2_at_station"] = ct_val
                new_row["cams_xco2_at_station"] = cams_val

                new_row["error_gen_minus_tccon"] = gen_val - tccon_val
                new_row["error_ct_minus_tccon"] = ct_val - tccon_val
                new_row["error_cams_minus_tccon"] = cams_val - tccon_val

                if OBS_SOURCE == "gosat2":
                    # Classification must use the actual GOSAT-2 observation mask.
                    m2d = mask_np[b, 0]
                    new_row["gosat2_group"] = classify_observation_constraint(m2d, iy, ix)
                    new_row["analysis_timestamp"] = str(batch["timestamp"][b])
                    new_row["flowda_spread_ppm_at_station"] = float(x_gen_std[b, 0, iy, ix] * trainer.std)
                    new_row["gosat2_xco2_at_station_if_direct"] = (
                        float(oco_ppm[b, 0, iy, ix]) if m2d[iy, ix] > 0.5 else np.nan)

                out_rows.append(new_row)

        done = min(start + TCCON_BATCH_SIZE, len(valid_items))
        print(f"[{region}-{split}] processed unique pt: {done}/{len(valid_items)}")

    if hasattr(ds, "close"):
        ds.close()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pd.DataFrame(out_rows), pd.DataFrame(failed_rows), plot_count


def calc_tccon_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    n = len(y_true)
    if not n:
        return {k: (0 if k == "N" else np.nan) for k in ("N", "Bias", "MAE", "RMSE", "R", "R2")}
    err = y_pred - y_true
    r = (float(np.corrcoef(y_true, y_pred)[0, 1]) if n >= 2 and
         np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12 else np.nan)
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {"N": n, "Bias": float(np.mean(err)), "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err**2))), "R": r,
            "R2": float(1 - np.sum(err**2) / ss_tot) if ss_tot > 1e-12 else np.nan}


def make_tccon_metrics_table(df, group_cols=None):
    if group_cols is None:
        group_cols = []

    pred_map = {
        "CT": "ct_xco2_at_station",
        "CAMS": "cams_xco2_at_station",
        "Gen": "gen_xco2_at_station",
    }

    rows = []

    if len(group_cols) == 0:
        grouped = [((), df)]
    else:
        grouped = list(df.groupby(group_cols))

    for keys, sub in grouped:
        if len(group_cols) == 0:
            group_info = {}
        else:
            if len(group_cols) == 1:
                keys = (keys,)
            group_info = dict(zip(group_cols, keys))

        for dataset_name, pred_col in pred_map.items():
            m = calc_tccon_metrics(sub["nearest_tccon_xco2"], sub[pred_col])
            rows.append({**group_info, "Dataset": dataset_name, **m})

    return pd.DataFrame(rows)


def add_tccon_gen_improvement_vs_ct(metrics_df):
    df = metrics_df.copy()

    base_cols = ["Dataset", "N", "Bias", "MAE", "RMSE", "R", "R2"]
    group_cols = [c for c in df.columns if c not in base_cols]

    rows = []

    if group_cols:
        grouped = list(df.groupby(group_cols, dropna=False))
    else:
        grouped = [((), df)]

    for _, sub in grouped:
        sub = sub.copy()

        ct = sub[sub["Dataset"] == "CT"]
        gen = sub[sub["Dataset"] == "Gen"]

        if len(ct) == 1 and len(gen) == 1:
            ct_rmse = float(ct["RMSE"].iloc[0])
            gen_rmse = float(gen["RMSE"].iloc[0])
            ct_mae = float(ct["MAE"].iloc[0])
            gen_mae = float(gen["MAE"].iloc[0])

            rmse_reduction = (ct_rmse - gen_rmse) / ct_rmse * 100 if ct_rmse > 1e-12 else np.nan
            mae_reduction = (ct_mae - gen_mae) / ct_mae * 100 if ct_mae > 1e-12 else np.nan
        else:
            rmse_reduction = np.nan
            mae_reduction = np.nan

        for _, r in sub.iterrows():
            rr = r.to_dict()
            rr["Gen_RMSE_reduction_vs_CT_pct"] = rmse_reduction if rr["Dataset"] == "Gen" else np.nan
            rr["Gen_MAE_reduction_vs_CT_pct"] = mae_reduction if rr["Dataset"] == "Gen" else np.nan
            rows.append(rr)

    return pd.DataFrame(rows)


def save_tccon_metrics(df_val):
    """Shared metrics for all satellite sources and observation-constraint groups."""
    group = "gosat2_group" if OBS_SOURCE == "gosat2" else "oco_group"
    result_dir = TCCON_VALIDATE_OUT_DIR
    by = {
        "metrics_all": [], "metrics_by_region": ["region_name"],
        "metrics_by_station": ["region_name", "station"],
        f"metrics_by_{group}": [group],
        f"metrics_by_region_{group}": ["region_name", group],
        f"metrics_by_station_{group}": ["region_name", "station", group],
    }
    for name, columns in by.items():
        table = add_tccon_gen_improvement_vs_ct(make_tccon_metrics_table(df_val, columns))
        path = os.path.join(result_dir, f"{name}.csv")
        table.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[SAVED] {path}")
        if not columns:
            print(table.to_string(index=False))


def run_tccon_validate_with_pth():
    set_global_seed(GOSAT2_TCCON_SEED if OBS_SOURCE == "gosat2" else TCCON_SEED)
    print("=" * 100)
    print("Start TCCON validation with saved PTH")
    print("=" * 100)
    print(f"TCCON_MATCH_TABLE: {TCCON_MATCH_TABLE}")
    print(f"TCCON_VALIDATE_OUT_DIR: {TCCON_VALIDATE_OUT_DIR}")
    print(f"REGIONS_TO_VALIDATE: {REGIONS_TO_VALIDATE}")
    print(f"TCCON_WINDOW_SIZE: {TCCON_WINDOW_SIZE}")
    print(f"TCCON_AUX_LAG_HOURS: {TCCON_AUX_LAG_HOURS}")
    print(f"TCCON_BATCH_SIZE: {TCCON_BATCH_SIZE}")
    print(f"TCCON_SAMPLE_STEPS: {TCCON_SAMPLE_STEPS}")
    print(f"TCCON_EXTRACT_METHOD: {TCCON_EXTRACT_METHOD}")
    print("=" * 100)

    if not os.path.exists(TCCON_MATCH_TABLE):
        raise FileNotFoundError(f"TCCON match table not found: {TCCON_MATCH_TABLE}")

    df_match = pd.read_csv(TCCON_MATCH_TABLE, keep_default_na=False)
    df_match = normalize_tccon_match_table(df_match)

    print(f"match table rows: {len(df_match)}")
    print(f"unique pt: {df_match['cache_path'].nunique()}")
    print(f"stations: {df_match['station'].nunique()}")
    if OBS_SOURCE != "gosat2":
        print(f"oco_group counts: {df_match['oco_group'].value_counts().to_dict()}")

    all_outputs = []
    all_failed = []

    for region in REGIONS_TO_VALIDATE:
        region_plot_count = 0

        for split in ["train", "val", "test"]:
            sub = df_match[(df_match["region"] == region) & (df_match["split"] == split)].copy()

            if sub.empty:
                continue

            out_df, failed_df, region_plot_count = run_tccon_region_split(
                region, split, sub, plot_count_start=region_plot_count
            )

            if not out_df.empty:
                all_outputs.append(out_df)

            if not failed_df.empty:
                all_failed.append(failed_df)

    if not all_outputs:
        raise RuntimeError("No TCCON validation results were produced. Check the match table, dataset files, and checkpoint paths.")

    df_val = pd.concat(all_outputs, ignore_index=True)
    df_val = df_val.sort_values(["region_name", "station", "cache_time"]).reset_index(drop=True)

    sample_name = "gosat2_tccon_validation_samples_with_predictions.csv" if OBS_SOURCE == "gosat2" else "tccon_validation_samples_with_predictions.csv"
    validation_samples_path = os.path.join(TCCON_VALIDATE_OUT_DIR, sample_name)
    df_val.to_csv(validation_samples_path, index=False, encoding="utf-8-sig")

    if all_failed:
        df_failed = pd.concat(all_failed, ignore_index=True)
    else:
        df_failed = pd.DataFrame(columns=["region", "split", "cache_path", "reason"])

    failed_path = os.path.join(TCCON_VALIDATE_OUT_DIR, "failed_pt_files.csv")
    df_failed.to_csv(failed_path, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("Saved sample-level validation table:")
    print(validation_samples_path)
    print(f"rows: {len(df_val)}")
    print(f"unique pt: {df_val['cache_path'].nunique()}")
    print(f"stations: {df_val['station'].nunique()}")
    group = "gosat2_group" if OBS_SOURCE == "gosat2" else "oco_group"
    print(f"{group} counts: {df_val[group].value_counts().to_dict()}")
    print("=" * 100)
    print(f"Failed PT table: {failed_path}, failed={len(df_failed)}")

    save_tccon_metrics(df_val)

    print("=" * 100)
    print("Done.")
    print("=" * 100)


def main_cli():
    import argparse
    parser = argparse.ArgumentParser(description="Independent TCCON validation for FlowDA")
    parser.add_argument("--source", choices=("oco2", "oco3", "gosat2"), default="oco2")
    args = parser.parse_args()
    global OBS_SOURCE, TCCON_MATCH_TABLE, TCCON_VALIDATE_OUT_DIR
    OBS_SOURCE = args.source
    TCCON_MATCH_TABLE = (GOSAT2_TCCON_MATCH_TABLE if OBS_SOURCE == "gosat2"
                         else get_tccon_match_table(OBS_SOURCE))
    TCCON_VALIDATE_OUT_DIR = (GOSAT2_TCCON_OUT_DIR if OBS_SOURCE == "gosat2"
                              else get_tccon_output_dir(OBS_SOURCE))
    TCCON_VALIDATE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    return run_tccon_validate_with_pth()


if __name__ == "__main__":
    main_cli()
