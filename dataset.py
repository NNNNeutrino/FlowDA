import glob
import json
import logging
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import Dataset


warnings.filterwarnings(
    "ignore",
    message="Passing method to .*Index.get_loc is deprecated.*",
    category=FutureWarning,
)


def safe_nan_to_num(tensor, nan_val=0.0):
    """Replace NaN/Inf values in a tensor with a specified finite value."""
    if not isinstance(tensor, torch.Tensor):
        return tensor
    out = tensor.clone()
    out[torch.isnan(out)] = nan_val
    out[torch.isinf(out)] = nan_val
    return out


class XCO2FlowDatasetPT(Dataset):
    """
    Dataset used by FlowDA.

    CO2 fields are loaded from cached PT files. ERA5 and CAMS surface-flux
    auxiliary fields are read at ``timestamp - aux_lag_hours``.

    Observation sources:
        - "sim": simulated observations
        - "oco2": real OCO-2 observations
        - "oco3": real OCO-3 observations
        - "gosat2": real GOSAT-2 observations (independent validation)

    Returned sample:
        x_ct          : normalized CT2025 background, [1, H, W]
        z_cams        : normalized CAMS reference, [1, H, W]
        y_oco         : normalized observations on valid observed grids, [1, H, W]
        mask          : observation mask, [1, H, W]
        era5          : normalized ERA5 fields, [10, H, W]
        surface_flux  : normalized CAMS surface flux, [1, H, W]
        timestamp     : target time
        aux_timestamp : auxiliary-field time
        doy           : normalized day of year
        tod           : normalized time of day
        base_mean     : sample-specific CT mean used for CO2 normalization
    """

    ERA5_CHANNELS = ["u850", "v850", "u700", "v700", "u500", "v500", "u10", "v10", "blh", "sp"]

    def __init__(
        self,
        cache_root,
        stats_path,
        flag="train",
        window_size=2,
        filter_residual=False,
        max_residual_ppm=0.7,
        shift_x=0,
        shift_y=0,
        rotate_k=0,
        use_era5=True,
        era5_pressure_dir=None,
        era5_single_dir=None,
        era5_region="ea",
        era5_stats_path=None,
        use_surface_flux=True,
        surface_flux_dir=None,
        surface_flux_region="ea",
        surface_flux_var="flux_apri",
        surface_flux_stats_path=None,
        aux_lag_hours=3,
        obs_source="oco2",
        split_csv=None,
        region=None,
        auto_use_gosat2_paths=True,
        gosat2_cache_root=None,
        gosat2_split_csv=None,
        gosat2_surface_flux_dir=None,
    ):
        self.obs_source = obs_source.lower()
        if self.obs_source not in {"oco2", "oco3", "gosat2", "sim"}:
            raise ValueError(f"Unsupported obs_source: {self.obs_source!r}")
        self.obs_key = f"y_{self.obs_source}"

        # GOSAT-2 uses a dedicated cache/split/flux dataset; the original
        # regional CO2 and ERA5 normalization statistics are reused.
        # Defaults are resolved in config.py, never hard-coded here.
        if self.obs_source == "gosat2" and auto_use_gosat2_paths:
            from config import GOSAT2_CACHE_ROOT, GOSAT2_SPLIT_CSV, GOSAT2_SURFACE_FLUX_ROOT

            cache_root = gosat2_cache_root or GOSAT2_CACHE_ROOT
            split_csv = gosat2_split_csv or GOSAT2_SPLIT_CSV
            if use_surface_flux:
                surface_flux_dir = gosat2_surface_flux_dir or GOSAT2_SURFACE_FLUX_ROOT

        self.flag = flag.lower()
        self.cache_root = str(cache_root)
        self.data_dir = os.path.join(self.cache_root, self.flag)
        self.split_csv = str(split_csv) if split_csv is not None else None
        self.region = (region or era5_region).upper()

        # Retained for compatibility with existing training scripts.
        self.filter_residual = filter_residual
        self.max_residual_ppm = max_residual_ppm

        self.window_size = window_size
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.rotate_k = rotate_k
        self.aux_lag_hours = aux_lag_hours

        self.use_era5 = use_era5
        self.era5_pressure_dir = str(era5_pressure_dir) if era5_pressure_dir is not None else None
        self.era5_single_dir = str(era5_single_dir) if era5_single_dir is not None else None
        self.era5_region = era5_region
        self.era5_stats_path = str(era5_stats_path) if era5_stats_path is not None else None
        self.era5_cache = {}
        self.era5_channels = list(self.ERA5_CHANNELS)
        self.era5_stats = None

        self.use_surface_flux = use_surface_flux
        self.surface_flux_dir = str(surface_flux_dir) if surface_flux_dir is not None else None
        self.surface_flux_region = surface_flux_region
        self.surface_flux_var = surface_flux_var
        self.surface_flux_stats_path = str(surface_flux_stats_path) if surface_flux_stats_path is not None else None
        self.surface_flux_channels = [self.surface_flux_var]
        self.surface_flux_stats = None

        self._load_file_list()
        self._load_co2_stats(stats_path)
        self._load_era5_stats()
        self._load_surface_flux_stats()

        logging.info(
            "[Dataset] %s-%s | N=%d | window=%d | shift=(%d,%d) | rotate_k=%d | aux_lag=%dh | obs=%s",
            self.region, self.flag, len(self.files), self.window_size, self.shift_x, self.shift_y,
            self.rotate_k, self.aux_lag_hours, self.obs_source,
        )

    def __len__(self):
        return len(self.files)

    # ------------------------------------------------------------------
    # Dataset metadata and statistics
    # ------------------------------------------------------------------
    def _load_file_list(self):
        if self.split_csv is not None:
            if not os.path.exists(self.split_csv):
                raise FileNotFoundError(f"Dataset split CSV not found: {self.split_csv}")

            split_df = pd.read_csv(self.split_csv, dtype=str, keep_default_na=False)
            required_columns = {"region", "split", "filename", "full_path"}
            missing_columns = required_columns - set(split_df.columns)
            if missing_columns:
                raise ValueError(f"Dataset split CSV is missing columns: {sorted(missing_columns)}")

            split_df["region"] = split_df["region"].str.strip().str.upper()
            split_df["split"] = split_df["split"].str.strip().str.lower()
            split_df = split_df[(split_df["region"] == self.region) & (split_df["split"] == self.flag)].copy()

            if split_df.empty:
                raise ValueError(f"No samples found for region={self.region}, split={self.flag}")

            split_df = split_df.sort_values("filename").reset_index(drop=True)
            self.file_paths = [
                self._resolve_pt_path(filename, recorded_path)
                for filename, recorded_path in zip(split_df["filename"], split_df["full_path"])
            ]
        else:
            if not os.path.exists(self.data_dir):
                raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")
            self.file_paths = sorted(
                os.path.join(self.data_dir, name)
                for name in os.listdir(self.data_dir)
                if name.endswith(".pt")
            )

        missing_files = [path for path in self.file_paths if not os.path.isfile(path)]
        if missing_files:
            raise FileNotFoundError(f"{len(missing_files)} PT files are missing, e.g. {missing_files[0]}")

        self.files = [os.path.basename(path) for path in self.file_paths]
        if len(self.files) != len(set(self.files)):
            raise ValueError(f"Duplicate PT filenames found in {self.region}-{self.flag}")

        self.file_path_map = dict(zip(self.files, self.file_paths))
        self.file_set = set(self.files)

    def _resolve_pt_path(self, filename, recorded_path):
        """Relocate PT files without changing membership in the split CSV.

        This also supports GOSAT-2 caches organized into region/year folders.
        """
        root = Path(self.cache_root)
        original = Path(str(recorded_path).replace("\\", "/"))
        filename = Path(str(filename).replace("\\", "/")).name

        candidates = []
        if not original.is_absolute():
            candidates.append(root / original)

        # Retain the directory structure following the original cache root.
        if root.name in original.parts:
            index = len(original.parts) - 1 - original.parts[::-1].index(root.name)
            candidates.append(root.joinpath(*original.parts[index + 1:]))

        candidates.extend([
            root / self.flag / filename,
            root / self.region.lower() / self.flag / filename,
            root / self.region.lower() / filename,
            root / original.parent.name / filename,
            root / original.parent.parent.name / original.parent.name / filename,
            root / filename,
            original,  # Original server path, if it is still available.
        ])
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        return str(candidates[0])

    def _load_co2_stats(self, stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
        self.mean_gb = stats["mean_gb"]
        self.std_gb = stats["std_gb"]

    def _load_era5_stats(self):
        if not self.use_era5:
            return

        if self.era5_pressure_dir is None:
            raise ValueError("era5_pressure_dir is required when use_era5=True")
        if self.era5_single_dir is None:
            raise ValueError("era5_single_dir is required when use_era5=True")
        if self.era5_stats_path is None:
            raise ValueError("era5_stats_path is required when use_era5=True")

        with open(self.era5_stats_path, "r") as f:
            stats_json = json.load(f)
        self.era5_stats = stats_json.get("stats", stats_json)

        for channel in self.era5_channels:
            if channel not in self.era5_stats:
                raise KeyError(f"ERA5 stats missing channel '{channel}'. Available: {list(self.era5_stats)}")

    def _load_surface_flux_stats(self):
        if not self.use_surface_flux:
            return

        if self.surface_flux_dir is None:
            raise ValueError("surface_flux_dir is required when use_surface_flux=True")
        if self.surface_flux_stats_path is None:
            raise ValueError("surface_flux_stats_path is required when use_surface_flux=True")
        if not os.path.exists(self.surface_flux_stats_path):
            raise FileNotFoundError(f"Surface-flux stats file not found: {self.surface_flux_stats_path}")

        with open(self.surface_flux_stats_path, "r") as f:
            stats_json = json.load(f)
        self.surface_flux_stats = stats_json.get("stats", stats_json)

        if self.surface_flux_var not in self.surface_flux_stats and "surface_flux" not in self.surface_flux_stats:
            raise KeyError(
                f"Surface-flux stats missing '{self.surface_flux_var}' or 'surface_flux'. "
                f"Available: {list(self.surface_flux_stats)}"
            )

    # ------------------------------------------------------------------
    # Observation mask
    # ------------------------------------------------------------------
    def _apply_spatial_transform(self, mask):
        """Apply optional rotation/translation to the observation mask only."""
        if self.rotate_k != 0:
            mask = torch.rot90(mask, k=self.rotate_k, dims=(-2, -1))

        if self.shift_x != 0 or self.shift_y != 0:
            mask = torch.roll(mask, shifts=(self.shift_y, self.shift_x), dims=(-2, -1))

            if self.shift_y > 0:
                mask[:, :self.shift_y, :] = 0
            elif self.shift_y < 0:
                mask[:, self.shift_y:, :] = 0

            if self.shift_x > 0:
                mask[:, :, :self.shift_x] = 0
            elif self.shift_x < 0:
                mask[:, :, self.shift_x:] = 0

        return mask

    def _get_temporal_stack(self, current_ts):
        curr_dt = datetime.strptime(current_ts, "%Y%m%d_%H%M")
        mask_list = []

        for i in range(-self.window_size, self.window_size + 1):
            target_dt = curr_dt + timedelta(hours=i * 3)
            target_fn = f"ROI_{target_dt.strftime('%Y%m%d_%H%M')}.pt"
            if target_fn not in self.file_set:
                continue

            target_fp = self.file_path_map[target_fn]
            try:
                data = torch.load(target_fp, map_location="cpu")
                if self.obs_key not in data:
                    continue
                y = safe_nan_to_num(data[self.obs_key].float(), nan_val=0.0).squeeze()
                if y.ndim != 2:
                    raise ValueError(f"{target_fp}: {self.obs_key} must be 2-D, got {tuple(y.shape)}")
                mask_list.append((y > 0.1).float())
            except Exception as exc:
                logging.warning("Failed to read temporal mask: %s | %s", target_fp, exc)

        if not mask_list:
            return torch.zeros((60, 60), dtype=torch.float32)

        stacked_mask = (torch.stack(mask_list).sum(dim=0) > 0).float()
        return self._apply_spatial_transform(stacked_mask.unsqueeze(0)).squeeze(0)

    # ------------------------------------------------------------------
    # Shared NetCDF helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_time_name(ds):
        for name in ("valid_time", "time"):
            if name in ds.coords or name in ds.variables:
                return name
        raise KeyError(f"No 'time' or 'valid_time' found. Coordinates: {list(ds.coords)}")

    @staticmethod
    def _get_level_name(ds):
        for name in ("pressure_level", "level", "isobaricInhPa"):
            if name in ds.coords:
                return name
        raise KeyError(f"No pressure-level coordinate found. Coordinates: {list(ds.coords)}")

    @staticmethod
    def _resize_to_target(arr, h, w):
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2-D field after squeeze, got shape={arr.shape}")
        arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
        return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)

    # ------------------------------------------------------------------
    # ERA5
    # ------------------------------------------------------------------
    def _find_era5_file(self, folder, yyyymm, kind):
        if folder is None:
            raise ValueError(f"ERA5 {kind} directory is empty")

        if kind == "pressure":
            patterns = [
                f"era5_pressure_uv_1deg_{self.era5_region}_{yyyymm}.nc",
                f"*pressure*{self.era5_region}*{yyyymm}.nc",
                f"*uv*{self.era5_region}*{yyyymm}.nc",
                f"*{yyyymm}.nc",
            ]
        elif kind == "single":
            patterns = [
                f"era5_single_1deg_{self.era5_region}_{yyyymm}.nc",
                f"*single*{self.era5_region}*{yyyymm}.nc",
                f"*{yyyymm}.nc",
            ]
        else:
            raise ValueError(f"Unknown ERA5 file type: {kind}")

        for pattern in patterns:
            matches = sorted(glob.glob(os.path.join(folder, pattern)))
            if matches:
                return matches[0]

        raise FileNotFoundError(f"ERA5 {kind} file not found: folder={folder}, yyyymm={yyyymm}, region={self.era5_region}")

    def _open_nc(self, path):
        if path not in self.era5_cache:
            self.era5_cache[path] = xr.open_dataset(path)
        return self.era5_cache[path]

    @staticmethod
    def _get_real_var_name(ds, var):
        aliases = {"u": ["u"], "v": ["v"], "u10": ["u10", "10u"], "v10": ["v10", "10v"], "blh": ["blh"], "sp": ["sp"]}
        for name in aliases.get(var, [var]):
            if name in ds.variables:
                return name
        raise KeyError(f"Variable '{var}' not found. Available: {list(ds.variables)}")

    def _normalize_era5(self, name, x):
        if self.era5_stats is None:
            return x
        if name not in self.era5_stats:
            raise KeyError(f"ERA5 stats missing '{name}'. Available: {list(self.era5_stats)}")

        mean = float(self.era5_stats[name]["mean"])
        std = float(self.era5_stats[name]["std"])
        if std < 1e-12:
            std = 1.0
        return (x - mean) / std

    def _get_era5_stack(self, ts_str, h, w):
        """Load normalized ERA5 fields at the supplied timestamp."""
        if not self.use_era5:
            return torch.empty((0, h, w), dtype=torch.float32)

        dt = datetime.strptime(ts_str, "%Y%m%d_%H%M")
        yyyymm = dt.strftime("%Y%m")

        ds_p = self._open_nc(self._find_era5_file(self.era5_pressure_dir, yyyymm, "pressure"))
        ds_s = self._open_nc(self._find_era5_file(self.era5_single_dir, yyyymm, "single"))

        time_p = self._get_time_name(ds_p)
        time_s = self._get_time_name(ds_s)
        level_name = self._get_level_name(ds_p)

        ds_p_t = ds_p.sel({time_p: np.datetime64(dt)}, method="nearest")
        ds_s_t = ds_s.sel({time_s: np.datetime64(dt)}, method="nearest")

        fields = []
        u_var = self._get_real_var_name(ds_p_t, "u")
        v_var = self._get_real_var_name(ds_p_t, "v")

        for lev in (850, 700, 500):
            u = ds_p_t[u_var].sel({level_name: lev}, method="nearest").values
            v = ds_p_t[v_var].sel({level_name: lev}, method="nearest").values
            fields.extend([
                self._normalize_era5(f"u{lev}", self._resize_to_target(u, h, w)),
                self._normalize_era5(f"v{lev}", self._resize_to_target(v, h, w)),
            ])

        for var in ("u10", "v10", "blh", "sp"):
            real_var = self._get_real_var_name(ds_s_t, var)
            x = self._resize_to_target(ds_s_t[real_var].values, h, w)
            fields.append(self._normalize_era5(var, x))

        return torch.stack(fields, dim=0).float()

    # ------------------------------------------------------------------
    # CAMS surface flux
    # ------------------------------------------------------------------
    def _find_surface_flux_file(self, ts_str):
        if self.surface_flux_dir is None:
            raise ValueError("surface_flux_dir is empty")

        candidate_dirs = [self.surface_flux_dir]
        region_subdir = os.path.join(self.surface_flux_dir, self.surface_flux_region)
        if os.path.isdir(region_subdir):
            candidate_dirs.append(region_subdir)

        patterns = [f"CAMS_SurfaceFlux_{ts_str}.nc", f"*SurfaceFlux*{ts_str}*.nc", f"*{ts_str}*.nc"]
        for folder in candidate_dirs:
            for pattern in patterns:
                matches = sorted(glob.glob(os.path.join(folder, pattern)))
                if matches:
                    return matches[0]

        raise FileNotFoundError(
            f"Surface-flux file not found: ts={ts_str}, dir={self.surface_flux_dir}, region={self.surface_flux_region}"
        )

    def _get_surface_flux_var_name(self, ds):
        candidates = [self.surface_flux_var, "surface_flux", "flux_apri", "flux_apos", "flux_foss"]
        for name in candidates:
            if name in ds.variables:
                return name
        raise KeyError(f"Surface-flux variable '{self.surface_flux_var}' not found. Available: {list(ds.variables)}")

    def _normalize_surface_flux(self, name, x):
        if self.surface_flux_stats is None:
            raise ValueError("surface_flux_stats is empty")

        if name in self.surface_flux_stats:
            stats_key = name
        elif self.surface_flux_var in self.surface_flux_stats:
            stats_key = self.surface_flux_var
        elif "surface_flux" in self.surface_flux_stats:
            stats_key = "surface_flux"
        else:
            raise KeyError(
                f"Surface-flux stats missing '{name}' or '{self.surface_flux_var}'. "
                f"Available: {list(self.surface_flux_stats)}"
            )

        mean = float(self.surface_flux_stats[stats_key]["mean"])
        std = float(self.surface_flux_stats[stats_key]["std"])
        if std < 1e-20:
            std = 1.0
        return (x - mean) / std

    def _get_surface_flux_stack(self, ts_str, h, w):
        """Load normalized CAMS surface flux at the supplied timestamp."""
        if not self.use_surface_flux:
            return torch.empty((0, h, w), dtype=torch.float32)

        dt = datetime.strptime(ts_str, "%Y%m%d_%H%M")
        flux_fp = self._find_surface_flux_file(ts_str)

        with xr.open_dataset(flux_fp) as ds:
            real_var = self._get_surface_flux_var_name(ds)
            da = ds[real_var]

            if "time" in da.dims:
                if "time" in ds.coords:
                    try:
                        da = da.sel(time=np.datetime64(dt), method="nearest")
                    except Exception:
                        da = da.isel(time=0)
                else:
                    da = da.isel(time=0)

            x = self._resize_to_target(da.values, h, w)
            x = self._normalize_surface_flux(real_var, x)

        return x.unsqueeze(0).float()

    # ------------------------------------------------------------------
    # Sample construction
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = torch.load(file_path, map_location="cpu")

        x_ct = safe_nan_to_num(data["x_ct"].float(), nan_val=self.mean_gb).squeeze()
        z_cams = safe_nan_to_num(data["z_cams"].float(), nan_val=self.mean_gb).squeeze()
        if x_ct.ndim != 2 or z_cams.ndim != 2:
            raise ValueError(f"CT/CAMS fields must be 2-D: {file_path}")

        if self.obs_key not in data:
            raise KeyError(f"Sample is missing '{self.obs_key}'. Available: {list(data.keys())}")

        y_oco_raw = safe_nan_to_num(data[self.obs_key].float(), nan_val=0.0).squeeze()
        if y_oco_raw.shape != x_ct.shape:
            raise ValueError(f"{self.obs_key} shape does not match CT in {file_path}")
        ts_str = data["timestamp"]
        h, w = x_ct.shape[-2:]

        dt = datetime.strptime(ts_str, "%Y%m%d_%H%M")
        aux_dt = dt - timedelta(hours=self.aux_lag_hours)
        aux_ts_str = aux_dt.strftime("%Y%m%d_%H%M")

        mask_final = self._get_temporal_stack(ts_str)
        y_oco_new = mask_final * y_oco_raw

        current_base_mean = x_ct.mean().item()
        x_ct_norm = (x_ct.unsqueeze(0) - current_base_mean) / self.std_gb
        z_cams_norm = (z_cams.unsqueeze(0) - current_base_mean) / self.std_gb

        y_oco_norm = torch.zeros_like(y_oco_new.unsqueeze(0))
        mask_bool = mask_final > 0.5
        if mask_bool.any():
            y_oco_norm[0, mask_bool] = (y_oco_new[mask_bool] - current_base_mean) / self.std_gb

        doy = torch.tensor((dt.timetuple().tm_yday - 1) / 365.0, dtype=torch.float32)
        tod = torch.tensor((dt.hour + dt.minute / 60.0) / 24.0, dtype=torch.float32)

        era5 = self._get_era5_stack(aux_ts_str, h, w)
        surface_flux = self._get_surface_flux_stack(aux_ts_str, h, w)

        return {
            "x_ct": x_ct_norm,
            "z_cams": z_cams_norm,
            "y_oco": y_oco_norm,
            "mask": mask_final.unsqueeze(0),
            "era5": era5,
            "surface_flux": surface_flux,
            "timestamp": ts_str,
            "aux_timestamp": aux_ts_str,
            "doy": doy,
            "tod": tod,
            "base_mean": current_base_mean,
        }

    def close(self):
        for ds in self.era5_cache.values():
            try:
                ds.close()
            except Exception:
                pass
        self.era5_cache.clear()
