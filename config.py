from pathlib import Path
import os

# Repository root (the directory containing this file)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_STORAGE_ROOT = PROJECT_ROOT.parent  # FlowDA/ and data/ are siblings


def _env_path(name, default):
    """Return a path from an environment variable, or use a portable default."""
    value = os.environ.get(name)
    path = Path(value).expanduser() if value else Path(default)
    return path.resolve()


# -----------------------------------------------------------------------------
# Root directories
# -----------------------------------------------------------------------------
# Defaults assume FlowDA/ is next to data/, dataset/, and outputs/.
# Override these with environment variables on other systems:
#   FLOWDA_DATA_ROOT
#   FLOWDA_STATS_ROOT
#   FLOWDA_CHECKPOINT_ROOT
#   FLOWDA_OUTPUT_ROOT
DATA_ROOT = _env_path("FLOWDA_DATA_ROOT", DATA_STORAGE_ROOT / "data")
STATS_ROOT = _env_path("FLOWDA_STATS_ROOT", DATA_STORAGE_ROOT / "stats")
CHECKPOINT_ROOT = _env_path("FLOWDA_CHECKPOINT_ROOT", PROJECT_ROOT / "checkpoints")
OUTPUT_ROOT = _env_path("FLOWDA_OUTPUT_ROOT", DATA_STORAGE_ROOT / "outputs")

DATA_SPLIT_CSV = DATA_ROOT / "dataset_split.csv"
LOG_DIR = OUTPUT_ROOT / "logs"
PLOT_DIR = OUTPUT_ROOT / "plots"
UNCERTAINTY_OUT_DIR = OUTPUT_ROOT / "uncertainty"


def checkpoint_path(region, window_size=0, aux_lag_hours=3, seed=42):
    """Return the pretrained checkpoint path for a region."""
    region = region.lower()
    return CHECKPOINT_ROOT / f"best_model_{region}.pth"


def get_region_paths(region, aux_lag_hours=3, window_size=0, seed=42):
    """Build all region-specific input and checkpoint paths from the roots above."""
    region = region.lower()
    return {
        "CACHE_DIR": DATA_ROOT / f"cache_{region}_region_interpolated",
        "STATS_JSON": STATS_ROOT / f"data_stats_{region}_interpolated.json",
        "ERA5_PRESSURE_DIR": DATA_ROOT / f"era5_{region}" / "pressure_levels",
        "ERA5_SINGLE_DIR": DATA_ROOT / f"era5_{region}" / "single_levels",
        "ERA5_STATS_JSON": STATS_ROOT / f"era5_stats_{region}_lag{aux_lag_hours}h.json",
        "SURFACE_FLUX_DIR": DATA_ROOT / "CAMS_surface_flux" / region,
        "SURFACE_FLUX_STATS_JSON": STATS_ROOT / f"surface_flux_stats_{region}_lag{aux_lag_hours}h.json",
        "MODEL_PATH": checkpoint_path(
            region,
            window_size=window_size,
            aux_lag_hours=aux_lag_hours,
            seed=seed,
        ),
    }


def get_tccon_match_table(obs_source):
    obs_source = obs_source.lower()
    return DATA_ROOT / "tccon_statistics" / obs_source / "station_pt_tccon_samples.csv"


def get_tccon_output_dir(obs_source):
    obs_source = obs_source.lower()
    return OUTPUT_ROOT / "tccon" / obs_source


# -----------------------------------------------------------------------------
# GOSAT-2 independent validation (GOSAT-2 observations when OCO-2 is absent).
# These products are distinct from the OCO-2/3 training and evaluation cache.
# Each default can be overridden without changing the public source code.
# -----------------------------------------------------------------------------
GOSAT2_CACHE_ROOT = _env_path(
    "FLOWDA_GOSAT2_CACHE_ROOT", DATA_ROOT / "cache_gosat2_only_no_oco2"
)
GOSAT2_SPLIT_CSV = _env_path(
    "FLOWDA_GOSAT2_SPLIT_CSV", GOSAT2_CACHE_ROOT / "gosat2_only_no_oco2_split.csv"
)
GOSAT2_SURFACE_FLUX_ROOT = _env_path(
    "FLOWDA_GOSAT2_FLUX_ROOT", DATA_ROOT / "CAMS_surface_flux_gosat2_only"
)
GOSAT2_TCCON_MATCH_TABLE = _env_path(
    "FLOWDA_GOSAT2_MATCH_TABLE", get_tccon_match_table("gosat2")
)
GOSAT2_TCCON_OUT_DIR = _env_path(
    "FLOWDA_GOSAT2_OUTPUT_ROOT", get_tccon_output_dir("gosat2") / "full"
)


def get_gosat2_region_paths(region):
    """Reuse the same regional full-model/normalization statistics as OCO-2."""
    paths = get_region_paths(region, aux_lag_hours=3, window_size=0, seed=42)
    paths.update({
        "CACHE_DIR": GOSAT2_CACHE_ROOT,
        "SPLIT_CSV": GOSAT2_SPLIT_CSV,
        "SURFACE_FLUX_DIR": GOSAT2_SURFACE_FLUX_ROOT,
    })
    return paths
