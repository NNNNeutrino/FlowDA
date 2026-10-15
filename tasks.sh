#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# 加载本地服务器的数据路径
if [[ -f ./run_local.sh ]]; then
    source ./run_local.sh
else
    echo "ERROR: run_local.sh not found!"
    exit 1
fi


# #!/usr/bin/env bash
# set -Eeuo pipefail

# # Run from the repository directory, regardless of the current working directory.
# cd "$(dirname "${BASH_SOURCE[0]}")"

# # Local server paths: optional, intentionally excluded from the GitHub repository.
# if [[ -f run_local.sh ]]; then
#     source ./run_local.sh
# fi

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-check}"

title() { printf '\n========== %s ==========\n' "$1"; }

check() {
    title "1. Python syntax"
    "$PYTHON_BIN" -m py_compile config.py dataset.py model.py main.py tccon_validate.py
    echo "[OK] All Python files compile."

    title "2. Imports, model forward, and path configuration"
    "$PYTHON_BIN" - <<'PY'
import torch
import config
import dataset
import model
import main
import tccon_validate

print("[OK] config / dataset / model / main / tccon_validate imports")

torch.set_num_threads(2)
network = model.FlowUNet().cpu().eval()
state = torch.zeros(1, 1, 16, 16)
mask = torch.zeros_like(state)
aux = torch.zeros(1, 4, 16, 16)
with torch.no_grad():
    prediction = network(state, state, state, mask,
                         torch.tensor([0.5]), torch.tensor([0.3]),
                         torch.tensor([0.5]), aux=aux)
assert prediction.shape == state.shape
print("[OK] FlowUNet forward pass:", tuple(prediction.shape))

paths = {
    "Dataset split": config.DATA_SPLIT_CSV,
    "OCO-2 TCCON": config.get_tccon_match_table("oco2"),
    "OCO-3 TCCON": config.get_tccon_match_table("oco3"),
    "GOSAT-2 split": config.GOSAT2_SPLIT_CSV,
    "GOSAT-2 TCCON": config.GOSAT2_TCCON_MATCH_TABLE,
}
for region in ("ea", "eu", "na"):
    p = config.get_region_paths(region)
    paths[f"{region.upper()} checkpoint"] = p["MODEL_PATH"]
    paths[f"{region.upper()} CO2 stats"] = p["STATS_JSON"]
for name, path in paths.items():
    print(f"[{'OK' if path.is_file() else 'MISSING'}] {name}: {path}")
print("[INFO] MISSING paths are reported; use 'smoke' or a validation mode for real-data testing.")
PY
}

smoke() {
    title "Read one real sample and load a checkpoint for each satellite"
    "$PYTHON_BIN" - "${1:-all}" <<'PY'
import sys
import pandas as pd
import torch
import config
import tccon_validate as validate
from main import build_aux_fields

sources = ("oco2", "oco3", "gosat2") if sys.argv[1] == "all" else (sys.argv[1],)
if any(source not in ("oco2", "oco3", "gosat2") for source in sources):
    raise ValueError("smoke source must be oco2, oco3, gosat2, or all")

torch.set_num_threads(2)
failed = []
for source in sources:
    data = None
    print(f"\n[SMOKE] source={source}", flush=True)
    try:
        validate.OBS_SOURCE = source
        split_path = config.GOSAT2_SPLIT_CSV if source == "gosat2" else config.DATA_SPLIT_CSV
        table = pd.read_csv(split_path, dtype=str, keep_default_na=False)
        table["region"] = table["region"].str.strip().str.lower()
        table["split"] = table["split"].str.strip().str.lower()
        choices = [(region, split) for region in ("ea", "eu", "na")
                   for split in ("test", "val", "train")
                   if ((table["region"] == region) & (table["split"] == split)).any()]
        if not choices:
            raise RuntimeError(f"No rows found in {split_path}")
        region, split = choices[0]
        data = validate.make_tccon_dataset(region, split)
        sample = data[0]
        trainer = validate.make_tccon_trainer(region)
        device = trainer.device
        x_ct = sample["x_ct"].unsqueeze(0).to(device)
        obs = sample["y_oco"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)
        batch = {key: sample[key].unsqueeze(0) for key in ("era5", "surface_flux")}
        aux = build_aux_fields(batch, device)
        doy = sample["doy"].view(1).to(device)
        tod = sample["tod"].view(1).to(device)
        with torch.no_grad():
            out = trainer.model(x_ct, x_ct, obs, mask, torch.tensor([0.5], device=device),
                                doy, tod, aux=aux)
        assert out.shape == x_ct.shape and torch.isfinite(out).all().item()
        print(f"[OK] {source}: {region}/{split}, sample={sample['timestamp']}, "
              f"observed_pixels={int(mask.sum())}, output={tuple(out.shape)}")
    except Exception as exc:
        failed.append(source)
        print(f"[FAILED] {source}: {type(exc).__name__}: {exc}", flush=True)
    finally:
        if data is not None:
            data.close()
if failed:
    raise SystemExit(f"Smoke test failed: {', '.join(failed)}")
print("\n[OK] Real-data smoke tests passed (not a full TCCON validation).")
PY
}

case "$MODE" in
    check)
        check
        ;;
    smoke)
        smoke "${2:-all}"
        ;;
    main)
        title "main.py (tasks controlled by RUN_TRAINING / RUN_UNCERTAINTY)"
        "$PYTHON_BIN" -u main.py
        ;;
    uncertainty)
        title "Existing checkpoints: simulated uncertainty evaluation (no training)"
        "$PYTHON_BIN" -u -c 'from main import run_all_uncertainty; run_all_uncertainty()'
        ;;
    oco2|oco3|gosat2)
        title "TCCON validation: $MODE"
        "$PYTHON_BIN" -u tccon_validate.py --source "$MODE"
        ;;
    all)
        check
        smoke all
        "$PYTHON_BIN" -u -c 'from main import run_all_uncertainty; run_all_uncertainty()'
        for source in oco2 oco3 gosat2; do
            title "TCCON validation: $source"
            "$PYTHON_BIN" -u tccon_validate.py --source "$source"
        done
        ;;
    *)
        echo "Usage: bash tasks.sh {check|smoke [oco2|oco3|gosat2|all]|main|uncertainty|oco2|oco3|gosat2|all}" >&2
        exit 2
        ;;
esac

echo "[DONE] $MODE"
