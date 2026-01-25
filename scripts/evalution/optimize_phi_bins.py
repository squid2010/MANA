#!/usr/bin/env python3
"""
Optimize phi bin cutpoints (Low / Medium / High) to maximize categorical metrics.

This script can:
- Load true and predicted phi values from a predictions file (HDF5/NPZ/CSV).
- Or run a trained MANA model over a phi HDF5 dataset to produce predictions.
- Grid-search pairs of thresholds (t1 < t2) in [0,1] to discretize continuous phi
  into three bins: Low (< t1), Med [t1, t2), High (>= t2).
- Evaluate categorical agreement between binned true phi and binned predicted phi
  using multiple metrics (accuracy, balanced accuracy, F1-macro, Cohen's kappa).
- Output the best thresholds for each metric and save candidate results to JSON/CSV.

Usage examples:
1) Use existing predictions file (HDF5/NPZ/CSV) with columns/datasets:
     phi_true, phi_pred
   python MANA/scripts/evalution/optimize_phi_bins.py --predictions predictions.h5

2) Run model on dataset and then optimize:
   python MANA/scripts/evalution/optimize_phi_bins.py --model path/to/model.pth \
       --phi_dataset data/phi/phi_data.h5

Notes:
- Only entries with finite true phi in [0,1] are considered for optimization,
  consistent with evaluation expectations.
- Default grid step is 0.02 (adjust with --step). Smaller steps increase runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Optional imports for running model inference
try:
    import torch
    from model.mana_model import MANA
    from torch_geometric.loader import DataLoader

    from data.dataset import DatasetConstructor

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


def load_predictions_from_h5(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load phi_true and phi_pred from an HDF5 file if present."""
    import h5py

    with h5py.File(str(path), "r") as f:
        # Common dataset names (try multiple)
        candidates = {}
        if "phi_pred" in f:
            candidates["phi_pred"] = f["phi_pred"][()]
        if "phi_prediction" in f:
            candidates["phi_pred"] = f["phi_prediction"][()]
        if (
            "phi" in f and "pred" in f["phi"].dtype.names
            if isinstance(f.get("phi"), h5py.Dataset) and f["phi"].dtype.names
            else False
        ):
            pass  # unlikely structure - skip

        # try simple names
        if "phi_pred" not in candidates:
            for key in ["phi_pred", "phi_prediction", "phi_predicted", "pred_phi"]:
                if key in f:
                    candidates["phi_pred"] = f[key][()]
                    break

        # true values
        for key in ["phi_true", "phi_delta", "phi", "phi_label"]:
            if key in f:
                candidates["phi_true"] = f[key][()]
                break

        # If not found, raise
        if "phi_true" not in candidates or "phi_pred" not in candidates:
            raise RuntimeError(
                f"Could not find phi_true and phi_pred datasets in {path}. "
                "Expected dataset names like 'phi_true'/'phi_delta' and 'phi_pred'."
            )

        return np.asarray(candidates["phi_true"]).ravel(), np.asarray(
            candidates["phi_pred"]
        ).ravel()


def load_predictions_from_npz_or_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load predictions from .npz or .csv"""
    if path.suffix.lower() == ".npz":
        data = np.load(str(path), allow_pickle=True)
        if "phi_true" in data and "phi_pred" in data:
            return data["phi_true"].ravel(), data["phi_pred"].ravel()
        # try alternate keys
        for a in ["true", "y_true", "phi", "phi_delta"]:
            for b in ["pred", "y_pred", "phi_pred", "phi_prediction"]:
                if a in data and b in data:
                    return data[a].ravel(), data[b].ravel()
        raise RuntimeError("NPZ file did not contain recognizable phi arrays.")
    else:
        # assume CSV
        import csv

        true_vals = []
        pred_vals = []
        with open(str(path), "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                # try common column names
                true_keys = ["phi_true", "phi_delta", "phi", "true", "y_true"]
                pred_keys = ["phi_pred", "pred_phi", "phi_prediction", "pred", "y_pred"]
                t = None
                p = None
                for k in true_keys:
                    if k in row and row[k] != "":
                        t = float(row[k])
                        break
                for k in pred_keys:
                    if k in row and row[k] != "":
                        p = float(row[k])
                        break
                if t is not None and p is not None:
                    true_vals.append(t)
                    pred_vals.append(p)
        if len(true_vals) == 0:
            raise RuntimeError(
                "CSV did not have recognizable columns for phi_true/phi_pred."
            )
        return np.array(true_vals), np.array(pred_vals)


def run_model_get_phi_preds(
    model_path: Path, dataset_path: Path, device: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the MANA model on the given phi dataset and return (true_phi, pred_phi).

    This attempts to reuse DatasetConstructor and MANA. If PyTorch is not available,
    this will raise.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch/MANA imports not available in this environment.")

    dev = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = MANA(num_atom_types=118, hidden_dim=128, tasks=["phi"]).to(dev)
    # load permissively
    state = torch.load(str(model_path), map_location=dev)
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    dataset = DatasetConstructor(str(dataset_path), split_by_mol_id=True)
    _, _, loader = dataset.get_dataloaders(num_workers=0)

    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(dev)
            preds = model(batch)
            if "phi" not in preds:
                raise RuntimeError("Model did not return 'phi' in forward pass.")
            pred_arr = preds["phi"].detach().cpu().numpy().ravel()
            # true phi values may be in batch.phi_delta
            if hasattr(batch, "phi_delta"):
                true_arr = batch.phi_delta.detach().cpu().numpy().ravel()
            else:
                true_arr = np.full_like(pred_arr, np.nan)
            y_true_all.append(true_arr)
            y_pred_all.append(pred_arr)

    return np.concatenate(y_true_all, axis=0), np.concatenate(y_pred_all, axis=0)


def filter_valid_phi(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep entries where true phi is NaN or in [0,1]. Discard others."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape[0] != y_pred.shape[0]:
        # If sizes mismatch, try to align by min length
        n = min(y_true.shape[0], y_pred.shape[0])
        y_true = y_true[:n]
        y_pred = y_pred[:n]
    finite = np.isfinite(y_true)
    mask = (~finite) | ((y_true >= 0.0) & (y_true <= 1.0))
    return y_true[mask], y_pred[mask]


def bin_values(vals: np.ndarray, t1: float, t2: float) -> np.ndarray:
    """Map continuous phi values to 3-class integers (0=Low,1=Med,2=High)."""
    vals = np.asarray(vals).ravel()
    out = np.full(vals.shape, -1, dtype=np.int32)
    out[vals < t1] = 0
    out[(vals >= t1) & (vals < t2)] = 1
    out[vals >= t2] = 2
    return out


def evaluate_bins(y_true: np.ndarray, y_pred: np.ndarray, t1: float, t2: float) -> dict:
    """Compute categorical metrics for given thresholds t1,t2."""
    y_true_b = bin_values(y_true, t1, t2)
    y_pred_b = bin_values(y_pred, t1, t2)

    # Only consider positions where true is assigned to a class (exclude NaN true => -1)
    valid = y_true_b >= 0
    if valid.sum() == 0:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "f1_macro": float("nan"),
            "kappa": float("nan"),
            "n": 0,
        }
    yt = y_true_b[valid]
    yp = y_pred_b[valid]

    # Compute metrics, handling degenerate cases
    try:
        acc = accuracy_score(yt, yp)
    except Exception:
        acc = float("nan")
    try:
        bacc = balanced_accuracy_score(yt, yp)
    except Exception:
        bacc = float("nan")
    try:
        f1 = f1_score(yt, yp, average="macro", labels=[0, 1, 2])
    except Exception:
        f1 = float("nan")
    try:
        kappa = cohen_kappa_score(yt, yp)
    except Exception:
        kappa = float("nan")

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "f1_macro": float(f1),
        "kappa": float(kappa),
        "n": int(valid.sum()),
    }


def grid_search_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    step: float = 0.02,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> List[dict]:
    """Search over t1,t2 (t_min..t_max) with given step and return list of results."""
    results = []
    # ensure step divides range reasonably
    eps = 1e-8
    t_vals = np.arange(t_min, t_max + eps, step)
    # avoid trivial extreme thresholds at exactly 0 or 1 if desired; keep full range though
    for i, t1 in enumerate(t_vals):
        for t2 in t_vals[i + 1 :]:
            if t2 <= t1 + 1e-6:
                continue
            res = evaluate_bins(y_true, y_pred, float(t1), float(t2))
            res.update({"t1": float(t1), "t2": float(t2)})
            results.append(res)
    return results


def select_best(results: Iterable[dict], metric: str = "f1_macro") -> dict:
    """Select the result with maximum value of metric. If tie, prefer higher n, then smaller t1 gap."""
    best = None
    for r in results:
        val = r.get(metric, float("nan"))
        if best is None:
            best = r
            continue
        # treat nan as -inf
        if np.isnan(val):
            continue
        bval = best.get(metric, float("nan"))
        if np.isnan(bval) or val > bval:
            best = r
        elif val == bval:
            # break ties by larger n
            if r.get("n", 0) > best.get("n", 0):
                best = r
    return best or {}


def save_results(results: List[dict], out_path: Path) -> None:
    """Save grid search results to JSON (and CSV)."""
    import csv

    out_json = str(out_path)
    out_csv = str(out_path.with_suffix(".csv"))
    # JSON
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    # CSV (flat)
    keys = [
        "t1",
        "t2",
        "n",
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "kappa",
    ]
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in results:
            out = {k: r.get(k, "") for k in keys}
            writer.writerow(out)


def main():
    parser = argparse.ArgumentParser(
        description="Optimize phi bins (Low/Med/High) for categorical metrics"
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="Path to predictions file (HDF5/NPZ/CSV) containing phi_true and phi_pred",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to model .pth - if provided will be used with --phi_dataset to generate predictions",
    )
    parser.add_argument(
        "--phi_dataset",
        default=None,
        help="Path to phi HDF5 dataset (used with --model to run inference)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.02,
        help="Grid step for threshold search (default 0.02)",
    )
    parser.add_argument(
        "--out",
        default="phi_bin_candidates.json",
        help="Output JSON file to save grid results",
    )
    parser.add_argument(
        "--metric",
        default="f1_macro",
        choices=["accuracy", "balanced_accuracy", "f1_macro", "kappa"],
        help="Primary metric to optimize",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string (e.g. cpu or cuda:0) when running model inference",
    )
    args = parser.parse_args()

    # Load or produce predictions
    if args.predictions:
        p = Path(args.predictions)
        if not p.exists():
            raise FileNotFoundError(f"Predictions file not found: {p}")
        if p.suffix.lower() in [".h5", ".hdf5"]:
            y_true, y_pred = load_predictions_from_h5(p)
        elif p.suffix.lower() in [".npz", ".npy", ".csv", ".txt"]:
            y_true, y_pred = load_predictions_from_npz_or_csv(p)
        else:
            # try h5 first then npz
            try:
                y_true, y_pred = load_predictions_from_h5(p)
            except Exception:
                y_true, y_pred = load_predictions_from_npz_or_csv(p)
    elif args.model and args.phi_dataset:
        if not HAS_TORCH:
            raise RuntimeError(
                "Model inference requested but PyTorch or model imports are not available."
            )
        y_true, y_pred = run_model_get_phi_preds(
            Path(args.model), Path(args.phi_dataset), device=args.device
        )
    else:
        raise RuntimeError(
            "You must provide either --predictions or both --model and --phi_dataset."
        )

    # Filter to valid true values (as evaluate.py does)
    y_true_f, y_pred_f = filter_valid_phi(y_true, y_pred)
    if y_true_f.size == 0:
        raise RuntimeError(
            "No valid phi entries found after filtering (phi must be NaN or in [0,1])."
        )

    # Grid search thresholds
    print(
        f"Running grid search step={args.step} over [{0.0}, {1.0}] for {y_true_f.size} valid samples..."
    )
    results = grid_search_thresholds(
        y_true_f, y_pred_f, step=float(args.step), t_min=0.0, t_max=1.0
    )

    # Select best by requested metric
    best = select_best(results, metric=args.metric)
    print("Best thresholds by metric", args.metric)
    print(json.dumps(best, indent=2))

    # Also provide best by other metrics
    for m in ["accuracy", "balanced_accuracy", "f1_macro", "kappa"]:
        b = select_best(results, metric=m)
        print(
            f"Best by {m}: t1={b.get('t1')}, t2={b.get('t2')}, {m}={b.get(m)}, n={b.get('n')}"
        )

    # Save results
    out_path = Path(args.out)
    save_results(results, out_path)
    print(
        f"Saved grid search results to: {out_path} and {out_path.with_suffix('.csv')}"
    )


if __name__ == "__main__":
    main()
