import os
from pathlib import Path
import numpy as np


def infer_valid_time(arr: np.ndarray, tol: float = 1e-12) -> int:
    """
    Infer non-padded time length from a saved spectrogram array shaped [F, T].
    Padding in prep_data_BIUREV.py is all-zero columns.
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array [F,T], got shape {arr.shape}")
    nonzero_cols = np.where(np.any(np.abs(arr) > tol, axis=0))[0]
    if nonzero_cols.size == 0:
        return 0
    return int(nonzero_cols[-1] + 1)


def get_stats_path(data_base_path: str, data_type: str) -> Path:
    return Path(data_base_path) / "norm_stats" / f"{data_type}_train_minmax.npz"


def normalize_to_minus1_1(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    denom = max(vmax - vmin, 1e-12)
    return (2.0 * (x - vmin) / denom - 1.0).astype(np.float32)


def denormalize_from_minus1_1(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return (((x + 1.0) * 0.5) * (vmax - vmin) + vmin).astype(np.float32)


def _iter_npy_files(root: Path):
    if not root.exists():
        return
    for dp, _, files in os.walk(root):
        for fn in sorted(files):
            if fn.endswith('.npy'):
                yield Path(dp) / fn


def compute_train_minmax(data_base_path: str, data_type: str) -> tuple[float, float]:
    roots = [
        Path(data_base_path) / f"clean_{data_type}" / "train",
        Path(data_base_path) / f"reverb_{data_type}" / "train",
    ]
    global_min = None
    global_max = None

    for root in roots:
        for fp in _iter_npy_files(root):
            arr = np.load(fp)
            valid_t = infer_valid_time(arr)
            if valid_t <= 0:
                continue
            vals = arr[:, :valid_t]
            amin = float(vals.min())
            amax = float(vals.max())
            global_min = amin if global_min is None else min(global_min, amin)
            global_max = amax if global_max is None else max(global_max, amax)

    if global_min is None or global_max is None:
        raise RuntimeError(
            f"Could not compute normalization stats from train split under {data_base_path} for data_type={data_type}"
        )
    if not np.isfinite(global_min) or not np.isfinite(global_max):
        raise RuntimeError("Non-finite normalization stats encountered")
    if global_max <= global_min:
        raise RuntimeError(f"Invalid normalization range: min={global_min}, max={global_max}")
    return float(global_min), float(global_max)


def load_or_create_train_minmax(data_base_path: str, data_type: str) -> dict:
    stats_path = get_stats_path(data_base_path, data_type)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    if stats_path.exists():
        z = np.load(stats_path)

        # Support old and new key names
        min_keys = ["min", "min_val", "vmin", "global_min"]
        max_keys = ["max", "max_val", "vmax", "global_max"]

        found_min = None
        found_max = None

        for k in min_keys:
            if k in z.files:
                found_min = float(z[k])
                break

        for k in max_keys:
            if k in z.files:
                found_max = float(z[k])
                break

        if found_min is None or found_max is None:
            raise KeyError(
                f"Normalization stats file {stats_path} does not contain supported keys. "
                f"Found keys: {z.files}"
            )

        return {
            "min": found_min,
            "max": found_max,
            "path": str(stats_path),
        }

    vmin, vmax = compute_train_minmax(data_base_path, data_type)
    np.savez(stats_path, min=np.float32(vmin), max=np.float32(vmax))
    return {
        "min": float(vmin),
        "max": float(vmax),
        "path": str(stats_path),
    }
