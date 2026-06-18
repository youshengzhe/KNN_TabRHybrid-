# coding: utf-8
"""
TabR-inspired residual retrieval for the ML final assignment.

This version keeps the original lightweight nearest-neighbor residual idea and
adds a second, very cheap context branch: local smoothing on the provided
Error_* signals. On this dataset, different targets benefit from different
context strategies, so we tune the final strategy per target on a validation
split of the training set.

The resulting prediction for each target can be one of:
1. identity baseline: Error_*
2. KNN residual correction
3. rolling-mean smoothing of Error_*
4. blend of KNN prediction and rolling-mean smoothing
5. train median fallback

This is not a full TabR reimplementation. It borrows the key practical lesson
from TabR: retrieval helps most when treated as a target-aware signal source,
not as a single monolithic predictor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


TARGET_COLUMNS = [
    "T_SONIC",
    "CO2_density",
    "CO2_density_fast_tmpr",
    "H2O_density",
    "H2O_sig_strgth",
    "CO2_sig_strgth",
]

ERROR_COLUMNS = [
    "Error_T_SONIC",
    "Error_CO2_density",
    "Error_CO2_density_fast_tmpr",
    "Error_H2O_density",
    "Error_H2O_sig_strgth",
    "Error_CO2_sig_strgth",
]

BASE_FEATURE_COLUMNS = [
    "Ux",
    "Uy",
    "Uz",
    "diag_sonic",
    "diag_irga",
    "T_SONIC_corr",
    "TA_1_1_1",
    "PA",
    "FW",
    *ERROR_COLUMNS,
]


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        result.append(resolved)
    return result


def build_search_dirs(root: Path | None, script_dir: Path) -> list[Path]:
    search_bases = [Path.cwd(), script_dir]
    if root is not None:
        search_bases.insert(0, root)

    search_dirs: list[Path] = []
    for base in unique_paths(search_bases):
        search_dirs.append(base)
        for parent in [base, *base.parents]:
            if not parent.exists():
                continue
            search_dirs.extend(path for path in parent.glob("001-*") if path.is_dir())
            search_dirs.extend(path for path in parent.glob("*/001-*") if path.is_dir())
    return unique_paths(search_dirs)


def find_dat_file(search_dirs: list[Path], series: str, with_truth: bool) -> Path:
    candidates: list[Path] = []
    for search_dir in search_dirs:
        candidates.extend(sorted(search_dir.rglob(f"*{series}.dat")))

    for path in unique_paths(candidates):
        cols = pd.read_csv(path, nrows=0).columns
        has_truth = all(col in cols for col in TARGET_COLUMNS)
        if has_truth == with_truth:
            return path

    searched = "\n".join(f"  - {path}" for path in search_dirs)
    raise FileNotFoundError(
        f"Cannot find *{series}.dat with_truth={with_truth}. Searched:\n{searched}"
    )


def try_find_dat_file(search_dirs: list[Path], series: str, with_truth: bool) -> Path | None:
    try:
        return find_dat_file(search_dirs, series, with_truth)
    except FileNotFoundError:
        return None


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df[BASE_FEATURE_COLUMNS].copy()
    if "TIMESTAMP" in df.columns:
        timestamp = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
        seconds = (
            timestamp.dt.hour * 3600
            + timestamp.dt.minute * 60
            + timestamp.dt.second
            + timestamp.dt.microsecond / 1_000_000
        ).fillna(0.0)
        angle = 2.0 * np.pi * seconds.to_numpy(dtype=float) / 86400.0
        features["time_sin"] = np.sin(angle)
        features["time_cos"] = np.cos(angle)
    return features


def mean_abs_error(pred: np.ndarray, true: np.ndarray) -> tuple[np.ndarray, float]:
    errors = np.abs(pred - true)
    return errors.mean(axis=0), float(errors.mean())


def choose_fit_indices(n_rows: int, max_train_rows: int, seed: int) -> np.ndarray:
    if max_train_rows <= 0 or max_train_rows >= n_rows:
        return np.arange(n_rows)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=max_train_rows, replace=False))


def smooth_error_columns(df: pd.DataFrame, window: int) -> np.ndarray:
    return (
        df[ERROR_COLUMNS]
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def weighted_residual_prediction(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    k: int,
    max_train_rows: int,
    batch_size: int,
    seed: int,
    algorithm: str,
) -> np.ndarray:
    fit_idx = choose_fit_indices(len(train_df), max_train_rows, seed)

    train_features = add_time_features(train_df)
    test_features = add_time_features(test_df)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_features.iloc[fit_idx].to_numpy(dtype=np.float32))
    x_test = scaler.transform(test_features.to_numpy(dtype=np.float32))

    y_train = train_df[TARGET_COLUMNS].to_numpy(dtype=np.float32)[fit_idx]
    error_train = train_df[ERROR_COLUMNS].to_numpy(dtype=np.float32)[fit_idx]
    residual_train = y_train - error_train

    n_neighbors = min(k, len(fit_idx))
    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        metric="euclidean",
        n_jobs=-1,
    )
    nn.fit(x_train)

    baseline = test_df[ERROR_COLUMNS].to_numpy(dtype=np.float32)
    pred = np.empty((len(test_df), len(TARGET_COLUMNS)), dtype=np.float32)

    for start in range(0, len(test_df), batch_size):
        end = min(start + batch_size, len(test_df))
        distances, indices = nn.kneighbors(x_test[start:end], return_distance=True)

        weights = 1.0 / (distances + 1e-6)
        weights = weights / weights.sum(axis=1, keepdims=True)
        correction = np.einsum("bk,bkd->bd", weights, residual_train[indices])
        pred[start:end] = baseline[start:end] + correction

        print(f"Processed {end}/{len(test_df)} rows")

    return pred.astype(float)


def tune_target_strategies(
    train_df: pd.DataFrame,
    *,
    k: int,
    max_train_rows: int,
    max_val_rows: int,
    batch_size: int,
    seed: int,
    algorithm: str,
    smooth_windows: list[int],
    blend_alphas: list[float],
) -> tuple[list[dict[str, float | str]], np.ndarray]:
    split = int(len(train_df) * 0.8)
    fit_df = train_df.iloc[:split].reset_index(drop=True)
    val_df = train_df.iloc[split:].reset_index(drop=True)

    if 0 < max_val_rows < len(val_df):
        rng = np.random.default_rng(seed + 1)
        val_idx = np.sort(rng.choice(len(val_df), size=max_val_rows, replace=False))
        val_df = val_df.iloc[val_idx].reset_index(drop=True)

    print(f"Tuning on {len(fit_df)} fit rows and {len(val_df)} validation rows")

    knn_pred = weighted_residual_prediction(
        fit_df,
        val_df,
        k=k,
        max_train_rows=max_train_rows,
        batch_size=batch_size,
        seed=seed,
        algorithm=algorithm,
    )

    baseline = val_df[ERROR_COLUMNS].to_numpy(dtype=float)
    true = val_df[TARGET_COLUMNS].to_numpy(dtype=float)
    medians = fit_df[TARGET_COLUMNS].median().to_numpy(dtype=float)

    smooth_cache = {window: smooth_error_columns(val_df, window) for window in smooth_windows}

    strategies: list[dict[str, float | str]] = []
    print("\nValidation strategy selection:")
    for target_i, target in enumerate(TARGET_COLUMNS):
        best = {
            "name": "identity",
            "window": 0.0,
            "alpha": 1.0,
            "mae": float(np.abs(baseline[:, target_i] - true[:, target_i]).mean()),
        }

        median_mae = float(np.abs(medians[target_i] - true[:, target_i]).mean())
        if median_mae < best["mae"]:
            best = {"name": "median", "window": 0.0, "alpha": 0.0, "mae": median_mae}

        knn_mae = float(np.abs(knn_pred[:, target_i] - true[:, target_i]).mean())
        if knn_mae < best["mae"]:
            best = {"name": "knn", "window": 0.0, "alpha": 1.0, "mae": knn_mae}

        for window, smooth_pred in smooth_cache.items():
            smooth_mae = float(np.abs(smooth_pred[:, target_i] - true[:, target_i]).mean())
            if smooth_mae < best["mae"]:
                best = {"name": "smooth", "window": float(window), "alpha": 0.0, "mae": smooth_mae}

            for alpha in blend_alphas:
                blended = alpha * knn_pred[:, target_i] + (1.0 - alpha) * smooth_pred[:, target_i]
                blend_mae = float(np.abs(blended - true[:, target_i]).mean())
                if blend_mae < best["mae"]:
                    best = {
                        "name": "blend",
                        "window": float(window),
                        "alpha": float(alpha),
                        "mae": blend_mae,
                    }

        strategies.append(best)
        print(
            f"  {target}: {best['name']}, window={int(best['window'])}, "
            f"alpha={best['alpha']:.2f}, val_mae={best['mae']:.6f}"
        )

    return strategies, medians


def apply_target_strategies(
    test_df: pd.DataFrame,
    knn_pred: np.ndarray,
    strategies: list[dict[str, float | str]],
    medians: np.ndarray,
) -> np.ndarray:
    pred = test_df[ERROR_COLUMNS].to_numpy(dtype=float).copy()
    smooth_cache: dict[int, np.ndarray] = {}

    for i, strategy in enumerate(strategies):
        name = str(strategy["name"])
        if name == "knn":
            pred[:, i] = knn_pred[:, i]
        elif name == "median":
            pred[:, i] = medians[i]
        elif name == "smooth":
            window = int(strategy["window"])
            if window not in smooth_cache:
                smooth_cache[window] = smooth_error_columns(test_df, window)
            pred[:, i] = smooth_cache[window][:, i]
        elif name == "blend":
            window = int(strategy["window"])
            alpha = float(strategy["alpha"])
            if window not in smooth_cache:
                smooth_cache[window] = smooth_error_columns(test_df, window)
            pred[:, i] = alpha * knn_pred[:, i] + (1.0 - alpha) * smooth_cache[window][:, i]

    return pred


def export_predictions(pred: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [" ".join(f"{value:.10g}" for value in row) for row in pred]
    pd.DataFrame({"Predicted_Value": rows}).to_csv(output_path, index=False)


def parse_int_list(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TabR-inspired KNN residual retrieval model")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Optional directory to search first for the train/test .dat files.",
    )
    parser.add_argument("--train-series", default="661")
    parser.add_argument("--test-series", default="662")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=50_000,
        help="Use 0 to retrieve from all training rows. Smaller values run faster.",
    )
    parser.add_argument("--max-val-rows", type=int, default=80_000)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--algorithm", default="kd_tree", choices=["auto", "ball_tree", "kd_tree", "brute"])
    parser.add_argument(
        "--smooth-windows",
        default="5,7,9,11,15,21,31,61",
        help="Comma-separated rolling windows to try for the smoothing branch.",
    )
    parser.add_argument(
        "--blend-alphas",
        default="0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95",
        help="Comma-separated blend weights for alpha * knn + (1-alpha) * smooth.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    search_dirs = build_search_dirs(args.root, script_dir)

    train_path = find_dat_file(search_dirs, args.train_series, with_truth=True)
    test_no_truth_path = find_dat_file(search_dirs, args.test_series, with_truth=False)
    test_truth_path = try_find_dat_file(search_dirs, args.test_series, with_truth=True)

    print("Search directories:")
    for search_dir in search_dirs:
        print(f"  {search_dir}")
    print(f"Train: {train_path}")
    print(f"Test: {test_no_truth_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_no_truth_path)

    strategies, medians = tune_target_strategies(
        train_df,
        k=args.k,
        max_train_rows=args.max_train_rows,
        max_val_rows=args.max_val_rows,
        batch_size=args.batch_size,
        seed=args.seed,
        algorithm=args.algorithm,
        smooth_windows=parse_int_list(args.smooth_windows),
        blend_alphas=parse_float_list(args.blend_alphas),
    )

    knn_pred = weighted_residual_prediction(
        train_df,
        test_df,
        k=args.k,
        max_train_rows=args.max_train_rows,
        batch_size=args.batch_size,
        seed=args.seed,
        algorithm=args.algorithm,
    )
    pred = apply_target_strategies(test_df, knn_pred, strategies, medians)

    output_path = args.output or script_dir / "result_KNN_TabRHybrid.csv"
    export_predictions(pred, output_path)
    print(f"Saved predictions to: {output_path}")

    if test_truth_path is not None:
        truth_df = pd.read_csv(test_truth_path, usecols=TARGET_COLUMNS)
        per_target, overall = mean_abs_error(pred, truth_df[TARGET_COLUMNS].to_numpy(dtype=float))
        print("\nPublic test MAE:")
        for target, mae in zip(TARGET_COLUMNS, per_target):
            print(f"  {target}: {mae:.6f}")
        print(f"  Overall: {overall:.6f}")
    else:
        print("No truth file found for the test series; skipped MAE evaluation.")


if __name__ == "__main__":
    main()
