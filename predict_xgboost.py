"""使用最新 XGBoost 模型預測 2025 測試集並輸出評估圖表。"""
import argparse
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATASET_PATH = Path("output/supervised/dataset.npz")
MANIFEST_PATH = Path("output/supervised/manifest.csv")
CLEANED_DATA_PATH = Path("output/西屯_2021_2025_cleaned.csv")
SCALER_PATH = Path("output/scaler.pkl")
XGBOOST_OUTPUT_DIR = Path("output/xgboost")
PREDICT_OUTPUT_DIR = XGBOOST_OUTPUT_DIR / "predict"
TARGET = "PM2.5"
TEST_START = pd.Timestamp("2025-01-01 00:00")
TEST_END = pd.Timestamp("2025-12-31 23:00")
PLOT_RESAMPLE_RULE = "W-SUN"
TEST_ZOOM_RESAMPLE_RULE = "D"


def _create_predict_dir(run_number: int) -> Path:
    """建立與訓練 run 對應的預測資料夾，避免覆寫既有結果。"""
    PREDICT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predict_dir = PREDICT_OUTPUT_DIR / f"p{run_number:03d}"
    try:
        predict_dir.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"預測資料夾已存在：{predict_dir}，為避免覆寫請先確認既有結果。"
        ) from error
    return predict_dir


def _find_model(model_run: str) -> tuple[Path, int]:
    match = re.fullmatch(r"(?:run_)?(\d+)", model_run)
    if not match:
        raise ValueError("模型 run 必須是例如 run_003 或 003。")

    run_number = int(match.group(1))
    run_dir = XGBOOST_OUTPUT_DIR / f"run_{run_number:03d}"
    model_path = run_dir / "xgboost_multioutput_model.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到指定模型：{model_path}")
    return model_path, run_number


def _flatten_windows(windows: np.ndarray) -> np.ndarray:
    if windows.ndim != 3:
        raise ValueError(f"輸入窗口必須是三維陣列，實際為 {windows.shape}")
    return windows.reshape(windows.shape[0], -1)


def _inverse_target(values: np.ndarray) -> np.ndarray:
    scaler_bundle = joblib.load(SCALER_PATH)
    feature_index = scaler_bundle["feature_cols"].index(TARGET)
    scaler = scaler_bundle["scaler"]
    return values * scaler.scale_[feature_index] + scaler.mean_[feature_index]


def _load_test_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    with np.load(DATASET_PATH, allow_pickle=False) as dataset:
        x_test = _flatten_windows(dataset["X_test"])
        y_test = dataset["y_test"]

    manifest = pd.read_csv(MANIFEST_PATH, parse_dates=["target_start", "target_end"])
    test_manifest = manifest[manifest["split"] == "test"].reset_index(drop=True)
    if len(test_manifest) != len(x_test):
        raise ValueError(
            f"測試窗口數量不一致：dataset={len(x_test)}, manifest={len(test_manifest)}"
        )
    if not len(x_test):
        raise ValueError("測試集沒有可預測的窗口。")
    if y_test.ndim != 2 or y_test.shape[1] != 24:
        raise ValueError(f"測試目標必須是 (samples, 24)，實際為 {y_test.shape}")
    if not np.isfinite(x_test).all() or not np.isfinite(y_test).all():
        raise ValueError("測試資料包含 NaN 或無限值。")
    return x_test, y_test, test_manifest


def _build_prediction_table(
    predictions: np.ndarray,
    test_manifest: pd.DataFrame,
    actual: pd.Series,
) -> pd.DataFrame:
    rows = []
    for sample_index, row in test_manifest.iterrows():
        timestamps = pd.date_range(
            row["target_start"], row["target_end"], freq="h"
        )
        for horizon, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "sample_index": sample_index,
                    "horizon": horizon + 1,
                    "Datetime": timestamp,
                    "actual": actual.get(timestamp, np.nan),
                    "predicted": predictions[sample_index, horizon],
                }
            )

    window_predictions = pd.DataFrame(rows)
    if window_predictions["actual"].isna().any():
        raise ValueError("測試預測時間找不到對應的真實 PM2.5 值。")

    # 窗口彼此重疊，同一小時可能有多個預測；以平均值作為時序圖與指標的單一預測。
    return (
        window_predictions.groupby("Datetime", as_index=False)
        .agg(
            actual=("actual", "first"),
            predicted=("predicted", "mean"),
            prediction_count=("predicted", "size"),
        )
        .sort_values("Datetime")
    )


def _calculate_metrics(prediction_table: pd.DataFrame) -> pd.DataFrame:
    actual = prediction_table["actual"]
    predicted = prediction_table["predicted"]
    rows = [
        {
            "scope": "overall",
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
            "R2": r2_score(actual, predicted),
        }
    ]
    return pd.DataFrame(rows)


def _weekly_mean(series: pd.Series) -> pd.Series:
    """將時間序列轉為每週平均，供時間序列圖降低取樣頻率。"""
    return series.sort_index().resample(PLOT_RESAMPLE_RULE).mean().dropna()


def _plot_full_overview(
    actual: pd.Series, prediction_table: pd.DataFrame, output_path: Path
) -> None:
    actual_weekly = _weekly_mean(actual)
    prediction_weekly = _weekly_mean(
        prediction_table.set_index("Datetime")["predicted"]
    )
    plt.figure(figsize=(16, 6))
    plt.plot(
        actual_weekly.index,
        actual_weekly.values,
        color="tab:blue",
        linewidth=1.5,
        label="Actual (weekly mean)",
    )
    plt.plot(
        prediction_weekly.index,
        prediction_weekly.values,
        color="tab:orange",
        linewidth=1.5,
        label="2025 Test Prediction (weekly mean)",
    )
    plt.axvline(TEST_START, color="tab:red", linestyle="--", label="Prediction start")
    plt.xlabel("Datetime")
    plt.ylabel("PM2.5")
    plt.title("Full Period PM2.5 Overview (Weekly Mean)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_test_zoom(prediction_table: pd.DataFrame, output_path: Path) -> None:
    test_table = prediction_table[
        prediction_table["Datetime"].between(TEST_START, TEST_END)
    ].set_index("Datetime")
    test_daily = (
        test_table[["actual", "predicted"]]
        .resample(TEST_ZOOM_RESAMPLE_RULE)
        .mean()
        .interpolate(method="time", limit_direction="both")
    )
    plt.figure(figsize=(16, 6))
    plt.plot(
        test_daily.index,
        test_daily["actual"],
        color="tab:blue",
        linewidth=1.5,
        label="Actual (daily mean)",
    )
    plt.plot(
        test_daily.index,
        test_daily["predicted"],
        color="tab:orange",
        linewidth=1.5,
        label="Prediction (daily mean)",
    )
    plt.xlabel("Datetime")
    plt.ylabel("PM2.5")
    plt.title("2025 Test Set: Actual vs Prediction (Daily Mean)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    metric_names = ["MAE", "RMSE", "R2"]
    metric_values = [metrics.iloc[0][name] for name in metric_names]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(metric_names, metric_values, color=["#2563eb", "#f59e0b", "#16a34a"])
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom")
    plt.ylabel("Score")
    plt.title("2025 Test Set Evaluation Metrics")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_scatter(prediction_table: pd.DataFrame, output_path: Path) -> None:
    actual = prediction_table["actual"]
    predicted = prediction_table["predicted"]
    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())
    plt.figure(figsize=(7, 7))
    plt.scatter(actual, predicted, s=12, alpha=0.45, color="tab:blue")
    plt.plot([lower, upper], [lower, upper], color="tab:red", linestyle="--", label="Ideal")
    plt.xlabel("Actual PM2.5")
    plt.ylabel("Predicted PM2.5")
    plt.title("Predicted vs Actual")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_residuals(prediction_table: pd.DataFrame, output_path: Path) -> None:
    residual = prediction_table["predicted"] - prediction_table["actual"]
    plt.figure(figsize=(16, 5))
    plt.axhline(0, color="black", linewidth=0.8)
    plt.scatter(
        prediction_table["Datetime"],
        residual,
        s=12,
        alpha=0.45,
        color="tab:purple",
    )
    plt.xlabel("Datetime")
    plt.ylabel("Residual (Predicted - Actual)")
    plt.title("Residuals Over Time")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def predict_model(model_run: str) -> None:
    """使用指定訓練 run 的模型預測測試集並輸出對應結果。"""
    model_path, run_number = _find_model(model_run)
    x_test, y_test, test_manifest = _load_test_data()
    model = joblib.load(model_path)
    predictions_scaled = model.predict(x_test)
    predictions = _inverse_target(predictions_scaled)

    cleaned = pd.read_csv(CLEANED_DATA_PATH, parse_dates=["Datetime"])
    actual = cleaned.set_index("Datetime")[TARGET].sort_index()
    prediction_table = _build_prediction_table(predictions, test_manifest, actual)
    metrics = _calculate_metrics(prediction_table)
    run_dir = _create_predict_dir(run_number)

    prediction_table.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(run_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    _plot_full_overview(actual, prediction_table, run_dir / "01_full_overview.png")
    _plot_test_zoom(prediction_table, run_dir / "02_test_zoom.png")
    _plot_metrics(metrics, run_dir / "03_metrics.png")
    _plot_scatter(prediction_table, run_dir / "04_predicted_vs_actual.png")
    _plot_residuals(prediction_table, run_dir / "05_residuals_over_time.png")

    print(f"使用模型：{model_path}")
    print(f"本次預測資料夾：{run_dir}")
    print(metrics.to_string(index=False))
    print(f"預測結果：{run_dir / 'predictions.csv'}")
    print(f"評估指標：{run_dir / 'metrics.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        required=True,
        help="要使用的訓練模型，例如 run_003 或 003；結果會輸出到 predict/p003。",
    )
    args = parser.parse_args()
    predict_model(args.run)