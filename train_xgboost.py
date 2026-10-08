"""以 24 小時窗口訓練單一 MultiOutputRegressor XGBoost 模型。"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


DATASET_PATH = Path("output/supervised/dataset.npz")
XGBOOST_OUTPUT_DIR = Path("output/xgboost")

N_ESTIMATORS = 300
MAX_DEPTH = 4
LEARNING_RATE = 0.05
EARLY_STOPPING_ROUNDS = 30
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
MIN_CHILD_WEIGHT = 10
REG_ALPHA = 0.1
REG_LAMBDA = 5.0
RANDOM_STATE = 42


def _create_run_dir() -> Path:
    """建立下一個訓練資料夾，避免覆寫任何既有訓練結果。"""
    XGBOOST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_number = 1
    while True:
        run_dir = XGBOOST_OUTPUT_DIR / f"run_{run_number:03d}"
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            run_number += 1


def _flatten_windows(windows: np.ndarray) -> np.ndarray:
    """將 (samples, input_hours, features) 轉成 XGBoost 所需的二維矩陣。"""
    if windows.ndim != 3:
        raise ValueError(f"輸入窗口必須是三維陣列，實際為 {windows.shape}")
    return windows.reshape(windows.shape[0], -1)


def _build_model() -> MultiOutputRegressor:
    estimator = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        min_child_weight=MIN_CHILD_WEIGHT,
        reg_alpha=REG_ALPHA,
        reg_lambda=REG_LAMBDA,
        eval_metric="mae",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    return MultiOutputRegressor(estimator, n_jobs=1)


def _fit_model(
    model: MultiOutputRegressor,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> MultiOutputRegressor:
    """逐一訓練每個輸出，讓每個 XGBoost 子模型正確使用 validation 早停。"""
    model.estimators_ = []
    for target_index in range(y_train.shape[1]):
        estimator = clone(model.estimator)
        estimator.fit(
            x_train,
            y_train[:, target_index],
            eval_set=[(x_validation, y_validation[:, target_index])],
            verbose=False,
        )
        model.estimators_.append(estimator)
    model.n_features_in_ = x_train.shape[1]
    return model


def _collect_loss_curve(
    model: MultiOutputRegressor,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> pd.DataFrame:
    rows = []
    trained_rounds = [
        estimator.best_iteration + 1
        for estimator in model.estimators_
        if estimator.best_iteration is not None
    ]
    max_boosting_round = max(trained_rounds, default=N_ESTIMATORS)
    for boosting_round in range(1, max_boosting_round + 1):
        train_predictions = np.column_stack(
            [
                estimator.predict(
                    x_train,
                    iteration_range=(0, min(boosting_round, estimator.best_iteration + 1)),
                )
                for estimator in model.estimators_
            ]
        )
        validation_predictions = np.column_stack(
            [
                estimator.predict(
                    x_validation,
                    iteration_range=(0, min(boosting_round, estimator.best_iteration + 1)),
                )
                for estimator in model.estimators_
            ]
        )
        train_mae = mean_absolute_error(y_train, train_predictions)
        validation_mae = mean_absolute_error(y_validation, validation_predictions)
        rows.append(
            {
                "boosting_round": boosting_round,
                "train_mae": train_mae,
                "validation_mae": validation_mae,
            }
        )
        print(
            f"Round {boosting_round:03d}/{max_boosting_round}: "
            f"train MAE={train_mae:.6f}, validation MAE={validation_mae:.6f}"
        )
    return pd.DataFrame(rows)


def _save_loss_outputs(loss: pd.DataFrame, run_dir: Path) -> None:
    loss_csv_path = run_dir / "xgboost_loss.csv"
    loss_plot_path = run_dir / "xgboost_loss_curve.png"
    loss.to_csv(loss_csv_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    plt.plot(loss["boosting_round"], loss["train_mae"], color="blue", label="Train MAE")
    plt.plot(
        loss["boosting_round"],
        loss["validation_mae"],
        color="red",
        label="Validation MAE",
        linewidth=1.5,
    )
    plt.xlabel("Boosting round")
    plt.ylabel("MAE")
    plt.title("XGBoost Multi-Output Training Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()


def train_model() -> None:
    """只使用 train 與 validation split 訓練並輸出模型與 loss 紀錄。"""
    with np.load(DATASET_PATH, allow_pickle=False) as dataset:
        x_train = _flatten_windows(dataset["X_train"])
        y_train = dataset["y_train"]
        x_validation = _flatten_windows(dataset["X_validation"])
        y_validation = dataset["y_validation"]

    if not x_train.shape[0] or not x_validation.shape[0]:
        raise ValueError("train 與 validation 都必須包含至少一筆樣本。")
    if y_train.ndim != 2 or y_validation.ndim != 2:
        raise ValueError("目標陣列必須是 (samples, forecast_horizon)。")
    if y_train.shape[1] != y_validation.shape[1]:
        raise ValueError("train 與 validation 的預測時數不一致。")
    if not np.isfinite(x_train).all() or not np.isfinite(y_train).all():
        raise ValueError("train 資料包含 NaN 或無限值。")
    if not np.isfinite(x_validation).all() or not np.isfinite(y_validation).all():
        raise ValueError("validation 資料包含 NaN 或無限值。")

    print(f"Train shape: X={x_train.shape}, y={y_train.shape}")
    print(f"Validation shape: X={x_validation.shape}, y={y_validation.shape}")
    print("開始訓練 MultiOutputRegressor...")

    model = _fit_model(
        _build_model(),
        x_train,
        y_train,
        x_validation,
        y_validation,
    )
    loss = _collect_loss_curve(
        model,
        x_train,
        y_train,
        x_validation,
        y_validation,
    )

    run_dir = _create_run_dir()
    model_path = run_dir / "xgboost_multioutput_model.pkl"
    joblib.dump(model, model_path)
    _save_loss_outputs(loss, run_dir)

    print(f"本次訓練資料夾：{run_dir}")
    print(f"模型已輸出：{model_path}")
    print(f"Loss CSV 已輸出：{run_dir / 'xgboost_loss.csv'}")
    print(f"Loss 圖已輸出：{run_dir / 'xgboost_loss_curve.png'}")
    print(
        "各輸出模型最佳輪數："
        f" {[estimator.best_iteration + 1 for estimator in model.estimators_]}"
    )


if __name__ == "__main__":
    train_model()