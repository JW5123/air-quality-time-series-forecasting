"""將逐時資料轉成 24 小時輸入、24 小時輸出的監督式樣本。"""
from pathlib import Path

import numpy as np
import pandas as pd


def _split_for_target(target_end: pd.Timestamp, splits: dict) -> str | None:
    for name, (start, end) in splits.items():
        start_ts = pd.Timestamp.min if start is None else pd.Timestamp(start)
        end_ts = pd.Timestamp.max if end is None else pd.Timestamp(end)
        if start_ts <= target_end <= end_ts:
            return name
    return None


def build_supervised_dataset(
    features: pd.DataFrame,
    quality_flags: pd.DataFrame,
    target: str,
    input_hours: int,
    horizon_hours: int,
    splits: dict,
    output_path: str | Path,
    manifest_path: str | Path,
) -> pd.DataFrame:
    """建立 X=(過去窗口), y=(未來窗口)，並輸出 split manifest。

    只有目標窗口完全落在同一個時間 split 才建立樣本，避免目標跨 split。
    ``quality_flags`` 必須包含 ``pm25_observed``、``long_gap``、``spike``。
    """
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("features index 必須是 DatetimeIndex")
    # 1. 檢核時間軸：必須是唯一、遞增且無中斷的連續「每小時」頻率
    if not features.index.is_monotonic_increasing or not features.index.is_unique:
        raise ValueError("features index 必須是唯一且遞增的 DatetimeIndex")
    if not features.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("features index 必須是連續每小時時間軸")
    # 2. 檢核品質標記：必須齊備三大 QC(品質標記) 欄位
    required_flags = {"pm25_observed", "long_gap", "spike"}
    missing_flags = required_flags - set(quality_flags.columns)
    if missing_flags:
        raise ValueError(f"quality_flags 缺少欄位: {sorted(missing_flags)}")

    quality_flags = quality_flags.reindex(features.index).fillna(False).astype(bool)
    values = features.to_numpy(dtype=np.float32)
    target_index = features.columns.get_loc(target)
    timestamps = features.index
    samples = {name: [] for name in splits}
    manifest_rows = []
    total_span = input_hours + horizon_hours  # 24 + 24 = 48 筆連續每小時資料

    for start in range(len(features) - total_span + 1):
        input_end = start + input_hours - 1
        target_start = input_end + 1
        target_end = target_start + horizon_hours - 1
        target_start_ts = timestamps[target_start]
        target_end_ts = timestamps[target_end]
        # 只接受完整落在同一個 split 的目標窗口，避免目標跨 Train/Val/Test 邊界
        split = _split_for_target(target_end_ts, splits)
        if split is None:
            continue
        split_start, split_end = splits[split]
        if split_start is not None and target_start_ts < pd.Timestamp(split_start):
            continue
        if split_end is not None and target_end_ts > pd.Timestamp(split_end):
            continue

        input_slice = slice(start, input_end + 1)  # [t-23 ~ t]
        target_slice = slice(target_start, target_end + 1)  # [t+1 ~ t+24]
        input_values = values[input_slice]
        target_values = values[target_slice, target_index]
        input_flags = quality_flags.iloc[input_slice]
        target_flags = quality_flags.iloc[target_slice]

        # 數值合法性 (不可包含 NaN 或 Infinite)
        if not np.isfinite(input_values).all() or not np.isfinite(target_values).all():
            continue
        # 輸入特徵品質 (排除長時段缺值、內插值、極端突波)
        if input_flags["long_gap"].any():
            continue
        if input_flags["pm25_observed"].eq(False).any():
            continue
        if input_flags["spike"].any():
            continue
        # 目標值品質 (預測目標 y 必須全為「真實觀測值」)
        if target_flags["pm25_observed"].eq(False).any():
            continue
        # 通過所有篩選才保留樣本
        samples[split].append((input_values, target_values))
        manifest_rows.append({
            "split": split,
            "input_start": timestamps[start],
            "input_end": timestamps[input_end],
            "target_start": target_start_ts,
            "target_end": target_end_ts,
        })

    # 建立輸出目錄，確保後續可以寫入壓縮資料檔
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}

    # 將各 split 的樣本整理成模型使用的 X/y 陣列
    for split, rows in samples.items():
        if rows:
            # X: (樣本數, 輸入窗口長度, 特徵數)
            # y: (樣本數, 預測窗口長度)
            arrays[f"X_{split}"] = np.stack([row[0] for row in rows])
            arrays[f"y_{split}"] = np.stack([row[1] for row in rows])
        else:
            # 沒有合格樣本時仍保留固定形狀，方便後續統一讀取
            arrays[f"X_{split}"] = np.empty((0, input_hours, features.shape[1]), dtype=np.float32)
            arrays[f"y_{split}"] = np.empty((0, horizon_hours), dtype=np.float32)

    # 儲存特徵名稱，確保載入資料時能對應每個欄位
    arrays["feature_names"] = np.asarray(features.columns, dtype=str)
    np.savez_compressed(output_path, **arrays)

    # 輸出每個樣本的時間範圍與 split 對應關係
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    return manifest