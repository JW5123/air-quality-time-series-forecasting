"""
數值特徵標準化模組
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def standardize_features(
    df: pd.DataFrame,
    feature_cols,
    train_end=None,
    scaler_path=None,
    log1p_cols=(),
):
    """
    對指定的數值型特徵做 StandardScaler 標準化 (平均值 0、標準差 1)。

    與舊版的差異：
    1. 整個 3 年只 fit「一個」scaler（舊版逐年各 fit 一個，導致同樣的 0 mm 雨量
       在 3 年分別被轉成不同數值，且年與年之間的濃度水準差異被抹平）。
    2. scaler 只用 train_end 之前的「訓練期」資料 fit，再套用到全部資料，
       避免驗證/測試期的均值、標準差洩漏到訓練過程。train_end=None 則用全部資料 fit。
    3. log1p_cols 內的欄位（如 RAINFALL）先 log1p 再標準化，降低極端豪雨的影響。
    4. df 內的 NaN（長缺口）會原樣保留；StandardScaler 計算統計量時會自動忽略 NaN。

    scaler_path 會存成 dict：{'scaler','feature_cols','log1p_cols','train_end'}，
    推論階段請用 inverse_standardize() 還原。
    """
    df = df.copy()
    feature_cols = list(feature_cols)

    for c in log1p_cols:
        if c in df.columns:
            df[c] = np.log1p(df[c].clip(lower=0))

    if train_end is not None:
        train_mask = df.index <= pd.Timestamp(train_end)
    else:
        train_mask = np.ones(len(df), dtype=bool)
    if not train_mask.any():
        raise ValueError('訓練期沒有任何資料，請檢查 TRAIN_END。')

    scaler = StandardScaler().fit(df.loc[train_mask, feature_cols])
    df[feature_cols] = scaler.transform(df[feature_cols])

    if scaler_path:
        joblib.dump(
            {
                'scaler': scaler,
                'feature_cols': feature_cols,
                'log1p_cols': [c for c in log1p_cols if c in df.columns],
                'train_end': train_end,
            },
            scaler_path,
        )
    return df, scaler


def inverse_standardize(df: pd.DataFrame, scaler_path) -> pd.DataFrame:
    """把標準化後的欄位還原成原始物理單位（含 log1p 的反轉）。"""
    bundle = joblib.load(scaler_path)
    cols = bundle['feature_cols']
    out = df.copy()
    out[cols] = bundle['scaler'].inverse_transform(out[cols])
    for c in bundle['log1p_cols']:
        out[c] = np.expm1(out[c])
    return out
