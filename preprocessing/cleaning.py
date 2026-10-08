"""
無效值清洗與異常值處理模組
"""
import numpy as np
import pandas as pd

# 環境部測站常見的無效值註記字元
INVALID_MARKERS = ['#', '*', 'x', 'A']


def _clean_invalid_chars(val, nr_as_zero: bool = False):
    """
    清洗單一儲存格：
    - 缺值 (NaN) 原樣回傳
    - 'NR' (No Rain)：只有在降雨欄位 (nr_as_zero=True) 才視為 0，其他欄位視為無效
    - 含無效註記字元 (#, *, x, A) 的值視為無效，轉為 NaN
    - 其餘嘗試轉為 float，失敗則回傳 NaN
    """
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()

    if val_str == 'NR':
        return 0.0 if nr_as_zero else np.nan

    if any(marker in val_str for marker in INVALID_MARKERS):
        return np.nan

    try:
        return float(val_str)
    except ValueError:
        return np.nan


def clean_invalid_values(df: pd.DataFrame, item_columns, nr_zero_columns=('RAINFALL',)) -> pd.DataFrame:
    """對指定的測項欄位逐一套用無效值清洗"""
    df = df.copy()
    for col in item_columns:
        nr_zero = col in nr_zero_columns
        df[col] = df[col].apply(lambda v: _clean_invalid_chars(v, nr_zero))
    return df


def apply_physical_ranges(df: pd.DataFrame, ranges: dict) -> pd.DataFrame:
    """超出物理合理範圍的值（例如 RH > 100、風向 > 360）設為 NaN，並印出各欄位筆數。"""
    df = df.copy()
    for col, (lo, hi) in ranges.items():
        if col not in df.columns:
            continue
        bad = (df[col] < lo) | (df[col] > hi)
        if bad.any():
            print(f'  物理範圍檢查 {col} [{lo}, {hi}]: {int(bad.sum())} 筆設為 NaN')
        df.loc[bad, col] = np.nan
    return df


def remove_negative_outliers(df: pd.DataFrame, columns, exclude=()) -> pd.DataFrame:
    """環境監測濃度理論上不應為負值，將負值視為異常並設為 NaN。
    exclude 內的欄位（例如氣溫）不檢查。"""
    df = df.copy()
    for col in columns:
        if col in exclude:
            continue
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            n_neg = int((df[col] < 0).sum())
            if n_neg:
                print(f'  負值檢查 {col}: {n_neg} 筆設為 NaN')
            df.loc[df[col] < 0, col] = np.nan
    return df


def flag_stuck_values(df: pd.DataFrame, min_hours: dict):
    """
    偵測儀器停滯：連續 N 小時數值「完全相同」者設為 NaN。
    前提：df 必須是完整的每小時時間軸（先 reindex_hourly），且尚未填補缺值。
    回傳 (處理後 df, 報告 DataFrame)。
    """
    df = df.copy()
    rows = []
    for col, n in min_hours.items():
        if col not in df.columns:
            continue
        s = df[col]
        run_id = (s != s.shift()).cumsum()        # NaN != NaN，所以缺值會切斷連續段
        run_len = s.groupby(run_id).transform('size')
        stuck = s.notna() & (run_len >= n)
        if stuck.any():
            for _, seg in s[stuck].groupby(run_id[stuck]):
                rows.append((col, seg.index[0], seg.index[-1], len(seg), seg.iloc[0]))
            df.loc[stuck, col] = np.nan
    report = pd.DataFrame(rows, columns=['Variable', 'Start', 'End', 'Hours', 'Value'])
    return df, report


def detect_isolated_spikes(df: pd.DataFrame, columns, k: float = 10.0, remove: bool = False):
    """
    偵測「單一小時孤立尖峰」：相對前一小時暴增、下一小時又立刻掉回。
    真實污染事件通常會持續數小時，孤立尖峰較像儀器校正或雜訊。
    remove=False 時只回報、不修改資料。
    """
    df = df.copy()
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col]
        d = s.diff()
        mad_scale = 1.4826 * (d - d.median()).abs().median()
        thr = max(k * mad_scale, 6 * d.std())
        up = (d > thr) & (d.shift(-1) < -thr * 0.5)
        down = (d < -thr) & (d.shift(-1) > thr * 0.5)
        spike = up | down
        for ts in s.index[spike.fillna(False)]:
            rows.append((col, ts, s.loc[ts], s.shift(1).loc[ts], s.shift(-1).loc[ts]))
        if remove:
            df.loc[spike.fillna(False), col] = np.nan
    report = pd.DataFrame(rows, columns=['Variable', 'Datetime', 'Value', 'Prev', 'Next'])
    return df, report
