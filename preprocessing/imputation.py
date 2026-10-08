"""
缺失值填補模組

設計原則（與舊版最大的差異）：
1. 只補「短缺口」(<= max_gap 小時)。更長的缺口整段保留 NaN。
   舊版用 interpolate(limit=4) 之後再 ffill()，會把長缺口的前 4 小時內插、
   其餘全部複製同一個值，形成長達數十小時「數值完全相同」的假資料
   （例如 NO2 連續 68 小時相同）。這些段落其實是儀器維修的長缺口。
2. 只在缺口「兩端都有資料」時內插（limit_area='inside'），
   不用 bfill，因此不會用未來值回填過去，也不會外插。
2. PM2.5 的長缺口（>= 24 小時）改用時序週期性補值：優先使用前一天
    同時段，該值缺失時再使用前一週同曜日同時段；找不到參考值則保留 NaN。
3. 風向 (度) 是循環量：350° 與 10° 之間線性內插會得到 180°（完全反向）。
   因此風向改為對 sin/cos 內插再轉回角度。
4. RAINFALL：短缺口補 0（假設無雨），長缺口保留 NaN，
   不把長時間的儀器缺測當成「沒下雨」。
"""
import numpy as np
import pandas as pd


def _gap_lengths(s: pd.Series) -> pd.Series:
    """每個 NaN 位置所屬的連續缺失長度；非 NaN 位置為 0。"""
    isna = s.isna()
    grp = (isna != isna.shift(fill_value=False)).cumsum()
    lengths = isna.groupby(grp).transform('sum')
    return lengths.where(isna, 0).astype(int)


def _fill_short_gaps(s: pd.Series, max_gap: int) -> pd.Series:
    gap = _gap_lengths(s)
    short = (gap > 0) & (gap <= max_gap)
    interpolated = s.interpolate(method='time', limit_area='inside')
    return s.where(~short, interpolated)


def _fill_pm25_long_gaps(s: pd.Series, max_gap: int) -> pd.Series:
    """以前一天或前一週同時段補 PM2.5 長缺口，再補短缺口。"""
    gap = _gap_lengths(s)
    long_gap = gap >= 24
    filled = s.copy()

    previous_day = s.shift(24)
    filled.loc[long_gap] = previous_day.loc[long_gap]

    remaining_long_gap = long_gap & filled.isna()
    previous_week = s.shift(24 * 7)
    filled.loc[remaining_long_gap] = previous_week.loc[remaining_long_gap]

    short_gap = (gap > 0) & (gap <= max_gap)
    interpolated = filled.interpolate(method='time', limit_area='inside')
    filled.loc[short_gap] = interpolated.loc[short_gap]
    return filled


def _impute_circular(deg: pd.Series, max_gap: int) -> pd.Series:
    rad = np.deg2rad(deg)
    sin = _fill_short_gaps(pd.Series(np.sin(rad), index=deg.index), max_gap)
    cos = _fill_short_gaps(pd.Series(np.cos(rad), index=deg.index), max_gap)
    out = pd.Series(np.rad2deg(np.arctan2(sin, cos)) % 360, index=deg.index)
    out[np.hypot(sin, cos) < 1e-6] = np.nan     # 兩端風向相反 → 平均後無方向
    return out


def impute_missing(
    df: pd.DataFrame,
    feature_cols,
    rainfall_col: str = 'RAINFALL',
    wind_dir_col: str = 'WD_HR',
    max_gap: int = 4,
    pm25_col: str = 'PM2.5',
) -> pd.DataFrame:
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            continue
        if col == wind_dir_col:
            df[col] = _impute_circular(df[col], max_gap)
        elif col == rainfall_col:
            gap = _gap_lengths(df[col])
            df.loc[(gap > 0) & (gap <= max_gap), col] = 0.0
        elif col == pm25_col:
            df[col] = _fill_pm25_long_gaps(df[col], max_gap)
        else:
            df[col] = _fill_short_gaps(df[col], max_gap)
    return df


def summarize_gaps(df: pd.DataFrame, cols, min_len: int = 1) -> pd.DataFrame:
    """列出各欄位的連續缺失區段 (Variable, Start, End, Hours)。"""
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        isna = df[c].isna()
        grp = (isna != isna.shift(fill_value=False)).cumsum()
        for _, seg in isna[isna].groupby(grp[isna]):
            if len(seg) >= min_len:
                rows.append((c, seg.index[0], seg.index[-1], len(seg)))
    return pd.DataFrame(rows, columns=['Variable', 'Start', 'End', 'Hours'])
