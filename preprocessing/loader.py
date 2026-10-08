"""
資料讀取與寬轉長格式模組
負責讀取環境部測站原始 CSV，並將寬格式（0~23 小時為欄位）
轉換為以 Datetime 為欄位、各測項為獨立欄位的寬表格。
"""
import pandas as pd

ONE_HOUR = pd.Timedelta(hours=1)

def load_wide_csv(input_csv_path: str) -> pd.DataFrame:
    """讀取原始 CSV，處理可能的編碼問題（優先 utf-8，失敗改 cp950）"""
    try:
        df_raw = pd.read_csv(input_csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df_raw = pd.read_csv(input_csv_path, encoding='cp950')
    return df_raw

def melt_and_pivot(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    將 0~23 小時欄位 Unpivot 成單一 Timestamp，
    再依「測項」Pivot 成獨立欄位，回傳含 Datetime 欄位的寬表格。
    假設原始欄位為: 測站, 日期, 測項, 0, 1, ..., 23

    注意：pivot_table 預設會丟掉「所有測項都是 NaN」的整列（整站該小時無資料），
    這會讓時間軸少幾個小時，所以之後必須呼叫 reindex_hourly() 補回完整時間軸。
    """
    hour_col_map = {}
    for col in df_raw.columns:
        try:
            hour = int(str(col))
        except ValueError:
            continue
        if 0 <= hour < 24:
            hour_col_map[hour] = col

    missing_hours = sorted(set(range(24)) - set(hour_col_map))
    if missing_hours:
        raise ValueError(f'原始資料缺少小時欄位: {missing_hours}')

    hour_cols = [hour_col_map[i] for i in range(24)]
    id_vars = [col for col in df_raw.columns if col not in hour_cols]

    df_melt = pd.melt(
        df_raw, id_vars=id_vars, value_vars=hour_cols,
        var_name='Hour', value_name='Value'
    )

    # 若檔案含多個測站，pivot 的 aggfunc='first' 會悄悄混在一起，這裡直接擋下
    station_col = '測站' if '測站' in df_melt.columns else None
    if station_col and df_melt[station_col].nunique() > 1:
        raise ValueError(
            f'檔案內含多個測站 {df_melt[station_col].unique().tolist()}，請先篩選單一測站。'
        )

    date_col = '日期' if '日期' in df_melt.columns else 'DATE'
    item_col = '測項' if '測項' in df_melt.columns else 'ITEM'

    df_melt['Hour'] = df_melt['Hour'].map(lambda value: int(str(value)))
    df_melt['Datetime'] = (
        pd.to_datetime(df_melt[date_col]) +
        pd.to_timedelta(df_melt['Hour'], unit='h')
    )

    df_pivot = df_melt.pivot_table(
        index='Datetime', columns=item_col, values='Value', aggfunc='first'
    )
    df_pivot.reset_index(inplace=True)
    return df_pivot


def load_years(input_csv_paths) -> pd.DataFrame:
    """讀取多個年度檔，直接合併成一份「單一時間軸」的寬表格（尚未清洗）。

    重點：3 年必須合併後再做清洗 / 填補 / 標準化，
    不可以逐年各處理一次，否則年與年的交界會被切斷、每年 scaler 也會不同。
    """
    frames = []
    for path in input_csv_paths:
        print(f'讀取: {path}')
        frames.append(melt_and_pivot(load_wide_csv(path)))
    if not frames:
        raise ValueError('至少需要提供一個原始資料檔案。')
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values('Datetime').drop_duplicates('Datetime', keep='first')
    return df


def reindex_hourly(df: pd.DataFrame):
    """將 DatetimeIndex 補成「每小時一筆」的完整時間軸（起點取當天 00:00、終點取最後一天 23:00）。
    缺的小時會以 NaN 補回，回傳 (補齊後的 df, 被補回的時間點 Index)。"""
    df = df[~df.index.duplicated(keep='first')].sort_index()
    full = pd.date_range(
        df.index.min().normalize(),
        df.index.max().normalize() + pd.Timedelta(hours=23),
        freq=ONE_HOUR, name='Datetime',
    )
    added = full.difference(df.index)
    return df.reindex(full), added
