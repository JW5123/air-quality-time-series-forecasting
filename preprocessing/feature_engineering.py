"""
特徵工程模組：風向角度轉換、時間週期性編碼 (Cyclical Encoding)
"""
import numpy as np
import pandas as pd


def encode_wind_direction(df: pd.DataFrame, wd_col: str = 'WD_HR') -> pd.DataFrame:
    """
    將風向角度 (0~360 度) 轉為 sin/cos 分量。
    切勿把角度直接當數值輸入模型，否則 0 度和 359 度會被誤判為
    距離很遠，實際上兩者方向幾乎相同。
    """
    df = df.copy()
    if wd_col not in df.columns:
        return df
    radians = np.deg2rad(df[wd_col])
    df[f'{wd_col}_sin'] = np.sin(radians)
    df[f'{wd_col}_cos'] = np.cos(radians)
    df.drop(columns=[wd_col], inplace=True)
    return df


def encode_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    加入 Hour / DayOfWeek / Month 的正餘弦週期編碼 (Cyclical Encoding)。
    三者皆為循環量（23 點後接 0 點、週日後接週一、12 月後接 1 月），
    若直接當作原始整數輸入，模型會誤判首尾之間距離很遠，
    因此統一轉換為 sin/cos 分量，讓週期首尾在數值空間上相鄰。
    """
    df = df.copy()
    hour = df.index.hour
    day_of_week = df.index.dayofweek
    month = df.index.month

    df['Hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['Hour_cos'] = np.cos(2 * np.pi * hour / 24.0)

    df['DayOfWeek_sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * day_of_week / 7.0)

    df['Month_sin'] = np.sin(2 * np.pi * (month - 1) / 12.0)
    df['Month_cos'] = np.cos(2 * np.pi * (month - 1) / 12.0)

    return df
