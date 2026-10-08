"""
環境部測站空氣品質資料前處理主程式

流程：
    讀取 5 年 -> 合併成單一時間軸 -> 無效值清洗 -> 補齊完整每小時時間軸
    -> 物理範圍/負值/儀器停滯/孤立尖峰檢查
    -> 補短缺口 + PM2.5 長缺口的時序週期性數值
  -> 【輸出 cleaned.csv + EDA 第一階段（清洗後、特徵工程前）】
  -> 風向 sin/cos + 時間週期特徵 -> 標準化（單一 scaler、只用訓練期 fit）
  -> 【輸出 processed.csv + EDA 第二階段（特徵工程後相關分析）】

用法：  python main.py            # 完整流程
        python main.py --no-eda   # 只做前處理，不畫圖
"""
import argparse

import pandas as pd

import config as cfg
from preprocessing.loader import load_years, reindex_hourly
from preprocessing.cleaning import (
    clean_invalid_values, apply_physical_ranges, remove_negative_outliers,
    flag_stuck_values, detect_isolated_spikes,
)
from preprocessing.imputation import impute_missing, summarize_gaps
from preprocessing.feature_engineering import encode_wind_direction, encode_cyclical_time_features
from preprocessing.scaling import standardize_features
from preprocessing.supervised import build_supervised_dataset
from eda.stage1_cleaned import run_stage1
from eda.stage2_features import run_stage2


def build_cleaned_dataset() -> pd.DataFrame:
    """階段一：原始檔 -> 清洗完成、尚未做特徵工程/標準化的資料（物理單位）"""
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 讀取 3 年並合併成單一時間軸
    df_pivot = load_years(cfg.INPUT_FILES)

    # 2. 清洗無效資料標記 (#, *, x, A, NR)
    print('清洗無效資料標記 (#, *, x, A, NR)...')
    item_columns = df_pivot.columns.drop('Datetime')
    df_pivot = clean_invalid_values(df_pivot, item_columns, nr_zero_columns=(cfg.RAINFALL_COL,))

    # 3. 篩選欄位，設定時間索引
    cols = [c for c in cfg.TARGET_COLS if c in df_pivot.columns]
    missing_cols = sorted(set(cfg.TARGET_COLS) - set(cols))
    if missing_cols:
        print(f'  警告：原始資料缺少欄位 {missing_cols}')
    df = df_pivot[['Datetime'] + cols].set_index('Datetime')
    df.columns.name = None      # pivot 殘留的欄位索引名稱「測項」會干擾後續 corr/stack 與 CSV 表頭

    # 4. 補齊完整每小時時間軸（原本會被 pivot 悄悄丟掉的整小時無資料列）
    df, added = reindex_hourly(df)
    print(f'補齊時間軸：共 {len(df)} 小時，補回 {len(added)} 個整小時缺漏 {list(added.strftime("%Y-%m-%d %H:%M"))}')

    # 5. 物理範圍 + 負值檢查
    df = apply_physical_ranges(df, cfg.PHYSICAL_RANGES)
    df = remove_negative_outliers(df, cols, exclude=cfg.NEGATIVE_CHECK_EXCLUDE)

    # 6. 儀器停滯（連續相同值）
    df, stuck = flag_stuck_values(df, cfg.STUCK_MIN_HOURS)
    stuck.to_csv(cfg.REPORT_DIR / 'stuck_values.csv', index=False, encoding='utf-8-sig')
    print(f'儀器停滯區段：{len(stuck)} 段（{int(stuck.Hours.sum()) if len(stuck) else 0} 小時）已設為 NaN，'
          f'明細見 {cfg.REPORT_DIR / "stuck_values.csv"}')

    # 7. 孤立尖峰（預設只回報）
    df, spikes = detect_isolated_spikes(df, cfg.SPIKE_CHECK_COLS, remove=cfg.REMOVE_SPIKES)
    spikes.to_csv(cfg.REPORT_DIR / 'isolated_spikes.csv', index=False, encoding='utf-8-sig')
    action = '已設為 NaN' if cfg.REMOVE_SPIKES else '僅回報（REMOVE_SPIKES=False）'
    print(f'孤立尖峰：{len(spikes)} 筆，{action}，明細見 {cfg.REPORT_DIR / "isolated_spikes.csv"}')

    # 保留補值與異常點來源，供後續建立窗口時排除受污染樣本。
    quality_flags = pd.DataFrame(index=df.index)
    quality_flags['pm25_observed'] = df[cfg.TARGET].notna()
    quality_flags['spike'] = df.index.isin(spikes['Datetime']) if len(spikes) else False
    quality_flags['long_gap'] = False
    for _, gap in summarize_gaps(df, cols, min_len=cfg.MAX_GAP_HOURS + 1).iterrows():
        quality_flags.loc[gap['Start']:gap['End'], 'long_gap'] = True

    # 8. 補短缺口，以及 PM2.5 的前一天/前一週同時段長缺口
    n_before = int(df[cols].isna().sum().sum())
    df = impute_missing(df, cols, cfg.RAINFALL_COL, cfg.WIND_DIR_COL, cfg.MAX_GAP_HOURS)
    n_after = int(df[cols].isna().sum().sum())
    print(f'缺值填補：短缺口與 PM2.5 長缺口週期性補值已處理，NaN {n_before} -> {n_after}')

    long_gaps = summarize_gaps(df, cols, min_len=1)
    long_gaps.to_csv(cfg.REPORT_DIR / 'remaining_gaps.csv', index=False, encoding='utf-8-sig')
    print('各欄位剩餘缺值比例 (%):')
    print((df[cols].isna().mean() * 100).round(2).to_string())

    df.to_csv(cfg.CLEANED_CSV, encoding='utf-8-sig')
    quality_flags.to_csv(cfg.QUALITY_FLAGS_CSV, encoding='utf-8-sig')
    print(f'清洗後資料已輸出: {cfg.CLEANED_CSV}')
    return df


def build_supervised_files(features: pd.DataFrame) -> None:
    """輸出 24 小時輸入、24 小時目標的 XGBoost-ready 窗口資料。"""
    flags = pd.read_csv(cfg.QUALITY_FLAGS_CSV, index_col='Datetime', parse_dates=True)
    manifest = build_supervised_dataset(
        features=features,
        quality_flags=flags,
        target=cfg.TARGET,
        input_hours=cfg.INPUT_WINDOW_HOURS,
        horizon_hours=cfg.FORECAST_HORIZON_HOURS,
        splits=cfg.SPLITS,
        output_path=cfg.SUPERVISED_DATA_DIR / 'dataset.npz',
        manifest_path=cfg.SUPERVISED_DATA_DIR / 'manifest.csv',
    )
    print(f'監督式窗口已輸出：{cfg.SUPERVISED_DATA_DIR / "dataset.npz"}')
    print(f'有效樣本數：{manifest.groupby("split").size().to_dict() if len(manifest) else {}}')


def build_feature_dataset(df_cleaned: pd.DataFrame) -> pd.DataFrame:
    """階段二：特徵工程 + 標準化"""
    df = df_cleaned.copy()

    print('轉換風向為 Sin/Cos 分量...')
    df = encode_wind_direction(df, cfg.WIND_DIR_COL)

    print('加入時間週期特徵 (Hour / DayOfWeek / Month)...')
    df = encode_cyclical_time_features(df)

    print(f'標準化：單一 scaler，只用 <= {cfg.TRAIN_END} 的資料 fit；log1p: {cfg.LOG1P_COLS}')
    numeric_cols = [c for c in df_cleaned.columns if c != cfg.WIND_DIR_COL]
    df, _ = standardize_features(
        df, numeric_cols, train_end=cfg.TRAIN_END,
        scaler_path=cfg.SCALER_PATH, log1p_cols=cfg.LOG1P_COLS,
    )

    df.to_csv(cfg.PROCESSED_CSV, encoding='utf-8-sig')
    print(f'特徵工程後資料已輸出: {cfg.PROCESSED_CSV}')
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-eda', action='store_true', help='只做前處理，不輸出 EDA 圖表')
    args = parser.parse_args()

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cleaned = build_cleaned_dataset()
    if not args.no_eda:
        print('\n===== EDA 階段一：清洗後、特徵工程前 =====')
        run_stage1(cleaned, cfg.EDA_STAGE1_DIR, cfg.TARGET)

    features = build_feature_dataset(cleaned)
    build_supervised_files(features)
    if not args.no_eda:
        print('\n===== EDA 階段二：特徵工程後相關分析 =====')
        run_stage2(features, cfg.EDA_STAGE2_DIR, cfg.TARGET, cfg.MAX_LAG)

    print('\n全部完成。')
