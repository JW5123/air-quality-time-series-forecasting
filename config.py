"""
集中管理所有可調參數。要換測站、換年份、調整門檻，只需要改這個檔案。
"""
from pathlib import Path

# ---------- 輸入 / 輸出 ----------
INPUT_FILES = [
    "dataset/西屯_2021.csv",
    "dataset/西屯_2022.csv",
    "dataset/西屯_2023.csv",
    "dataset/西屯_2024.csv",
    "dataset/西屯_2025.csv",
]
OUTPUT_DIR = Path("output")

CLEANED_CSV = OUTPUT_DIR / "西屯_2021_2025_cleaned.csv"      # 清洗後、特徵工程前（物理單位）
PROCESSED_CSV = OUTPUT_DIR / "西屯_2021_2025_processed.csv"  # 特徵工程 + 標準化後（模型輸入）
SCALER_PATH = OUTPUT_DIR / "scaler.pkl"                       # 只有「一個」scaler，橫跨 3 年
QUALITY_FLAGS_CSV = OUTPUT_DIR / "西屯_2021_2025_quality_flags.csv"
SUPERVISED_DATA_DIR = OUTPUT_DIR / "supervised"
REPORT_DIR = OUTPUT_DIR / "reports"
EDA_STAGE1_DIR = OUTPUT_DIR / "eda_1_cleaned"
EDA_STAGE2_DIR = OUTPUT_DIR / "eda_2_features"

# ---------- 欄位 ----------
TARGET_COLS = [
    "PM2.5", "PM10", "O3", "NO2", "SO2", "CO",
    "WS_HR", "AMB_TEMP", "RH", "RAINFALL", "WD_HR",
]
TARGET = "PM2.5"                      # 預測目標（EDA 的 ACF/PACF/STL/Lag 都以它為主）
RAINFALL_COL = "RAINFALL"
WIND_DIR_COL = "WD_HR"
WIND_SPEED_COL = "WS_HR"

# ---------- 清洗 ----------
# 物理合理範圍，超出視為無效 (NaN)。單位依環境部：PM µg/m3、氣體 ppb/ppm、風速 m/s、溫度 °C、RH %
PHYSICAL_RANGES = {
    "RH": (0, 100),
    "WD_HR": (0, 360),
    "WS_HR": (0, 60),
    "AMB_TEMP": (-10, 45),
}
# 負值視為無效的欄位排除清單（溫度本來就可以為負）
NEGATIVE_CHECK_EXCLUDE = ["AMB_TEMP"]

# 連續多少小時「數值完全相同」視為儀器停滯 → 設為 NaN。
# 門檻依解析度而定：SO2/CO/RH 解析度較粗，門檻放寬。RAINFALL、WD_HR 不檢查（無雨/靜風本來就會連續相同）。
STUCK_MIN_HOURS = {
    "PM2.5": 12, "PM10": 12, "O3": 12, "NO2": 12,
    "SO2": 24, "CO": 24, "WS_HR": 12, "AMB_TEMP": 12, "RH": 24,
}

# 單一小時的孤立尖峰（前後都正常）：預設只回報不刪除，確認是儀器異常後再改成 True
SPIKE_CHECK_COLS = ["PM2.5", "PM10", "O3", "NO2", "SO2", "CO"]
REMOVE_SPIKES = False

# ---------- 缺值填補 ----------
# 只補「連續缺失 <= MAX_GAP_HOURS」的短缺口；更長的缺口保留 NaN（不再用 ffill 硬填）
MAX_GAP_HOURS = 4

# ---------- 標準化 ----------
# scaler 只用「訓練期」fit，避免測試期統計量洩漏。訓練期之後的資料視為驗證/測試。
TRAIN_END = "2023-12-31 23:00"
# 嚴重右偏（大量 0 + 極端豪雨）的欄位先做 log1p 再標準化；不需要就改成 []
LOG1P_COLS = ["RAINFALL"]

# ---------- 監督式資料集 ----------
INPUT_WINDOW_HOURS = 24
FORECAST_HORIZON_HOURS = 24
SPLITS = {
    "train": ("2021-01-01 00:00", "2023-12-31 23:00"),
    "validation": ("2024-01-01 00:00", "2024-12-31 23:00"),
    "test": ("2025-01-01 00:00", "2025-12-31 23:00"),
}

# ---------- EDA ----------
MAX_LAG = 72   # Lag-Correlation 最大落後小時數
