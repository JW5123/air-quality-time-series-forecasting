"""EDA 共用工具：繪圖風格、欄位標籤、存檔。"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # 不需要顯示視窗，直接存檔
import matplotlib.pyplot as plt
import seaborn as sns

LABELS = {
    'PM2.5': 'PM2.5 (µg/m³)', 'PM10': 'PM10 (µg/m³)',
    'O3': 'O3 (ppb)', 'NO2': 'NO2 (ppb)', 'SO2': 'SO2 (ppb)', 'CO': 'CO (ppm)',
    'WS_HR': 'Wind speed (m/s)', 'AMB_TEMP': 'Temperature (°C)',
    'RH': 'RH (%)', 'RAINFALL': 'Rainfall (mm)', 'WD_HR': 'Wind direction (°)',
}
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
DOW_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


def setup_style():
    sns.set_theme(style='whitegrid', context='notebook')
    plt.rcParams.update({
        'savefig.dpi': 150,
        'axes.titlesize': 11,
        'axes.titleweight': 'bold',
        'axes.labelsize': 10,
        'font.sans-serif': ['Microsoft JhengHei', 'Noto Sans CJK TC', 'PingFang TC',
                            'Arial Unicode MS', 'DejaVu Sans'],
        'axes.unicode_minus': False,
    })


def label(col: str) -> str:
    return LABELS.get(col, col)


def save(fig, out_dir, name: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name, bbox_inches='tight')
    plt.close(fig)


def run_steps(steps, out_dir):
    """依序執行 (檔名, 函式)；單一圖表失敗不會中斷整個流程。"""
    for name, fn in steps:
        try:
            fn()
            print(f'  ✓ {name}')
        except Exception as e:   # noqa: BLE001
            plt.close('all')
            print(f'  ✗ {name} 失敗: {type(e).__name__}: {e}')
    print(f'圖表輸出資料夾: {out_dir}')


def draw_corr_heatmap(ax, corr, annot=True, fmt='.2f', cbar=True, annot_size=8, mask_upper=True):
    import numpy as np
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1) if mask_upper else None
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                annot=annot, fmt=fmt, annot_kws={'size': annot_size},
                square=True, linewidths=.4, cbar=cbar,
                cbar_kws={'shrink': .7, 'label': 'r'} if cbar else None, ax=ax)
