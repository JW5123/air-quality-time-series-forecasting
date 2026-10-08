"""
EDA 階段二：特徵工程（風向 sin/cos + 時間週期）與標準化之後的相關分析。
輸入為模型實際使用的資料（processed），用來：
  1. 檢查所有特徵之間的相關性（共線性、特徵篩選）
  2. 驗證 sin/cos 編碼正確（Cyclical 2D）
  3. 用 Lag-Correlation 決定要建立哪些 lag 特徵
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .common import setup_style, save, run_steps, draw_corr_heatmap, MONTH_NAMES, DOW_NAMES

TIME_FEATURES = ['Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 'Month_sin', 'Month_cos']


# ----------------------------------------------------------------------------
# 1. 特徵相關性
# ----------------------------------------------------------------------------
def plot_full_correlation(df, out):
    corr = df.corr()
    n = len(corr)
    fig, ax = plt.subplots(figsize=(0.75 * n + 3, 0.68 * n + 2))
    draw_corr_heatmap(ax, corr, annot_size=7)
    ax.set_title('所有模型特徵相關係數熱圖（特徵工程後，Pearson）')
    save(fig, out, '11_correlation_heatmap_features.png')

    # 高共線性配對（|r| >= 0.8）
    pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
             .rename('r').reset_index().rename(columns={'level_0': 'A', 'level_1': 'B'}))
    high = pairs[pairs.r.abs() >= 0.8].sort_values('r', key=abs, ascending=False)
    high.to_csv(out / 'high_collinearity_pairs.csv', index=False)
    if len(high):
        print('    高共線性配對 (|r|>=0.8): ' + '; '.join(f'{a}~{b} ({r:+.2f})' for a, b, r in high.itertuples(index=False)))


def plot_target_correlation(df, target, out):
    others = [c for c in df.columns if c != target]
    pear = df[others].corrwith(df[target])
    spear = df[others].corrwith(df[target], method='spearman')
    order = pear.abs().sort_values().index
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(order) + 2))
    ax.barh(y - .2, pear[order], height=.4, label='Pearson')
    ax.barh(y + .2, spear[order], height=.4, label='Spearman')
    ax.set_yticks(y); ax.set_yticklabels(order); ax.axvline(0, color='k', lw=.6)
    ax.set_xlabel(f'與 {target} 的相關係數（lag 0）'); ax.legend()
    ax.set_title(f'特徵與目標值 {target} 的相關係數')
    save(fig, out, '12_target_correlation_bar.png')


# ----------------------------------------------------------------------------
# 2. Cyclical 2D：驗證 sin/cos 編碼
# ----------------------------------------------------------------------------
def _unit_circle(ax):
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), ls='--', lw=.8, color='gray')
    ax.set_aspect('equal'); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.axhline(0, color='k', lw=.3); ax.axvline(0, color='k', lw=.3)


def _cyclic_panel(ax, cos, sin, key, names, cmap, title, expected):
    _unit_circle(ax)
    sc = ax.scatter(cos, sin, c=key, cmap=cmap, s=25, alpha=.35, vmin=key.min(), vmax=key.max())
    uniq = pd.DataFrame({'c': cos, 's': sin, 'k': key}).drop_duplicates('k').sort_values('k')
    for _, r in uniq.iterrows():
        ax.annotate(names[int(r.k)], (r.c, r.s), xytext=(4, 4), textcoords='offset points', fontsize=8, weight='bold')
    radius = np.hypot(cos, sin)
    err = np.nanmax(np.abs(np.c_[cos, sin] - expected))
    ax.set_title(f'{title}\n半徑 {radius.min():.3f}-{radius.max():.3f}，與重算值的最大誤差 = {err:.1e}')
    ax.set_xlabel('cos'); ax.set_ylabel('sin')


def plot_cyclical_2d(df, out):
    idx = df.index
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))

    h = idx.hour.values
    exp = np.c_[np.cos(2 * np.pi * h / 24), np.sin(2 * np.pi * h / 24)]
    _cyclic_panel(axes[0, 0], df['Hour_cos'].values, df['Hour_sin'].values, h, [f'{i}h' for i in range(24)],
                  'twilight_shifted', 'Hour of day', exp)

    d = idx.dayofweek.values
    exp = np.c_[np.cos(2 * np.pi * d / 7), np.sin(2 * np.pi * d / 7)]
    _cyclic_panel(axes[0, 1], df['DayOfWeek_cos'].values, df['DayOfWeek_sin'].values, d, DOW_NAMES,
                  'twilight_shifted', 'Day of week', exp)

    m = idx.month.values - 1
    exp = np.c_[np.cos(2 * np.pi * m / 12), np.sin(2 * np.pi * m / 12)]
    _cyclic_panel(axes[1, 0], df['Month_cos'].values, df['Month_sin'].values, m, MONTH_NAMES,
                  'twilight_shifted', 'Month', exp)

    ax = axes[1, 1]
    _unit_circle(ax)
    ok = df[['WD_HR_cos', 'WD_HR_sin']].dropna()
    hb = ax.hexbin(ok['WD_HR_cos'], ok['WD_HR_sin'], gridsize=40, cmap='viridis', mincnt=1, bins='log', extent=(-1.1, 1.1, -1.1, 1.1))
    fig.colorbar(hb, ax=ax, shrink=.7, label='hours (log)')
    for deg, nm in [(0, 'N (0°)'), (90, 'E (90°)'), (180, 'S (180°)'), (270, 'W (270°)')]:
        ax.annotate(nm, (np.cos(np.deg2rad(deg)) * 1.15, np.sin(np.deg2rad(deg)) * 1.15), ha='center', weight='bold')
    r = np.hypot(ok['WD_HR_cos'], ok['WD_HR_sin'])
    ax.set_title(f'風向（密度）\n半徑 {r.min():.3f}-{r.max():.3f}（應接近 1）')
    ax.set_xlabel('cos'); ax.set_ylabel('sin')

    fig.suptitle('週期特徵 2D 檢查：所有點應位於單位圓上，時間相鄰點也應在圓上相鄰',
                 fontweight='bold', y=1.0)
    fig.tight_layout()
    save(fig, out, '13_cyclical_2d.png')


# ----------------------------------------------------------------------------
# 3. Lag-Correlation：決定 lag 特徵
# ----------------------------------------------------------------------------
def compute_lag_correlation(df, target, cols, max_lag):
    y = df[target]
    res = {}
    for c in cols:
        res[c] = [np.nan if (c == target and lag == 0) else y.corr(df[c].shift(lag))
                  for lag in range(max_lag + 1)]
    return pd.DataFrame(res, index=pd.Index(range(max_lag + 1), name='lag_hours')).T


def plot_lag_correlation(df, target, out, max_lag=72):
    """corr( target(t), feature(t - lag) )：lag > 0 表示「特徵領先目標 lag 小時」。
    前提：df 索引為完整的每小時時間軸，shift 才等於「往前 N 小時」。"""
    cols = [c for c in df.columns if c not in TIME_FEATURES]
    lc = compute_lag_correlation(df, target, cols, max_lag)
    lc.to_csv(out / 'lag_correlation_table.csv', encoding='utf-8-sig')

    strength = lc.iloc[:, 1:].abs().max(axis=1).sort_values(ascending=False)
    lc = lc.loc[strength.index]
    lim = float(np.ceil(np.nanmax(np.abs(lc.values)) * 10) / 10)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1])
    ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])
    sns.heatmap(lc, cmap='coolwarm', vmin=-lim, vmax=lim, center=0, ax=ax1,
                cbar_kws={'label': 'r', 'shrink': .8}, xticklabels=6, linewidths=0)
    ax1.set_xlabel('落後時間（小時）：t-lag 的特徵 vs. t 的目標值')
    ax1.set_title(f'與 {target} 的落後相關係數熱圖（依 lag >= 1 的最強相關排序；目標值列 = 自相關）')

    top = [c for c in lc.index if c != target][:6]
    for c in [target] + top:
        ax2.plot(lc.columns[1:] if c == target else lc.columns, lc.loc[c].iloc[1:] if c == target else lc.loc[c],
                 lw=2.2 if c == target else 1.3, label=c + (' (autocorr)' if c == target else ''))
    for k in range(24, max_lag + 1, 24):
        ax2.axvline(k, ls=':', lw=.6, color='gray')
    ax2.axhline(0, color='k', lw=.5)
    ax2.set_xlabel('落後時間（小時）'); ax2.set_ylabel('r'); ax2.legend(ncol=4, fontsize=9)
    ax2.set_title('落後相關曲線：目標值自相關與落後訊號最強的 6 個特徵')
    fig.tight_layout()
    save(fig, out, '14_lag_correlation.png')

    # 每個特徵：lag0、最佳 lag、常用 lag 的相關係數
    rows = []
    for c in lc.index:
        s = lc.loc[c]; s1 = s.iloc[1:]
        best = int(s1.abs().idxmax())
        rows.append({'feature': c, 'r_lag0': s.iloc[0], 'best_lag_h': best, 'r_at_best_lag': s[best],
                     **{f'r_lag{k}': s[k] for k in (1, 3, 6, 12, 24, 48) if k <= max_lag}})
    summ = pd.DataFrame(rows)
    summ.to_csv(out / 'lag_summary.csv', index=False, encoding='utf-8-sig')
    print('    最佳 lag 摘要: ' + '; '.join(
        f'{r.feature}: {r.best_lag_h}h (r={r.r_at_best_lag:+.2f})' for r in summ.head(6).itertuples()))


# ----------------------------------------------------------------------------
def run_stage2(df, out_dir, target='PM2.5', max_lag=72):
    from pathlib import Path
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    steps = [
        ('11 correlation heatmap (all features)', lambda: plot_full_correlation(df, out_dir)),
        ('12 target correlation bar', lambda: plot_target_correlation(df, target, out_dir)),
        ('13 cyclical 2D', lambda: plot_cyclical_2d(df, out_dir)),
        ('14 lag-correlation', lambda: plot_lag_correlation(df, target, out_dir, max_lag)),
    ]
    run_steps(steps, out_dir)
