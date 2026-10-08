"""
EDA 階段一：資料清洗後、特徵工程前（物理單位、風向仍為度數）。
目的：確認資料品質、分佈、週期性、時間相依結構與風場特性。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import STL
try:
    from statsmodels.tsa.seasonal import MSTL
except ImportError:      # statsmodels < 0.14
    MSTL = None

from .common import (setup_style, label, save, run_steps, draw_corr_heatmap,
                    MONTH_NAMES)

WD, WS = 'WD_HR', 'WS_HR'
POLLUTANTS = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']


def _continuous_cols(df):
    """不含風向（循環量，不適合 Pearson / boxplot）的連續欄位"""
    return [c for c in df.columns if c != WD]


# ----------------------------------------------------------------------------
# 0. 資料品質：缺值總覽 + 原始時間序列
# ----------------------------------------------------------------------------
def plot_missing_overview(df, out):
    cols = list(df.columns)
    monthly = (df[cols].isna().groupby(df.index.to_period('M')).mean().T * 100)
    pct_total = df[cols].isna().mean() * 100
    monthly.index = [f'{c} ({pct_total[c]:.1f}%)' for c in monthly.index]
    annot = monthly.round(0).astype(int).astype(str).where(monthly > 0, '')
    fig, ax = plt.subplots(figsize=(max(12, 0.42 * monthly.shape[1] + 4), 0.55 * len(cols) + 2))
    sns.heatmap(monthly, cmap='Reds', vmin=0, vmax=100, annot=annot, fmt='',
                annot_kws={'size': 7}, linewidths=.3, linecolor='white',
                cbar_kws={'label': '% missing in month'}, ax=ax)
    ax.set_xticklabels([str(p) for p in monthly.columns], rotation=90, fontsize=8)
    ax.set_title('清洗後每月缺值比例（剩餘 NaN = 超過填補上限的缺口）')
    save(fig, out, '00_missing_overview.png')


def plot_timeseries_overview(df, out):
    cols = _continuous_cols(df)
    fig, axes = plt.subplots(len(cols), 1, figsize=(15, 1.9 * len(cols)), sharex=True)
    for ax, c in zip(np.atleast_1d(axes), cols):
        ax.plot(df.index, df[c], lw=.4, alpha=.55, color='tab:blue', label='hourly')
        ax.plot(df.index, df[c].rolling(24 * 7, min_periods=24).mean(),
                lw=1.3, color='tab:red', label='7-day mean')
        ax.set_ylabel(label(c), fontsize=8)
        ax.margins(x=0)
    np.atleast_1d(axes)[0].legend(loc='upper right', ncol=2, fontsize=8)
    np.atleast_1d(axes)[0].set_title('原始時間序列（缺口為真實 NaN，未填補）')
    save(fig, out, '01_timeseries_overview.png')


def plot_target_detail(df, target, out):
    s = df[target]
    daily = s.resample('D').mean()
    fig, axes = plt.subplots(3, 1, figsize=(15, 11))

    axes[0].plot(daily.index, daily, lw=.6, color='tab:blue', label='daily mean')
    axes[0].plot(daily.index, daily.rolling(30, min_periods=10).mean(),
                 lw=1.8, color='tab:red', label='30-day rolling mean')
    axes[0].set_title(f'{target}：每日平均')
    axes[0].set_ylabel(label(target)); axes[0].legend(); axes[0].margins(x=0)

    m = s.resample('MS').mean()
    tbl = pd.DataFrame({'year': m.index.year, 'month': m.index.month, 'v': m.values}).pivot(
        index='month', columns='year', values='v')
    for y in tbl.columns:
        axes[1].plot(tbl.index, tbl[y], marker='o', label=str(y))
    axes[1].set_xticks(range(1, 13)); axes[1].set_xticklabels(MONTH_NAMES)
    axes[1].set_title(f'{target}：各年份每月平均（年度比較）')
    axes[1].set_ylabel(label(target)); axes[1].legend(title='Year')

    roll = s.rolling(24 * 14, min_periods=24 * 10).mean()
    if roll.notna().any():
        end = roll.idxmax(); start = end - pd.Timedelta(days=14)
        w = s[start:end]
        axes[2].plot(w.index, w, lw=1, marker='.', ms=3)
        axes[2].set_title(f'{target}：污染最高 14 天的逐時放大圖（{start:%Y-%m-%d} 至 {end:%Y-%m-%d}）')
        axes[2].set_ylabel(label(target))
    fig.tight_layout()
    save(fig, out, '01b_target_detail.png')


# ----------------------------------------------------------------------------
# 1. 分佈：Histogram + KDE
# ----------------------------------------------------------------------------
def plot_histograms(df, out):
    cols = list(df.columns)
    ncols = 3; nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.4 * nrows))
    axes = axes.ravel()
    for ax, c in zip(axes, cols):
        x = df[c].dropna()
        if c == 'RAINFALL':
            wet = x[x > 0]
            sns.histplot(wet, log_scale=True, kde=True, bins=40, ax=ax, color='tab:blue')
            ax.set_title(f'{c}（僅降雨時段，x 軸為對數尺度）')
            ax.text(.97, .95, f'zero share={(x == 0).mean() * 100:.0f}%\nskew={x.skew():.1f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=8)
        else:
            sns.histplot(x, kde=True, bins=36 if c == WD else 50, ax=ax, color='tab:blue')
            ax.axvline(x.median(), color='red', ls='--', lw=1)
            ax.set_title(c)
            ax.text(.97, .95, f'skew={x.skew():.2f}\nmedian={x.median():.1f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=8)
        ax.set_xlabel(label(c))
    for ax in axes[len(cols):]:
        ax.axis('off')
    fig.suptitle('直方圖與 KDE（紅色虛線 = 中位數）', y=1.0, fontweight='bold')
    fig.tight_layout()
    save(fig, out, '02_histogram_kde.png')


# ----------------------------------------------------------------------------
# 2. 週期性：日內 / 季節 Boxplot
# ----------------------------------------------------------------------------
def _grouped_boxplots(df, group, xticklabels, xlabel, title, out, name, rain_agg, rain_desc):
    cols = _continuous_cols(df)
    ncols = 2; nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.3 * nrows))
    axes = axes.ravel()
    for ax, c in zip(axes, cols):
        if c == 'RAINFALL':
            agg = rain_agg(df[c])
            ax.bar(agg.index, agg.values, color='tab:blue', alpha=.8)
            ax.set_title(f'{c}（{rain_desc}；零值過多，箱型圖參考性有限）')
        else:
            tmp = pd.DataFrame({'g': group, 'v': df[c].values})
            sns.boxplot(data=tmp, x='g', y='v', ax=ax, showfliers=False,
                        color='#8fb8de', linewidth=.8)
            ax.set_title(c)
        ax.set_xlabel(xlabel); ax.set_ylabel(label(c), fontsize=8)
        ax.set_xticks(range(len(xticklabels)) if c != 'RAINFALL' else sorted(set(group)))
        ax.set_xticklabels(xticklabels, fontsize=8)
    for ax in axes[len(cols):]:
        ax.axis('off')
    fig.suptitle(title, y=1.0, fontweight='bold')
    fig.tight_layout()
    save(fig, out, name)


def plot_diurnal_boxplot(df, out):
    hour = df.index.hour
    _grouped_boxplots(
        df, hour, [str(h) for h in range(24)], 'Hour of day',
        '日內變化型態（隱藏離群值）', out, '03_diurnal_boxplot.png',
        rain_agg=lambda s: s.groupby(s.index.hour).mean(), rain_desc='mean mm per hour')


def plot_seasonal_boxplot(df, out):
    month = df.index.month
    def monthly_total(s):
        tot = s.resample('MS').sum(min_count=1)
        return tot.groupby(tot.index.month).mean()
    _grouped_boxplots(
        df, month, MONTH_NAMES, 'Month',
        '季節變化型態（隱藏離群值）', out, '04_seasonal_boxplot.png',
        rain_agg=monthly_total, rain_desc='mean monthly total')


# ----------------------------------------------------------------------------
# 3. 時間相依：ACF / PACF / STL
# ----------------------------------------------------------------------------
def _diag_series(df, target):
    """ACF/PACF/STL 需要無缺值序列：僅用於診斷，線性內插全部缺口（不會寫回資料）。"""
    return df[target].interpolate(limit_direction='both')


def plot_acf_pacf(df, target, out, lags=72, long_lags=24 * 14):
    s = _diag_series(df, target)
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[1, :])
    plot_acf(s, lags=lags, ax=ax1, zero=False, title=f'{target} 自相關 ACF（落後 1-{lags} 小時）')
    plot_pacf(s, lags=lags, ax=ax2, zero=False, method='ywm', title=f'{target} 偏自相關 PACF（落後 1-{lags} 小時）')
    plot_acf(s, lags=long_lags, ax=ax3, zero=False, title=f'{target} 自相關 ACF（最長 {long_lags // 24} 天；每 24 小時標示）')
    for k in range(24, long_lags + 1, 24):
        ax3.axvline(k, ls=':', lw=.5, color='gray')
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel('落後時間（小時）')
    fig.text(.5, -.01, 'Series linearly interpolated over all gaps for this diagnostic only.',
             ha='center', fontsize=8, color='gray')
    fig.tight_layout()
    save(fig, out, '05_acf_pacf.png')

    vals, conf = pacf(s, nlags=lags, method='ywm', alpha=.05)
    band = 1.96 / np.sqrt(len(s))
    sig = [(k, vals[k]) for k in range(1, lags + 1) if abs(vals[k]) > band]
    top = sorted(sig, key=lambda t: -abs(t[1]))[:8]
    print(f'    PACF 顯著落後（|r|>{band:.3f}）最強的前幾個 lag: '
          + ', '.join(f'{k}h({v:+.2f})' for k, v in top))


def plot_stl(df, target, out):
    s = _diag_series(df, target)

    # (a) 小時資料：日週期 + 週週期
    if MSTL is not None:
        res = MSTL(s, periods=(24, 168), stl_kwargs={'robust': True}).fit()
        seas = res.seasonal
        s24, s168 = seas.iloc[:, 0], seas.iloc[:, 1]
    else:
        res = STL(s, period=24, robust=True).fit()
        s24, s168 = res.seasonal, None
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1, 1])
    ax0 = fig.add_subplot(gs[0, :]); ax0.plot(s.index, s, lw=.3, alpha=.5, label='observed')
    ax0.plot(res.trend.index, res.trend, lw=1.6, color='red', label='trend')
    ax0.set_title(f'STL/MSTL（逐時）：觀測值與趨勢'); ax0.legend(); ax0.set_ylabel(label(target))
    ax1 = fig.add_subplot(gs[1, 0]); prof = s24.groupby(s.index.hour).mean()
    ax1.plot(prof.index, prof.values, marker='o'); ax1.set_xlabel('一天中的小時')
    ax1.set_title('每日季節成分（平均型態）')
    ax2 = fig.add_subplot(gs[1, 1])
    if s168 is not None:
        prof2 = s168.groupby(s.index.dayofweek * 24 + s.index.hour).mean()
        ax2.plot(prof2.index, prof2.values, lw=1)
        ax2.set_xticks(range(0, 169, 24)); ax2.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', ''])
        ax2.set_title('每週季節成分（平均型態）')
    else:
        ax2.axis('off')
    ax3 = fig.add_subplot(gs[2, :]); ax3.plot(res.resid.index, res.resid, lw=.3)
    ax3.set_title('殘差'); ax3.axhline(0, color='k', lw=.5)
    fig.tight_layout()
    save(fig, out, '06_stl_hourly.png')

    # (b) 日均值：年週期 + 趨勢
    daily = s.resample('D').mean()
    if len(daily) >= 2 * 365:
        d = STL(daily, period=365, robust=True).fit()
        fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
        for ax, (t, v) in zip(axes, [('Observed (daily mean)', d.observed), ('Trend', d.trend),
                                     ('Annual seasonal', d.seasonal), ('Residual', d.resid)]):
            ax.plot(v.index, v, lw=.8); ax.set_title(t); ax.margins(x=0)
        fig.suptitle(f'{target} 每日 STL（週期 = 365 天）', fontweight='bold', y=1.0)
        fig.tight_layout()
        save(fig, out, '06b_stl_daily_annual.png')


# ----------------------------------------------------------------------------
# 4. 變數關係：相關矩陣、散佈圖 + 迴歸
# ----------------------------------------------------------------------------
def plot_correlation_heatmap(df, out):
    cols = _continuous_cols(df)
    pear = df[cols].corr(); spear = df[cols].corr(method='spearman')
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
    draw_corr_heatmap(axes[0], pear, cbar=False); axes[0].set_title('Pearson（線性相關）')
    draw_corr_heatmap(axes[1], spear); axes[1].set_title('Spearman（排名相關；對偏態與離群值較穩健）')
    fig.suptitle('相關係數熱圖（清洗後資料、物理單位；第二階段以 sin/cos 表示風向）',
                 y=1.0, fontweight='bold')
    fig.tight_layout()
    save(fig, out, '07_correlation_heatmap.png')


def plot_scatter_regression(df, target, out):
    others = [c for c in _continuous_cols(df) if c != target]
    ncols = 3; nrows = int(np.ceil(len(others) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.1 * nrows))
    axes = axes.ravel()
    for ax, c in zip(axes, others):
        d = df[[c, target]].dropna()
        pr = d[c].corr(d[target]); sr = d[c].corr(d[target], method='spearman')
        ds = d.sample(min(len(d), 6000), random_state=0)
        x = np.log1p(ds[c]) if c == 'RAINFALL' else ds[c]
        sns.regplot(x=x, y=ds[target], ax=ax, ci=95, n_boot=200,
                    scatter_kws=dict(s=4, alpha=.15, edgecolor='none'),
                    line_kws=dict(color='red', lw=1.6))
        ax.set_xlabel(('log1p ' if c == 'RAINFALL' else '') + label(c)); ax.set_ylabel(label(target))
        ax.set_title(f'{c}：Pearson r={pr:.2f}、Spearman r={sr:.2f}')
    for ax in axes[len(others):]:
        ax.axis('off')
    fig.suptitle(f'{target} 與其他變數的關係（隨機抽樣 6,000 點，紅線 = OLS 擬合）', y=1.0, fontweight='bold')
    fig.tight_layout()
    save(fig, out, '08_scatter_regression.png')


# ----------------------------------------------------------------------------
# 5. 風場：Wind rose、Pollutant polar plot
# ----------------------------------------------------------------------------
WS_EDGES = [0, 1, 2, 3, 4, 6, np.inf]
WS_LABELS = ['0–1', '1–2', '2–3', '3–4', '4–6', '6+']


def _draw_wind_rose(ax, wd, ws, n_sectors=16, legend=True):
    d = pd.DataFrame({'wd': wd, 'ws': ws}).dropna()
    width = 360 / n_sectors
    sector = (((d.wd + width / 2) % 360) // width).astype(int)
    cat = pd.cut(d.ws, bins=WS_EDGES, labels=WS_LABELS, right=False, include_lowest=True)
    freq = pd.crosstab(sector, cat, normalize=True).reindex(
        index=range(n_sectors), columns=WS_LABELS, fill_value=0) * 100
    theta = np.deg2rad(np.arange(n_sectors) * width)
    bottom = np.zeros(n_sectors)
    colors = plt.cm.viridis_r(np.linspace(.1, .95, len(WS_LABELS)))
    for col, colr in zip(WS_LABELS, colors):
        ax.bar(theta, freq[col].values, width=np.deg2rad(width) * .9, bottom=bottom,
               color=colr, edgecolor='white', linewidth=.4, label=col)
        bottom += freq[col].values
    ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    # ax.yaxis.set_major_formatter(lambda v, _: f'{v:.0f}%')
    
    ax.set_yticklabels([])  # 隱藏預設容易被遮擋的標籤
    for t in ax.get_yticks():
        if t > 0:  # 避免在原點疊加 0% 文字
            # 手動在 22.5 度角（N 與 NE 中間）畫上百分比，並強制設定 zorder=10 與半透明白底
            ax.text(np.deg2rad(22.5), t, f'{t:.0f}%', fontsize=8,
                    ha='center', va='center', zorder=10,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, lw=0))
            
    if legend:
        ax.legend(title='m/s', loc='lower left', bbox_to_anchor=(1.08, 0), fontsize=8)


def plot_wind_rose(df, out):
    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={'projection': 'polar'})
    _draw_wind_rose(ax, df[WD], df[WS])
    ax.set_title('風玫瑰圖（風的來向；柱長 = 時數百分比）', pad=18)
    save(fig, out, '09_wind_rose.png')

    seasons = {'Winter (Dec–Feb)': [12, 1, 2], 'Spring (Mar–May)': [3, 4, 5],
               'Summer (Jun–Aug)': [6, 7, 8], 'Autumn (Sep–Nov)': [9, 10, 11]}
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), subplot_kw={'projection': 'polar'})
    for i, (ax, (name, months)) in enumerate(zip(axes.ravel(), seasons.items())):
        sub = df[df.index.month.isin(months)]
        _draw_wind_rose(ax, sub[WD], sub[WS], legend=(i == 1))
        ax.set_title(name, pad=14)
    fig.suptitle('季節風玫瑰圖', fontweight='bold')
    fig.tight_layout()
    save(fig, out, '09b_wind_rose_seasonal.png')


def plot_pollutant_polar(df, out, n_sectors=24, n_ws=8, min_count=5):
    cols = [c for c in POLLUTANTS if c in df.columns]
    d = df[[WD, WS] + cols].dropna(subset=[WD, WS])
    ws_max = d[WS].quantile(.98)
    ws_edges = np.linspace(0, ws_max, n_ws + 1)
    width = 360 / n_sectors
    sector = (((d[WD] + width / 2) % 360) // width).astype(int)
    ws_idx = np.clip(np.digitize(d[WS].clip(upper=ws_max - 1e-9), ws_edges) - 1, 0, n_ws - 1)
    theta_edges = np.deg2rad(np.arange(n_sectors + 1) * width - width / 2)

    ncols = 3; nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows), subplot_kw={'projection': 'polar'})
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        g = d[c].groupby([sector.values, ws_idx]).agg(['mean', 'count'])
        Z = np.full((n_ws, n_sectors), np.nan)
        for (si, wi), row in g.iterrows():
            if row['count'] >= min_count:
                Z[wi, si] = row['mean']
        mesh = ax.pcolormesh(theta_edges, ws_edges, Z, cmap='YlOrRd', shading='flat')
        ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
        ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
        ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
        ax.set_title(c, pad=14)
        ticks = np.round(np.linspace(0, ws_max, 5)[1:], 1)
        ax.set_yticks(ticks); ax.set_yticklabels([])
        ax.grid(color='white', alpha=.5, lw=.5)
        for r in ticks:   # 手動疊上風速刻度（內建刻度標籤會被色塊蓋住）
            ax.text(np.deg2rad(22.5), r, f'{r:g}', fontsize=7, ha='center', va='center', zorder=10,
                    bbox=dict(boxstyle='round,pad=.15', fc='white', alpha=.7, lw=0))
        cb = fig.colorbar(mesh, ax=ax, shrink=.75, pad=.1); cb.set_label(label(c))
    for ax in axes[len(cols):]:
        ax.axis('off')
    fig.suptitle('污染物極座標圖：依風向（角度）與風速（半徑，m/s）顯示平均濃度；'
                 f'少於 {min_count} 小時的格點留白', y=1.0, fontweight='bold')
    fig.tight_layout()
    save(fig, out, '10_pollutant_polar_plot.png')


# ----------------------------------------------------------------------------
def run_stage1(df, out_dir, target='PM2.5'):
    setup_style()
    steps = [
        ('00 missing overview', lambda: plot_missing_overview(df, out_dir)),
        ('01 time series overview', lambda: plot_timeseries_overview(df, out_dir)),
        ('01b target detail', lambda: plot_target_detail(df, target, out_dir)),
        ('02 histogram + KDE', lambda: plot_histograms(df, out_dir)),
        ('03 diurnal boxplot', lambda: plot_diurnal_boxplot(df, out_dir)),
        ('04 seasonal boxplot', lambda: plot_seasonal_boxplot(df, out_dir)),
        ('05 ACF / PACF', lambda: plot_acf_pacf(df, target, out_dir)),
        ('06 STL', lambda: plot_stl(df, target, out_dir)),
        ('07 correlation heatmap', lambda: plot_correlation_heatmap(df, out_dir)),
        ('08 scatter + regression', lambda: plot_scatter_regression(df, target, out_dir)),
        ('09 wind rose', lambda: plot_wind_rose(df, out_dir)),
        ('10 pollutant polar plot', lambda: plot_pollutant_polar(df, out_dir)),
    ]
    run_steps(steps, out_dir)
