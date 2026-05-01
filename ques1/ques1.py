import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate, stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
import platform

warnings.filterwarnings('ignore')

# ============================================================
# 0. 终极中文显示解决方案 (自动匹配操作系统)
# ============================================================
sys_os = platform.system()
if sys_os == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', '黑体']
elif sys_os == 'Darwin': # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC']
else: # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC']
    
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号


# 设置全局绘图参数
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.frameon': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

COLORS = {
    'primary': '#2E5B88',
    'secondary': '#E85D4C',
    'tertiary': '#4A9B7F',
    'neutral': '#7F7F7F'
}
FIG_SINGLE = (6, 4.5)
FIG_DOUBLE = (12, 4.5)

# ============================================================
# 1. 数据加载与亚网格对齐 (1分钟样条插值)
# ============================================================
print("正在加载并处理数据...")
df = pd.read_excel("附件1：两组位移时序数据-问题1.xlsx")
df['时间'] = pd.to_datetime(df['时间'])
df = df.sort_values('时间').reset_index(drop=True)
df = df.rename(columns={'数据A_光纤位移计数据_mm': 'A', '数据B_振弦式位移计数据_mm': 'B'})

t0 = df['时间'].iloc[0]
t_minutes = (df['时间'] - t0).dt.total_seconds().values / 60.0
t_1min = np.arange(t_minutes[0], t_minutes[-1] + 1, 1)

f_A = interpolate.CubicSpline(t_minutes, df['A'].values, bc_type='natural')
f_B = interpolate.CubicSpline(t_minutes, df['B'].values, bc_type='natural')
A_1min, B_1min = f_A(t_1min), f_B(t_1min)

# ============================================================
# 2. 互相关分析 (CCF)
# ============================================================
max_lag_minutes = 60
lags = np.arange(-max_lag_minutes, max_lag_minutes + 1)
corr = np.zeros(len(lags))

for i, lag in enumerate(lags):
    a_seg = A_1min[-lag:] if lag < 0 else (A_1min[:-lag] if lag > 0 else A_1min)
    b_seg = B_1min[:lag] if lag < 0 else (B_1min[lag:] if lag > 0 else B_1min)
    n = min(len(a_seg), len(b_seg))
    corr[i] = np.corrcoef(a_seg[:n], b_seg[:n])[0, 1]

opt_lag = lags[np.argmax(np.abs(corr))]
max_corr = np.max(np.abs(corr))

# ============================================================
# 3. OLS 回归与预测
# ============================================================
print("正在训练 OLS 回归模型...")
X = df['A'].values.reshape(-1, 1) # 由于时滞为0，直接使用原数据对齐
y = df['B'].values

model = LinearRegression()
model.fit(X, y)
b0, b1 = model.intercept_, model.coef_[0]
y_pred = model.predict(X)
r2 = model.score(X, y)

# 预测表 1.1 的数据
test_vals = np.array([7.132, 18.526, 84.337, 123.554, 167.667])
corrected = model.predict(test_vals.reshape(-1, 1))

# ============================================================
# 4. 可视化生图逻辑 (无乱码版)
# ============================================================
print("正在生成诊断图表...")

# 图1：互相关分析图
fig, ax = plt.subplots(figsize=FIG_SINGLE)
ax.plot(lags, corr, color=COLORS['primary'], linewidth=2)
ax.scatter([opt_lag], [max_corr], color=COLORS['secondary'], s=80, zorder=5)
ax.axvline(x=opt_lag, color=COLORS['secondary'], linestyle='--', alpha=0.6)
ax.set_title("图1：亚网格级互相关分析 (CCF)")
ax.set_xlabel('延迟 (分钟)')
ax.set_ylabel('Pearson 相关系数')
ax.text(opt_lag+2, max_corr-0.0002, f'最优延迟={opt_lag}min\nr={max_corr:.6f}', color=COLORS['secondary'])
plt.tight_layout()
plt.savefig('问题1_CCF分析.png')
plt.close()

# 图2：校正散点图
fig, ax = plt.subplots(figsize=FIG_SINGLE)
ax.scatter(df['A'], df['B'], s=2, alpha=0.3, color=COLORS['primary'], label='原始数据点')
x_line = np.linspace(0, 400, 100)
ax.plot(x_line, model.predict(x_line.reshape(-1, 1)), color=COLORS['secondary'], linewidth=2, label=f'拟合线: B={b0:.3f}+{b1:.3f}A')
ax.plot([0, 400], [0, 400], 'k--', alpha=0.4, label='y=x (理想等值线)')
ax.scatter(test_vals, corrected, color=COLORS['tertiary'], s=80, marker='s', edgecolors='white', zorder=10, label='待校正点')
ax.set_title("图2：位移数据 OLS 回归校正图")
ax.set_xlabel('A 光纤位移 (mm)')
ax.set_ylabel('B 振弦位移 (mm)')
ax.legend()
plt.tight_layout()
plt.savefig('问题1_校正散点图.png')
plt.close()

# 图3：残差诊断图
fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)
res = y - y_pred
axes[0].scatter(y_pred, res, s=2, alpha=0.3, color=COLORS['primary'])
axes[0].axhline(y=0, color=COLORS['secondary'], linestyle='--', linewidth=1.5)
axes[0].set_title('(a) 残差分布图')
axes[0].set_xlabel('预测值 (mm)')
axes[0].set_ylabel('残差 (mm)')

stats.probplot(res, dist="norm", plot=axes[1])
axes[1].get_lines()[1].set_color(COLORS['secondary'])
axes[1].get_lines()[0].set_markerfacecolor(COLORS['primary'])
axes[1].set_title('(b) 残差 Q-Q 图')
plt.tight_layout()
plt.savefig('问题1_残差诊断.png')
plt.close()

# 打印最终结果
print("\n" + "="*40)
print("【问题1 最终校正结果】")
print(f"回归方程: B = {b1:.4f} * A {b0:+.4f}")
print(f"模型 R² : {r2:.6f}")
print("-" * 40)
print(f"{'校正前(A)':>12s} | {'校正后(B)':>12s}")
print("-" * 40)
for a, b in zip(test_vals, corrected):
    print(f"{a:>12.3f} | {b:>12.3f}")
print("="*40)
print("图表已成功生成并保存在当前目录下！")