import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
import platform

# 屏蔽不必要的警告
warnings.filterwarnings('ignore')

# ============================================================
# 0. 终极环境配置 (自动修复字体乱码 & 路径配置)
# ============================================================
input_dir = r'./'  
output_dir = r'./eda'
os.makedirs(output_dir, exist_ok=True)

# 1. 设置 Seaborn 主题风格
sns.set_theme(style='ticks')

# 2. 动态匹配操作系统字体，解决“白框”乱码
sys_os = platform.system()
if sys_os == 'Windows':
    target_fonts = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
elif sys_os == 'Darwin': # macOS
    target_fonts = ['PingFang SC', 'Heiti TC', 'Arial Unicode MS']
else: # Linux
    target_fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC']

plt.rcParams.update({
    'font.sans-serif': target_fonts + ['sans-serif'],
    'axes.unicode_minus': False,  
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {'primary': '#2E5B88', 'secondary': '#E85D4C', 'tertiary': '#4A9B7F', 'neutral': '#7F7F7F', 'light': '#B8D4E8'}

# 文件路径拼接
file_p1 = os.path.join(input_dir, "附件1：两组位移时序数据-问题1.xlsx")
file_p2 = os.path.join(input_dir, "附件2：位移时序数据-问题2.xlsx")
file_p3 = os.path.join(input_dir, "附件3：监测数据（训练集与实验集）-问题3.xlsx")
file_p4 = os.path.join(input_dir, "附件4：监测数据（训练集与实验集）-问题4.xlsx")
file_p5 = os.path.join(input_dir, "附件5：监测数据-问题5.xlsx")

print(f"✅ 系统检测: {sys_os} | 使用字体: {plt.rcParams['font.sans-serif'][0]}")

# ============================================================
# 1. 附件1 EDA：两组位移详细分析
# ============================================================
print("正在生成附件1详细图表...")
df1 = pd.read_excel(file_p1).rename(columns={'数据A_光纤位移计数据_mm': '位移A', '数据B_振弦式位移计数据_mm': '位移B'})
df1['时间'] = pd.to_datetime(df1['时间'])

# 图1-1：双变量时序图
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df1['时间'], df1['位移A'], color=COLORS['primary'], linewidth=0.6, alpha=0.8, label='光纤位移计(A)')
ax.plot(df1['时间'], df1['位移B'], color=COLORS['secondary'], linewidth=0.6, alpha=0.8, label='振弦式位移计(B)')
ax.set_title("附件1：两组位移计原始时序对比")
ax.set_xlabel('时间'), ax.set_ylabel('位移 (mm)')
ax.legend()
fig.savefig(os.path.join(output_dir, '附件1_时序图.png'))
plt.close()

# 图1-2：分布直方图+KDE
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for i, (col, color) in enumerate(zip(['位移A', '位移B'], [COLORS['primary'], COLORS['secondary']])):
    sns.histplot(df1[col], bins=60, color=color, kde=True, ax=axes[i], alpha=0.6)
    axes[i].set_title(f'{col} 分布特征')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, '附件1_分布图.png'))
plt.close()

# 图1-3：箱线图
fig, ax = plt.subplots(figsize=(5, 4))
sns.boxplot(data=df1[['位移A', '位移B']], palette=[COLORS['primary'], COLORS['secondary']], ax=ax)
ax.set_xticklabels(['光纤位移计(A)', '振弦式位移计(B)'])
ax.set_title("附件1：位移离群值检测")
fig.savefig(os.path.join(output_dir, '附件1_箱线图.png'))
plt.close()

# ============================================================
# 2. 附件2 EDA：表面位移详细分析
# ============================================================
print("正在生成附件2详细图表...")
df2 = pd.read_excel(file_p2)

# 图2-1：趋势图
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df2['编号'], df2['表面位移_mm'], color=COLORS['primary'], linewidth=0.5)
ax.set_title("附件2：表面位移时序趋势")
ax.set_xlabel('样本编号'), ax.set_ylabel('表面位移 (mm)')
fig.savefig(os.path.join(output_dir, '附件2_时序图.png'))
plt.close()

# 图2-2：分布图
fig, ax = plt.subplots(figsize=(5, 4))
sns.histplot(df2['表面位移_mm'], bins=60, color=COLORS['primary'], kde=True, ax=ax)
ax.set_title("附件2：表面位移密度分布")
fig.savefig(os.path.join(output_dir, '附件2_分布图.png'))
plt.close()

# ============================================================
# 3. 附件3 EDA：因素相关性详细分析
# ============================================================
print("正在生成附件3详细图表...")
df3_train = pd.read_excel(file_p3, sheet_name='训练集')
name_map = {'a:降雨量_mm':'降雨量','b:孔隙水压力_kPa':'孔压','c:微震事件数':'微震','d:深部位移_mm':'深部位移','e:表面位移_mm':'表面位移'}
df3_plot = df3_train.rename(columns=name_map)

# 图3-1：热力图
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(df3_plot[['降雨量','孔压','微震','深部位移','表面位移']].corr(), 
            annot=True, fmt='.3f', cmap='RdBu_r', center=0, ax=ax)
ax.set_title("附件3：多监测因素相关性矩阵")
fig.savefig(os.path.join(output_dir, '附件3_相关性热力图.png'))
plt.close()

# 图3-2：深部 vs 表面位移
fig, ax = plt.subplots(figsize=(5, 4))
sns.regplot(data=df3_plot, x='深部位移', y='表面位移', scatter_kws={'s':10, 'alpha':0.3}, line_kws={'color':'red'}, ax=ax)
ax.set_title("附件3：深部位移与表面位移关联分析")
fig.savefig(os.path.join(output_dir, '附件3_深部vs表面位移.png'))
plt.close()

# ============================================================
# 4. 全量对比分析 (看板渲染)
# ============================================================
print("正在生成最终全量对比看板...")
df4 = pd.read_excel(file_p4)
df5 = pd.read_excel(file_p5)

fig = plt.figure(figsize=(15, 12))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# (a) 附件3相关性
ax1 = fig.add_subplot(gs[0, 0])
corr3 = df3_plot[['降雨量','孔压','微震','深部位移','表面位移']].corr()['表面位移'].drop('表面位移')
corr3.plot(kind='barh', color=COLORS['primary'], ax=ax1)
ax1.set_title("(a) 附件3相关性系数")

# (b) 4vs5分布
ax2 = fig.add_subplot(gs[0, 1])
sns.boxplot(data=[df4['表面位移_mm'].dropna(), df5['表面位移_mm'].dropna()], palette=[COLORS['light'], COLORS['tertiary']], ax=ax2)
ax2.set_xticklabels(['附件4', '附件5'])
ax2.set_title("(b) 跨年份表面位移分布")

# (c) 附件1相关性
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(df1['位移A'], df1['位移B'], s=1, alpha=0.1, color=COLORS['primary'])
ax3.set_title("(c) 附件1：A vs B相关性")

# (d) 附件4热力图
ax4 = fig.add_subplot(gs[1, 0])
sns.heatmap(df4[['表面位移_mm','降雨量_mm','孔隙水压力_kPa','微震事件数']].corr(), annot=True, fmt='.2f', cmap='RdBu_r', cbar=False, ax=ax4)
ax4.set_title("(d) 附件4相关矩阵")

# (e) 附件5热力图
ax5 = fig.add_subplot(gs[1, 1])
sns.heatmap(df5[['表面位移_mm','降雨量_mm','孔隙水压力_kPa','微震事件数','干湿入渗系数']].corr(), annot=True, fmt='.2f', cmap='RdBu_r', cbar=False, ax=ax5)
ax5.set_title("(e) 附件5相关矩阵")

# (f) 缺失率对比
ax6 = fig.add_subplot(gs[1, 2])
miss = [df3_train.isna().mean().mean()*100, df4.isna().mean().mean()*100, df5.isna().mean().mean()*100]
ax6.bar(['附件3', '附件4', '附件5'], miss, color=COLORS['secondary'])
ax6.set_ylabel('缺失率 (%)'), ax6.set_title("(f) 数据缺失情况对比")

# (g) 附件5分箱分析
ax7 = fig.add_subplot(gs[2, 0])
if '干湿入渗系数' in df5.columns:
    df5['入渗等级'] = pd.qcut(df5['干湿入渗系数'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    sns.barplot(data=df5, x='入渗等级', y='表面位移_mm', color=COLORS['tertiary'], ax=ax7)
    ax7.set_title("(g) 入渗等级 vs 位移")

# (h) 附件3回归分析
ax8 = fig.add_subplot(gs[2, 1])
ax8.scatter(df3_plot['深部位移'], df3_plot['表面位移'], s=1, alpha=0.2, color=COLORS['primary'])
ax8.set_title("(h) 深部 vs 表面")

# (i) 附件3实验集缺失
ax9 = fig.add_subplot(gs[2, 2])
pd.read_excel(file_p3, sheet_name='实验集').isna().sum().drop('编号').plot(kind='barh', color=COLORS['neutral'], ax=ax9)
ax9.set_title("(i) 实验集缺失维度")

plt.tight_layout()
fig.savefig(os.path.join(output_dir, '综合对比分析_全量数据.png'))
plt.close()

print(f"\n✅ 渲染完成！共生成 8 张图片，存放在 {output_dir} 文件夹下。")