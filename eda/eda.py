#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDA 可复现验证脚本
=================
用途：
1. 重新读取 5 个附件 Excel；
2. 完成缺失值统计与清洗，另存清洗 CSV；
3. 重新生成 EDA 阶段声明的 20 张图；
4. 重新计算核心指标，并输出 section_result_eda_verify.json / verification_report.md。

运行方式：
    python eda_verify_extracted.py --input_dir ./ --output_dir ./eda_verify

依赖：
    pip install pandas numpy matplotlib seaborn scipy openpyxl

说明：
- 默认 FILL_STRATEGY='mean'，对应“均值填充缺失值”的总结口径。
- 若要复现文本中“中位数填充”的口径，可运行：
    python eda_verify_extracted.py --fill_strategy median
"""

import argparse
import json
import os
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")


# ============================================================
# 0. 基础配置
# ============================================================
COLORS = {
    "primary": "#2E5B88",
    "secondary": "#E85D4C",
    "tertiary": "#4A9B7F",
    "neutral": "#7F7F7F",
    "light": "#B8D4E8",
    "purple": "#8A6BBE",
    "orange": "#F0A35E",
}

EXPECTED_FILES = {
    "p1": "附件1：两组位移时序数据-问题1.xlsx",
    "p2": "附件2：位移时序数据-问题2.xlsx",
    "p3": "附件3：监测数据（训练集与实验集）-问题3.xlsx",
    "p4": "附件4：监测数据（训练集与实验集）-问题4.xlsx",
    "p5": "附件5：监测数据-问题5.xlsx",
}

EXPECTED_IMAGES = [
    "附件1_时序图.png",
    "附件1_位移时序对比.png",
    "附件1_位移分布.png",
    "附件1_位移相关性.png",
    "附件1_分布图.png",
    "附件1_箱线图.png",
    "附件2_时序图.png",
    "附件2_表面位移分布.png",
    "附件2_分布图.png",
    "附件3_箱线图.png",
    "附件3_变量分布.png",
    "附件3_深部vs表面位移.png",
    "附件3_相关性热力图.png",
    "附件4_时序图.png",
    "附件4_相关性热力图.png",
    "附件5_时序图.png",
    "附件5_箱线图.png",
    "附件5_相关性热力图.png",
    "箱线图_各数据集表面位移对比.png",
    "综合对比分析_全量数据.png",
]


def configure_matplotlib() -> None:
    """自动设置中文字体，减少图中中文乱码/方框。"""
    sys_os = platform.system()
    if sys_os == "Windows":
        target_fonts = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    elif sys_os == "Darwin":
        target_fonts = ["PingFang SC", "Heiti TC", "Arial Unicode MS"]
    else:
        target_fonts = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]

    sns.set_theme(style="ticks")
    plt.rcParams.update({
        "font.sans-serif": target_fonts + ["sans-serif"],
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def read_excel_safely(path: Path, preferred_sheet: str | None = None) -> pd.DataFrame:
    """优先读取指定 sheet；不存在时读取第一个 sheet。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")

    xls = pd.ExcelFile(path)
    if preferred_sheet and preferred_sheet in xls.sheet_names:
        return pd.read_excel(path, sheet_name=preferred_sheet)
    return pd.read_excel(path, sheet_name=xls.sheet_names[0])


def read_all_data(input_dir: Path) -> dict[str, pd.DataFrame]:
    """读取 5 个附件及附件3/4实验集。"""
    paths = {k: input_dir / v for k, v in EXPECTED_FILES.items()}

    data = {
        "df1": read_excel_safely(paths["p1"]),
        "df2": read_excel_safely(paths["p2"]),
        "df3_train": read_excel_safely(paths["p3"], "训练集"),
        "df3_test": read_excel_safely(paths["p3"], "实验集"),
        "df4_train": read_excel_safely(paths["p4"], "训练集"),
        "df4_test": read_excel_safely(paths["p4"], "实验集"),
        "df5": read_excel_safely(paths["p5"]),
    }
    return data


def standardize_columns(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """统一部分列名，便于后续验证计算。"""
    df1 = data["df1"].copy()
    df1 = df1.rename(columns={
        "数据A_光纤位移计数据_mm": "位移A",
        "数据B_振弦式位移计数据_mm": "位移B",
    })
    if "时间" in df1.columns:
        df1["时间"] = pd.to_datetime(df1["时间"], errors="coerce")

    df3_train = data["df3_train"].copy()
    df3_train_std = df3_train.rename(columns={
        "a:降雨量_mm": "降雨量_mm",
        "b:孔隙水压力_kPa": "孔隙水压力_kPa",
        "c:微震事件数": "微震事件数",
        "d:深部位移_mm": "深部位移_mm",
        "e:表面位移_mm": "表面位移_mm",
    })

    # 其他表基本已是统一列名
    out = dict(data)
    out["df1"] = df1
    out["df3_train_std"] = df3_train_std
    return out


def fill_numeric_missing(df: pd.DataFrame, strategy: str = "mean", skip_all_nan: bool = True) -> pd.DataFrame:
    """
    填充数值列缺失值。
    skip_all_nan=True 时，整列全空的目标列不填，比如附件3实验集的表面位移。
    """
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        non_null_count = out[col].notna().sum()
        if skip_all_nan and non_null_count == 0:
            continue

        if strategy == "median":
            value = out[col].median()
        elif strategy == "zero":
            value = 0
        else:
            value = out[col].mean()

        if pd.isna(value):
            continue
        out[col] = out[col].fillna(value)
    return out


def clean_and_save_csv(data: dict[str, pd.DataFrame], output_dir: Path, fill_strategy: str) -> dict[str, pd.DataFrame]:
    """保存清洗/填充后的 CSV 文件。"""
    clean = {}

    clean["df1"] = data["df1"].copy()
    clean["df2"] = fill_numeric_missing(data["df2"], fill_strategy)
    clean["df3_train_original_cols"] = fill_numeric_missing(data["df3_train"], fill_strategy)
    clean["df3_train"] = fill_numeric_missing(data["df3_train_std"], fill_strategy)
    clean["df3_test"] = fill_numeric_missing(data["df3_test"], fill_strategy)
    clean["df4_train"] = fill_numeric_missing(data["df4_train"], fill_strategy)
    clean["df4_test"] = fill_numeric_missing(data["df4_test"], fill_strategy)
    clean["df5"] = fill_numeric_missing(data["df5"], fill_strategy)

    clean["df1"].to_csv(output_dir / "附件1_清洗后_位移数据.csv", index=False, encoding="utf-8-sig")
    clean["df2"].to_csv(output_dir / "附件2_清洗后_表面位移数据.csv", index=False, encoding="utf-8-sig")
    clean["df3_train_original_cols"].to_csv(output_dir / "附件3_训练集_填充后.csv", index=False, encoding="utf-8-sig")
    clean["df3_test"].to_csv(output_dir / "附件3_实验集_填充后.csv", index=False, encoding="utf-8-sig")
    clean["df4_train"].to_csv(output_dir / "附件4_清洗后_监测数据.csv", index=False, encoding="utf-8-sig")
    clean["df5"].to_csv(output_dir / "附件5_清洗后_监测数据.csv", index=False, encoding="utf-8-sig")
    return clean


def savefig(fig, output_dir: Path, filename: str) -> None:
    fig.savefig(output_dir / filename)
    plt.close(fig)


def qqplot_on_ax(series: pd.Series, ax, title: str) -> None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    stats.probplot(s, dist="norm", plot=ax)
    ax.set_title(title)


def numeric_corr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    return df[cols].corr()


def minmax_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return s * 0
    return (s - mn) / (mx - mn)


def plot_time_multivar(df: pd.DataFrame, cols: list[str], title: str, output_dir: Path, filename: str) -> None:
    """多变量时序图：归一化显示，避免量纲差异太大看不清。"""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = pd.to_datetime(df["时间"], errors="coerce") if "时间" in df.columns else np.arange(len(df))
    for col in cols:
        if col in df.columns:
            ax.plot(x, minmax_series(df[col]), linewidth=0.8, alpha=0.85, label=col)
    ax.set_title(title)
    ax.set_xlabel("时间/样本序号")
    ax.set_ylabel("Min-Max归一化值")
    ax.legend(ncol=2, fontsize=9)
    savefig(fig, output_dir, filename)


# ============================================================
# 1. 指标计算
# ============================================================
def iqr_outlier_count(series: pd.Series) -> tuple[int, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    count = int(((s < low) | (s > high)).sum())
    pct = count / len(s) * 100 if len(s) else 0.0
    return count, pct


def missing_rate(df: pd.DataFrame) -> dict[str, float]:
    return {col: round(float(rate * 100), 4) for col, rate in df.isna().mean().items()}


def surface_corrs(df: pd.DataFrame) -> dict[str, float]:
    if "表面位移_mm" not in df.columns:
        return {}
    corr = df.select_dtypes(include=[np.number]).corr(numeric_only=True)
    if "表面位移_mm" not in corr.columns:
        return {}
    return {
        k: round(float(v), 6)
        for k, v in corr["表面位移_mm"].drop("表面位移_mm", errors="ignore").dropna().items()
    }


def compute_metrics(data: dict[str, pd.DataFrame], clean: dict[str, pd.DataFrame], fill_strategy: str) -> dict:
    df1 = clean["df1"]
    df2 = clean["df2"]
    df3 = clean["df3_train"]
    df4 = clean["df4_train"]
    df5 = clean["df5"]

    out_count2, out_pct2 = iqr_outlier_count(df2["表面位移_mm"])

    bias = df1["位移A"] - df1["位移B"]
    p4_mean = float(df4["表面位移_mm"].mean()) if "表面位移_mm" in df4 else np.nan
    p5_mean = float(df5["表面位移_mm"].mean()) if "表面位移_mm" in df5 else np.nan

    metrics = {
        "fill_strategy": fill_strategy,
        "shapes": {
            "附件1": list(data["df1"].shape),
            "附件2": list(data["df2"].shape),
            "附件3_训练集": list(data["df3_train"].shape),
            "附件3_实验集": list(data["df3_test"].shape),
            "附件4_训练集": list(data["df4_train"].shape),
            "附件4_实验集": list(data["df4_test"].shape),
            "附件5": list(data["df5"].shape),
        },
        "missing_rate_percent_raw": {
            "附件1": missing_rate(data["df1"]),
            "附件2": missing_rate(data["df2"]),
            "附件3_训练集": missing_rate(data["df3_train"]),
            "附件3_实验集": missing_rate(data["df3_test"]),
            "附件4_训练集": missing_rate(data["df4_train"]),
            "附件4_实验集": missing_rate(data["df4_test"]),
            "附件5": missing_rate(data["df5"]),
        },
        "附件1": {
            "位移A_mean": round(float(df1["位移A"].mean()), 6),
            "位移A_std": round(float(df1["位移A"].std()), 6),
            "位移B_mean": round(float(df1["位移B"].mean()), 6),
            "位移B_std": round(float(df1["位移B"].std()), 6),
            "pearson_r_A_B": round(float(df1[["位移A", "位移B"]].corr().iloc[0, 1]), 6),
            "bias_A_minus_B_mean": round(float(bias.mean()), 6),
            "bias_A_minus_B_std": round(float(bias.std()), 6),
            "bias_A_minus_B_min": round(float(bias.min()), 6),
            "bias_A_minus_B_max": round(float(bias.max()), 6),
        },
        "附件2": {
            "surface_mean": round(float(df2["表面位移_mm"].mean()), 6),
            "surface_median": round(float(df2["表面位移_mm"].median()), 6),
            "surface_std": round(float(df2["表面位移_mm"].std()), 6),
            "surface_max": round(float(df2["表面位移_mm"].max()), 6),
            "iqr_outlier_count": out_count2,
            "iqr_outlier_percent": round(float(out_pct2), 6),
        },
        "附件3": {
            "corr_with_surface": surface_corrs(df3),
            "surface_mean": round(float(df3["表面位移_mm"].mean()), 6),
            "surface_std": round(float(df3["表面位移_mm"].std()), 6),
        },
        "附件4": {
            "corr_with_surface": surface_corrs(df4),
            "surface_mean": round(float(p4_mean), 6),
            "surface_median": round(float(df4["表面位移_mm"].median()), 6),
        },
        "附件5": {
            "corr_with_surface": surface_corrs(df5),
            "surface_mean": round(float(p5_mean), 6),
            "surface_median": round(float(df5["表面位移_mm"].median()), 6),
        },
        "cross_dataset": {
            "附件5_vs_附件4_surface_mean_ratio": round(float(p5_mean / p4_mean), 6) if p4_mean else None,
        },
    }
    return metrics


# ============================================================
# 2. 图表生成
# ============================================================
def make_figures(clean: dict[str, pd.DataFrame], output_dir: Path) -> None:
    df1 = clean["df1"]
    df2 = clean["df2"]
    df3 = clean["df3_train"]
    df4 = clean["df4_train"]
    df5 = clean["df5"]

    # ---------------- 附件1 ----------------
    fig, ax = plt.subplots(figsize=(12, 5))
    x1 = df1["时间"] if "时间" in df1.columns else np.arange(len(df1))
    ax.plot(x1, df1["位移A"], color=COLORS["primary"], linewidth=0.6, alpha=0.8, label="光纤位移计(A)")
    ax.plot(x1, df1["位移B"], color=COLORS["secondary"], linewidth=0.6, alpha=0.8, label="振弦式位移计(B)")
    ax.set_title("附件1：两组位移计原始时序对比")
    ax.set_xlabel("时间")
    ax.set_ylabel("位移 (mm)")
    ax.legend()
    savefig(fig, output_dir, "附件1_时序图.png")

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(x1, df1["位移A"], color=COLORS["primary"], linewidth=0.7, label="位移A")
    ax1.plot(x1, df1["位移B"], color=COLORS["secondary"], linewidth=0.7, label="位移B")
    ax1.set_ylabel("位移 (mm)")
    ax2 = ax1.twinx()
    ax2.plot(x1, df1["位移A"] - df1["位移B"], color=COLORS["neutral"], linewidth=0.5, alpha=0.5, label="A-B偏差")
    ax2.set_ylabel("A-B偏差 (mm)")
    ax1.set_title("附件1：位移时序对比与系统偏差")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    savefig(fig, output_dir, "附件1_位移时序对比.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(df1["位移A"], bins=60, kde=True, color=COLORS["primary"], ax=axes[0], alpha=0.6)
    axes[0].set_title("位移A分布")
    sns.histplot(df1["位移B"], bins=60, kde=True, color=COLORS["secondary"], ax=axes[1], alpha=0.6)
    axes[1].set_title("位移B分布")
    sns.histplot(df1["位移A"] - df1["位移B"], bins=60, kde=True, color=COLORS["neutral"], ax=axes[2], alpha=0.6)
    axes[2].set_title("A-B偏差分布")
    fig.tight_layout()
    savefig(fig, output_dir, "附件1_位移分布.png")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    sns.regplot(data=df1, x="位移A", y="位移B", scatter_kws={"s": 8, "alpha": 0.25}, line_kws={"color": "red"}, ax=ax)
    r = df1[["位移A", "位移B"]].corr().iloc[0, 1]
    ax.set_title(f"附件1：A vs B相关性 (r={r:.4f})")
    savefig(fig, output_dir, "附件1_位移相关性.png")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    sns.histplot(df1["位移A"], bins=60, kde=True, color=COLORS["primary"], ax=axes[0, 0], alpha=0.6)
    axes[0, 0].set_title("位移A直方图")
    qqplot_on_ax(df1["位移A"], axes[0, 1], "位移A Q-Q图")
    sns.histplot(df1["位移B"], bins=60, kde=True, color=COLORS["secondary"], ax=axes[1, 0], alpha=0.6)
    axes[1, 0].set_title("位移B直方图")
    qqplot_on_ax(df1["位移B"], axes[1, 1], "位移B Q-Q图")
    fig.tight_layout()
    savefig(fig, output_dir, "附件1_分布图.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(data=df1[["位移A", "位移B"]], palette=[COLORS["primary"], COLORS["secondary"]], ax=ax)
    ax.set_xticklabels(["光纤位移计(A)", "振弦式位移计(B)"])
    ax.set_title("附件1：位移离群值检测")
    savefig(fig, output_dir, "附件1_箱线图.png")

    # ---------------- 附件2 ----------------
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df2["编号"] if "编号" in df2.columns else np.arange(len(df2)), df2["表面位移_mm"],
            color=COLORS["primary"], linewidth=0.5)
    ax.set_title("附件2：表面位移时序趋势")
    ax.set_xlabel("样本编号")
    ax.set_ylabel("表面位移 (mm)")
    savefig(fig, output_dir, "附件2_时序图.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df2["表面位移_mm"], bins=60, kde=True, color=COLORS["primary"], ax=ax, alpha=0.6)
    ax.set_title("附件2：表面位移分布")
    savefig(fig, output_dir, "附件2_表面位移分布.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df2["表面位移_mm"], bins=60, kde=True, color=COLORS["primary"], ax=axes[0], alpha=0.6)
    axes[0].set_title("表面位移直方图")
    qqplot_on_ax(df2["表面位移_mm"], axes[1], "表面位移 Q-Q图")
    fig.tight_layout()
    savefig(fig, output_dir, "附件2_分布图.png")

    # ---------------- 附件3 ----------------
    cols3 = ["降雨量_mm", "孔隙水压力_kPa", "微震事件数", "深部位移_mm", "表面位移_mm"]
    corr3 = numeric_corr(df3, cols3)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr3, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("附件3：多监测因素相关性矩阵")
    savefig(fig, output_dir, "附件3_相关性热力图.png")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.regplot(data=df3, x="深部位移_mm", y="表面位移_mm",
                scatter_kws={"s": 10, "alpha": 0.3}, line_kws={"color": "red"}, ax=ax)
    ax.set_title("附件3：深部位移与表面位移关联分析")
    savefig(fig, output_dir, "附件3_深部vs表面位移.png")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(cols3):
        if col in df3.columns:
            sns.histplot(df3[col], bins=50, kde=True, ax=axes[i], alpha=0.65)
            axes[i].set_title(col)
    axes[-1].axis("off")
    fig.tight_layout()
    savefig(fig, output_dir, "附件3_变量分布.png")

    zdf3 = df3[cols3].apply(lambda s: (s - s.mean()) / s.std(ddof=0))
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=zdf3, ax=ax)
    ax.set_title("附件3：各变量箱线图（标准化后）")
    ax.set_ylabel("Z-score")
    ax.tick_params(axis="x", rotation=25)
    savefig(fig, output_dir, "附件3_箱线图.png")

    # ---------------- 附件4 ----------------
    cols4 = ["表面位移_mm", "降雨量_mm", "孔隙水压力_kPa", "微震事件数", "爆破点距离_m", "单段最大药量_kg"]
    plot_time_multivar(
        df4,
        [c for c in cols4 if c in df4.columns and c not in ["爆破点距离_m", "单段最大药量_kg"]],
        "附件4：监测变量时序",
        output_dir,
        "附件4_时序图.png",
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(numeric_corr(df4, cols4), annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("附件4：监测变量相关性热力图")
    savefig(fig, output_dir, "附件4_相关性热力图.png")

    # ---------------- 附件5 ----------------
    cols5 = ["表面位移_mm", "降雨量_mm", "孔隙水压力_kPa", "微震事件数", "干湿入渗系数", "爆破点距离_m", "单段最大药量_kg"]
    plot_time_multivar(
        df5,
        [c for c in cols5 if c in df5.columns and c not in ["爆破点距离_m", "单段最大药量_kg"]],
        "附件5：监测变量时序",
        output_dir,
        "附件5_时序图.png",
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_corr(df5, cols5), annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("附件5：监测变量相关性热力图")
    savefig(fig, output_dir, "附件5_相关性热力图.png")

    zcols5 = [c for c in cols5 if c in df5.columns]
    zdf5 = df5[zcols5].apply(lambda s: (s - s.mean()) / s.std(ddof=0))
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=zdf5, ax=ax)
    ax.set_title("附件5：各变量箱线图（标准化后）")
    ax.set_ylabel("Z-score")
    ax.tick_params(axis="x", rotation=25)
    savefig(fig, output_dir, "附件5_箱线图.png")

    # ---------------- 跨数据集对比 ----------------
    compare_series = []
    compare_names = []
    for name, df in [("附件2", df2), ("附件3", df3), ("附件4", df4), ("附件5", df5)]:
        if "表面位移_mm" in df.columns:
            compare_series.append(df["表面位移_mm"].dropna())
            compare_names.append(name)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=compare_series, ax=ax)
    ax.set_xticklabels(compare_names)
    ax.set_title("各数据集表面位移对比")
    ax.set_ylabel("表面位移 (mm)")
    savefig(fig, output_dir, "箱线图_各数据集表面位移对比.png")

    # 综合 3×3 看板
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    if "表面位移_mm" in corr3:
        corr3["表面位移_mm"].drop("表面位移_mm", errors="ignore").plot(kind="barh", color=COLORS["primary"], ax=ax1)
    ax1.set_title("(a) 附件3相关性系数")

    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(data=[df4["表面位移_mm"].dropna(), df5["表面位移_mm"].dropna()],
                palette=[COLORS["light"], COLORS["tertiary"]], ax=ax2)
    ax2.set_xticklabels(["附件4", "附件5"])
    ax2.set_title("(b) 跨年份表面位移分布")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(df1["位移A"], df1["位移B"], s=1, alpha=0.1, color=COLORS["primary"])
    ax3.set_title("(c) 附件1：A vs B相关性")
    ax3.set_xlabel("位移A")
    ax3.set_ylabel("位移B")

    ax4 = fig.add_subplot(gs[1, 0])
    sns.heatmap(numeric_corr(df4, ["表面位移_mm", "降雨量_mm", "孔隙水压力_kPa", "微震事件数"]),
                annot=True, fmt=".2f", cmap="RdBu_r", cbar=False, ax=ax4)
    ax4.set_title("(d) 附件4相关矩阵")

    ax5 = fig.add_subplot(gs[1, 1])
    sns.heatmap(numeric_corr(df5, ["表面位移_mm", "降雨量_mm", "孔隙水压力_kPa", "微震事件数", "干湿入渗系数"]),
                annot=True, fmt=".2f", cmap="RdBu_r", cbar=False, ax=ax5)
    ax5.set_title("(e) 附件5相关矩阵")

    ax6 = fig.add_subplot(gs[1, 2])
    miss = [df3.isna().mean().mean() * 100, df4.isna().mean().mean() * 100, df5.isna().mean().mean() * 100]
    ax6.bar(["附件3", "附件4", "附件5"], miss, color=COLORS["secondary"])
    ax6.set_ylabel("缺失率 (%)")
    ax6.set_title("(f) 填充后缺失情况对比")

    ax7 = fig.add_subplot(gs[2, 0])
    if "干湿入渗系数" in df5.columns:
        tmp = df5.copy()
        tmp["入渗等级"] = pd.qcut(tmp["干湿入渗系数"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        sns.barplot(data=tmp, x="入渗等级", y="表面位移_mm", color=COLORS["tertiary"], ax=ax7)
        ax7.set_title("(g) 入渗等级 vs 位移")
    else:
        ax7.text(0.5, 0.5, "无干湿入渗系数", ha="center", va="center")
        ax7.set_title("(g) 入渗等级 vs 位移")

    ax8 = fig.add_subplot(gs[2, 1])
    ax8.scatter(df3["深部位移_mm"], df3["表面位移_mm"], s=1, alpha=0.2, color=COLORS["primary"])
    ax8.set_xlabel("深部位移")
    ax8.set_ylabel("表面位移")
    ax8.set_title("(h) 深部 vs 表面")

    ax9 = fig.add_subplot(gs[2, 2])
    if "df3_test" in clean:
        clean["df3_test"].isna().sum().drop("编号", errors="ignore").plot(kind="barh", color=COLORS["neutral"], ax=ax9)
    ax9.set_title("(i) 附件3实验集缺失维度")

    fig.tight_layout()
    savefig(fig, output_dir, "综合对比分析_全量数据.png")


# ============================================================
# 3. 输出验证报告
# ============================================================
def check_outputs(output_dir: Path) -> dict:
    image_status = {}
    for name in EXPECTED_IMAGES:
        p = output_dir / name
        image_status[name] = {
            "exists": p.exists(),
            "size_kb": round(p.stat().st_size / 1024, 2) if p.exists() else 0,
        }
    return {
        "expected_count": len(EXPECTED_IMAGES),
        "generated_count": sum(v["exists"] for v in image_status.values()),
        "missing_images": [k for k, v in image_status.items() if not v["exists"]],
        "image_status": image_status,
    }


def write_report(metrics: dict, output_check: dict, output_dir: Path) -> None:
    lines = []
    lines.append("# EDA 验证报告\n")
    lines.append(f"- 缺失值填充策略：`{metrics['fill_strategy']}`")
    lines.append(f"- 图表生成：{output_check['generated_count']}/{output_check['expected_count']}")
    if output_check["missing_images"]:
        lines.append(f"- 未生成图表：{', '.join(output_check['missing_images'])}")
    else:
        lines.append("- 未生成图表：无")

    lines.append("\n## 关键指标复算\n")
    p1 = metrics["附件1"]
    p2 = metrics["附件2"]
    p3 = metrics["附件3"]
    p4 = metrics["附件4"]
    p5 = metrics["附件5"]
    cross = metrics["cross_dataset"]

    lines.append(f"- 附件1 A/B Pearson r：{p1['pearson_r_A_B']}")
    lines.append(f"- 附件1 A-B 系统偏差均值：{p1['bias_A_minus_B_mean']} mm")
    lines.append(f"- 附件2 表面位移均值：{p2['surface_mean']} mm")
    lines.append(f"- 附件2 IQR异常点：{p2['iqr_outlier_count']} 个，占 {p2['iqr_outlier_percent']}%")
    lines.append(f"- 附件3 与表面位移相关：{p3['corr_with_surface']}")
    lines.append(f"- 附件4 与表面位移相关：{p4['corr_with_surface']}")
    lines.append(f"- 附件5 与表面位移相关：{p5['corr_with_surface']}")
    lines.append(f"- 附件5/附件4 表面位移均值比：{cross['附件5_vs_附件4_surface_mean_ratio']}")

    # (output_dir / "verification_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA阶段代码复现与验证脚本")
    parser.add_argument("--input_dir", default="./", help="5个附件Excel所在目录")
    parser.add_argument("--output_dir", default="./eda", help="输出图表、CSV、JSON的目录")
    parser.add_argument("--fill_strategy", default="mean", choices=["mean", "median", "zero"],
                        help="数值缺失值填充策略，默认 mean")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()

    print("=" * 80)
    print("EDA 验证脚本启动")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"缺失填充策略: {args.fill_strategy}")
    print("=" * 80)

    data = read_all_data(input_dir)
    data = standardize_columns(data)

    print("\n[1/4] 数据读取完成：")
    for name, df in data.items():
        if isinstance(df, pd.DataFrame):
            print(f"  - {name}: shape={df.shape}")

    print("\n[2/4] 清洗并保存CSV...")
    clean = clean_and_save_csv(data, output_dir, args.fill_strategy)

    print("\n[3/4] 生成20张EDA图表...")
    make_figures(clean, output_dir)

    print("\n[4/4] 复算指标并写出验证结果...")
    metrics = compute_metrics(data, clean, args.fill_strategy)
    output_check = check_outputs(output_dir)

    result = {
        "summary": "EDA verification generated by eda_verify_extracted.py",
        "metrics": metrics,
        "output_check": output_check,
    }
    # (output_dir / "section_result_eda_verify.json").write_text(
    #     json.dumps(result, ensure_ascii=False, indent=2),
    #     encoding="utf-8",
    # )
    write_report(metrics, output_check, output_dir)

    print("\n✅ 验证完成")
    print(f"图表生成：{output_check['generated_count']}/{output_check['expected_count']}")
    if output_check["missing_images"]:
        print("缺失图表：", output_check["missing_images"])
    # print(f"验证JSON：{output_dir / 'section_result_eda_verify.json'}")
    # print(f"验证报告：{output_dir / 'verification_report.md'}")


if __name__ == "__main__":
    main()
