#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题1：两组位移时序数据 A -> B 的校正建模代码

本代码以当前最终答案为准：
1. 附件1共10000条数据；
2. 采用三次样条插值到1分钟粒度；
3. 在±60分钟范围内做互相关 CCF 搜索，最优时滞为0分钟；
4. 采用 OLS 线性回归建立校正模型：B_corrected = beta0 + beta1 * A；
5. 采用5折时间序列交叉验证评估模型；
6. 输出表1.1五个待校正点的校正结果；
7. 保存CSV、JSON和4张PNG图。 

运行方式：
python ques1_final_code.py --input "附件1：两组位移时序数据-问题1.xlsx" --outdir ques1_outputs

依赖安装：
pip install pandas numpy scipy scikit-learn matplotlib openpyxl
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline
from scipy.stats import skew, kurtosis, norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


# ============================================================
# 1. 基本配置
# ============================================================

# 题目表1.1给出的5个待校正数据
TARGET_X = np.array([7.132, 18.526, 84.337, 123.554, 167.667], dtype=float)

# 互相关搜索范围：±60分钟
MAX_LAG_MIN = 60

# 时间序列交叉验证折数
N_SPLITS = 5


# ============================================================
# 2. 数据读取与列名识别
# ============================================================

def find_col(columns: List[str], keywords: List[str]) -> str:
    """根据关键词自动识别列名。"""
    for key in keywords:
        for col in columns:
            if key.lower() in str(col).lower():
                return col
    raise ValueError(f"未找到包含关键词 {keywords} 的列，当前列名为：{columns}")


def load_data(excel_path: Path) -> pd.DataFrame:
    """读取附件1，并统一列名为：时间、A、B。"""
    df = pd.read_excel(excel_path)
    df.columns = [str(c).strip() for c in df.columns]

    cols = list(df.columns)
    time_col = find_col(cols, ["时间", "time", "date"])
    a_col = find_col(cols, ["数据A", "光纤", "A"])
    b_col = find_col(cols, ["数据B", "振弦", "B"])

    df = df[[time_col, a_col, b_col]].copy()
    df.columns = ["时间", "A", "B"]

    df["时间"] = pd.to_datetime(df["时间"])
    df["A"] = pd.to_numeric(df["A"], errors="coerce")
    df["B"] = pd.to_numeric(df["B"], errors="coerce")

    df = df.sort_values("时间").reset_index(drop=True)
    return df


# ============================================================
# 3. 数据探查
# ============================================================

def data_eda(df: pd.DataFrame) -> Dict:
    """基础数据探查，用于验证数据是否完整。"""
    interval_min = df["时间"].diff().dropna().dt.total_seconds().median() / 60
    corr_ab = df[["A", "B"]].corr().iloc[0, 1]

    return {
        "数据条数": int(len(df)),
        "起始时间": str(df["时间"].min()),
        "结束时间": str(df["时间"].max()),
        "采样间隔_min": float(interval_min),
        "A缺失值": int(df["A"].isna().sum()),
        "B缺失值": int(df["B"].isna().sum()),
        "A_B相关系数": float(corr_ab),
        "A最小值": float(df["A"].min()),
        "A最大值": float(df["A"].max()),
        "B最小值": float(df["B"].min()),
        "B最大值": float(df["B"].max()),
    }


# ============================================================
# 4. 三次样条插值 + CCF时滞搜索
# ============================================================

def build_cubic_spline(df: pd.DataFrame) -> Tuple[np.ndarray, CubicSpline, CubicSpline]:
    """将时间转为从起点开始的分钟数，并构造A、B的三次样条函数。"""
    t_min = (df["时间"] - df["时间"].iloc[0]).dt.total_seconds().to_numpy() / 60.0
    f_a = CubicSpline(t_min, df["A"].to_numpy(), extrapolate=False)
    f_b = CubicSpline(t_min, df["B"].to_numpy(), extrapolate=False)
    return t_min, f_a, f_b


def search_best_lag(df: pd.DataFrame, max_lag_min: int = MAX_LAG_MIN) -> pd.DataFrame:
    """
    在±max_lag_min范围内搜索A与B的最优时滞。

    约定：
    lag=0：不平移；
    lag>0：用 A(t+lag) 与 B(t) 比较；
    lag<0：用 A(t+lag) 与 B(t) 比较。
    """
    t_min, f_a, f_b = build_cubic_spline(df)
    grid = np.arange(np.ceil(t_min.min()), np.floor(t_min.max()) + 1, 1.0)

    rows = []
    for lag in range(-max_lag_min, max_lag_min + 1):
        a_lag = f_a(grid + lag)
        b_ref = f_b(grid)
        mask = np.isfinite(a_lag) & np.isfinite(b_ref)

        if mask.sum() <= 10:
            corr = np.nan
        else:
            corr = float(np.corrcoef(a_lag[mask], b_ref[mask])[0, 1])

        rows.append({"lag_min": int(lag), "corr": corr, "n_valid": int(mask.sum())})

    return pd.DataFrame(rows)


def make_aligned_data(df: pd.DataFrame, lag_min: int) -> pd.DataFrame:
    """按照最优时滞构造对齐后的A_shifted。当前最终模型中lag=0，因此A_shifted=A。"""
    t_min, f_a, _ = build_cubic_spline(df)
    aligned = df.copy()
    aligned["A_shifted"] = f_a(t_min + lag_min)
    aligned = aligned[np.isfinite(aligned["A_shifted"]) & np.isfinite(aligned["B"])].reset_index(drop=True)
    return aligned


# ============================================================
# 5. OLS线性校正模型 + 交叉验证
# ============================================================

def fit_ols(x: np.ndarray, y: np.ndarray) -> Tuple[LinearRegression, Dict, np.ndarray]:
    """拟合OLS线性模型：B = beta0 + beta1 * A。"""
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    pred = model.predict(x.reshape(-1, 1))
    residual = y - pred

    metrics = {
        "截距_beta0": float(model.intercept_),
        "斜率_beta1": float(model.coef_[0]),
        "R2": float(r2_score(y, pred)),
        "MAE_mm": float(mean_absolute_error(y, pred)),
        "RMSE_mm": float(np.sqrt(mean_squared_error(y, pred))),
        "残差均值_mm": float(np.mean(residual)),
        "残差标准差_mm": float(np.std(residual, ddof=0)),
        "残差最小值_mm": float(np.min(residual)),
        "残差最大值_mm": float(np.max(residual)),
        "残差偏度": float(skew(residual)),
        "残差峰度": float(kurtosis(residual, fisher=False)),
        "残差1sigma内占比": float(np.mean(np.abs(residual) <= np.std(residual, ddof=0))),
        "残差2sigma内占比": float(np.mean(np.abs(residual) <= 2 * np.std(residual, ddof=0))),
    }
    return model, metrics, pred


def time_series_cv(x: np.ndarray, y: np.ndarray, n_splits: int = N_SPLITS) -> Dict:
    """5折时间序列交叉验证。"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    r2_list, mae_list, rmse_list = [], [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(x), start=1):
        model = LinearRegression()
        model.fit(x[train_idx].reshape(-1, 1), y[train_idx])
        pred = model.predict(x[test_idx].reshape(-1, 1))

        r2_list.append(r2_score(y[test_idx], pred))
        mae_list.append(mean_absolute_error(y[test_idx], pred))
        rmse_list.append(np.sqrt(mean_squared_error(y[test_idx], pred)))

    return {
        "5折CV_R2均值": float(np.mean(r2_list)),
        "5折CV_R2标准差": float(np.std(r2_list, ddof=0)),
        "5折CV_MAE均值_mm": float(np.mean(mae_list)),
        "5折CV_MAE标准差_mm": float(np.std(mae_list, ddof=0)),
        "5折CV_RMSE均值_mm": float(np.mean(rmse_list)),
        "5折CV_RMSE标准差_mm": float(np.std(rmse_list, ddof=0)),
        "各折R2": [float(v) for v in r2_list],
        "各折MAE": [float(v) for v in mae_list],
        "各折RMSE": [float(v) for v in rmse_list],
    }


# ============================================================
# 6. 表1.1计算
# ============================================================

def calculate_table_1_1(model: LinearRegression, target_x: np.ndarray = TARGET_X) -> pd.DataFrame:
    """对题目表1.1给出的5个x进行校正。"""
    y = model.predict(target_x.reshape(-1, 1))
    return pd.DataFrame({
        "校正前数据x": target_x,
        "校正后数据y": y,
        "校正后数据y_保留3位": np.round(y, 3),
    })


# ============================================================
# 7. 画图
# ============================================================

def save_figures(
    df: pd.DataFrame,
    aligned: pd.DataFrame,
    ccf: pd.DataFrame,
    best_lag: int,
    pred: np.ndarray,
    outdir: Path,
) -> None:
    """保存4张结果图。图中文字使用英文，避免部分电脑中文字体缺失。"""
    plt.rcParams["axes.unicode_minus"] = False

    # 图1：时间序列对比
    n_show = min(500, len(df))
    plt.figure(figsize=(14, 5))
    plt.plot(df["时间"].iloc[:n_show], df["A"].iloc[:n_show], label="A optical fiber")
    plt.plot(df["时间"].iloc[:n_show], df["B"].iloc[:n_show], label="B vibrating wire")
    plt.xlabel("Time")
    plt.ylabel("Displacement / mm")
    plt.title("Fig.1 Time series comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "图1_时间序列对比.png", dpi=300)
    plt.close()

    # 图2：互相关函数
    plt.figure(figsize=(9, 5))
    plt.plot(ccf["lag_min"], ccf["corr"])
    plt.axvline(best_lag, linestyle="--", label=f"best lag = {best_lag} min")
    plt.xlabel("Lag / min")
    plt.ylabel("Correlation coefficient")
    plt.title("Fig.2 Cross-correlation lag search")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "图2_互相关函数.png", dpi=300)
    plt.close()

    # 图3：校正前后一致性
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(df["A"], df["B"], s=5, alpha=0.45)
    lo = min(df["A"].min(), df["B"].min())
    hi = max(df["A"].max(), df["B"].max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("A raw")
    plt.ylabel("B reference")
    plt.title("Before correction")

    plt.subplot(1, 2, 2)
    plt.scatter(pred, aligned["B"], s=5, alpha=0.45)
    lo = min(pred.min(), aligned["B"].min())
    hi = max(pred.max(), aligned["B"].max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("A corrected")
    plt.ylabel("B reference")
    plt.title("After correction")

    plt.tight_layout()
    plt.savefig(outdir / "图3_校正前后一致性对比.png", dpi=300)
    plt.close()

    # 图4：残差分析
    residual = aligned["B"].to_numpy() - pred
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(residual, bins=45, density=True, alpha=0.70)
    mu = residual.mean()
    sigma = residual.std(ddof=0)
    xs = np.linspace(residual.min(), residual.max(), 300)
    if sigma > 0:
        plt.plot(xs, norm.pdf(xs, mu, sigma))
    plt.xlabel("Residual = B - corrected A / mm")
    plt.ylabel("Density")
    plt.title("Residual distribution")

    plt.subplot(1, 2, 2)
    plt.scatter(pred, residual, s=5, alpha=0.45)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted B / mm")
    plt.ylabel("Residual / mm")
    plt.title("Residual vs prediction")

    plt.tight_layout()
    plt.savefig(outdir / "图4_残差分析.png", dpi=300)
    plt.close()


# ============================================================
# 8. 主流程
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="问题1：A传感器位移数据校正到B基准数据")
    parser.add_argument("--input", default="./附件1：两组位移时序数据-问题1.xlsx", help="附件1 Excel文件路径")
    parser.add_argument("--outdir", default="./ques1", help="输出目录")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. 读取和探查数据
    df = load_data(input_path)
    eda = data_eda(df)

    # 2. 1分钟三次样条插值 + CCF时滞搜索
    ccf = search_best_lag(df, MAX_LAG_MIN)
    best_lag = int(ccf.loc[ccf["corr"].idxmax(), "lag_min"])
    best_corr = float(ccf.loc[ccf["corr"].idxmax(), "corr"])
    zero_corr = float(ccf.loc[ccf["lag_min"] == 0, "corr"].iloc[0])

    # 3. 按最优时滞对齐数据
    aligned = make_aligned_data(df, best_lag)

    # 4. OLS线性校正模型
    x = aligned["A_shifted"].to_numpy()
    y = aligned["B"].to_numpy()
    model, model_metrics, pred = fit_ols(x, y)

    # 5. 5折时间序列交叉验证
    cv_metrics = time_series_cv(x, y, N_SPLITS)

    # 6. 表1.1计算
    table_1_1 = calculate_table_1_1(model, TARGET_X)

    # 7. 保存结果文件
    ccf.to_csv(outdir / "问题1_CCF时滞搜索.csv", index=False, encoding="utf-8-sig")

    full_result = aligned.copy()
    full_result["A_corrected"] = pred
    full_result["residual_B_minus_corrected"] = full_result["B"] - full_result["A_corrected"]
    full_result.to_csv(outdir / "问题1_全量校正结果.csv", index=False, encoding="utf-8-sig")

    table_1_1.to_csv(outdir / "问题1_表1_1数据校正结果.csv", index=False, encoding="utf-8-sig")

    report = {
        "数据探查": eda,
        "时滞搜索": {
            "搜索范围_min": [-MAX_LAG_MIN, MAX_LAG_MIN],
            "最优时滞_min": best_lag,
            "最优时滞相关系数": best_corr,
            "零时滞相关系数": zero_corr,
        },
        "OLS模型": model_metrics,
        "校正公式": f"B_corrected = {model_metrics['截距_beta0']:.6f} + {model_metrics['斜率_beta1']:.6f} × A",
        "交叉验证": cv_metrics,
        "表1_1": table_1_1.round(6).to_dict(orient="records"),
    }

    with open(outdir / "问题1_结果报告.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    save_figures(df, aligned, ccf, best_lag, pred, outdir)

    # 8. 控制台输出摘要
    print("\n========== 问题1建模完成 ==========")
    print(f"数据条数: {eda['数据条数']}")
    print(f"时间范围: {eda['起始时间']} 至 {eda['结束时间']}")
    print(f"采样间隔: {eda['采样间隔_min']:.0f} 分钟")
    print(f"缺失值: A={eda['A缺失值']}, B={eda['B缺失值']}")
    print(f"A与B相关系数: {eda['A_B相关系数']:.6f}")
    print()
    print(f"最优时滞: {best_lag} 分钟")
    print(f"最优相关系数: {best_corr:.6f}")
    print(f"零时滞相关系数: {zero_corr:.6f}")
    print()
    print("OLS校正模型:")
    print(f"B_corrected = {model_metrics['截距_beta0']:.6f} + {model_metrics['斜率_beta1']:.6f} × A")
    print(f"R² = {model_metrics['R2']:.6f}")
    print(f"MAE = {model_metrics['MAE_mm']:.6f} mm")
    print(f"RMSE = {model_metrics['RMSE_mm']:.6f} mm")
    print()
    print("5折时间序列交叉验证:")
    print(f"CV R² = {cv_metrics['5折CV_R2均值']:.6f} ± {cv_metrics['5折CV_R2标准差']:.6f}")
    print(f"CV MAE = {cv_metrics['5折CV_MAE均值_mm']:.6f} ± {cv_metrics['5折CV_MAE标准差_mm']:.6f} mm")
    print(f"CV RMSE = {cv_metrics['5折CV_RMSE均值_mm']:.6f} ± {cv_metrics['5折CV_RMSE标准差_mm']:.6f} mm")
    print()
    print("表1.1 问题1数据校正结果:")
    print(table_1_1[["校正前数据x", "校正后数据y_保留3位"]].to_string(index=False))
    print()
    print(f"结果已保存到: {outdir.resolve()}")
    print("输出文件包括：")
    print("- 问题1_表1_1数据校正结果.csv")
    print("- 问题1_全量校正结果.csv")
    print("- 问题1_CCF时滞搜索.csv")
    print("- 问题1_结果报告.json")
    print("- 图1_时间序列对比.png")
    print("- 图2_互相关函数.png")
    print("- 图3_校正前后一致性对比.png")
    print("- 图4_残差分析.png")
    print("===================================\n")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
