#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题2：三段式形变阶段识别与分段建模

运行示例：
    python ques2_clean_run.py \
        --input "附件2：位移时序数据-问题2.xlsx" \
        --output-dir "ques2_outputs"

说明：
- 索引使用 Python/CSV 的 0-based 索引：T1=8144, T2=9590。
- 如果要把预测位移中的负值按物理约束裁剪为 0，加 --clip-prediction。
- 代码不依赖 ruptures，避免环境安装问题。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


DT_HOURS = 10 / 60  # 10 min = 1/6 h
START_TIME = pd.Timestamp("2024-05-04 00:00:00")
STAGE_LABELS = {
    "Ⅰ": "缓慢匀速形变",
    "Ⅱ": "加速形变",
    "Ⅲ": "快速形变",
}
COLORS = {
    "Ⅰ": "#2E5B88",
    "Ⅱ": "#E85D4C",
    "Ⅲ": "#4A9B7F",
    "gray": "#7F7F7F",
}


@dataclass
class StageFit:
    stage: str
    start: int
    end: int              # Python slice end, exclusive
    model_type: str
    equation: str
    params: Dict[str, float]
    r2: float
    mae: float
    rmse: float
    avg_velocity: float
    fitted: np.ndarray
    residuals: np.ndarray


def configure_matplotlib() -> None:
    """统一图片风格；尽量支持中文字体。"""
    plt.rcParams.update({
        "font.sans-serif": ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def load_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path.resolve()}")

    df = pd.read_excel(input_path)
    required = {"编号", "表面位移_mm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"输入文件缺少列：{missing}；当前列为：{list(df.columns)}")

    df = df.copy()
    df["时间"] = pd.date_range(start=START_TIME, periods=len(df), freq="10min")
    df["时间_h"] = np.arange(len(df), dtype=float) * DT_HOURS
    return df


def fit_one_stage(stage: str, start: int, end: int, t_hours: np.ndarray, disp: np.ndarray) -> StageFit:
    t_seg = t_hours[start:end]
    d_seg = disp[start:end]
    t0 = t_seg[0]

    if stage == "Ⅰ":
        # 线性模型：d = a t + b
        coeffs = np.polyfit(t_seg, d_seg, 1)
        fitted = np.polyval(coeffs, t_seg)
        model_type = "线性(Linear)"
        equation = f"d = {coeffs[0]:.6f}t + {coeffs[1]:.4f}"
        params = {"a_斜率_mm_per_h": float(coeffs[0]), "b_截距_mm": float(coeffs[1])}

    elif stage == "Ⅱ":
        # 二次多项式：d = a t² + b t + c
        coeffs = np.polyfit(t_seg, d_seg, 2)
        fitted = np.polyval(coeffs, t_seg)
        model_type = "二次多项式(Quadratic)"
        equation = f"d = {coeffs[0]:.8f}t² + {coeffs[1]:.6f}t + {coeffs[2]:.4f}"
        params = {
            "a_二次项": float(coeffs[0]),
            "b_一次项": float(coeffs[1]),
            "c_截距": float(coeffs[2]),
            "acceleration_2a_mm_per_h2": float(2 * coeffs[0]),
        }

    elif stage == "Ⅲ":
        # Saito指数模型：d = alpha * exp(beta * Δt), Δt = t - T2
        t_rel = t_seg - t0
        log_d = np.log(np.maximum(d_seg, 1e-9))
        beta, log_alpha = np.polyfit(t_rel, log_d, 1)
        alpha = float(np.exp(log_alpha))
        fitted = alpha * np.exp(beta * t_rel)
        model_type = "指数Saito(Exponential)"
        equation = f"d = {alpha:.4f}·exp({beta:.6f}·Δt), Δt=t-{t0:.1f}"
        params = {"alpha_mm": alpha, "beta_per_h": float(beta), "t0_h": float(t0)}

    else:
        raise ValueError(f"未知阶段：{stage}")

    residuals = d_seg - fitted
    avg_velocity = (d_seg[-1] - d_seg[0]) / (t_seg[-1] - t_seg[0])

    return StageFit(
        stage=stage,
        start=start,
        end=end,
        model_type=model_type,
        equation=equation,
        params=params,
        r2=float(r2_score(d_seg, fitted)),
        mae=float(mean_absolute_error(d_seg, fitted)),
        rmse=float(np.sqrt(mean_squared_error(d_seg, fitted))),
        avg_velocity=float(avg_velocity),
        fitted=fitted,
        residuals=residuals,
    )


def fit_piecewise(df: pd.DataFrame, t1_idx: int, t2_idx: int) -> Tuple[List[StageFit], np.ndarray, np.ndarray, List[str]]:
    n = len(df)
    if not (0 < t1_idx < t2_idx < n):
        raise ValueError(f"变点索引非法：T1={t1_idx}, T2={t2_idx}, n={n}")

    t_hours = df["时间_h"].to_numpy(float)
    disp = df["表面位移_mm"].to_numpy(float)

    segments = [("Ⅰ", 0, t1_idx), ("Ⅱ", t1_idx, t2_idx), ("Ⅲ", t2_idx, n)]
    fits = [fit_one_stage(stage, start, end, t_hours, disp) for stage, start, end in segments]

    all_fitted = np.zeros(n, dtype=float)
    all_residuals = np.zeros(n, dtype=float)
    all_stages: List[str] = [""] * n
    for fit in fits:
        all_fitted[fit.start:fit.end] = fit.fitted
        all_residuals[fit.start:fit.end] = fit.residuals
        all_stages[fit.start:fit.end] = [fit.stage] * (fit.end - fit.start)

    return fits, all_fitted, all_residuals, all_stages


def detect_noise_candidates(disp: np.ndarray) -> Dict[str, object]:
    """输出瞬时跳变候选，用来支撑“噪声/工程扰动”判别准则。"""
    step = np.diff(disp)
    abs_step = np.abs(step)
    mu = float(abs_step.mean())
    sigma = float(abs_step.std(ddof=1))
    threshold_mu3 = mu + 3 * sigma
    threshold_p995 = float(np.percentile(abs_step, 99.5))
    threshold = max(threshold_mu3, threshold_p995)
    idx = np.where(abs_step > threshold)[0] + 1  # 跳变发生在后一观测点

    return {
        "单步位移增量绝对值均值": mu,
        "单步位移增量绝对值标准差": sigma,
        "μ+3σ阈值_mm": float(threshold_mu3),
        "P99.5阈值_mm": threshold_p995,
        "实际采用阈值_mm": float(threshold),
        "瞬时跳变候选数量": int(len(idx)),
        "瞬时跳变候选索引_前20个": idx[:20].tolist(),
    }


def save_outputs(
    df: pd.DataFrame,
    fits: List[StageFit],
    all_fitted: np.ndarray,
    all_residuals: np.ndarray,
    all_stages: List[str],
    output_dir: Path,
    clip_prediction: bool,
) -> Tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    result_df = pd.DataFrame({
        "编号": df["编号"].to_numpy(),
        "时间_h": df["时间_h"].to_numpy(),
        "表面位移_mm": df["表面位移_mm"].to_numpy(),
        "拟合值_mm": all_fitted,
        "残差_mm": all_residuals,
        "阶段": all_stages,
        "阶段标签": [STAGE_LABELS[s] for s in all_stages],
    })
    result_csv = output_dir / "问题2_三阶段模型结果.csv"
    result_df.to_csv(result_csv, index=False, encoding="utf-8-sig", float_format="%.6f")

    pred_values = all_fitted.copy()
    if clip_prediction:
        pred_values = np.maximum(pred_values, 0.0)

    pred_df = pd.DataFrame({
        "编号": df["编号"].to_numpy(),
        "时间小时": df["时间_h"].to_numpy(),
        "预测位移_mm": pred_values,
    })
    pred_csv = output_dir / "问题2_预测结果.csv"
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig", float_format="%.6f")

    # 汇总表，方便直接复制到论文或答题卡
    disp = df["表面位移_mm"].to_numpy(float)
    overall_r2 = r2_score(disp, all_fitted)
    rows = []
    for fit in fits:
        sub = result_df.iloc[fit.start:fit.end]
        rows.append({
            "阶段": fit.stage,
            "阶段标签": STAGE_LABELS[fit.stage],
            "索引范围": f"{fit.start}~{fit.end-1}",
            "时间范围_h": f"{sub['时间_h'].iloc[0]:.1f}~{sub['时间_h'].iloc[-1]:.1f}",
            "位移范围_mm": f"{sub['表面位移_mm'].iloc[0]:.2f}~{sub['表面位移_mm'].iloc[-1]:.2f}",
            "位移变化量_mm": f"{sub['表面位移_mm'].iloc[-1] - sub['表面位移_mm'].iloc[0]:.2f}",
            "持续时间_h": f"{sub['时间_h'].iloc[-1] - sub['时间_h'].iloc[0]:.1f}",
            "平均速度_mm_h": f"{fit.avg_velocity:.4f}",
            "模型类型": fit.model_type,
            "模型方程": fit.equation,
            "R2": f"{fit.r2:.4f}",
            "MAE_mm": f"{fit.mae:.4f}",
            "RMSE_mm": f"{fit.rmse:.4f}",
        })
    summary_df = pd.DataFrame(rows)
    summary_df.loc[len(summary_df)] = {
        "阶段": "整体",
        "阶段标签": "三段联合",
        "索引范围": "0~9999",
        "时间范围_h": f"0.0~{df['时间_h'].iloc[-1]:.1f}",
        "位移范围_mm": f"{disp[0]:.2f}~{disp[-1]:.2f}",
        "位移变化量_mm": f"{disp[-1] - disp[0]:.2f}",
        "持续时间_h": f"{df['时间_h'].iloc[-1]:.1f}",
        "平均速度_mm_h": "",
        "模型类型": "线性+二次+指数Saito",
        "模型方程": "",
        "R2": f"{overall_r2:.4f}",
        "MAE_mm": f"{np.mean(np.abs(all_residuals)):.4f}",
        "RMSE_mm": f"{np.sqrt(np.mean(all_residuals**2)):.4f}",
    }
    summary_csv = output_dir / "问题2_模型汇总表.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    return result_csv, pred_csv, summary_csv


def plot_all(
    df: pd.DataFrame,
    fits: List[StageFit],
    all_fitted: np.ndarray,
    all_residuals: np.ndarray,
    t1_idx: int,
    t2_idx: int,
    output_dir: Path,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    t = df["时间_h"].to_numpy(float)
    d = df["表面位移_mm"].to_numpy(float)
    n = len(df)
    fit_map = {fit.stage: fit for fit in fits}
    created: List[Path] = []

    # 图1：三阶段主图
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 1.2, 1.2], hspace=0.3)

    ax = fig.add_subplot(gs[0])
    for stage, start, end in [("Ⅰ", 0, t1_idx), ("Ⅱ", t1_idx, t2_idx), ("Ⅲ", t2_idx, n)]:
        ax.axvspan(t[start], t[end - 1], alpha=0.08, color=COLORS[stage])
        ax.text(t[start + (end-start)//2], d.max() * 0.95, f"Stage {stage}",
                ha="center", color=COLORS[stage], fontweight="bold")
    ax.plot(t, d, lw=0.6, alpha=0.65, color=COLORS["Ⅰ"], label="Observed")
    ax.plot(t, all_fitted, lw=0.9, color=COLORS["Ⅱ"], label="Fitted")
    for idx, name, color in [(t1_idx, "T₁", COLORS["Ⅱ"]), (t2_idx, "T₂", COLORS["Ⅲ"] )]:
        ax.axvline(t[idx], ls="--", lw=1.0, color=color, alpha=0.75)
        ax.annotate(f"{name}\n({t[idx]:.1f}h)", xy=(t[idx], d[idx]), xytext=(t[idx] - 120, d[idx] + 120),
                    fontsize=8, color=color, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
    ax.set_ylabel("Displacement (mm)")
    ax.legend(loc="upper left")
    ax.set_title("问题2 三阶段位移演化与分段拟合")

    ax = fig.add_subplot(gs[1])
    velocities = np.diff(d) / DT_HOURS
    v_smooth = uniform_filter1d(velocities.astype(float), size=80)
    ax.plot(t[:-1], v_smooth, color=COLORS["Ⅱ"], lw=0.75, alpha=0.75, label="Velocity (smoothed)")
    for stage, start, end in [("Ⅰ", 0, t1_idx), ("Ⅱ", t1_idx, t2_idx), ("Ⅲ", t2_idx, n - 1)]:
        avg_v = fit_map[stage].avg_velocity
        ax.axhline(avg_v, color=COLORS[stage], ls=":", lw=1.0, alpha=0.8)
        ax.text(t[start + (end-start)//2], avg_v + 0.25, f"{avg_v:.2f}", ha="center", fontsize=8, color=COLORS[stage])
    for idx in [t1_idx, t2_idx]:
        ax.axvline(t[idx], color=COLORS["gray"], ls=":", lw=0.8, alpha=0.6)
    ax.set_ylabel("Velocity (mm/h)")
    ax.legend(loc="upper left")

    ax = fig.add_subplot(gs[2])
    for stage, start, end in [("Ⅰ", 0, t1_idx), ("Ⅱ", t1_idx, t2_idx), ("Ⅲ", t2_idx, n)]:
        ax.scatter(t[start:end], all_residuals[start:end], s=2, alpha=0.25, color=COLORS[stage], label=f"Stage {stage}")
    std_r = np.std(all_residuals)
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(3 * std_r, color="red", ls="--", lw=0.6, alpha=0.5, label="±3σ")
    ax.axhline(-3 * std_r, color="red", ls="--", lw=0.6, alpha=0.5)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Residual (mm)")
    ax.legend(loc="lower right", ncol=2, fontsize=7)

    fig.tight_layout()
    path = output_dir / "问题2_三阶段划分主图.png"
    fig.savefig(path)
    plt.close(fig)
    created.append(path)

    # 图2：分阶段细节
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, fit in zip(axes, fits):
        start, end = fit.start, fit.end
        ax.scatter(t[start:end], d[start:end], s=2, alpha=0.3, color=COLORS[fit.stage], label="Observed")
        ax.plot(t[start:end], all_fitted[start:end], color="black", lw=1.4, label="Fitted")
        text = f"{fit.model_type}\nR²={fit.r2:.4f}\nMAE={fit.mae:.4f}mm\nRMSE={fit.rmse:.4f}mm\nv={fit.avg_velocity:.4f}mm/h"
        ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.85))
        ax.set_title(f"阶段{fit.stage}：{STAGE_LABELS[fit.stage]}", color=COLORS[fit.stage])
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Displacement (mm)")
        ax.legend(loc="lower right")
    fig.tight_layout()
    path = output_dir / "问题2_分阶段模型细节.png"
    fig.savefig(path)
    plt.close(fig)
    created.append(path)

    # 图3：速度演化
    fig, ax = plt.subplots(figsize=(6, 4))
    velocities = np.diff(d) / DT_HOURS
    v_smooth = uniform_filter1d(velocities.astype(float), size=80)
    ax.plot(t[:-1], v_smooth, color=COLORS["Ⅱ"], lw=0.8, alpha=0.75, label="Velocity (smoothed)")
    for fit in fits:
        avg_v = fit.avg_velocity
        mid_t = t[fit.start + (fit.end-fit.start)//2]
        ax.axhline(avg_v, color=COLORS[fit.stage], ls="--", lw=1.0, alpha=0.7)
        ax.text(mid_t, avg_v + 0.2, f"Stage {fit.stage}: {avg_v:.2f} mm/h", ha="center", fontsize=8, color=COLORS[fit.stage])
    for idx in [t1_idx, t2_idx]:
        ax.axvline(t[idx], color=COLORS["gray"], ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Velocity (mm/h)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = output_dir / "问题2_速度演化.png"
    fig.savefig(path)
    plt.close(fig)
    created.append(path)

    # 图4：残差分析
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(all_fitted, all_residuals, s=3, alpha=0.3, color=COLORS["Ⅰ"])
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].set_xlabel("Fitted values (mm)")
    axes[0].set_ylabel("Residuals (mm)")
    axes[0].set_title("残差-拟合值")
    for fit in fits:
        axes[1].hist(all_residuals[fit.start:fit.end], bins=30, alpha=0.4, density=True,
                     color=COLORS[fit.stage], label=f"Stage {fit.stage}")
    axes[1].set_xlabel("Residuals (mm)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("分阶段残差分布")
    axes[1].legend()
    fig.tight_layout()
    path = output_dir / "问题2_残差分析.png"
    fig.savefig(path)
    plt.close(fig)
    created.append(path)

    # 图5：CUSUM检测辅助图
    fig, ax = plt.subplots(figsize=(6, 4))
    velocities = np.diff(d) / DT_HOURS
    cumsum = np.cumsum(velocities - velocities.mean())
    ax.plot(t[1:], cumsum, color=COLORS["Ⅰ"], lw=0.8)
    for idx, name, color in [(t1_idx, "T₁", COLORS["Ⅱ"]), (t2_idx, "T₂", COLORS["Ⅲ"] )]:
        ax.axvline(t[idx], color=color, ls="--", lw=1.0, alpha=0.7, label=f"{name}={t[idx]:.1f}h")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("CUSUM of velocity")
    ax.set_title("CUSUM检测辅助图")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "问题2_CUSUM检测.png"
    fig.savefig(path)
    plt.close(fig)
    created.append(path)

    return created


def print_summary(df: pd.DataFrame, fits: List[StageFit], all_fitted: np.ndarray, all_residuals: np.ndarray,
                  t1_idx: int, t2_idx: int, noise_report: Dict[str, object]) -> None:
    t = df["时间_h"].to_numpy(float)
    d = df["表面位移_mm"].to_numpy(float)
    overall_r2 = r2_score(d, all_fitted)

    print("=" * 72)
    print("【问题2：三段式形变阶段识别与建模】")
    print("=" * 72)
    print(f"数据量：{len(df)} 点；时间范围：{t[0]:.1f}h ~ {t[-1]:.1f}h；位移范围：{d.min():.3f} ~ {d.max():.3f} mm")
    print(f"T₁：索引 {t1_idx}，时间 {t[t1_idx]:.1f}h，位移 {d[t1_idx]:.2f}mm")
    print(f"T₂：索引 {t2_idx}，时间 {t[t2_idx]:.1f}h，位移 {d[t2_idx]:.2f}mm")

    print("\n【分阶段模型】")
    for fit in fits:
        print(f"阶段{fit.stage}（{STAGE_LABELS[fit.stage]}，索引{fit.start}~{fit.end-1}）：")
        print(f"  模型：{fit.model_type}；{fit.equation}")
        print(f"  R²={fit.r2:.4f}, MAE={fit.mae:.4f}mm, RMSE={fit.rmse:.4f}mm, 平均速度={fit.avg_velocity:.4f}mm/h")
    print(f"整体R²={overall_r2:.4f}, 整体MAE={np.mean(np.abs(all_residuals)):.4f}mm, 整体RMSE={np.sqrt(np.mean(all_residuals**2)):.4f}mm")

    v1, v2, v3 = [fit.avg_velocity for fit in fits]
    print("\n【速度增长】")
    print(f"Ⅰ→Ⅱ：{v2 / v1:.2f}x；Ⅱ→Ⅲ：{v3 / v2:.2f}x；Ⅰ→Ⅲ：{v3 / v1:.1f}x")

    print("\n【噪声/工程扰动判别辅助量】")
    for k, v in noise_report.items():
        print(f"  {k}: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="问题2：三段式形变阶段识别与分段建模")
    parser.add_argument("--input", default="附件2：位移时序数据-问题2.xlsx", help="附件2 Excel文件路径")
    parser.add_argument("--output-dir", default="./ques2", help="输出文件夹")
    parser.add_argument("--t1-idx", type=int, default=8144, help="T1变点0-based索引，默认8144")
    parser.add_argument("--t2-idx", type=int, default=9590, help="T2变点0-based索引，默认9590")
    parser.add_argument("--clip-prediction", action="store_true", help="将预测CSV中的负位移裁剪为0；不影响拟合/残差/R²")
    parser.add_argument("--no-plots", action="store_true", help="只生成CSV，不生成图片")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    df = load_data(input_path)
    fits, all_fitted, all_residuals, all_stages = fit_piecewise(df, args.t1_idx, args.t2_idx)
    noise_report = detect_noise_candidates(df["表面位移_mm"].to_numpy(float))

    result_csv, pred_csv, summary_csv = save_outputs(
        df, fits, all_fitted, all_residuals, all_stages,
        output_dir=output_dir,
        clip_prediction=args.clip_prediction,
    )

    image_paths: List[Path] = []
    if not args.no_plots:
        image_paths = plot_all(df, fits, all_fitted, all_residuals, args.t1_idx, args.t2_idx, output_dir)

    print_summary(df, fits, all_fitted, all_residuals, args.t1_idx, args.t2_idx, noise_report)

    print("\n【输出文件】")
    for p in [result_csv, pred_csv, summary_csv] + image_paths:
        print(f"  ✓ {p}")


if __name__ == "__main__":
    main()
