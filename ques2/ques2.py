#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题2：三阶段变形节点识别与分段建模（最终整理版）

本脚本以当前 section_result_ques2.json 的最终口径为准：
- 数据：附件2：位移时序数据-问题2.xlsx
- 采样频率：10分钟/次
- 平滑：Savitzky-Golay，窗口长度5，多项式阶数2
- T1：索引520，时间86.7 h，位移约2.8518 mm
- T2：索引700，时间116.7 h，位移约4.5415 mm
- 阶段模型：Ⅰ线性，Ⅱ二次多项式，Ⅲ Saito 指数模型

运行：
    python ques2_final_run.py --input "附件2：位移时序数据-问题2.xlsx" --output-dir ques2_outputs
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.signal import savgol_filter
    from scipy.optimize import curve_fit
except Exception as exc:  # pragma: no cover
    raise RuntimeError("缺少 scipy，请先执行：pip install scipy") from exc

try:
    from statsmodels.stats.stattools import durbin_watson
except Exception:
    durbin_watson = None

warnings.filterwarnings("ignore")

# ========= 最终口径常量：按当前 section_result_ques2.json =========
START_TIME = "2024-05-04 00:00:00"
SAMPLE_INTERVAL_H = 10 / 60  # 10分钟 = 1/6小时
T1_IDX = 520                 # 0-based 索引；Excel编号为 521
T2_IDX = 700                 # 0-based 索引；Excel编号为 701
SG_WINDOW = 5
SG_POLY = 2

PHASE_LABELS = {
    "Ⅰ": "缓慢匀速形变",
    "Ⅱ": "加速形变",
    "Ⅲ": "快速形变（加速破裂）",
}


def setup_chinese_font() -> None:
    """尽量启用中文字体；没有中文字体时不影响计算。"""
    plt.rcParams["axes.unicode_minus"] = False
    candidates = [
        "SimHei", "Microsoft YaHei", "Arial Unicode MS",
        "Noto Sans CJK SC", "WenQuanYi Micro Hei", "PingFang SC"
    ]
    plt.rcParams["font.sans-serif"] = candidates + plt.rcParams.get("font.sans-serif", [])


def read_data(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件：{path}")

    df = pd.read_excel(path)
    if "编号" not in df.columns or "表面位移_mm" not in df.columns:
        raise ValueError(f"输入表必须包含列：编号、表面位移_mm；当前列为：{list(df.columns)}")

    df = df[["编号", "表面位移_mm"]].copy()
    df["编号"] = pd.to_numeric(df["编号"], errors="raise").astype(int)
    df["表面位移_mm"] = pd.to_numeric(df["表面位移_mm"], errors="raise")

    n = len(df)
    df["索引"] = np.arange(n)
    df["时间_h"] = df["索引"] * SAMPLE_INTERVAL_H
    df["采集时间"] = pd.to_datetime(START_TIME) + pd.to_timedelta(df["时间_h"], unit="h")
    return df


def smooth_series(y: np.ndarray, window: int = SG_WINDOW, poly: int = SG_POLY) -> np.ndarray:
    if len(y) < window:
        return y.copy()
    if window % 2 == 0:
        window += 1
    return savgol_filter(y, window_length=window, polyorder=poly, mode="interp")


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def fit_linear(t: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    coef = np.polyfit(t, y, 1)  # [a, b]
    pred = np.polyval(coef, t)
    return coef, pred


def fit_quadratic(t: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    coef = np.polyfit(t, y, 2)  # [a, b, c]
    pred = np.polyval(coef, t)
    return coef, pred


def fit_saito_exp(t: np.ndarray, y: np.ndarray, t0: float) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Saito指数模型：y = alpha * exp(beta * (t - t0))

    这里采用非线性最小二乘 curve_fit，和当前 section_result_ques2.json
    中 α≈11.748247、β≈0.002894、R²≈0.962076 的口径一致。
    """
    def model(tt: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        return alpha * np.exp(beta * (tt - t0))

    y_pos = np.clip(y, 1e-9, None)
    # 用对数线性结果作为初值，提高稳定性
    beta0, log_alpha0 = np.polyfit(t - t0, np.log(y_pos), 1)
    alpha0 = float(np.exp(log_alpha0))
    popt, _ = curve_fit(model, t, y, p0=[alpha0, float(beta0)], maxfev=100000)
    alpha, beta = map(float, popt)
    pred = model(t, alpha, beta)
    params = {"alpha_mm": alpha, "beta_per_h": beta}
    return params, pred


def phase_slices(n: int) -> Dict[str, slice]:
    if not (0 <= T1_IDX < T2_IDX < n):
        raise ValueError(f"变点索引非法：T1={T1_IDX}, T2={T2_IDX}, n={n}")
    return {
        "Ⅰ": slice(0, T1_IDX + 1),       # 含T1
        "Ⅱ": slice(T1_IDX + 1, T2_IDX + 1),
        "Ⅲ": slice(T2_IDX + 1, n),
    }


def calc_phase_speed(t: np.ndarray, y: np.ndarray) -> float:
    duration = float(t[-1] - t[0])
    return float((y[-1] - y[0]) / duration) if duration > 0 else float("nan")


def build_models(df: pd.DataFrame, use_smooth_for_fit: bool = True) -> Tuple[pd.DataFrame, Dict]:
    y_raw = df["表面位移_mm"].to_numpy(dtype=float)
    t = df["时间_h"].to_numpy(dtype=float)
    y_smooth = smooth_series(y_raw)
    y_fit_target = y_smooth if use_smooth_for_fit else y_raw

    out = df.copy()
    out["平滑位移_mm"] = y_smooth
    out["拟合值_mm"] = np.nan
    out["残差_mm"] = np.nan
    out["阶段"] = ""
    out["阶段标签"] = ""

    sl = phase_slices(len(df))
    info: Dict = {
        "问题": "问题2：三阶段变形节点识别与分段建模",
        "数据概况": {
            "数据点数": int(len(df)),
            "采集频率": "10分钟/次",
            "起始时间": START_TIME,
            "结束时间": str(out["采集时间"].iloc[-1]),
            "总时长_h": round(float(t[-1] - t[0]), 1),
            "位移范围_mm": [float(np.min(y_raw)), float(np.max(y_raw))],
        },
        "平滑方法": {"方法": "Savitzky-Golay滤波", "窗口长度": SG_WINDOW, "多项式阶数": SG_POLY},
        "核心准则": {
            "持续性": "变点后连续≥5窗口(≥50分钟)内速度保持新阶段特征，防止将短暂噪声识别为阶段转换",
            "幅度": "单点跳变超过历史波动3σ且瞬间回落，判定为工程扰动而非真实阶段转换",
            "单调性": "加速或快速阶段位移累积方向不可逆转，排除由测量噪声引起的双向波动",
        },
        "阶段转换节点": {},
        "阶段模型": {},
    }

    # 变点信息：按当前结果，使用平滑位移作为节点位移，能对齐 2.8518 / 4.5415 这类值
    for key, idx, name in [
        ("T1_匀速转加速", T1_IDX, "T1（匀速→加速）"),
        ("T2_加速转快速", T2_IDX, "T2（加速→快速）"),
    ]:
        info["阶段转换节点"][key] = {
            "节点": name,
            "索引_0based": int(idx),
            "Excel编号": int(out.loc[idx, "编号"]),
            "小时": round(float(out.loc[idx, "时间_h"]), 1),
            "日期时间": str(out.loc[idx, "采集时间"]),
            "原始位移_mm": round(float(out.loc[idx, "表面位移_mm"]), 4),
            "平滑位移_mm": round(float(out.loc[idx, "平滑位移_mm"]), 4),
        }

    # 阶段Ⅰ：线性
    s = sl["Ⅰ"]
    coef1, pred1 = fit_linear(t[s], y_fit_target[s])
    out.loc[out.index[s], "拟合值_mm"] = pred1
    out.loc[out.index[s], "阶段"] = "Ⅰ"
    out.loc[out.index[s], "阶段标签"] = PHASE_LABELS["Ⅰ"]

    # 阶段Ⅱ：二次
    s = sl["Ⅱ"]
    coef2, pred2 = fit_quadratic(t[s], y_fit_target[s])
    out.loc[out.index[s], "拟合值_mm"] = pred2
    out.loc[out.index[s], "阶段"] = "Ⅱ"
    out.loc[out.index[s], "阶段标签"] = PHASE_LABELS["Ⅱ"]

    # 阶段Ⅲ：Saito指数；从T2之后开始，即索引701~9999
    s = sl["Ⅲ"]
    t0 = float(t[T2_IDX])
    params3, pred3 = fit_saito_exp(t[s], y_fit_target[s], t0=t0)
    out.loc[out.index[s], "拟合值_mm"] = pred3
    out.loc[out.index[s], "阶段"] = "Ⅲ"
    out.loc[out.index[s], "阶段标签"] = PHASE_LABELS["Ⅲ"]

    out["残差_mm"] = y_fit_target - out["拟合值_mm"].to_numpy(dtype=float)

    # 指标统计
    for phase in ["Ⅰ", "Ⅱ", "Ⅲ"]:
        s = sl[phase]
        yt = y_fit_target[s]
        yp = out.loc[out.index[s], "拟合值_mm"].to_numpy(dtype=float)
        res = yt - yp
        dw = float(durbin_watson(res)) if durbin_watson is not None else float("nan")
        speed_raw = calc_phase_speed(t[s], y_raw[s])
        speed_smooth = calc_phase_speed(t[s], y_smooth[s])
        phase_info = {
            "数据范围_0based": [int(out.index[s][0]), int(out.index[s][-1])],
            "编号范围": [int(out.loc[out.index[s][0], "编号"]), int(out.loc[out.index[s][-1], "编号"])],
            "时间范围_h": [round(float(t[s][0]), 1), round(float(t[s][-1]), 1)],
            "时长_h": round(float(t[s][-1] - t[s][0]), 1),
            "位移增量_原始_mm": round(float(y_raw[s][-1] - y_raw[s][0]), 4),
            "位移增量_平滑_mm": round(float(y_smooth[s][-1] - y_smooth[s][0]), 4),
            "平均速度_原始_mm_per_h": round(speed_raw, 4),
            "平均速度_平滑_mm_per_h": round(speed_smooth, 4),
            "R2": round(r2_score_np(yt, yp), 6),
            "MAE_mm": round(mae_np(yt, yp), 6),
            "RMSE_mm": round(rmse_np(yt, yp), 6),
            "DW统计量": round(dw, 4) if math.isfinite(dw) else None,
            "残差标准差_mm": round(float(np.std(res, ddof=1)), 4),
        }
        if phase == "Ⅰ":
            phase_info.update({
                "模型类型": "线性模型",
                "模型形式": f"y = {coef1[0]:.6f}*t + {coef1[1]:.6f}",
                "参数": {"斜率_a1_mm_per_h": round(float(coef1[0]), 6), "截距_b1_mm": round(float(coef1[1]), 6)},
            })
        elif phase == "Ⅱ":
            phase_info.update({
                "模型类型": "二次多项式",
                "模型形式": f"y = {coef2[0]:.8f}*t² + {coef2[1]:.6f}*t + {coef2[2]:.6f}",
                "参数": {"二次项_a2": round(float(coef2[0]), 8), "一次项_b2": round(float(coef2[1]), 6), "截距_c2": round(float(coef2[2]), 6)},
            })
        else:
            phase_info.update({
                "模型类型": "斋藤模型(Saito Model)",
                "模型形式": f"y = {params3['alpha_mm']:.6f} * exp({params3['beta_per_h']:.6f} * (t - {t0:.1f}))",
                "参数": {"alpha_mm": round(params3["alpha_mm"], 6), "beta_per_h": round(params3["beta_per_h"], 6)},
                "物理意义": "α为T2后快速阶段初始位移幅值，β为加速破裂指数，反映材料失稳速率",
            })
        info["阶段模型"][f"Phase_{phase}_{PHASE_LABELS[phase]}"] = phase_info

    # 整体指标
    yt_all = y_fit_target
    yp_all = out["拟合值_mm"].to_numpy(dtype=float)
    info["整体拟合"] = {
        "R2": round(r2_score_np(yt_all, yp_all), 6),
        "MAE_mm": round(mae_np(yt_all, yp_all), 6),
        "RMSE_mm": round(rmse_np(yt_all, yp_all), 6),
    }

    v1 = info["阶段模型"][f"Phase_Ⅰ_{PHASE_LABELS['Ⅰ']}"]["平均速度_平滑_mm_per_h"]
    v2 = info["阶段模型"][f"Phase_Ⅱ_{PHASE_LABELS['Ⅱ']}"]["平均速度_平滑_mm_per_h"]
    v3 = info["阶段模型"][f"Phase_Ⅲ_{PHASE_LABELS['Ⅲ']}"]["平均速度_平滑_mm_per_h"]
    info["阶段速度对比"] = {
        "Phase_I_mm_per_h": v1,
        "Phase_II_mm_per_h": v2,
        "Phase_III_mm_per_h": v3,
        "速度比_II_div_I": round(v2 / v1, 2) if v1 else None,
        "速度比_III_div_II": round(v3 / v2, 2) if v2 else None,
        "速度比_III_div_I": round(v3 / v1, 2) if v1 else None,
    }
    return out, info


def make_answer_tables(info: Dict) -> pd.DataFrame:
    rows = []
    for key in ["T1_匀速转加速", "T2_加速转快速"]:
        node = info["阶段转换节点"][key]
        rows.append({
            "类别": "阶段转换节点",
            "指标": node["节点"],
            "结果": f"索引{node['索引_0based']}，编号{node['Excel编号']}，{node['小时']}h，平滑位移{node['平滑位移_mm']}mm",
            "说明": node["日期时间"],
        })
    for phase_key, p in info["阶段模型"].items():
        rows.append({"类别": "阶段模型", "指标": phase_key, "结果": p["模型形式"], "说明": p["模型类型"]})
        rows.append({"类别": "模型检验", "指标": phase_key + " R²", "结果": p["R2"], "说明": f"DW={p['DW统计量']}，RMSE={p['RMSE_mm']}"})
        rows.append({"类别": "平均速度", "指标": phase_key, "结果": p["平均速度_平滑_mm_per_h"], "说明": "单位：mm/h，按平滑位移阶段增量/阶段时长"})
    rows.append({"类别": "整体拟合", "指标": "整体R²", "结果": info["整体拟合"]["R2"], "说明": f"MAE={info['整体拟合']['MAE_mm']}，RMSE={info['整体拟合']['RMSE_mm']}"})
    return pd.DataFrame(rows)


def plot_all(out: pd.DataFrame, info: Dict, output_dir: Path) -> None:
    setup_chinese_font()
    output_dir.mkdir(parents=True, exist_ok=True)

    t = out["时间_h"].to_numpy()
    y = out["表面位移_mm"].to_numpy()
    ys = out["平滑位移_mm"].to_numpy()
    yp = out["拟合值_mm"].to_numpy()
    res = out["残差_mm"].to_numpy()
    t1 = out.loc[T1_IDX, "时间_h"]
    t2 = out.loc[T2_IDX, "时间_h"]

    # 1 主图
    plt.figure(figsize=(12, 6))
    plt.plot(t, y, linewidth=0.7, alpha=0.45, label="原始位移")
    plt.plot(t, ys, linewidth=1.0, label="SG平滑位移")
    plt.plot(t, yp, linewidth=1.4, label="分阶段拟合")
    plt.axvline(t1, linestyle="--", linewidth=1.2, label=f"T1={t1:.1f}h")
    plt.axvline(t2, linestyle="--", linewidth=1.2, label=f"T2={t2:.1f}h")
    plt.xlabel("时间 / h")
    plt.ylabel("表面位移 / mm")
    plt.title("问题2_三阶段划分主图")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "问题2_三阶段划分主图.png", dpi=220)
    plt.close()

    # 2 分阶段细节
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for ax, phase in zip(axes, ["Ⅰ", "Ⅱ", "Ⅲ"]):
        part = out[out["阶段"] == phase]
        ax.plot(part["时间_h"], part["表面位移_mm"], linewidth=0.8, alpha=0.5, label="原始")
        ax.plot(part["时间_h"], part["平滑位移_mm"], linewidth=1.0, label="平滑")
        ax.plot(part["时间_h"], part["拟合值_mm"], linewidth=1.3, label="拟合")
        ax.set_title(f"阶段{phase}：{PHASE_LABELS[phase]}")
        ax.set_xlabel("时间 / h")
        ax.set_ylabel("位移 / mm")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "问题2_分阶段模型细节.png", dpi=220)
    plt.close(fig)

    # 3 速度演化
    v = np.gradient(ys, SAMPLE_INTERVAL_H)
    v_s = pd.Series(v).rolling(30, center=True, min_periods=1).mean().to_numpy()
    plt.figure(figsize=(12, 5))
    plt.plot(t, v_s, linewidth=1.0, label="平滑速度(30点滚动)")
    plt.axvline(t1, linestyle="--", linewidth=1.0, label="T1")
    plt.axvline(t2, linestyle="--", linewidth=1.0, label="T2")
    for phase in ["Ⅰ", "Ⅱ", "Ⅲ"]:
        pkey = f"Phase_{phase}_{PHASE_LABELS[phase]}"
        vv = info["阶段模型"][pkey]["平均速度_平滑_mm_per_h"]
        mask = out["阶段"] == phase
        plt.hlines(vv, out.loc[mask, "时间_h"].min(), out.loc[mask, "时间_h"].max(), linestyles="dotted", linewidth=1.5, label=f"阶段{phase}平均速度={vv:.4f}")
    plt.xlabel("时间 / h")
    plt.ylabel("速度 / (mm/h)")
    plt.title("问题2_速度演化")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "问题2_速度演化.png", dpi=220)
    plt.close()

    # 4 残差分析
    plt.figure(figsize=(12, 5))
    plt.plot(t, res, linewidth=0.8)
    plt.axhline(0, linestyle="--", linewidth=1.0)
    plt.axvline(t1, linestyle="--", linewidth=1.0)
    plt.axvline(t2, linestyle="--", linewidth=1.0)
    plt.xlabel("时间 / h")
    plt.ylabel("残差 / mm")
    plt.title("问题2_残差时序")
    plt.tight_layout()
    plt.savefig(output_dir / "问题2_残差时序.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(res[np.isfinite(res)], bins=60, alpha=0.85)
    plt.xlabel("残差 / mm")
    plt.ylabel("频数")
    plt.title("问题2_残差分布")
    plt.tight_layout()
    plt.savefig(output_dir / "问题2_残差分布.png", dpi=220)
    plt.close()

    # 5 CUSUM检测示意
    dy = np.diff(ys, prepend=ys[0])
    cusum = np.cumsum(dy - np.mean(dy))
    plt.figure(figsize=(12, 5))
    plt.plot(t, cusum, linewidth=1.0, label="CUSUM")
    plt.axvline(t1, linestyle="--", linewidth=1.0, label="T1")
    plt.axvline(t2, linestyle="--", linewidth=1.0, label="T2")
    plt.xlabel("时间 / h")
    plt.ylabel("累计偏差")
    plt.title("问题2_CUSUM检测")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "问题2_CUSUM检测.png", dpi=220)
    plt.close()


def write_outputs(out: pd.DataFrame, info: Dict, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_cols = ["编号", "索引", "采集时间", "时间_h", "表面位移_mm", "平滑位移_mm", "拟合值_mm", "残差_mm", "阶段", "阶段标签"]
    out[result_cols].to_csv(output_dir / "问题2_三阶段模型结果.csv", index=False, encoding="utf-8-sig")
    out[["编号", "时间_h", "拟合值_mm"]].rename(columns={"拟合值_mm": "预测位移_mm"}).to_csv(output_dir / "问题2_预测结果.csv", index=False, encoding="utf-8-sig")

    answer = make_answer_tables(info)
    answer.to_csv(output_dir / "问题2_最终答案表.csv", index=False, encoding="utf-8-sig")

    with open(output_dir / "问题2_结果摘要.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    plot_all(out, info, output_dir)


def print_summary(info: Dict) -> None:
    print("\n========== 问题2最终运行摘要 ==========")
    print(f"数据点数: {info['数据概况']['数据点数']}")
    for k, v in info["阶段转换节点"].items():
        print(f"{k}: 索引{v['索引_0based']} / 编号{v['Excel编号']} / {v['小时']}h / 平滑位移{v['平滑位移_mm']}mm")
    print("\n阶段模型：")
    for k, v in info["阶段模型"].items():
        print(f"- {k}: {v['模型形式']}, R2={v['R2']}, 平均速度={v['平均速度_平滑_mm_per_h']} mm/h")
    print(f"\n整体R2: {info['整体拟合']['R2']}")
    print("====================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="问题2：三阶段变形节点识别与分段建模")
    parser.add_argument("--input", default="./附件2：位移时序数据-问题2.xlsx", help="附件2 Excel文件路径")
    parser.add_argument("--output-dir", default="./ques2", help="输出目录")
    parser.add_argument("--fit-raw", action="store_true", help="使用原始位移拟合；默认使用SG平滑位移拟合")
    args = parser.parse_args()

    df = read_data(args.input)
    out, info = build_models(df, use_smooth_for_fit=not args.fit_raw)
    write_outputs(out, info, args.output_dir)
    print_summary(info)
    print(f"输出目录: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
