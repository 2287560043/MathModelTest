#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题2输出验证脚本

运行示例：
    python ques2_validate_outputs.py \
        --result-csv ques2_outputs/问题2_三阶段模型结果.csv \
        --pred-csv ques2_outputs/问题2_预测结果.csv \
        --section-result section_result_ques2.json \
        --section-packet section_packet_ques2.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED = {
    "n": 10000,
    "t1_idx": 8144,
    "t2_idx": 9590,
    "t1_time": 1357.3,
    "t2_time": 1598.3,
    "t1_disp": 358.59,
    "t2_disp": 845.39,
    "stage_counts": {"Ⅰ": 8144, "Ⅱ": 1446, "Ⅲ": 410},
    "velocities": {"Ⅰ": 0.2619, "Ⅱ": 2.0180, "Ⅲ": 6.7301},
    "r2": {"Ⅰ": 0.9446, "Ⅱ": 0.9999, "Ⅲ": 0.9981},
    "overall_r2": 0.9922,
}


def ok(name: str) -> None:
    print(f"✓ {name}")


def warn(name: str) -> None:
    print(f"⚠ {name}")


def fail(name: str) -> None:
    print(f"✗ {name}")


def near(a: float, b: float, tol: float) -> bool:
    return abs(float(a) - float(b)) <= tol


def validate_result_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)

    required = ["编号", "时间_h", "表面位移_mm", "拟合值_mm", "残差_mm", "阶段", "阶段标签"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        fail(f"结果CSV缺少列：{missing}")
    else:
        ok("结果CSV列名完整")

    if len(df) == EXPECTED["n"]:
        ok("结果CSV行数=10000")
    else:
        fail(f"结果CSV行数={len(df)}，应为10000")

    counts = df["阶段"].value_counts().sort_index().to_dict()
    if counts == EXPECTED["stage_counts"]:
        ok(f"阶段数量正确：{counts}")
    else:
        fail(f"阶段数量不对：{counts}，期望 {EXPECTED['stage_counts']}")

    stages = df["阶段"].to_numpy()
    t1_idx = int(np.where(stages == "Ⅱ")[0][0]) if "Ⅱ" in set(stages) else -1
    t2_idx = int(np.where(stages == "Ⅲ")[0][0]) if "Ⅲ" in set(stages) else -1
    if t1_idx == EXPECTED["t1_idx"] and t2_idx == EXPECTED["t2_idx"]:
        ok("T1/T2索引正确：8144/9590")
    else:
        fail(f"T1/T2索引不对：{t1_idx}/{t2_idx}")

    t1_time = df.loc[EXPECTED["t1_idx"], "时间_h"]
    t2_time = df.loc[EXPECTED["t2_idx"], "时间_h"]
    t1_disp = df.loc[EXPECTED["t1_idx"], "表面位移_mm"]
    t2_disp = df.loc[EXPECTED["t2_idx"], "表面位移_mm"]
    if near(t1_time, EXPECTED["t1_time"], 0.05) and near(t2_time, EXPECTED["t2_time"], 0.05):
        ok("T1/T2时间正确：约1357.3h/1598.3h")
    else:
        fail(f"T1/T2时间不对：{t1_time:.4f}/{t2_time:.4f}")
    if near(t1_disp, EXPECTED["t1_disp"], 0.05) and near(t2_disp, EXPECTED["t2_disp"], 0.05):
        ok("T1/T2位移正确：约358.59mm/845.39mm")
    else:
        fail(f"T1/T2位移不对：{t1_disp:.4f}/{t2_disp:.4f}")

    for sid in ["Ⅰ", "Ⅱ", "Ⅲ"]:
        sub = df[df["阶段"] == sid]
        v = (sub["表面位移_mm"].iloc[-1] - sub["表面位移_mm"].iloc[0]) / (sub["时间_h"].iloc[-1] - sub["时间_h"].iloc[0])
        ss_res = np.sum(sub["残差_mm"].to_numpy() ** 2)
        y = sub["表面位移_mm"].to_numpy()
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        if near(v, EXPECTED["velocities"][sid], 5e-4):
            ok(f"阶段{sid}平均速度正确：{v:.4f} mm/h")
        else:
            fail(f"阶段{sid}平均速度={v:.4f}，期望约{EXPECTED['velocities'][sid]:.4f}")
        if near(r2, EXPECTED["r2"][sid], 5e-4):
            ok(f"阶段{sid} R²正确：{r2:.4f}")
        else:
            warn(f"阶段{sid} R²={r2:.4f}，期望约{EXPECTED['r2'][sid]:.4f}；检查模型形式/变点是否被改动")

    ss_res = np.sum(df["残差_mm"].to_numpy() ** 2)
    y = df["表面位移_mm"].to_numpy()
    ss_tot = np.sum((y - y.mean()) ** 2)
    overall = 1 - ss_res / ss_tot
    if near(overall, EXPECTED["overall_r2"], 8e-4):
        ok(f"整体R²正确：{overall:.4f}")
    else:
        warn(f"整体R²={overall:.4f}，期望约{EXPECTED['overall_r2']:.4f}")

    return df


def validate_pred_csv(path: Path) -> None:
    if not path.exists():
        warn(f"预测CSV不存在：{path}")
        return
    pred = pd.read_csv(path)
    required = ["编号", "时间小时", "预测位移_mm"]
    missing = [c for c in required if c not in pred.columns]
    if missing:
        fail(f"预测CSV缺少列：{missing}")
    else:
        ok("预测CSV列名完整")
    if len(pred) == EXPECTED["n"]:
        ok("预测CSV行数=10000")
    else:
        warn(f"预测CSV行数={len(pred)}；如做了外推预测则可以不同，否则应为10000")
    n_neg = int((pred["预测位移_mm"] < 0).sum())
    if n_neg == 0:
        ok("预测CSV无负位移")
    else:
        warn(f"预测CSV存在 {n_neg} 个负位移；若要物理约束，运行主脚本时加 --clip-prediction")


def scan_json(path: Path, label: str) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    has_new = all(s in text for s in ["8144", "9590", "1357.3", "1598.3", "0.9922"])
    has_old = any(s in text for s in ["1094.5", "1584.2", "0.9960", "6.2648", "0.2152"])

    print(f"\n【扫描 {label}】")
    if has_new:
        ok("包含新版核心值：8144/9590/1357.3h/1598.3h/0.9922")
    else:
        warn("未完整包含新版核心值")
    if has_old:
        warn("发现旧版残留值：1094.5h/1584.2h/0.9960/旧速度等；论文正文或JSON需要统一")
    else:
        ok("未发现常见旧版残留值")

    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "metrics" in obj:
            metrics = {m.get("name"): str(m.get("value")) for m in obj.get("metrics", []) if isinstance(m, dict)}
            if metrics.get("变点T₁索引") == "8144" and metrics.get("整体R²") == "0.9922":
                ok("metrics字段为新版结果")
            else:
                warn(f"metrics字段可能不是新版结果：T1={metrics.get('变点T₁索引')}, overallR2={metrics.get('整体R²')}")
    except Exception as exc:
        warn(f"JSON解析失败：{exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证问题2输出是否与最终答案一致")
    parser.add_argument("--result-csv", default="./ques2/问题2_三阶段模型结果.csv")
    parser.add_argument("--pred-csv", default="./ques2/问题2_预测结果.csv")
    parser.add_argument("--section-result", default="section_result_ques2.json")
    parser.add_argument("--section-packet", default="section_packet_ques2.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 72)
    print("【问题2输出验证】")
    print("=" * 72)
    validate_result_csv(Path(args.result_csv))
    validate_pred_csv(Path(args.pred_csv))
    scan_json(Path(args.section_result), "section_result_ques2.json")
    scan_json(Path(args.section_packet), "section_packet_ques2.json")


if __name__ == "__main__":
    main()
