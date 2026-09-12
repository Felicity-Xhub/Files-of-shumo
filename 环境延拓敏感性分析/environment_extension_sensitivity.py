#!/usr/bin/env python3
"""A题问题3、4：仅有前4小时环境数据时的边界延拓情景复算。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.sparse import bmat, diags
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


SPLIT_TIME_S = 4.0 * 3600.0
HORIZON_S = 72.0 * 3600.0
TARGET = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="问题3、4的4小时后环境延拓敏感性分析。")
    parser.add_argument("--environment", type=Path, default=Path("outputs/stage3/environment_si.csv"))
    parser.add_argument("--radius", type=Path, default=Path("outputs/stage3/radius_si.csv"))
    parser.add_argument("--q3-model", type=Path, default=Path("q3_model_solve.py"))
    parser.add_argument("--q4-model", type=Path, default=Path("q4_model_solve.py"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/environment_extension_sensitivity_20260912"))
    parser.add_argument("--q3-step-cm", type=float, default=0.0015625)
    parser.add_argument("--q4-intervals", type=int, default=1280)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-11)
    parser.add_argument("--max-step-s", type=float, default=60.0)
    parser.add_argument("--curve-step-s", type=float, default=300.0)
    return parser.parse_args()


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path.resolve())
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模型：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_full_environment(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["time_s", "air_temperature_K", "air_moisture_kgkg"]
    if list(frame.columns) != required:
        raise ValueError(f"环境数据字段应为{required}。")
    if frame[required].isna().any().any() or not np.isfinite(frame[required].to_numpy()).all():
        raise ValueError("环境数据存在缺失值或非有限值。")
    if not frame.time_s.is_monotonic_increasing or frame.time_s.duplicated().any():
        raise ValueError("环境时间必须严格递增且无重复。")
    if frame.time_s.iloc[0] != 0.0 or frame.time_s.iloc[-1] < SPLIT_TIME_S:
        raise ValueError("环境数据必须覆盖0至4小时。")
    if not np.any(np.isclose(frame.time_s.to_numpy(), SPLIT_TIME_S)):
        raise ValueError("环境数据缺少4小时端点。")
    return frame.loc[frame.time_s <= SPLIT_TIME_S].copy()


def build_scenarios(environment_frame: pd.DataFrame) -> tuple[list[dict[str, float | str]], dict[str, float | int | str]]:
    last_hour = environment_frame.loc[
        (environment_frame.time_s >= SPLIT_TIME_S - 3600.0)
        & (environment_frame.time_s <= SPLIT_TIME_S)
    ].copy()
    if len(last_hour) < 2:
        raise ValueError("末1小时数据不足以计算样本标准差。")
    mean_t = float(last_hour.air_temperature_K.mean())
    sd_t = float(last_hour.air_temperature_K.std(ddof=1))
    mean_c = float(last_hour.air_moisture_kgkg.mean())
    sd_c = float(last_hour.air_moisture_kgkg.std(ddof=1))
    final = environment_frame.iloc[-1]
    scenarios: list[dict[str, float | str]] = [
        {"scenario": "S0", "description": "保持4小时末次观测值", "temperature_k": float(final.air_temperature_K), "moisture": float(final.air_moisture_kgkg)},
        {"scenario": "S1", "description": "保持末1小时观测均值", "temperature_k": mean_t, "moisture": mean_c},
        {"scenario": "S2", "description": "温度均值-1个样本标准差，环境水分均值+1个样本标准差", "temperature_k": mean_t - sd_t, "moisture": mean_c + sd_c},
        {"scenario": "S3", "description": "温度均值+1个样本标准差，环境水分均值-1个样本标准差", "temperature_k": mean_t + sd_t, "moisture": mean_c - sd_c},
    ]
    stats: dict[str, float | int | str] = {
        "time_range_s": "[10800, 14400]",
        "time_range_h": "[3, 4]",
        "endpoint_rule": "两端均包含；按附件原始60 s观测点逐点统计",
        "sample_count": int(len(last_hour)),
        "temperature_mean_k": mean_t,
        "temperature_mean_c": mean_t - 273.15,
        "temperature_sample_sd_k": sd_t,
        "temperature_sample_sd_c": sd_t,
        "moisture_mean_kgkg": mean_c,
        "moisture_sample_sd_kgkg": sd_c,
        "last_temperature_k": float(final.air_temperature_K),
        "last_temperature_c": float(final.air_temperature_K - 273.15),
        "last_moisture_kgkg": float(final.air_moisture_kgkg),
    }
    return scenarios, stats


def sparsity_matrix(count: int):
    tri = diags(
        [np.ones(count - 1), np.ones(count), np.ones(count - 1)],
        [-1, 0, 1], shape=(count, count), format="csc",
    )
    return bmat([[tri, tri], [tri, tri]], format="csc")


def post_environment_function(temperature_k: float, moisture: float):
    def environment_at(_time_s: float | np.ndarray, _environment: Any):
        t = np.asarray(_time_s)
        if t.ndim == 0:
            return float(temperature_k), float(moisture)
        return np.full_like(t, temperature_k, dtype=float), np.full_like(t, moisture, dtype=float)
    return environment_at


def validate_solution(solution: Any, label: str) -> None:
    if not solution.success or solution.sol is None:
        raise RuntimeError(f"{label}求解失败：{solution.message}")
    if not np.isfinite(solution.y).all():
        raise RuntimeError(f"{label}产生非有限值。")


def solve_pre_segment(module: Any, grid: Any, environment: Any, radius_data: Any | None, args: argparse.Namespace):
    count = len(grid.radii) if hasattr(grid, "radii") else len(grid.x)
    initial = np.concatenate([
        np.full(count, module.INITIAL_TEMPERATURE_K),
        np.full(count, module.INITIAL_MOISTURE),
    ])
    rhs = module.make_rhs(grid, environment) if radius_data is None else module.make_rhs(grid, environment, radius_data)
    solution = solve_ivp(
        rhs, (0.0, SPLIT_TIME_S), initial, method="BDF", dense_output=True,
        rtol=args.rtol, atol=args.atol, max_step=args.max_step_s,
        jac_sparsity=sparsity_matrix(count),
    )
    validate_solution(solution, "0至4小时预积分")
    if abs(float(solution.t[-1]) - SPLIT_TIME_S) > 1e-7:
        raise RuntimeError("预积分未精确覆盖4小时分段点。")
    return solution, np.asarray(solution.y[:, -1], dtype=float)


def solve_post_segment(
    module: Any, grid: Any, environment: Any, radius_data: Any | None,
    state_at_4h: np.ndarray, scenario: dict[str, float | str], args: argparse.Namespace,
):
    count = len(grid.radii) if hasattr(grid, "radii") else len(grid.x)
    original_environment_at = module.environment_at
    module.environment_at = post_environment_function(float(scenario["temperature_k"]), float(scenario["moisture"]))
    try:
        rhs = module.make_rhs(grid, environment) if radius_data is None else module.make_rhs(grid, environment, radius_data)
        solution = solve_ivp(
            rhs, (SPLIT_TIME_S, HORIZON_S), state_at_4h, method="BDF", dense_output=True,
            events=module.make_threshold_event(count), rtol=args.rtol, atol=args.atol,
            max_step=args.max_step_s, jac_sparsity=sparsity_matrix(count),
        )
    finally:
        module.environment_at = original_environment_at
    validate_solution(solution, f"{scenario['scenario']}的4小时后积分")
    event_time = None
    if len(solution.t_events) == 1 and len(solution.t_events[0]) == 1:
        event_time = float(solution.t_events[0][0])
        event_state = np.asarray(solution.y_events[0][0], dtype=float)
        if not np.isfinite(event_state).all():
            raise RuntimeError("阈值事件状态存在非有限值。")
        if abs(float(np.max(event_state[count:])) - TARGET) > 5e-8:
            raise RuntimeError("阈值事件未由全节点最大含水率=0.15触发。")
    elif float(solution.t[-1]) < HORIZON_S - 1e-7:
        raise RuntimeError("无阈值事件且积分未覆盖72小时。")
    max_72h = None if event_time is not None else float(np.max(solution.sol(HORIZON_S)[count:]))
    if max_72h is not None and not np.isfinite(max_72h):
        raise RuntimeError("72小时最大含水率为非有限值。")
    return solution, event_time, max_72h


def curve_frame(pre: Any, post: Any, count: int, scenario_id: str, curve_step_s: float) -> pd.DataFrame:
    post_end = float(post.t[-1])
    times = np.arange(0.0, post_end + 0.1, curve_step_s)
    if times.size == 0 or times[-1] < post_end - 1e-7:
        times = np.append(times, post_end)
    elif abs(times[-1] - post_end) > 1e-7:
        times = np.append(times[times < post_end], post_end)
    values = np.empty(times.size)
    pre_mask = times <= SPLIT_TIME_S + 1e-9
    if np.any(pre_mask):
        values[pre_mask] = np.max(pre.sol(np.minimum(times[pre_mask], SPLIT_TIME_S))[count:, :], axis=0)
    if np.any(~pre_mask):
        values[~pre_mask] = np.max(post.sol(times[~pre_mask])[count:, :], axis=0)
    if not np.isfinite(values).all():
        raise RuntimeError("曲线采样出现非有限值。")
    return pd.DataFrame({"scenario": scenario_id, "time_s": times, "time_h": times / 3600.0, "global_max_moisture_kgkg": values})


def plot_curves(frame: pd.DataFrame, question: str, path: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=160)
    colors = {"S0": "#1f77b4", "S1": "#2ca02c", "S2": "#d62728", "S3": "#ff7f0e"}
    for scenario, group in frame.groupby("scenario", sort=False):
        ax.plot(group.time_h, group.global_max_moisture_kgkg, lw=2.0, color=colors[scenario], label=scenario)
        last = group.iloc[-1]
        if abs(float(last.global_max_moisture_kgkg) - TARGET) < 2e-5:
            ax.scatter([last.time_h], [last.global_max_moisture_kgkg], s=32, color=colors[scenario], zorder=4)
    ax.axhline(TARGET, color="#333333", ls="--", lw=1.4, label="阈值 0.15")
    ax.axvline(4.0, color="#777777", ls=":", lw=1.2, label="4 h 分段点")
    ax.set(xlabel="时间 / h", ylabel="全域最大含水率 / (kg/kg)", title=f"{question}：不同环境延拓情景下的全域最大含水率")
    ax.set_xlim(0.0, max(72.0, float(frame.time_h.max())) if (frame.time_h.max() >= 71.9) else float(frame.time_h.max()) * 1.03)
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, frameon=False)
    # 主图保留完整干燥过程，插图放大阈值附近，避免四条曲线重叠而看不清时间差。
    inset = inset_axes(ax, width="38%", height="42%", loc="center right", borderpad=1.8)
    end_times = []
    for scenario, group in frame.groupby("scenario", sort=False):
        inset.plot(group.time_h, group.global_max_moisture_kgkg, lw=1.7, color=colors[scenario])
        end_times.append(float(group.time_h.iloc[-1]))
    inset.axhline(TARGET, color="#333333", ls="--", lw=1.0)
    inset.set_xlim(min(end_times) - 1.0, max(end_times) + 0.6)
    inset.set_ylim(0.145, 0.172)
    inset.set_title("阈值附近放大", fontsize=9)
    inset.tick_params(labelsize=8)
    inset.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env_frame = load_full_environment(args.environment)
    environment = tuple(env_frame[c].to_numpy(dtype=float) for c in env_frame.columns)
    scenarios, stats = build_scenarios(env_frame)
    q3 = import_module(args.q3_model, "q3_sensitivity_model")
    q4 = import_module(args.q4_model, "q4_sensitivity_model")
    radius_data = q4.load_radius(args.radius)
    if float(radius_data[0][-1]) < HORIZON_S:
        raise RuntimeError("附件2收缩曲线未覆盖72小时。")

    q3_grid = q3.build_grid(args.q3_step_cm)
    q4_grid = q4.build_grid(args.q4_intervals)
    q3_pre, q3_state = solve_pre_segment(q3, q3_grid, environment, None, args)
    q4_pre, q4_state = solve_pre_segment(q4, q4_grid, environment, radius_data, args)

    results: list[dict[str, Any]] = []
    q3_curves: list[pd.DataFrame] = []
    q4_curves: list[pd.DataFrame] = []
    for scenario in scenarios:
        print(f"正在计算 {scenario['scenario']} ...", flush=True)
        q3_post, q3_event, q3_max72 = solve_post_segment(q3, q3_grid, environment, None, q3_state, scenario, args)
        q4_post, q4_event, q4_max72 = solve_post_segment(q4, q4_grid, environment, radius_data, q4_state, scenario, args)
        q3_curves.append(curve_frame(q3_pre, q3_post, len(q3_grid.radii), str(scenario["scenario"]), args.curve_step_s))
        q4_curves.append(curve_frame(q4_pre, q4_post, len(q4_grid.x), str(scenario["scenario"]), args.curve_step_s))
        results.append({
            **scenario,
            "post4h_temperature_c": float(scenario["temperature_k"]) - 273.15,
            "q3_status": "达到阈值" if q3_event is not None else "72小时内未达标",
            "q3_event_time_s": q3_event,
            "q3_event_time_h": None if q3_event is None else q3_event / 3600.0,
            "q3_max_moisture_72h": q3_max72,
            "q4_status": "达到阈值" if q4_event is not None else "72小时内未达标",
            "q4_event_time_s": q4_event,
            "q4_event_time_h": None if q4_event is None else q4_event / 3600.0,
            "q4_max_moisture_72h": q4_max72,
        })

    summary = pd.DataFrame(results)
    for q in ("q3", "q4"):
        baseline = summary.loc[summary.scenario == "S0", f"{q}_event_time_s"].iloc[0]
        if pd.isna(baseline):
            summary[f"{q}_delta_vs_s0_s"] = np.nan
            summary[f"{q}_delta_vs_s0_h"] = np.nan
            summary[f"{q}_delta_vs_s0_percent"] = np.nan
        else:
            summary[f"{q}_delta_vs_s0_s"] = summary[f"{q}_event_time_s"] - baseline
            summary[f"{q}_delta_vs_s0_h"] = summary[f"{q}_delta_vs_s0_s"] / 3600.0
            summary[f"{q}_delta_vs_s0_percent"] = summary[f"{q}_delta_vs_s0_s"] / baseline * 100.0
    summary["q3_grid"] = f"固定半径径向网格：{len(q3_grid.radii)}节点，Δr={q3_grid.dr*100:.7f} cm"
    summary["q4_grid"] = f"归一化移动域：{len(q4_grid.x)}节点，{args.q4_intervals}区间"
    summary["solver"] = f"BDF；rtol={args.rtol:g}；atol={args.atol:g}；max_step={args.max_step_s:g} s；4 h分段"
    summary["q4_scope_note"] = "给定收缩曲线条件下的边界情景分析"

    q3_curve = pd.concat(q3_curves, ignore_index=True)
    q4_curve = pd.concat(q4_curves, ignore_index=True)
    summary.to_csv(args.output_dir / "环境延拓敏感性汇总.csv", index=False, encoding="utf-8-sig")
    q3_curve.to_csv(args.output_dir / "第三问全域最大含水率曲线.csv", index=False, encoding="utf-8-sig")
    q4_curve.to_csv(args.output_dir / "第四问全域最大含水率曲线.csv", index=False, encoding="utf-8-sig")
    plot_curves(q3_curve, "第三问", args.output_dir / "第三问环境延拓敏感性曲线.png")
    plot_curves(q4_curve, "第四问", args.output_dir / "第四问环境延拓敏感性曲线.png")

    payload = {
        "environment_statistics": stats,
        "scenario_definition_note": "S2、S3为人工组合的情景扰动，不代表实际发生概率、置信区间或严格最坏情况。",
        "split_integration": "0至4小时采用附件原始环境数据及线性插值；4小时状态作为各情景后段积分的共同初值，边界跳变在分段处显式处理。",
        "q3_configuration": {"nodes": len(q3_grid.radii), "step_cm": args.q3_step_cm, "rtol": args.rtol, "atol": args.atol, "max_step_s": args.max_step_s},
        "q4_configuration": {"nodes": len(q4_grid.x), "intervals": args.q4_intervals, "rtol": args.rtol, "atol": args.atol, "max_step_s": args.max_step_s, "radius_horizon_s": float(radius_data[0][-1]), "scope_note": "给定收缩曲线条件下的边界情景分析"},
        "event_definition": "临界时间由全计算节点最大含水率从上向下穿越0.15的终止事件确定。",
        "results": summary.replace({np.nan: None}).to_dict(orient="records"),
    }
    (args.output_dir / "计算说明与结果.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary[["scenario", "post4h_temperature_c", "moisture", "q3_status", "q3_event_time_h", "q3_delta_vs_s0_percent", "q4_status", "q4_event_time_h", "q4_delta_vs_s0_percent"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
