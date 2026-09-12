#!/usr/bin/env python3
"""A题问题4：附件2收缩半径驱动的归一化移动边界热湿模型。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.integrate import solve_ivp
from scipy.sparse import bmat, diags


INITIAL_RADIUS_M = 0.02
INITIAL_TEMPERATURE_K = 301.15
INITIAL_MOISTURE = 2.55
TARGET_MOISTURE = 0.15
# 使认证终点在四位小数报告中也能显示为小于0.1500。
CERTIFICATION_MARGIN = 5.0e-5
HEAT_TRANSFER_COEFFICIENT = 25.0
MASS_TRANSFER_COEFFICIENT = 8.0e-7
BOUNDARY_CONSTANT_AFTER_S = 4.0 * 3600.0
FIXED_OUTPUT_RADII_M = np.arange(0.0, INITIAL_RADIUS_M, 0.001)
TABLE_FIXED_RADII_M = np.array([0.0, 0.005, 0.010])


@dataclass
class Grid:
    x: np.ndarray
    reference_volumes: np.ndarray
    east_areas_x: np.ndarray
    dx: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="求解A题问题4的移动边界热湿模型。")
    parser.add_argument(
        "--environment", type=Path, default=Path("outputs/stage3/environment_si.csv"),
        help="阶段三环境边界CSV；4 h后自动保持末值。",
    )
    parser.add_argument(
        "--radius-data", type=Path, default=Path("outputs/stage3/radius_si.csv"),
        help="阶段三半径CSV，字段为time_s,radius_m。",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/stage4_q4"), help="输出目录。"
    )
    parser.add_argument("--intervals", type=int, default=1280)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-11)
    parser.add_argument("--skip-diagnostics", action="store_true")
    return parser.parse_args()


def build_grid(intervals: int) -> Grid:
    if intervals < 10:
        raise ValueError("归一化网格区间数至少为10。")
    dx = 1.0 / intervals
    x = np.linspace(0.0, 1.0, intervals + 1)
    west_faces = np.maximum(0.0, x - dx / 2.0)
    east_faces = np.minimum(1.0, x + dx / 2.0)
    reference_volumes = np.pi * (east_faces**2 - west_faces**2)
    east_areas_x = 2.0 * np.pi * east_faces
    return Grid(x, reference_volumes, east_areas_x, dx)


def load_environment(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"未找到环境数据：{path}")
    frame = pd.read_csv(path)
    required = ["time_s", "air_temperature_K", "air_moisture_kgkg"]
    if list(frame.columns) != required:
        raise ValueError(f"环境数据字段应为{required}，实际为{list(frame.columns)}。")
    if frame[required].isna().any().any():
        raise ValueError("环境数据存在缺失值。")
    if not frame.time_s.is_monotonic_increasing or frame.time_s.duplicated().any():
        raise ValueError("环境时间必须严格递增且无重复。")
    if frame.time_s.iloc[0] != 0 or frame.time_s.iloc[-1] < BOUNDARY_CONSTANT_AFTER_S:
        raise ValueError("环境数据必须从0 s开始并至少覆盖4 h。")
    frame = frame.loc[frame.time_s <= BOUNDARY_CONSTANT_AFTER_S].copy()
    if frame.time_s.iloc[-1] != BOUNDARY_CONSTANT_AFTER_S:
        raise ValueError("环境数据缺少4 h边界值。")
    return tuple(frame[c].to_numpy(dtype=float) for c in required)  # type: ignore[return-value]


def load_radius(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"未找到半径数据：{path}")
    frame = pd.read_csv(path)
    required = ["time_s", "radius_m"]
    if list(frame.columns) != required:
        raise ValueError(f"半径数据字段应为{required}，实际为{list(frame.columns)}。")
    if frame[required].isna().any().any():
        raise ValueError("半径数据存在缺失值。")
    if not frame.time_s.is_monotonic_increasing or frame.time_s.duplicated().any():
        raise ValueError("半径时间必须严格递增且无重复。")
    if frame.time_s.iloc[0] != 0 or not np.isclose(frame.radius_m.iloc[0], INITIAL_RADIUS_M):
        raise ValueError("半径数据必须从t=0、R=0.02 m开始。")
    radius = frame.radius_m.to_numpy(dtype=float)
    if np.any(radius <= 0) or np.any(np.diff(radius) > 1e-12):
        raise ValueError("半径必须为正且随时间非增。")
    return frame.time_s.to_numpy(dtype=float), radius


def density(c: np.ndarray) -> np.ndarray:
    return 760.0 + 90.0 * c


def heat_capacity(c: np.ndarray) -> np.ndarray:
    return 1850.0 + 2150.0 * c / (c + 1.0)


def conductivity(c: np.ndarray) -> np.ndarray:
    return 0.12 + 0.20 * c / (c + 1.0)


def diffusivity(c: np.ndarray, temperature_k: np.ndarray, scale: float = 1.0) -> np.ndarray:
    safe_c = np.maximum(c, 1e-12)
    safe_t = np.maximum(temperature_k, 1.0)
    return scale * 4.2e-4 * np.exp(-0.30 / safe_c) * np.exp(-3850.0 / safe_t)


def harmonic_mean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return 2.0 * left * right / np.maximum(left + right, 1e-30)


def environment_at(
    time_s: float | np.ndarray, environment: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> tuple[float | np.ndarray, float | np.ndarray]:
    times, temperatures, moistures = environment
    return np.interp(time_s, times, temperatures), np.interp(time_s, times, moistures)


def surface_equilibrium_moisture(air_moisture: float | np.ndarray) -> float | np.ndarray:
    """有效表面平衡含水率近似。

    题目未给出吸附等温线，故将空气含水率直接作为表面平衡含水率。
    这是显式模型假设，而非把两种物理量视为已知严格等价关系。
    """
    return air_moisture


def radius_at(time_s: float | np.ndarray, radius_data: tuple[np.ndarray, np.ndarray]):
    return np.interp(time_s, radius_data[0], radius_data[1])


def make_rhs(
    grid: Grid,
    environment: tuple[np.ndarray, np.ndarray, np.ndarray],
    radius_data: tuple[np.ndarray, np.ndarray],
    h_scale: float = 1.0,
    hm_scale: float = 1.0,
    d_scale: float = 1.0,
    radius_scale: float = 1.0,
):
    count = len(grid.x)

    def rhs(time_s: float, state: np.ndarray) -> np.ndarray:
        temperature = state[:count]
        moisture = state[count:]
        if np.any(moisture <= 0.0):
            raise RuntimeError("迭代中出现非物理含水率。")
        air_t, air_c = environment_at(time_s, environment)
        surface_c_eq = surface_equilibrium_moisture(air_c)
        base_radius = float(radius_at(time_s, radius_data))
        # 扰动只作用于相对初始半径的收缩量，保证R(0)=2 cm。
        radius = INITIAL_RADIUS_M - radius_scale * (INITIAL_RADIUS_M - base_radius)
        if radius <= 0:
            raise RuntimeError("半径扰动产生非正半径。")
        physical_volumes = radius**2 * grid.reference_volumes

        d_temperature = np.zeros(count)
        local_k = conductivity(moisture)
        face_k = harmonic_mean(local_k[:-1], local_k[1:])
        heat_flux = face_k * grid.east_areas_x[:-1] / grid.dx * np.diff(temperature)
        d_temperature[:-1] += heat_flux
        d_temperature[1:] -= heat_flux
        d_temperature[-1] += (
            h_scale * HEAT_TRANSFER_COEFFICIENT * radius * grid.east_areas_x[-1]
            * (float(air_t) - temperature[-1])
        )
        d_temperature /= density(moisture) * heat_capacity(moisture) * physical_volumes

        d_moisture = np.zeros(count)
        local_d = diffusivity(moisture, temperature, d_scale)
        face_d = harmonic_mean(local_d[:-1], local_d[1:])
        mass_flux = face_d * grid.east_areas_x[:-1] / grid.dx * np.diff(moisture)
        d_moisture[:-1] += mass_flux
        d_moisture[1:] -= mass_flux
        d_moisture[-1] += (
            hm_scale * MASS_TRANSFER_COEFFICIENT * radius * grid.east_areas_x[-1]
            * (float(surface_c_eq) - moisture[-1])
        )
        d_moisture /= physical_volumes
        return np.concatenate([d_temperature, d_moisture])

    return rhs


def make_threshold_event(count: int):
    def threshold_event(_time_s: float, state: np.ndarray) -> float:
        return float(np.max(state[count:]) - TARGET_MOISTURE)

    threshold_event.terminal = True
    threshold_event.direction = -1.0
    return threshold_event


def solve_to_threshold(
    intervals: int,
    environment: tuple[np.ndarray, np.ndarray, np.ndarray],
    radius_data: tuple[np.ndarray, np.ndarray],
    rtol: float,
    atol: float,
    h_scale: float = 1.0,
    hm_scale: float = 1.0,
    d_scale: float = 1.0,
    radius_scale: float = 1.0,
) -> tuple[Grid, Any, float]:
    grid = build_grid(intervals)
    count = len(grid.x)
    initial = np.concatenate([
        np.full(count, INITIAL_TEMPERATURE_K), np.full(count, INITIAL_MOISTURE)
    ])
    tri = diags(
        [np.ones(count - 1), np.ones(count), np.ones(count - 1)],
        [-1, 0, 1], shape=(count, count), format="csc",
    )
    sparsity = bmat([[tri, tri], [tri, tri]], format="csc")
    max_time_s = float(radius_data[0][-1])
    solution = solve_ivp(
        make_rhs(grid, environment, radius_data, h_scale, hm_scale, d_scale, radius_scale),
        (0.0, max_time_s), initial, method="BDF", dense_output=True,
        events=make_threshold_event(count), rtol=rtol, atol=atol,
        max_step=60.0, jac_sparsity=sparsity,
    )
    if not solution.success:
        raise RuntimeError(f"BDF求解失败：{solution.message}")
    if solution.sol is None:
        raise RuntimeError("求解器未返回连续插值解。")
    if len(solution.t_events) != 1 or len(solution.t_events[0]) != 1:
        raise RuntimeError("在附件2覆盖的72 h内未定位到烘干阈值，不能外推半径数据。")
    return grid, solution, float(solution.t_events[0][0])


def solve_over_horizon(
    intervals: int,
    environment: tuple[np.ndarray, np.ndarray, np.ndarray],
    radius_data: tuple[np.ndarray, np.ndarray],
    rtol: float,
    atol: float,
    end_time_s: float,
    h_scale: float = 1.0,
    hm_scale: float = 1.0,
    d_scale: float = 1.0,
    radius_scale: float = 1.0,
) -> tuple[Grid, Any]:
    """积分到指定时刻，不用阈值事件提前终止。"""
    if end_time_s <= 0.0 or end_time_s > float(radius_data[0][-1]):
        raise ValueError("积分终点必须在附件2半径数据覆盖范围内。")
    grid = build_grid(intervals)
    count = len(grid.x)
    initial = np.concatenate([
        np.full(count, INITIAL_TEMPERATURE_K), np.full(count, INITIAL_MOISTURE)
    ])
    tri = diags(
        [np.ones(count - 1), np.ones(count), np.ones(count - 1)],
        [-1, 0, 1], shape=(count, count), format="csc",
    )
    sparsity = bmat([[tri, tri], [tri, tri]], format="csc")
    solution = solve_ivp(
        make_rhs(grid, environment, radius_data, h_scale, hm_scale, d_scale, radius_scale),
        (0.0, end_time_s), initial, method="BDF", dense_output=True,
        rtol=rtol, atol=atol, max_step=60.0, jac_sparsity=sparsity,
    )
    if not solution.success or solution.sol is None:
        raise RuntimeError(f"BDF求解失败：{solution.message}")
    return grid, solution


def sample_at_x(grid: Grid, solution: Any, times: np.ndarray, x_values: np.ndarray):
    count = len(grid.x)
    states = solution.sol(times)
    out_t = np.empty((len(times), len(x_values)))
    out_c = np.empty_like(out_t)
    for j in range(len(times)):
        out_t[j] = np.interp(x_values, grid.x, states[:count, j])
        out_c[j] = np.interp(x_values, grid.x, states[count:, j])
    return out_t, out_c


def sample_physical_outputs(
    grid: Grid, solution: Any, times: np.ndarray, radius_data: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    count = len(grid.x)
    moisture = np.full((len(times), len(FIXED_OUTPUT_RADII_M) + 1), np.nan)
    radii = radius_at(times, radius_data)
    for start in range(0, len(times), 600):
        stop = min(start + 600, len(times))
        states = solution.sol(times[start:stop])[count:, :]
        for local, row in enumerate(range(start, stop)):
            valid = FIXED_OUTPUT_RADII_M <= radii[row] + 1e-12
            moisture[row, :-1][valid] = np.interp(
                FIXED_OUTPUT_RADII_M[valid] / radii[row], grid.x, states[:, local]
            )
            moisture[row, -1] = states[-1, local]
    return radii, moisture


def certification_time_on_output_grid(event_time_s: float) -> float:
    """取临界时刻后至少120 s的60 s网格认证候选时刻。"""
    return float(np.ceil((event_time_s + 120.0) / 60.0) * 60.0)


def make_output_times(certification_time_s: float) -> np.ndarray:
    """从0 s开始、每60 s输出，终点为认证时刻。"""
    return np.arange(0.0, certification_time_s + 0.1, 60.0)


def event_metrics(
    grid: Grid, solution: Any, event_time_s: float, radius_data: tuple[np.ndarray, np.ndarray]
) -> dict[str, float]:
    count = len(grid.x)
    state = solution.sol(np.array([event_time_s]))[:, 0]
    moisture = state[count:]
    max_index = int(np.argmax(moisture))
    radius = float(radius_at(event_time_s, radius_data))
    return {
        "event_time_s": event_time_s,
        "event_time_h": event_time_s / 3600.0,
        "event_time_days": event_time_s / 86400.0,
        "radius_at_event_cm": radius * 100.0,
        "critical_radius_cm": float(grid.x[max_index] * radius * 100.0),
        "max_moisture_kgkg": float(moisture[max_index]),
        "center_moisture_kgkg": float(moisture[0]),
        "surface_moisture_kgkg": float(moisture[-1]),
        "reference_area_weighted_moisture_kgkg": float(
            np.dot(moisture, grid.reference_volumes) / grid.reference_volumes.sum()
        ),
        "center_temperature_C": float(state[0] - 273.15),
        "surface_temperature_C": float(state[count - 1] - 273.15),
    }


def convergence_analysis(environment, radius_data, rtol: float, atol: float) -> pd.DataFrame:
    intervals_list = [160, 320, 640, 1280]
    records = []
    for intervals in intervals_list:
        grid, solution, event_time = solve_to_threshold(
            intervals, environment, radius_data, rtol, atol
        )
        records.append((intervals, solution, event_time))
    reference = records[-1][2]
    return pd.DataFrame([{
        "intervals": intervals,
        "nodes": intervals + 1,
        "initial_physical_step_cm": 2.0 / intervals,
        "event_time_h": event_time / 3600.0,
        "event_time_difference_s_vs_finest": event_time - reference,
        "nfev": int(solution.nfev), "njev": int(solution.njev), "nlu": int(solution.nlu),
    } for intervals, solution, event_time in records])


def sensitivity_analysis(environment, radius_data, rtol: float, atol: float) -> pd.DataFrame:
    scenarios = [
        ("基准", 1.0, 1.0, 1.0, 1.0), ("h -10%", 0.9, 1.0, 1.0, 1.0),
        ("h +10%", 1.1, 1.0, 1.0, 1.0), ("hm -10%", 1.0, 0.9, 1.0, 1.0),
        ("hm +10%", 1.0, 1.1, 1.0, 1.0), ("D -10%", 1.0, 1.0, 0.9, 1.0),
        ("D +10%", 1.0, 1.0, 1.1, 1.0), ("收缩量 -10%", 1.0, 1.0, 1.0, 0.9),
        ("收缩量 +10%", 1.0, 1.0, 1.0, 1.1),
    ]
    rows = []
    for name, hs, hms, ds, rs in scenarios:
        grid, solution, event_time = solve_to_threshold(
            160, environment, radius_data, rtol, atol,
            h_scale=hs, hm_scale=hms, d_scale=ds, radius_scale=rs,
        )
        metrics = event_metrics(grid, solution, event_time, radius_data)
        rows.append({
            "scenario": name, "h_scale": hs, "hm_scale": hms, "D_scale": ds,
            "shrinkage_scale": rs, "event_time_h": metrics["event_time_h"],
            "base_radius_at_event_cm": metrics["radius_at_event_cm"],
            "surface_moisture_at_event": metrics["surface_moisture_kgkg"],
        })
    return pd.DataFrame(rows)


def conservation_report(grid: Grid, solution: Any, environment, radius_data) -> dict[str, float]:
    count = len(grid.x)
    state = solution.y
    total = state[count:, :].T @ grid.reference_volumes
    _, air_c = environment_at(solution.t, environment)
    surface_c_eq = surface_equilibrium_moisture(air_c)
    radii = radius_at(solution.t, radius_data)
    boundary_rate = (
        MASS_TRANSFER_COEFFICIENT * grid.east_areas_x[-1] / radii
        * (surface_c_eq - state[-1, :])
    )
    cumulative = np.zeros(len(solution.t))
    cumulative[1:] = np.cumsum(
        0.5 * (boundary_rate[1:] + boundary_rate[:-1]) * np.diff(solution.t)
    )
    residual = total - total[0] - cumulative
    scale = max(abs(total[-1] - total[0]), 1e-15)
    return {
        "max_absolute_reference_balance_residual": float(np.max(np.abs(residual))),
        "max_relative_reference_balance_residual": float(np.max(np.abs(residual)) / scale),
    }


def configure_plotting() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_radius(radius_data, event_time_s: float, path: Path) -> None:
    configure_plotting()
    fig, axis = plt.subplots(figsize=(8.0, 4.5))
    axis.plot(radius_data[0] / 3600.0, radius_data[1] * 100.0, color="#1F4E78")
    axis.axvline(event_time_s / 3600.0, color="#C00000", linestyle="--", label="烘干结束")
    axis.scatter([event_time_s / 3600.0], [radius_at(event_time_s, radius_data) * 100.0], color="#C00000")
    axis.set_xlabel("时间（h）"); axis.set_ylabel("药材半径（cm）")
    axis.set_title("附件2半径变化与烘干结束时刻")
    axis.grid(alpha=0.25); axis.legend(); fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_profiles(grid: Grid, solution: Any, event_time_s: float, radius_data, path: Path) -> None:
    configure_plotting()
    regular = np.arange(0.0, np.floor(event_time_s / 21600.0) * 21600.0 + 1.0, 21600.0)
    times = np.unique(np.append(regular, event_time_s))
    states = solution.sol(times)
    count = len(grid.x)
    fig, axis = plt.subplots(figsize=(8.2, 4.7))
    for j, time_s in enumerate(times):
        physical_r_cm = grid.x * float(radius_at(time_s, radius_data)) * 100.0
        axis.plot(physical_r_cm, states[count:, j], label=f"{time_s / 3600:.1f} h")
    axis.axhline(TARGET_MOISTURE, color="#C00000", linestyle="--", label="阈值0.15")
    axis.set_xlabel("到中心距离（cm）"); axis.set_ylabel("水分浓度（kg/kg）")
    axis.set_title("收缩过程中水分浓度径向分布")
    axis.grid(alpha=0.25); axis.legend(ncol=2, fontsize=8); fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_flow(path: Path) -> None:
    configure_plotting()
    fig, axis = plt.subplots(figsize=(12, 2.5))
    axis.set_xlim(0, 12); axis.set_ylim(0, 2.5); axis.axis("off")
    labels = [
        "附件1环境\n附件2半径", "变换x=r/R(t)\n固定计算域", "附录4变物性\n有限体积离散",
        "BDF推进\n更新R(t)", "全域事件定位\nmax C=0.15", "收敛/敏感性\n结果导出",
    ]
    xs = [0.2, 2.2, 4.2, 6.2, 8.2, 10.2]
    for x, label in zip(xs, labels):
        axis.add_patch(FancyBboxPatch(
            (x, 0.75), 1.6, 1.0, boxstyle="round,pad=0.08",
            facecolor="#EAF2F8", edgecolor="#1F4E78", linewidth=1.4,
        ))
        axis.text(x + 0.8, 1.25, label, ha="center", va="center", fontsize=10)
    for x in xs[:-1]:
        axis.add_patch(FancyArrowPatch(
            (x + 1.62, 1.25), (x + 1.98, 1.25), arrowstyle="-|>",
            mutation_scale=13, color="#555555", linewidth=1.2,
        ))
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def main() -> int:
    args = parse_args()
    environment = load_environment(args.environment.expanduser().resolve())
    radius_data = load_radius(args.radius_data.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    grid, event_solution, event_time = solve_to_threshold(
        args.intervals, environment, radius_data, args.rtol, args.atol
    )
    metrics = event_metrics(grid, event_solution, event_time, radius_data)
    count = len(grid.x)
    certification_time = certification_time_on_output_grid(event_time)
    output_times = make_output_times(certification_time)
    grid, solution = solve_over_horizon(
        args.intervals, environment, radius_data, args.rtol, args.atol, certification_time
    )
    certification_state = solution.sol(np.array([certification_time]))[count:, 0]
    certification_max = float(np.max(certification_state))
    if certification_max >= TARGET_MOISTURE - CERTIFICATION_MARGIN:
        raise RuntimeError("认证终点未以数值裕量严格低于含水率阈值。")
    output_radii, output_moisture = sample_physical_outputs(
        grid, solution, output_times, radius_data
    )
    columns = [f"{r * 100:.1f}" for r in FIXED_OUTPUT_RADII_M] + ["药材表面"]
    full_frame = pd.DataFrame(output_moisture, columns=columns)
    full_frame.insert(0, "时间/s", output_times)
    full_frame.to_csv(output_dir / "q4_moisture_full.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"time_s": output_times, "radius_cm": output_radii * 100.0}).to_csv(
        output_dir / "q4_radius_at_output.csv", index=False, encoding="utf-8-sig"
    )

    regular_table_times = np.arange(21600.0, np.floor(event_time / 21600.0) * 21600.0 + 1.0, 21600.0)
    table_times = np.append(regular_table_times, certification_time)
    table = np.empty((len(table_times), len(TABLE_FIXED_RADII_M) + 1))
    for j, time_s in enumerate(table_times):
        radius = float(radius_at(time_s, radius_data))
        state_c = solution.sol(np.array([time_s]))[count:, 0]
        table[j, :-1] = np.interp(TABLE_FIXED_RADII_M / radius, grid.x, state_c)
        table[j, -1] = state_c[-1]
    paper = pd.DataFrame(
        table,
        index=[f"{t / 3600:.0f}" for t in regular_table_times] + ["烘干结束时间（认证）"],
        columns=["0", "0.5", "1", "药材表面"],
    )
    paper.index.name = "时间/h"
    paper.to_csv(output_dir / "q4_table_moisture.csv", encoding="utf-8-sig", float_format="%.4f")

    event_state = event_solution.sol(np.array([event_time]))[count:, 0]
    before_max = float(np.max(event_solution.sol(np.array([max(0.0, event_time - 1.0)]))[count:, 0]))
    if np.max(event_state) > TARGET_MOISTURE + 2e-8 or before_max <= TARGET_MOISTURE:
        raise RuntimeError("全域阈值首次穿越校验失败。")
    if not np.isfinite(output_moisture[~np.isnan(output_moisture)]).all():
        raise RuntimeError("结果存在非有限数。")

    if args.skip_diagnostics:
        convergence = pd.DataFrame()
        sensitivity = pd.DataFrame()
    else:
        convergence = convergence_analysis(environment, radius_data, args.rtol, args.atol)
        sensitivity = sensitivity_analysis(environment, radius_data, args.rtol, args.atol)
        convergence.to_csv(output_dir / "q4_convergence.csv", index=False, encoding="utf-8-sig")
        sensitivity.to_csv(output_dir / "q4_sensitivity.csv", index=False, encoding="utf-8-sig")
    balance = conservation_report(grid, solution, environment, radius_data)
    baseline_grid, baseline_solution = solve_over_horizon(
        args.intervals, environment, radius_data, args.rtol, args.atol,
        float(radius_data[0][-1]), radius_scale=0.0,
    )
    baseline_final = baseline_solution.sol(np.array([radius_data[0][-1]]))[len(baseline_grid.x):, 0]
    pd.DataFrame([
        {"scenario": "附件2动态收缩", "grid_intervals": args.intervals,
         "observation_time_h": certification_time / 3600.0, "event_reached": True,
         "center_moisture_kgkg": float(certification_state[0])},
        {"scenario": "固定半径2 cm", "grid_intervals": args.intervals,
         "observation_time_h": float(radius_data[0][-1] / 3600.0), "event_reached": False,
         "center_moisture_kgkg": float(baseline_final[0])},
    ]).to_csv(output_dir / "q4_ablation.csv", index=False, encoding="utf-8-sig")
    plot_radius(radius_data, certification_time, output_dir / "q4_radius_history.png")
    plot_profiles(grid, solution, certification_time, radius_data, output_dir / "q4_radial_profiles.png")
    plot_flow(output_dir / "q4_model_flow.png")

    summary = {
        "model": "shrinking-radius normalized-coordinate variable-property radial finite volume",
        "criterion": "first time max_r C(r,t) <= 0.15 kg/kg",
        "parameters": {
            "initial_radius_m": INITIAL_RADIUS_M,
            "initial_temperature_K": INITIAL_TEMPERATURE_K,
            "initial_moisture_kgkg": INITIAL_MOISTURE,
            "target_moisture_kgkg": TARGET_MOISTURE,
            "heat_transfer_coefficient_W_m2K": HEAT_TRANSFER_COEFFICIENT,
            "mass_transfer_coefficient_m_s": MASS_TRANSFER_COEFFICIENT,
            "boundary_constant_after_s": BOUNDARY_CONSTANT_AFTER_S,
        },
        "numerics": {
            "intervals": args.intervals, "nodes": count,
            "initial_physical_step_cm": 2.0 / args.intervals,
            "rtol": args.rtol, "atol": args.atol, "max_step_s": 60.0,
            "nfev": int(solution.nfev), "njev": int(solution.njev), "nlu": int(solution.nlu),
            "output_rows": len(output_times),
        },
        "event": metrics,
        "event_checks": {
            "max_moisture_one_second_before": before_max,
            "max_moisture_at_event": float(np.max(event_state)),
            "all_nodes_at_or_below_threshold": bool(np.all(event_state <= TARGET_MOISTURE + 2e-8)),
        },
        "certification": {
            "certification_time_s": certification_time,
            "certification_time_h": certification_time / 3600.0,
            "strict_margin_kgkg": CERTIFICATION_MARGIN,
            "max_moisture_kgkg": certification_max,
            "all_nodes_strictly_below_target": bool(
                np.all(certification_state < TARGET_MOISTURE - CERTIFICATION_MARGIN)
            ),
        },
        "model_assumptions": {
            "coordinate": "x=r/R(t) is a uniformly shrinking material coordinate; the state is followed with the material, so no separate Eulerian coordinate-velocity term is added.",
            "surface_equilibrium": "air moisture is used as an effective surface-equilibrium moisture approximation because no sorption isotherm is supplied.",
            "post_4h_environment": "air temperature and moisture remain at the 4 h values.",
            "geometry": "one-dimensional radial cylindrical transfer; end effects are neglected.",
            "energy": "latent-heat coupling is not explicitly included.",
        },
        "boundary_after_4h": {
            "air_temperature_C": float(environment[1][-1] - 273.15),
            "air_moisture_kgkg": float(environment[2][-1]),
        },
        "radius_data": {
            "observations": len(radius_data[0]),
            "last_time_h": float(radius_data[0][-1] / 3600.0),
            "last_radius_cm": float(radius_data[1][-1] * 100.0),
        },
        "conservation": balance,
        "convergence": convergence.to_dict(orient="records"),
        "sensitivity": sensitivity.to_dict(orient="records"),
    }
    (output_dir / "q4_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"求解成功：{count}个归一化径向节点，{len(output_times)}个输出时刻。")
    print(f"临界时刻={event_time / 3600:.9f} h，半径={metrics['radius_at_event_cm']:.6f} cm")
    print(f"认证烘干时间={certification_time / 3600:.9f} h，最大水分={certification_max:.9f} kg/kg")
    print(
        f"中心/表面水分={metrics['center_moisture_kgkg']:.9f}/"
        f"{metrics['surface_moisture_kgkg']:.9f} kg/kg"
    )
    print(f"最大相对参考域水分守恒残差={balance['max_relative_reference_balance_residual']:.3e}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2)
