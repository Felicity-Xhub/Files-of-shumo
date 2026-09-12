#!/usr/bin/env python3
"""A题问题3：固定半径变物性径向热湿模型与全域阈值事件定位。"""

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


RADIUS_M = 0.02
INITIAL_TEMPERATURE_K = 301.15
INITIAL_MOISTURE = 2.55
TARGET_MOISTURE = 0.15
HEAT_TRANSFER_COEFFICIENT = 25.0
MASS_TRANSFER_COEFFICIENT = 8.0e-7
BOUNDARY_CONSTANT_AFTER_S = 4.0 * 3600.0
DEFAULT_MAX_TIME_S = 10.0 * 24.0 * 3600.0
OUTPUT_RADII_M = np.arange(0.0, RADIUS_M + 0.0005, 0.001)
TABLE_RADII_M = np.array([0.0, 0.005, 0.010, 0.015, 0.020])
PROVISIONAL_REPORT_TIME_S = 206100.0
REPORT_ROUNDING_LIMIT = 0.14995


@dataclass
class Grid:
    radii: np.ndarray
    volumes: np.ndarray
    east_areas: np.ndarray
    dr: float


@dataclass
class PiecewiseSolution:
    """由若干真实积分区间组成的连续解；禁止区间外外推。"""

    segments: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("分段解至少需要一个积分区间。")
        for segment in self.segments:
            if not segment.success or segment.sol is None:
                raise RuntimeError("不能拼接失败或缺少连续插值的积分结果。")
        self.t_min = float(self.segments[0].t[0])
        self.t_max = float(self.segments[-1].t[-1])
        self.success = True
        self.nfev = int(sum(segment.nfev for segment in self.segments))
        self.njev = int(sum(segment.njev for segment in self.segments))
        self.nlu = int(sum(segment.nlu for segment in self.segments))

    def sol(self, times: np.ndarray | float) -> np.ndarray:
        sample_times = np.atleast_1d(np.asarray(times, dtype=float))
        if not np.isfinite(sample_times).all():
            raise ValueError("采样时间存在非有限值。")
        tolerance = 1e-8 * max(1.0, abs(self.t_max))
        if sample_times.min() < self.t_min - tolerance or sample_times.max() > self.t_max + tolerance:
            raise ValueError(
                f"采样时间必须位于真实积分区间[{self.t_min:.9f}, {self.t_max:.9f}] s内。"
            )
        state_count = self.segments[0].y.shape[0]
        values = np.empty((state_count, sample_times.size), dtype=float)
        assigned = np.zeros(sample_times.size, dtype=bool)
        for index, segment in enumerate(self.segments):
            left = float(segment.t[0])
            right = float(segment.t[-1])
            if index == 0:
                mask = (sample_times >= left - tolerance) & (sample_times <= right + tolerance)
            else:
                mask = (~assigned) & (sample_times > left - tolerance) & (sample_times <= right + tolerance)
            if np.any(mask):
                values[:, mask] = segment.sol(np.clip(sample_times[mask], left, right))
                assigned[mask] = True
        if not assigned.all() or not np.isfinite(values).all():
            raise RuntimeError("分段连续解采样失败或产生非有限值。")
        return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="求解A题问题3并定位全域含水率阈值事件。")
    parser.add_argument(
        "--environment", type=Path, default=Path("outputs/stage3/environment_si.csv"),
        help="阶段三环境边界CSV；4 h后自动保持末值。",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/stage4_q3"), help="输出目录。"
    )
    parser.add_argument("--grid-step-cm", type=float, default=0.00078125)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-11)
    parser.add_argument("--max-time-days", type=float, default=10.0)
    parser.add_argument(
        "--skip-sensitivity", action="store_true",
        help="跳过参数敏感性分析，仅用于缩短一次正式网格复算。",
    )
    parser.add_argument(
        "--skip-diagnostics", action="store_true",
        help="跳过收敛性和敏感性情景的重复计算，仅用于导出已复核模型的结果文件。",
    )
    return parser.parse_args()


def build_grid(step_cm: float) -> Grid:
    if step_cm <= 0:
        raise ValueError("网格步长必须为正。")
    dr = step_cm / 100.0
    intervals = int(round(RADIUS_M / dr))
    if not np.isclose(intervals * dr, RADIUS_M, atol=1e-12, rtol=0):
        raise ValueError("网格步长必须整除2 cm。")
    radii = np.linspace(0.0, RADIUS_M, intervals + 1)
    west_faces = np.maximum(0.0, radii - dr / 2.0)
    east_faces = np.minimum(RADIUS_M, radii + dr / 2.0)
    volumes = np.pi * (east_faces**2 - west_faces**2)
    east_areas = 2.0 * np.pi * east_faces
    return Grid(radii, volumes, east_areas, dr)


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


def density(c: np.ndarray) -> np.ndarray:
    return 650.0 + 128.0 * c


def heat_capacity(c: np.ndarray) -> np.ndarray:
    return 1450.0 + 2736.0 * c / (c + 1.0)


def conductivity(c: np.ndarray) -> np.ndarray:
    return 0.21 + 0.38 * c / (c + 1.0)


def diffusivity(c: np.ndarray, temperature_k: np.ndarray, scale: float = 1.0) -> np.ndarray:
    safe_c = np.maximum(c, 1e-12)
    safe_t = np.maximum(temperature_k, 1.0)
    return scale * 2.4e-3 * np.exp(-0.45 / safe_c) * np.exp(-3850.0 / safe_t)


def harmonic_mean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return 2.0 * left * right / np.maximum(left + right, 1e-30)


def environment_at(
    time_s: float | np.ndarray, environment: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> tuple[float | np.ndarray, float | np.ndarray]:
    times, temperatures, moistures = environment
    # np.interp在区间右侧自动保持最后一个值，落实用户确认的4 h后恒定规则。
    return np.interp(time_s, times, temperatures), np.interp(time_s, times, moistures)


def make_rhs(
    grid: Grid,
    environment: tuple[np.ndarray, np.ndarray, np.ndarray],
    h_scale: float = 1.0,
    hm_scale: float = 1.0,
    d_scale: float = 1.0,
):
    count = len(grid.radii)

    def rhs(time_s: float, state: np.ndarray) -> np.ndarray:
        temperature = state[:count]
        moisture = state[count:]
        if np.any(moisture <= 0.0):
            raise RuntimeError("迭代中出现非物理含水率。")
        air_t, air_c = environment_at(time_s, environment)

        d_temperature = np.zeros(count)
        local_k = conductivity(moisture)
        face_k = harmonic_mean(local_k[:-1], local_k[1:])
        heat_flux = face_k * grid.east_areas[:-1] / grid.dr * np.diff(temperature)
        d_temperature[:-1] += heat_flux
        d_temperature[1:] -= heat_flux
        d_temperature[-1] += (
            h_scale * HEAT_TRANSFER_COEFFICIENT * grid.east_areas[-1]
            * (float(air_t) - temperature[-1])
        )
        d_temperature /= density(moisture) * heat_capacity(moisture) * grid.volumes

        d_moisture = np.zeros(count)
        local_d = diffusivity(moisture, temperature, d_scale)
        face_d = harmonic_mean(local_d[:-1], local_d[1:])
        mass_flux = face_d * grid.east_areas[:-1] / grid.dr * np.diff(moisture)
        d_moisture[:-1] += mass_flux
        d_moisture[1:] -= mass_flux
        d_moisture[-1] += (
            hm_scale * MASS_TRANSFER_COEFFICIENT * grid.east_areas[-1]
            * (float(air_c) - moisture[-1])
        )
        d_moisture /= grid.volumes
        return np.concatenate([d_temperature, d_moisture])

    return rhs


def make_threshold_event(count: int):
    def threshold_event(_time_s: float, state: np.ndarray) -> float:
        return float(np.max(state[count:]) - TARGET_MOISTURE)

    threshold_event.terminal = True
    threshold_event.direction = -1.0
    return threshold_event


def solve_to_threshold(
    step_cm: float,
    environment: tuple[np.ndarray, np.ndarray, np.ndarray],
    rtol: float,
    atol: float,
    max_time_s: float = DEFAULT_MAX_TIME_S,
    h_scale: float = 1.0,
    hm_scale: float = 1.0,
    d_scale: float = 1.0,
) -> tuple[Grid, Any, float, np.ndarray]:
    grid = build_grid(step_cm)
    count = len(grid.radii)
    initial = np.concatenate([
        np.full(count, INITIAL_TEMPERATURE_K),
        np.full(count, INITIAL_MOISTURE),
    ])
    tri = diags(
        [np.ones(count - 1), np.ones(count), np.ones(count - 1)],
        [-1, 0, 1], shape=(count, count), format="csc",
    )
    sparsity = bmat([[tri, tri], [tri, tri]], format="csc")
    solution = solve_ivp(
        make_rhs(grid, environment, h_scale, hm_scale, d_scale),
        (0.0, max_time_s), initial, method="BDF", dense_output=True,
        events=make_threshold_event(count), rtol=rtol, atol=atol,
        max_step=60.0, jac_sparsity=sparsity,
    )
    if not solution.success:
        raise RuntimeError(f"BDF求解失败：{solution.message}")
    if solution.sol is None:
        raise RuntimeError("求解器未返回连续插值解。")
    if len(solution.t_events) != 1 or len(solution.t_events[0]) != 1:
        raise RuntimeError(f"在{max_time_s / 86400:.2f}天内未定位到烘干阈值。")
    event_time = float(solution.t_events[0][0])
    if len(solution.y_events) != 1 or len(solution.y_events[0]) != 1:
        raise RuntimeError("阈值事件已触发，但未返回对应事件状态。")
    event_state = np.asarray(solution.y_events[0][0], dtype=float)
    if not np.isfinite(event_state).all():
        raise RuntimeError("阈值事件状态存在非有限值。")
    return grid, solution, event_time, event_state


def continue_from_event(
    grid: Grid,
    environment: tuple[np.ndarray, np.ndarray, np.ndarray],
    event_time_s: float,
    event_state: np.ndarray,
    report_time_s: float,
    rtol: float,
    atol: float,
    h_scale: float = 1.0,
    hm_scale: float = 1.0,
    d_scale: float = 1.0,
) -> Any:
    """关闭终止事件，从事件状态真实积分到指定报告时间。"""
    if report_time_s <= event_time_s:
        raise ValueError("报告时间必须晚于数值临界事件。")
    count = len(grid.radii)
    tri = diags(
        [np.ones(count - 1), np.ones(count), np.ones(count - 1)],
        [-1, 0, 1], shape=(count, count), format="csc",
    )
    sparsity = bmat([[tri, tri], [tri, tri]], format="csc")
    solution = solve_ivp(
        make_rhs(grid, environment, h_scale, hm_scale, d_scale),
        (event_time_s, report_time_s), event_state, method="BDF", dense_output=True,
        events=None, rtol=rtol, atol=atol, max_step=60.0, jac_sparsity=sparsity,
    )
    if not solution.success:
        raise RuntimeError(f"事件后续积分失败：{solution.message}")
    if solution.sol is None or float(solution.t[-1]) < report_time_s - 1e-8:
        raise RuntimeError("事件后续积分未覆盖报告时间。")
    if not np.isfinite(solution.y).all():
        raise RuntimeError("事件后续积分产生非有限值。")
    return solution


def sample_solution(
    grid: Grid, solution: Any, times: np.ndarray, radii: np.ndarray,
    radius_at_time: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not getattr(solution, "success", False) or getattr(solution, "sol", None) is None:
        raise RuntimeError("不能从失败或缺少连续插值的求解结果生成报表。")
    times = np.asarray(times, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if times.ndim != 1 or radii.ndim != 1 or len(times) == 0 or len(radii) == 0:
        raise ValueError("采样时间和位置必须是一维非空数组。")
    if not np.isfinite(times).all() or not np.isfinite(radii).all() or np.any(radii < 0.0):
        raise ValueError("采样时间或位置存在非有限值/负半径。")
    lower = float(solution.t_min) if hasattr(solution, "t_min") else float(solution.t[0])
    upper = float(solution.t_max) if hasattr(solution, "t_max") else float(solution.t[-1])
    tolerance = 1e-8 * max(1.0, abs(upper))
    if times.min() < lower - tolerance or times.max() > upper + tolerance:
        raise ValueError(f"采样时间必须位于真实积分区间[{lower:.9f}, {upper:.9f}] s内。")
    result_t = np.full((len(times), len(radii)), np.nan)
    result_c = np.full_like(result_t, np.nan)
    count = len(grid.radii)
    for start in range(0, len(times), 600):
        stop = min(start + 600, len(times))
        state = solution.sol(times[start:stop])
        if state.shape != (2 * count, stop - start) or not np.isfinite(state).all():
            raise RuntimeError("连续解返回维度错误或非有限值。")
        for local_index in range(stop - start):
            time_index = start + local_index
            domain_radius = (
                float(radius_at_time(times[time_index]))
                if radius_at_time is not None else float(grid.radii[-1])
            )
            if not np.isfinite(domain_radius) or domain_radius <= 0.0:
                raise RuntimeError("移动域半径无效。")
            valid = radii <= min(domain_radius, float(grid.radii[-1])) + 1e-12
            if np.any(valid):
                result_t[time_index, valid] = np.interp(
                    radii[valid], grid.radii, state[:count, local_index]
                )
                result_c[time_index, valid] = np.interp(
                    radii[valid], grid.radii, state[count:, local_index]
                )
    return result_t, result_c


def make_output_times(report_time_s: float) -> np.ndarray:
    if report_time_s < 60.0 or not np.isclose(report_time_s / 60.0, round(report_time_s / 60.0)):
        raise ValueError("报告时间必须是至少60 s的整分钟输出点。")
    return np.arange(60.0, report_time_s + 0.1, 60.0)


def make_output_frame(times: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(values, columns=[f"{r * 100:.1f}" for r in OUTPUT_RADII_M])
    frame.insert(0, "时间/s", times)
    return frame


def event_metrics(grid: Grid, solution: Any, event_time_s: float) -> dict[str, float | int]:
    state = solution.sol(np.array([event_time_s]))[:, 0]
    count = len(grid.radii)
    moisture = state[count:]
    max_index = int(np.argmax(moisture))
    return {
        "event_time_s": event_time_s,
        "event_time_h": event_time_s / 3600.0,
        "event_time_days": event_time_s / 86400.0,
        "critical_radius_cm": float(grid.radii[max_index] * 100.0),
        "max_moisture_kgkg": float(moisture[max_index]),
        "center_moisture_kgkg": float(moisture[0]),
        "surface_moisture_kgkg": float(moisture[-1]),
        "volume_weighted_moisture_kgkg": float(np.dot(moisture, grid.volumes) / grid.volumes.sum()),
        "center_temperature_C": float(state[0] - 273.15),
        "surface_temperature_C": float(state[count - 1] - 273.15),
    }


def convergence_analysis(
    environment: tuple[np.ndarray, np.ndarray, np.ndarray], rtol: float, atol: float,
    max_time_s: float,
) -> pd.DataFrame:
    steps = [0.00625, 0.003125, 0.0015625, 0.00078125]
    rows = []
    reference_time = None
    records = []
    for step in steps:
        grid, solution, event_time, _event_state = solve_to_threshold(
            step, environment, rtol, atol, max_time_s=max_time_s
        )
        metrics = event_metrics(grid, solution, event_time)
        records.append((step, grid, solution, metrics))
        if step == steps[-1]:
            reference_time = event_time
    assert reference_time is not None
    for step, grid, solution, metrics in records:
        rows.append({
            "grid_step_cm": step,
            "radial_nodes": len(grid.radii),
            "event_time_h": metrics["event_time_h"],
            "event_time_difference_s_vs_finest": float(metrics["event_time_s"]) - reference_time,
            "critical_radius_cm": metrics["critical_radius_cm"],
            "nfev": int(solution.nfev), "njev": int(solution.njev), "nlu": int(solution.nlu),
        })
    return pd.DataFrame(rows)


def sensitivity_analysis(
    environment: tuple[np.ndarray, np.ndarray, np.ndarray], rtol: float, atol: float,
    max_time_s: float,
) -> pd.DataFrame:
    scenarios = [
        ("基准", 1.0, 1.0, 1.0), ("h -10%", 0.9, 1.0, 1.0),
        ("h +10%", 1.1, 1.0, 1.0), ("hm -10%", 1.0, 0.9, 1.0),
        ("hm +10%", 1.0, 1.1, 1.0), ("D -10%", 1.0, 1.0, 0.9),
        ("D +10%", 1.0, 1.0, 1.1),
    ]
    rows = []
    for name, h_scale, hm_scale, d_scale in scenarios:
        grid, solution, event_time, _event_state = solve_to_threshold(
            0.003125, environment, rtol, atol, max_time_s=max_time_s,
            h_scale=h_scale, hm_scale=hm_scale, d_scale=d_scale,
        )
        metrics = event_metrics(grid, solution, event_time)
        rows.append({
            "scenario": name, "h_scale": h_scale, "hm_scale": hm_scale, "D_scale": d_scale,
            "event_time_h": metrics["event_time_h"],
            "critical_radius_cm": metrics["critical_radius_cm"],
            "surface_moisture_at_event": metrics["surface_moisture_kgkg"],
        })
    return pd.DataFrame(rows)


def conservation_report(
    grid: Grid, solution: Any, environment: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> dict[str, float]:
    count = len(grid.radii)
    state = solution.y
    total = state[count:, :].T @ grid.volumes
    _, air_c = environment_at(solution.t, environment)
    boundary_rate = MASS_TRANSFER_COEFFICIENT * grid.east_areas[-1] * (air_c - state[-1, :])
    cumulative = np.zeros(len(solution.t))
    cumulative[1:] = np.cumsum(
        0.5 * (boundary_rate[1:] + boundary_rate[:-1]) * np.diff(solution.t)
    )
    residual = total - total[0] - cumulative
    scale = max(abs(total[-1] - total[0]), 1e-15)
    return {
        "max_absolute_balance_residual": float(np.max(np.abs(residual))),
        "max_relative_balance_residual": float(np.max(np.abs(residual)) / scale),
    }


def configure_plotting() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_profiles(
    grid: Grid, solution: Any, event_time_s: float, path: Path
) -> None:
    configure_plotting()
    regular = np.arange(0.0, np.floor(event_time_s / 43200.0) * 43200.0 + 1.0, 43200.0)
    selected = np.unique(np.append(regular, event_time_s))
    temperature, moisture = sample_solution(grid, solution, selected, OUTPUT_RADII_M)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    radii_cm = OUTPUT_RADII_M * 100.0
    for index, time_s in enumerate(selected):
        label = f"{time_s / 3600:.1f} h"
        axes[0].plot(radii_cm, temperature[index] - 273.15, label=label)
        axes[1].plot(radii_cm, moisture[index], label=label)
    axes[0].set_title("药材径向温度分布")
    axes[0].set_ylabel("温度（°C）")
    axes[1].set_title("药材径向水分浓度分布")
    axes[1].set_ylabel("水分浓度（kg/kg）")
    axes[1].axhline(TARGET_MOISTURE, color="#C00000", linestyle="--", linewidth=1.2, label="阈值0.15")
    for axis in axes:
        axis.set_xlabel("到中心距离（cm）")
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_max_moisture(solution: Any, count: int, event_time_s: float, path: Path) -> None:
    configure_plotting()
    times = np.linspace(0.0, event_time_s, 1200)
    moisture = solution.sol(times)[count:, :]
    maximum = np.max(moisture, axis=0)
    fig, axis = plt.subplots(figsize=(8.5, 4.6))
    axis.plot(times / 3600.0, maximum, color="#1F4E78", linewidth=2.0)
    axis.axhline(TARGET_MOISTURE, color="#C00000", linestyle="--", label="烘干阈值0.15 kg/kg")
    axis.axvline(event_time_s / 3600.0, color="#548235", linestyle=":", label=f"结束{event_time_s / 3600:.4f} h")
    axis.set_xlabel("时间（h）")
    axis.set_ylabel("药材内最大水分浓度（kg/kg）")
    axis.set_title("全域阈值事件定位")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_flow(path: Path) -> None:
    configure_plotting()
    fig, axis = plt.subplots(figsize=(12, 2.5))
    axis.set_xlim(0, 12); axis.set_ylim(0, 2.5); axis.axis("off")
    labels = [
        "读取附件1\n4 h后保持末值", "建立固定半径\n变物性PDE", "有限体积离散\nBDF隐式推进",
        "计算全域\n最大含水率", "事件根定位\nmax C=0.15", "收敛/敏感性\n结果导出",
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
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.max_time_days <= 0:
        raise ValueError("最大搜索天数必须为正。")
    environment = load_environment(args.environment.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    max_time_s = args.max_time_days * 86400.0

    grid, event_solution, event_time, event_state = solve_to_threshold(
        args.grid_step_cm, environment, args.rtol, args.atol, max_time_s=max_time_s
    )
    metrics = event_metrics(grid, event_solution, event_time)
    count = len(grid.radii)

    report_time = max(
        PROVISIONAL_REPORT_TIME_S,
        np.ceil((event_time + 1e-9) / 60.0) * 60.0,
    )
    continuation_segments: list[Any] = []
    continuation_start = event_time
    continuation_state = event_state.copy()
    while True:
        continuation = continue_from_event(
            grid, environment, continuation_start, continuation_state,
            report_time, args.rtol, args.atol,
        )
        continuation_segments.append(continuation)
        report_state = continuation.sol(np.array([report_time]))[:, 0]
        if not np.isfinite(report_state).all():
            raise RuntimeError("报告时刻状态存在非有限值。")
        report_max_moisture = float(np.max(report_state[count:]))
        if report_max_moisture < REPORT_ROUNDING_LIMIT:
            break
        continuation_start = report_time
        continuation_state = report_state
        report_time += 60.0
        if report_time > max_time_s:
            raise RuntimeError("在最大搜索时长内未找到四位小数明确低于0.1500的报告点。")

    report_solution = PiecewiseSolution((event_solution, *continuation_segments))
    output_times = make_output_times(report_time)
    _, output_moisture = sample_solution(grid, report_solution, output_times, OUTPUT_RADII_M)
    if not np.isfinite(output_moisture).all():
        raise RuntimeError("结果存在NaN或无穷值。")
    if output_moisture.min() < environment[2].min() - 1e-6 or output_moisture.max() > INITIAL_MOISTURE + 1e-7:
        raise RuntimeError("水分结果违反物理包络。")

    event_moisture = event_state[count:]
    before_time = max(0.0, event_time - 1.0)
    before_max = float(np.max(event_solution.sol(np.array([before_time]))[count:, 0]))
    if np.max(event_moisture) > TARGET_MOISTURE + 2e-8:
        raise RuntimeError("事件时刻仍有位置超过目标含水率。")
    if before_max <= TARGET_MOISTURE:
        raise RuntimeError("阈值事件未满足首次穿越校验。")

    full_frame = make_output_frame(output_times, output_moisture)
    full_frame.to_csv(output_dir / "q3_moisture_full.csv", index=False, encoding="utf-8-sig")

    regular_table_times = np.arange(6.0 * 3600.0, report_time - 1e-8, 6.0 * 3600.0)
    table_times = np.append(regular_table_times, report_time)
    _, table_moisture = sample_solution(grid, report_solution, table_times, TABLE_RADII_M)
    paper = pd.DataFrame(
        table_moisture,
        index=[f"{t / 3600:.0f}" for t in regular_table_times]
        + [f"保守报告时间（{report_time / 3600:.2f} h）"],
        columns=["0", "0.5", "1", "1.5", "2"],
    )
    paper.index.name = "时间/h"
    paper.to_csv(output_dir / "q3_table_moisture.csv", encoding="utf-8-sig", float_format="%.4f")

    convergence = (
        pd.DataFrame()
        if args.skip_diagnostics
        else convergence_analysis(environment, args.rtol, args.atol, max_time_s)
    )
    sensitivity = (
        pd.DataFrame()
        if args.skip_diagnostics or args.skip_sensitivity
        else sensitivity_analysis(environment, args.rtol, args.atol, max_time_s)
    )
    balance = conservation_report(grid, event_solution, environment)
    convergence.to_csv(output_dir / "q3_convergence.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(output_dir / "q3_sensitivity.csv", index=False, encoding="utf-8-sig")

    plot_profiles(grid, event_solution, event_time, output_dir / "q3_radial_profiles.png")
    plot_max_moisture(event_solution, count, event_time, output_dir / "q3_threshold_event.png")
    plot_flow(output_dir / "q3_model_flow.png")

    summary = {
        "model": "fixed-radius variable-property radial finite volume with global threshold event",
        "criterion": "first time max_r C(r,t) <= 0.15 kg/kg",
        "parameters": {
            "radius_m": RADIUS_M,
            "initial_temperature_K": INITIAL_TEMPERATURE_K,
            "initial_moisture_kgkg": INITIAL_MOISTURE,
            "target_moisture_kgkg": TARGET_MOISTURE,
            "heat_transfer_coefficient_W_m2K": HEAT_TRANSFER_COEFFICIENT,
            "mass_transfer_coefficient_m_s": MASS_TRANSFER_COEFFICIENT,
            "boundary_constant_after_s": BOUNDARY_CONSTANT_AFTER_S,
        },
        "numerics": {
            "grid_step_cm": args.grid_step_cm, "radial_nodes": count,
            "rtol": args.rtol, "atol": args.atol, "max_step_s": 60.0,
            "event_nfev": int(event_solution.nfev),
            "continuation_nfev": int(sum(s.nfev for s in continuation_segments)),
            "total_nfev": int(report_solution.nfev),
            "output_rows": len(output_times),
        },
        "theoretical_event": metrics,
        "conservative_report": {
            "report_time_s": report_time,
            "report_time_h": report_time / 3600.0,
            "max_full_grid_moisture_kgkg": report_max_moisture,
            "strictly_below_0_15": bool(report_max_moisture < TARGET_MOISTURE),
            "rounds_below_0_1500": bool(report_max_moisture < REPORT_ROUNDING_LIMIT),
            "provisional_time_s": PROVISIONAL_REPORT_TIME_S,
        },
        "event_checks": {
            "max_moisture_one_second_before": before_max,
            "max_moisture_at_event": float(np.max(event_moisture)),
            "all_nodes_at_or_below_threshold": bool(np.all(event_moisture <= TARGET_MOISTURE + 2e-8)),
        },
        "boundary_after_4h": {
            "air_temperature_C": float(environment[1][-1] - 273.15),
            "air_moisture_kgkg": float(environment[2][-1]),
        },
        "conservation": balance,
        "convergence": convergence.to_dict(orient="records"),
        "sensitivity": sensitivity.to_dict(orient="records"),
    }
    (output_dir / "q3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )

    required = [
        "q3_moisture_full.csv", "q3_table_moisture.csv", "q3_convergence.csv",
        "q3_sensitivity.csv", "q3_summary.json", "q3_radial_profiles.png",
        "q3_threshold_event.png", "q3_model_flow.png",
    ]
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"输出完整性校验失败：{missing}")
    print(f"求解成功：{count}个径向节点，{len(output_times)}个输出时刻。")
    print(
        f"BDF统计：事件段nfev={event_solution.nfev}，"
        f"续积分段nfev={sum(s.nfev for s in continuation_segments)}"
    )
    print(f"数值临界时间={event_time / 3600:.9f} h ({event_time:.6f} s)")
    print(
        f"保守报告时间={report_time / 3600:.9f} h ({report_time:.0f} s)，"
        f"全节点最大未舍入含水率={report_max_moisture:.12f}"
    )
    print(
        f"关键位置={metrics['critical_radius_cm']:.6f} cm；"
        f"中心/表面水分={metrics['center_moisture_kgkg']:.9f}/"
        f"{metrics['surface_moisture_kgkg']:.9f} kg/kg"
    )
    print(f"最大相对水分守恒残差={balance['max_relative_balance_residual']:.3e}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2)
