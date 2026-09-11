#!/usr/bin/env python3
"""A题问题2：变物性圆柱径向有限体积模型与BDF隐式求解。"""

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
HEAT_TRANSFER_COEFFICIENT = 25.0
MASS_TRANSFER_COEFFICIENT = 8.0e-7
END_TIME_S = 10800
OUTPUT_RADII_M = np.arange(0.0, RADIUS_M + 0.0005, 0.001)
TABLE_TIMES_S = np.arange(1800, END_TIME_S + 1, 1800, dtype=float)
TABLE_RADII_M = np.array([0.0, 0.005, 0.010, 0.015, 0.020])


@dataclass
class Grid:
    radii: np.ndarray
    volumes: np.ndarray
    east_areas: np.ndarray
    dr: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="求解A题问题2并生成完整结果。")
    parser.add_argument(
        "--environment", type=Path, default=Path("outputs/stage3/environment_si.csv"),
        help="阶段三环境边界CSV。",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/stage4_q2"),
        help="输出目录。",
    )
    parser.add_argument("--grid-step-cm", type=float, default=0.003125)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-11)
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
    if frame.time_s.iloc[0] > 0 or frame.time_s.iloc[-1] < END_TIME_S:
        raise ValueError("环境数据未覆盖0–10800 s。")
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


def make_rhs(
    grid: Grid, environment: tuple[np.ndarray, np.ndarray, np.ndarray],
    h_scale: float = 1.0, hm_scale: float = 1.0, d_scale: float = 1.0,
):
    env_time, env_temperature, env_moisture = environment
    count = len(grid.radii)

    def rhs(time_s: float, state: np.ndarray) -> np.ndarray:
        temperature = state[:count]
        moisture = state[count:]
        if np.any(moisture <= 0) or np.any(moisture <= -0.999999):
            raise RuntimeError("迭代中出现非物理含水率。")
        air_t = float(np.interp(time_s, env_time, env_temperature))
        air_c = float(np.interp(time_s, env_time, env_moisture))

        d_temperature = np.zeros(count)
        d_moisture = np.zeros(count)

        local_k = conductivity(moisture)
        face_k = harmonic_mean(local_k[:-1], local_k[1:])
        heat_flux = face_k * grid.east_areas[:-1] / grid.dr * np.diff(temperature)
        d_temperature[:-1] += heat_flux
        d_temperature[1:] -= heat_flux
        d_temperature[-1] += (
            h_scale * HEAT_TRANSFER_COEFFICIENT * grid.east_areas[-1]
            * (air_t - temperature[-1])
        )
        d_temperature /= density(moisture) * heat_capacity(moisture) * grid.volumes

        local_d = diffusivity(moisture, temperature, d_scale)
        face_d = harmonic_mean(local_d[:-1], local_d[1:])
        mass_flux = face_d * grid.east_areas[:-1] / grid.dr * np.diff(moisture)
        d_moisture[:-1] += mass_flux
        d_moisture[1:] -= mass_flux
        d_moisture[-1] += (
            hm_scale * MASS_TRANSFER_COEFFICIENT * grid.east_areas[-1]
            * (air_c - moisture[-1])
        )
        d_moisture /= grid.volumes
        return np.concatenate([d_temperature, d_moisture])

    return rhs


def solve_model(
    step_cm: float, environment: tuple[np.ndarray, np.ndarray, np.ndarray],
    rtol: float, atol: float, h_scale: float = 1.0, hm_scale: float = 1.0,
    d_scale: float = 1.0,
) -> tuple[Grid, Any]:
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
        (0.0, END_TIME_S), initial, method="BDF", dense_output=True,
        rtol=rtol, atol=atol, max_step=20.0, jac_sparsity=sparsity,
    )
    if not solution.success:
        raise RuntimeError(f"BDF求解失败：{solution.message}")
    if solution.sol is None:
        raise RuntimeError("求解器未返回连续插值解。")
    return grid, solution


def sample_solution(
    grid: Grid, solution: Any, times: np.ndarray, radii: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    result_t = np.empty((len(times), len(radii)))
    result_c = np.empty_like(result_t)
    count = len(grid.radii)
    chunk_size = 600
    for start in range(0, len(times), chunk_size):
        stop = min(start + chunk_size, len(times))
        state = solution.sol(times[start:stop])
        for local_index in range(stop - start):
            result_t[start + local_index] = np.interp(
                radii, grid.radii, state[:count, local_index]
            )
            result_c[start + local_index] = np.interp(
                radii, grid.radii, state[count:, local_index]
            )
    return result_t, result_c


def make_output_frame(times: np.ndarray, radii: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(values, columns=[f"{r * 100:.1f}" for r in radii])
    frame.insert(0, "时间/s", times.astype(int))
    return frame


def convergence_analysis(
    environment: tuple[np.ndarray, np.ndarray, np.ndarray], rtol: float, atol: float
) -> pd.DataFrame:
    steps = [0.025, 0.0125, 0.00625, 0.003125]
    fields: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    stats: dict[float, Any] = {}
    for step in steps:
        grid, solution = solve_model(step, environment, rtol, atol)
        fields[step] = sample_solution(grid, solution, TABLE_TIMES_S, TABLE_RADII_M)
        stats[step] = solution
    reference_t, reference_c = fields[0.003125]
    rows = []
    for step in steps:
        values_t, values_c = fields[step]
        rows.append({
            "grid_step_cm": step,
            "radial_nodes": int(round(2.0 / step)) + 1,
            "max_abs_temperature_difference_C": float(np.max(np.abs(values_t - reference_t))),
            "max_abs_moisture_difference_kgkg": float(np.max(np.abs(values_c - reference_c))),
            "nfev": int(stats[step].nfev),
            "njev": int(stats[step].njev),
            "nlu": int(stats[step].nlu),
        })
    return pd.DataFrame(rows)


def sensitivity_analysis(
    environment: tuple[np.ndarray, np.ndarray, np.ndarray], rtol: float, atol: float
) -> pd.DataFrame:
    scenarios = [
        ("基准", 1.0, 1.0, 1.0), ("h -10%", 0.9, 1.0, 1.0),
        ("h +10%", 1.1, 1.0, 1.0), ("hm -10%", 1.0, 0.9, 1.0),
        ("hm +10%", 1.0, 1.1, 1.0), ("D -10%", 1.0, 1.0, 0.9),
        ("D +10%", 1.0, 1.0, 1.1),
    ]
    rows = []
    for name, h_scale, hm_scale, d_scale in scenarios:
        grid, solution = solve_model(
            0.0125, environment, rtol, atol, h_scale, hm_scale, d_scale
        )
        state = solution.sol(np.array([END_TIME_S]))[:, 0]
        count = len(grid.radii)
        rows.append({
            "scenario": name,
            "h_scale": h_scale, "hm_scale": hm_scale, "D_scale": d_scale,
            "center_temperature_C_3h": float(state[0] - 273.15),
            "surface_temperature_C_3h": float(state[count - 1] - 273.15),
            "center_moisture_3h": float(state[count]),
            "surface_moisture_3h": float(state[-1]),
            "volume_weighted_moisture_3h": float(
                np.dot(state[count:], grid.volumes) / grid.volumes.sum()
            ),
        })
    return pd.DataFrame(rows)


def conservation_report(
    grid: Grid, solution: Any, environment: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> dict[str, float]:
    count = len(grid.radii)
    state = solution.y
    total = state[count:, :].T @ grid.volumes
    air_c = np.interp(solution.t, environment[0], environment[2])
    boundary_rate = (
        MASS_TRANSFER_COEFFICIENT * grid.east_areas[-1]
        * (air_c - state[-1, :])
    )
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


def plot_profiles(times: np.ndarray, temperature_c: np.ndarray, moisture: np.ndarray, path: Path) -> None:
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    radii_cm = OUTPUT_RADII_M * 100
    selected = np.arange(0, END_TIME_S + 1, 1800)
    for time_s in selected:
        index = int(time_s)
        axes[0].plot(radii_cm, temperature_c[index], label=f"{time_s / 3600:.1f} h")
        axes[1].plot(radii_cm, moisture[index], label=f"{time_s / 3600:.1f} h")
    axes[0].set_title("3 h内药材径向温度分布")
    axes[0].set_ylabel("温度（°C）")
    axes[1].set_title("3 h内药材径向水分浓度分布")
    axes[1].set_ylabel("水分浓度（kg/kg）")
    for axis in axes:
        axis.set_xlabel("到中心距离（cm）")
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_flow(path: Path) -> None:
    configure_plotting()
    fig, axis = plt.subplots(figsize=(12, 2.5))
    axis.set_xlim(0, 12); axis.set_ylim(0, 2.5); axis.axis("off")
    labels = [
        "附件1\n边界插值", "由C更新\nρ、cp、k", "求解径向\n温度场",
        "由C、T更新D\n求解水分场", "BDF隐式迭代\n至3 h", "收敛/守恒\n与结果导出",
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
    environment = load_environment(args.environment.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    grid, solution = solve_model(args.grid_step_cm, environment, args.rtol, args.atol)
    output_times = np.arange(0, END_TIME_S + 1, dtype=float)
    temperature_k, moisture = sample_solution(grid, solution, output_times, OUTPUT_RADII_M)
    temperature_c = temperature_k - 273.15

    if not np.isfinite(temperature_c).all() or not np.isfinite(moisture).all():
        raise RuntimeError("结果存在NaN或无穷值。")
    if temperature_c.min() < 27.999 or temperature_c.max() > environment[1].max() - 273.15 + 1e-6:
        raise RuntimeError("温度结果违反物理包络。")
    if moisture.min() < environment[2].min() - 1e-6 or moisture.max() > INITIAL_MOISTURE + 1e-7:
        raise RuntimeError("水分结果违反物理包络。")

    temperature_frame = make_output_frame(output_times, OUTPUT_RADII_M, temperature_c)
    moisture_frame = make_output_frame(output_times, OUTPUT_RADII_M, moisture)
    temperature_frame.to_csv(output_dir / "q2_temperature_full.csv", index=False, encoding="utf-8-sig")
    moisture_frame.to_csv(output_dir / "q2_moisture_full.csv", index=False, encoding="utf-8-sig")

    table_rows = TABLE_TIMES_S.astype(int)
    table_cols = (TABLE_RADII_M / 0.001).round().astype(int)
    paper_t = pd.DataFrame(
        temperature_c[table_rows][:, table_cols], index=TABLE_TIMES_S / 3600,
        columns=["0", "0.5", "1", "1.5", "2"],
    )
    paper_c = pd.DataFrame(
        moisture[table_rows][:, table_cols], index=TABLE_TIMES_S / 3600,
        columns=["0", "0.5", "1", "1.5", "2"],
    )
    paper_t.index.name = "时间/h"; paper_c.index.name = "时间/h"
    paper_t.to_csv(output_dir / "q2_table_temperature.csv", encoding="utf-8-sig", float_format="%.4f")
    paper_c.to_csv(output_dir / "q2_table_moisture.csv", encoding="utf-8-sig", float_format="%.4f")

    convergence = convergence_analysis(environment, args.rtol, args.atol)
    sensitivity = sensitivity_analysis(environment, args.rtol, args.atol)
    balance = conservation_report(grid, solution, environment)
    convergence.to_csv(output_dir / "q2_convergence.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(output_dir / "q2_sensitivity.csv", index=False, encoding="utf-8-sig")
    plot_profiles(output_times, temperature_c, moisture, output_dir / "q2_radial_profiles.png")
    plot_flow(output_dir / "q2_model_flow.png")

    final_state = solution.sol(np.array([END_TIME_S]))[:, 0]
    count = len(grid.radii)
    summary = {
        "model": "variable-property radial finite volume with coefficient coupling",
        "parameters": {
            "radius_m": RADIUS_M,
            "initial_temperature_K": INITIAL_TEMPERATURE_K,
            "initial_moisture_kgkg": INITIAL_MOISTURE,
            "heat_transfer_coefficient_W_m2K": HEAT_TRANSFER_COEFFICIENT,
            "mass_transfer_coefficient_m_s": MASS_TRANSFER_COEFFICIENT,
            "rho_formula": "650+128*C kg/m3",
            "cp_formula": "1450+2736*C/(C+1) J/(kg K)",
            "k_formula": "0.21+0.38*C/(C+1) W/(m K)",
            "D_formula": "2.4e-3*exp(-0.45/C)*exp(-3850/T) m2/s",
        },
        "numerics": {
            "grid_step_cm": args.grid_step_cm, "radial_nodes": count,
            "rtol": args.rtol, "atol": args.atol, "max_step_s": 20.0,
            "nfev": int(solution.nfev), "njev": int(solution.njev), "nlu": int(solution.nlu),
        },
        "results_3h": {
            "center_temperature_C": float(final_state[0] - 273.15),
            "surface_temperature_C": float(final_state[count - 1] - 273.15),
            "center_moisture_kgkg": float(final_state[count]),
            "surface_moisture_kgkg": float(final_state[-1]),
            "volume_weighted_temperature_C": float(
                np.dot(final_state[:count] - 273.15, grid.volumes) / grid.volumes.sum()
            ),
            "volume_weighted_moisture_kgkg": float(
                np.dot(final_state[count:], grid.volumes) / grid.volumes.sum()
            ),
            "air_temperature_C": float(np.interp(END_TIME_S, environment[0], environment[1]) - 273.15),
            "air_moisture_kgkg": float(np.interp(END_TIME_S, environment[0], environment[2])),
        },
        "conservation": balance,
        "convergence": convergence.to_dict(orient="records"),
        "sensitivity": sensitivity.to_dict(orient="records"),
    }
    (output_dir / "q2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )

    required = [
        "q2_temperature_full.csv", "q2_moisture_full.csv", "q2_table_temperature.csv",
        "q2_table_moisture.csv", "q2_convergence.csv", "q2_sensitivity.csv",
        "q2_summary.json", "q2_radial_profiles.png", "q2_model_flow.png",
    ]
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"输出完整性校验失败：{missing}")
    print(f"求解成功：{count}个径向节点，10801个输出时刻。")
    print(f"BDF统计：nfev={solution.nfev}, njev={solution.njev}, nlu={solution.nlu}")
    print(
        f"3 h中心/表面温度={summary['results_3h']['center_temperature_C']:.6f}/"
        f"{summary['results_3h']['surface_temperature_C']:.6f} °C；"
        f"中心/表面水分={summary['results_3h']['center_moisture_kgkg']:.6f}/"
        f"{summary['results_3h']['surface_moisture_kgkg']:.6f} kg/kg"
    )
    print(f"最大相对水分守恒残差={balance['max_relative_balance_residual']:.3e}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2)
