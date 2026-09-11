#!/usr/bin/env python3
"""A题问题1：圆柱径向有限体积 + BDF自适应隐式积分。

输入：阶段三生成的 environment_si.csv，或通过 --environment 指定同结构文件。
输出：完整1 s × 0.1 cm结果、论文表格、收敛报告、守恒报告和图形。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.integrate import solve_ivp
from scipy.sparse import block_diag, diags


RADIUS_M = 0.02
INITIAL_TEMPERATURE_K = 301.15
INITIAL_MOISTURE = 2.55
DENSITY = 820.0
HEAT_CAPACITY = 2600.0
THERMAL_CONDUCTIVITY = 0.36
HEAT_TRANSFER_COEFFICIENT = 25.0
MASS_TRANSFER_COEFFICIENT = 8.0e-7
END_TIME_S = 1800
OUTPUT_RADII_M = np.arange(0.0, RADIUS_M + 0.0005, 0.001)
TABLE_TIMES_S = np.array([100, 300, 600, 900, 1200, 1500, 1800], dtype=float)
TABLE_RADII_M = np.array([0.0, 0.005, 0.010, 0.015, 0.020])


@dataclass
class Grid:
    radii: np.ndarray
    west_faces: np.ndarray
    east_faces: np.ndarray
    volumes: np.ndarray
    west_areas: np.ndarray
    east_areas: np.ndarray
    dr: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="求解A题问题1并输出可复现结果。")
    parser.add_argument(
        "--environment", type=Path, default=Path("outputs/stage3/environment_si.csv"),
        help="阶段三环境边界CSV，默认 outputs/stage3/environment_si.csv。",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/stage4_q1"),
        help="输出目录，默认 outputs/stage4_q1。",
    )
    parser.add_argument(
        "--grid-step-cm", type=float, default=0.0015625,
        help="生产计算径向步长（cm），默认0.0015625；必须整除2 cm。",
    )
    parser.add_argument("--rtol", type=float, default=1e-10, help="BDF相对误差容限。")
    parser.add_argument("--atol", type=float, default=1e-12, help="BDF绝对误差容限。")
    return parser.parse_args()


def build_grid(step_cm: float) -> Grid:
    if step_cm <= 0:
        raise ValueError("网格步长必须为正。")
    dr = step_cm / 100.0
    cells_float = RADIUS_M / dr
    cells = int(round(cells_float))
    if not np.isclose(cells_float, cells, rtol=0.0, atol=1e-10):
        raise ValueError(f"网格步长 {step_cm} cm 不能整除半径2 cm。")
    radii = np.linspace(0.0, RADIUS_M, cells + 1)
    west_faces = np.maximum(0.0, radii - dr / 2.0)
    east_faces = np.minimum(RADIUS_M, radii + dr / 2.0)
    volumes = np.pi * (east_faces**2 - west_faces**2)  # 单位长度圆环体积
    west_areas = 2.0 * np.pi * west_faces              # 单位长度界面面积
    east_areas = 2.0 * np.pi * east_faces
    return Grid(radii, west_faces, east_faces, volumes, west_areas, east_areas, dr)


def load_environment(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"未找到环境数据 {path}。请先运行阶段三脚本，或用 --environment 指定文件。"
        )
    frame = pd.read_csv(path)
    required = ["time_s", "air_temperature_K", "air_moisture_kgkg"]
    if list(frame.columns) != required:
        raise ValueError(f"环境数据字段应为 {required}，实际为 {list(frame.columns)}。")
    if frame[required].isna().any().any():
        raise ValueError("环境数据存在缺失值，停止求解。")
    if not frame["time_s"].is_monotonic_increasing or frame["time_s"].duplicated().any():
        raise ValueError("环境时间必须严格递增且无重复。")
    if frame["time_s"].iloc[0] > 0 or frame["time_s"].iloc[-1] < END_TIME_S:
        raise ValueError("环境数据未覆盖问题1所需的0–1800 s。")
    return tuple(frame[column].to_numpy(dtype=float) for column in required)  # type: ignore[return-value]


def diffusivity(moisture: np.ndarray, scale: float = 1.0) -> np.ndarray:
    safe = np.maximum(moisture, 1e-12)
    return scale * 7.0e-9 * np.exp(-0.89 / safe)


def harmonic_mean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return 2.0 * left * right / np.maximum(left + right, 1e-30)


def make_rhs(
    grid: Grid, env_time: np.ndarray, env_temperature: np.ndarray, env_moisture: np.ndarray,
    heat_transfer_scale: float = 1.0, mass_transfer_scale: float = 1.0,
    diffusivity_scale: float = 1.0,
):
    count = len(grid.radii)

    def rhs(time_s: float, state: np.ndarray) -> np.ndarray:
        temperature = state[:count]
        moisture = state[count:]
        air_t = float(np.interp(time_s, env_time, env_temperature))
        air_c = float(np.interp(time_s, env_time, env_moisture))

        d_temperature = np.zeros_like(temperature)
        d_moisture = np.zeros_like(moisture)

        conductance_t = THERMAL_CONDUCTIVITY * grid.east_areas[:-1] / grid.dr
        heat_flux = conductance_t * (temperature[1:] - temperature[:-1])
        d_temperature[:-1] += heat_flux
        d_temperature[1:] -= heat_flux
        d_temperature[-1] += (
            heat_transfer_scale * HEAT_TRANSFER_COEFFICIENT
            * grid.east_areas[-1] * (air_t - temperature[-1])
        )
        d_temperature /= DENSITY * HEAT_CAPACITY * grid.volumes

        local_d = diffusivity(moisture, diffusivity_scale)
        face_d = harmonic_mean(local_d[:-1], local_d[1:])
        conductance_c = face_d * grid.east_areas[:-1] / grid.dr
        moisture_flux = conductance_c * (moisture[1:] - moisture[:-1])
        d_moisture[:-1] += moisture_flux
        d_moisture[1:] -= moisture_flux
        d_moisture[-1] += (
            mass_transfer_scale * MASS_TRANSFER_COEFFICIENT
            * grid.east_areas[-1] * (air_c - moisture[-1])
        )
        d_moisture /= grid.volumes
        return np.concatenate([d_temperature, d_moisture])

    return rhs


def solve_model(
    step_cm: float, eval_times: np.ndarray, environment: tuple[np.ndarray, np.ndarray, np.ndarray],
    rtol: float, atol: float, heat_transfer_scale: float = 1.0,
    mass_transfer_scale: float = 1.0, diffusivity_scale: float = 1.0,
) -> tuple[Grid, Any]:
    grid = build_grid(step_cm)
    initial = np.concatenate([
        np.full(len(grid.radii), INITIAL_TEMPERATURE_K),
        np.full(len(grid.radii), INITIAL_MOISTURE),
    ])
    count = len(grid.radii)
    tri = diags(
        [np.ones(count - 1), np.ones(count), np.ones(count - 1)],
        offsets=[-1, 0, 1], shape=(count, count), format="csc",
    )
    jacobian_sparsity = block_diag((tri, tri), format="csc")
    solution = solve_ivp(
        make_rhs(
            grid, *environment, heat_transfer_scale=heat_transfer_scale,
            mass_transfer_scale=mass_transfer_scale, diffusivity_scale=diffusivity_scale,
        ),
        (0.0, END_TIME_S), initial, method="BDF", t_eval=eval_times,
        rtol=rtol, atol=atol, max_step=10.0, jac_sparsity=jacobian_sparsity,
    )
    if not solution.success:
        raise RuntimeError(f"BDF求解失败：{solution.message}")
    if not np.isfinite(solution.y).all():
        raise RuntimeError("数值解出现NaN或无穷值。")
    return grid, solution


def interpolate_fields(grid: Grid, solution: Any, output_radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = len(grid.radii)
    temperatures = np.vstack([
        np.interp(output_radii, grid.radii, solution.y[:count, index])
        for index in range(solution.y.shape[1])
    ])
    moistures = np.vstack([
        np.interp(output_radii, grid.radii, solution.y[count:, index])
        for index in range(solution.y.shape[1])
    ])
    return temperatures, moistures


def output_frame(times: np.ndarray, radii_m: np.ndarray, values: np.ndarray, kelvin_to_celsius: bool) -> pd.DataFrame:
    displayed = values - 273.15 if kelvin_to_celsius else values
    columns = [f"{radius * 100:.1f}" for radius in radii_m]
    frame = pd.DataFrame(displayed, columns=columns)
    frame.insert(0, "时间/s", times.astype(int))
    return frame


def convergence_analysis(
    environment: tuple[np.ndarray, np.ndarray, np.ndarray], rtol: float, atol: float
) -> pd.DataFrame:
    steps = [0.0125, 0.00625, 0.003125, 0.0015625]
    fields: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    solver_stats: dict[float, Any] = {}
    for step in steps:
        grid, solution = solve_model(step, TABLE_TIMES_S, environment, rtol, atol)
        fields[step] = interpolate_fields(grid, solution, TABLE_RADII_M)
        solver_stats[step] = solution
    reference_t, reference_c = fields[0.0015625]
    rows = []
    for step in steps:
        values_t, values_c = fields[step]
        rows.append({
            "grid_step_cm": step,
            "radial_nodes": int(round(2.0 / step)) + 1,
            "max_abs_temperature_difference_C": float(np.max(np.abs(values_t - reference_t))),
            "max_abs_moisture_difference_kgkg": float(np.max(np.abs(values_c - reference_c))),
            "nfev": int(solver_stats[step].nfev),
            "njev": int(solver_stats[step].njev),
            "nlu": int(solver_stats[step].nlu),
        })
    return pd.DataFrame(rows)


def conservation_report(
    grid: Grid, solution: Any, environment: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> dict[str, float]:
    count = len(grid.radii)
    moisture = solution.y[count:, :].T
    total = moisture @ grid.volumes
    air_c = np.interp(solution.t, environment[0], environment[2])
    boundary_rate = (
        MASS_TRANSFER_COEFFICIENT * grid.east_areas[-1] * (air_c - moisture[:, -1])
    )
    cumulative = np.zeros_like(solution.t)
    cumulative[1:] = np.cumsum(
        0.5 * (boundary_rate[1:] + boundary_rate[:-1]) * np.diff(solution.t)
    )
    residual = total - total[0] - cumulative
    scale = max(abs(total[-1] - total[0]), 1e-15)
    return {
        "initial_integrated_moisture_per_unit_length": float(total[0]),
        "final_integrated_moisture_per_unit_length": float(total[-1]),
        "max_absolute_balance_residual": float(np.max(np.abs(residual))),
        "max_relative_balance_residual": float(np.max(np.abs(residual)) / scale),
    }


def sensitivity_analysis(
    environment: tuple[np.ndarray, np.ndarray, np.ndarray], rtol: float, atol: float
) -> pd.DataFrame:
    scenarios = [
        ("基准", 1.0, 1.0, 1.0),
        ("h -10%", 0.9, 1.0, 1.0),
        ("h +10%", 1.1, 1.0, 1.0),
        ("hm -10%", 1.0, 0.9, 1.0),
        ("hm +10%", 1.0, 1.1, 1.0),
        ("D -10%", 1.0, 1.0, 0.9),
        ("D +10%", 1.0, 1.0, 1.1),
    ]
    rows: list[dict[str, float | str]] = []
    for name, h_scale, hm_scale, d_scale in scenarios:
        grid, solution = solve_model(
            0.00625, np.array([END_TIME_S], dtype=float), environment, rtol, atol,
            heat_transfer_scale=h_scale, mass_transfer_scale=hm_scale,
            diffusivity_scale=d_scale,
        )
        count = len(grid.radii)
        rows.append({
            "scenario": name,
            "h_scale": h_scale,
            "hm_scale": hm_scale,
            "D_scale": d_scale,
            "center_temperature_C_1800s": float(solution.y[0, -1] - 273.15),
            "surface_temperature_C_1800s": float(solution.y[count - 1, -1] - 273.15),
            "center_moisture_1800s": float(solution.y[count, -1]),
            "surface_moisture_1800s": float(solution.y[-1, -1]),
            "volume_weighted_moisture_1800s": float(
                np.dot(solution.y[count:, -1], grid.volumes) / grid.volumes.sum()
            ),
        })
    return pd.DataFrame(rows)


def configure_plotting() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_profiles(
    times: np.ndarray, temperature_c: np.ndarray, moisture: np.ndarray, output_dir: Path
) -> None:
    configure_plotting()
    selected = [0, 100, 300, 600, 900, 1200, 1500, 1800]
    radii_cm = OUTPUT_RADII_M * 100.0
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for time_s in selected:
        index = int(np.where(times == time_s)[0][0])
        axes[0].plot(radii_cm, temperature_c[index], label=f"{time_s} s")
        axes[1].plot(radii_cm, moisture[index], label=f"{time_s} s")
    axes[0].set_title("药材径向温度分布")
    axes[0].set_xlabel("到中心距离（cm）")
    axes[0].set_ylabel("温度（°C）")
    axes[1].set_title("药材径向水分浓度分布")
    axes[1].set_xlabel("到中心距离（cm）")
    axes[1].set_ylabel("水分浓度（kg/kg）")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "q1_radial_profiles.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_flowchart(path: Path) -> None:
    configure_plotting()
    fig, axis = plt.subplots(figsize=(12, 2.5))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 2.5)
    axis.axis("off")
    labels = [
        "附件1\n线性插值", "设置初值与\n对流边界", "径向有限体积\n空间离散",
        "BDF自适应\n隐式积分", "网格/守恒\n检验", "按题意导出\n表格与Excel",
    ]
    xs = [0.2, 2.2, 4.2, 6.2, 8.2, 10.2]
    for x, label in zip(xs, labels):
        box = FancyBboxPatch(
            (x, 0.75), 1.6, 1.0, boxstyle="round,pad=0.08",
            facecolor="#EAF2F8", edgecolor="#1F4E78", linewidth=1.4,
        )
        axis.add_patch(box)
        axis.text(x + 0.8, 1.25, label, ha="center", va="center", fontsize=10)
    for x in xs[:-1]:
        axis.add_patch(FancyArrowPatch(
            (x + 1.62, 1.25), (x + 1.98, 1.25), arrowstyle="-|>",
            mutation_scale=13, color="#555555", linewidth=1.2,
        ))
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", transparent=False)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    environment_path = args.environment.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = load_environment(environment_path)

    eval_times = np.arange(0, END_TIME_S + 1, dtype=float)
    grid, solution = solve_model(
        args.grid_step_cm, eval_times, environment, args.rtol, args.atol
    )
    temperature_k, moisture = interpolate_fields(grid, solution, OUTPUT_RADII_M)
    temperature_c = temperature_k - 273.15

    if temperature_c.min() < 27.999 or temperature_c.max() > environment[1].max() + 1e-6:
        raise RuntimeError("温度解违反预期最大值范围。")
    if moisture.min() < environment[2].min() - 1e-6 or moisture.max() > INITIAL_MOISTURE + 1e-8:
        raise RuntimeError("水分解违反预期最大值范围。")
    if np.any(np.diff(moisture, axis=0) > 1e-7):
        raise RuntimeError("检测到超过容差的节点水分回升，需检查数值解。")

    temperature_frame = output_frame(eval_times, OUTPUT_RADII_M, temperature_k, True)
    moisture_frame = output_frame(eval_times, OUTPUT_RADII_M, moisture, False)
    temperature_frame.to_csv(output_dir / "q1_temperature_full.csv", index=False, encoding="utf-8-sig")
    moisture_frame.to_csv(output_dir / "q1_moisture_full.csv", index=False, encoding="utf-8-sig")

    table_indices = TABLE_TIMES_S.astype(int)
    radius_indices = (TABLE_RADII_M / 0.001).round().astype(int)
    paper_temperature = pd.DataFrame(
        temperature_c[table_indices][:, radius_indices],
        index=TABLE_TIMES_S.astype(int), columns=["0", "0.5", "1", "1.5", "2"],
    )
    paper_moisture = pd.DataFrame(
        moisture[table_indices][:, radius_indices],
        index=TABLE_TIMES_S.astype(int), columns=["0", "0.5", "1", "1.5", "2"],
    )
    paper_temperature.index.name = "时间/s"
    paper_moisture.index.name = "时间/s"
    paper_temperature.to_csv(output_dir / "q1_table_temperature.csv", encoding="utf-8-sig", float_format="%.4f")
    paper_moisture.to_csv(output_dir / "q1_table_moisture.csv", encoding="utf-8-sig", float_format="%.4f")

    convergence = convergence_analysis(environment, args.rtol, args.atol)
    convergence.to_csv(output_dir / "q1_convergence.csv", index=False, encoding="utf-8-sig")
    sensitivity = sensitivity_analysis(environment, args.rtol, args.atol)
    sensitivity.to_csv(output_dir / "q1_sensitivity.csv", index=False, encoding="utf-8-sig")
    balance = conservation_report(grid, solution, environment)
    volume_weighted_temperature_c = float(
        np.dot(solution.y[:len(grid.radii), -1] - 273.15, grid.volumes) / grid.volumes.sum()
    )
    volume_weighted_moisture = float(
        np.dot(solution.y[len(grid.radii):, -1], grid.volumes) / grid.volumes.sum()
    )
    plot_profiles(eval_times.astype(int), temperature_c, moisture, output_dir)
    plot_flowchart(output_dir / "q1_model_flow.png")

    summary = {
        "input": str(environment_path),
        "model": "one-dimensional radial vertex-centered finite volume",
        "time_integrator": "scipy.solve_ivp BDF",
        "parameters": {
            "radius_m": RADIUS_M,
            "initial_temperature_K": INITIAL_TEMPERATURE_K,
            "initial_moisture_kgkg": INITIAL_MOISTURE,
            "density_kg_m3": DENSITY,
            "heat_capacity_J_kgK": HEAT_CAPACITY,
            "thermal_conductivity_W_mK": THERMAL_CONDUCTIVITY,
            "heat_transfer_coefficient_W_m2K": HEAT_TRANSFER_COEFFICIENT,
            "mass_transfer_coefficient_m_s": MASS_TRANSFER_COEFFICIENT,
            "diffusivity_formula": "7e-9*exp(-0.89/C) m2/s",
        },
        "numerics": {
            "grid_step_cm": args.grid_step_cm,
            "radial_nodes": len(grid.radii),
            "rtol": args.rtol,
            "atol": args.atol,
            "max_step_s": 10.0,
            "nfev": int(solution.nfev),
            "njev": int(solution.njev),
            "nlu": int(solution.nlu),
        },
        "results": {
            "temperature_C_at_1800s_center": float(temperature_c[-1, 0]),
            "temperature_C_at_1800s_surface": float(temperature_c[-1, -1]),
            "moisture_at_1800s_center": float(moisture[-1, 0]),
            "moisture_at_1800s_surface": float(moisture[-1, -1]),
            "air_temperature_C_at_1800s": float(np.interp(1800, environment[0], environment[1]) - 273.15),
            "air_moisture_at_1800s": float(np.interp(1800, environment[0], environment[2])),
            "volume_weighted_temperature_C_at_1800s": volume_weighted_temperature_c,
            "volume_weighted_moisture_at_1800s": volume_weighted_moisture,
        },
        "conservation": balance,
        "convergence": convergence.to_dict(orient="records"),
        "sensitivity": sensitivity.to_dict(orient="records"),
    }
    (output_dir / "q1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )

    expected = [
        "q1_temperature_full.csv", "q1_moisture_full.csv",
        "q1_table_temperature.csv", "q1_table_moisture.csv",
        "q1_convergence.csv", "q1_sensitivity.csv", "q1_summary.json",
        "q1_radial_profiles.png", "q1_model_flow.png",
    ]
    missing = [name for name in expected if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"输出完整性校验失败：{missing}")

    print(f"求解成功：{len(grid.radii)}个径向节点，{len(eval_times)}个输出时刻。")
    print(f"BDF统计：nfev={solution.nfev}, njev={solution.njev}, nlu={solution.nlu}")
    print(
        "1800 s：中心/表面温度="
        f"{temperature_c[-1, 0]:.6f}/{temperature_c[-1, -1]:.6f} °C；"
        "中心/表面水分="
        f"{moisture[-1, 0]:.6f}/{moisture[-1, -1]:.6f} kg/kg"
    )
    print(f"最大相对水分守恒残差={balance['max_relative_balance_residual']:.3e}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2)
