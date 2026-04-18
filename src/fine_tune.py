#!/usr/bin/env python3
"""Grid-search tuner for CBF-RM QP gains based on time-to-goal."""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import itertools
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from utils import StandardizedEnvironment


SCENARIOS = ("doorway", "hallway", "intersection")
DEFAULT_NUM_ROBOTS = 2

# Full search: 3^5 = 243 combinations.
DEFAULT_QP_GAIN_RANGES = {
	"gamma_gain": [0.8, 1.2, 1.6],
	"alpha_gain": [4.5, 5.7, 7.0],
	"beta_gain": [1.0, 1.5, 2.0],
	"p_weight": [8.0, 12.0, 16.0],
	"q_weight": [0.12, 0.24, 0.40],
}

# Quick search: 2^5 = 32 combinations.
QUICK_QP_GAIN_RANGES = {
	"gamma_gain": [1.0, 1.2],
	"alpha_gain": [5.0, 5.7],
	"beta_gain": [1.2, 1.5],
	"p_weight": [10.0, 12.0],
	"q_weight": [0.20, 0.24],
}

# Non-QP parameters are kept fixed to the defaults in methods/CBF-RM/app.py.
FIXED_PARAMS = {
	"dt": 0.1,
	"T": 30.0,
	"obs_sense_range": 3.0,
	"phi_risk": 1.0,
	"c_risk": 0.3,
	"t_risk": 0.8,
	"eps_D": 0.01,
	"k_psi": 2.5,
	"omega_c": 0.4,
	"clip_u": 1.0,
	"clip_omega": 2.0,
	"goal_threshold": 0.3,
}


def load_cbf_rm_module():
	"""Load the CBF-RM app module from methods/CBF-RM/app.py."""
	module_path = Path(__file__).resolve().parent / "methods" / "CBF-RM" / "app.py"
	spec = importlib.util.spec_from_file_location("cbf_rm_app", module_path)
	if spec is None or spec.loader is None:
		raise RuntimeError(f"Could not load module spec from {module_path}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def iter_gain_combinations(gain_ranges: Dict[str, Iterable[float]]):
	"""Yield dictionaries with one gain value per gain name."""
	gain_names = list(gain_ranges.keys())
	value_lists = [list(gain_ranges[name]) for name in gain_names]

	for combo in itertools.product(*value_lists):
		yield dict(zip(gain_names, combo))


def count_total_combinations(gain_ranges: Dict[str, Iterable[float]]) -> int:
	"""Count how many total grid points are in the gain ranges."""
	total = 1
	for values in gain_ranges.values():
		total *= len(list(values))
	return total


def get_default_agent_arrays(scenario: str, num_robots: int) -> Tuple[np.ndarray, np.ndarray]:
	"""Get default start/goal arrays from standardized defaults."""
	positions = StandardizedEnvironment.get_standard_agent_positions(scenario, num_robots)
	if len(positions) < num_robots:
		raise ValueError(
			f"Scenario '{scenario}' only has {len(positions)} default positions, "
			f"but num_robots={num_robots}."
		)

	x0 = np.array([pos["start"] for pos in positions], dtype=float)
	goals = np.array([pos["goal"] for pos in positions], dtype=float)
	return x0, goals


def run_single_scenario(
	cbf_rm_module,
	scenario: str,
	num_robots: int,
	gains: Dict[str, float],
	fixed_params: Dict[str, float],
) -> Dict[str, Any]:
	"""Run one scenario and return TTG-focused metrics."""
	x0, goals = get_default_agent_arrays(scenario, num_robots)

	dt = float(fixed_params["dt"])
	horizon_steps = int(round(float(fixed_params["T"]) / dt))

	agent_radius = StandardizedEnvironment.DEFAULT_AGENT_RADIUS
	d_safe = 2.0 * agent_radius + 0.08

	# Mute simulation stdout during tuning to keep logs readable.
	with contextlib.redirect_stdout(io.StringIO()):
		result = cbf_rm_module.run_cbf_rm_simulation(
			scenario=scenario,
			N=num_robots,
			X0=x0,
			G=goals,
			dt=dt,
			K=horizon_steps,
			d_safe=d_safe,
			obs_sense_range=float(fixed_params["obs_sense_range"]),
			gamma_gain=float(gains["gamma_gain"]),
			alpha_gain=float(gains["alpha_gain"]),
			beta_gain=float(gains["beta_gain"]),
			p_weight=float(gains["p_weight"]),
			q_weight=float(gains["q_weight"]),
			phi_risk=float(fixed_params["phi_risk"]),
			c_risk=float(fixed_params["c_risk"]),
			t_risk=float(fixed_params["t_risk"]),
			eps_D=float(fixed_params["eps_D"]),
			k_psi=float(fixed_params["k_psi"]),
			omega_c=float(fixed_params["omega_c"]),
			clip_u=float(fixed_params["clip_u"]),
			clip_omega=float(fixed_params["clip_omega"]),
			goal_threshold=float(fixed_params["goal_threshold"]),
			verbose_mode=False,
		)

	(
		_,
		_,
		_,
		_,
		_,
		_,
		_,
		infeasible_count,
		effective_steps,
		ttg_steps,
		ttg_seconds,
		reached_goal,
		_,
	) = result

	ttg_steps_array = np.asarray(ttg_steps)
	ttg_seconds_array = np.asarray(ttg_seconds)
	reached_goal_array = np.asarray(reached_goal)
	infeasible_count_array = np.asarray(infeasible_count)

	return {
		"mean_ttg": float(np.mean(ttg_seconds_array)),
		"makespan_ttg": float(np.max(ttg_seconds_array)),
		"success_rate": float(np.mean(reached_goal_array.astype(float))),
		"infeasible_total": int(np.sum(infeasible_count_array)),
		"effective_steps": int(effective_steps),
		"ttg_steps": [int(v) for v in ttg_steps_array.tolist()],
		"ttg_seconds": [float(v) for v in ttg_seconds_array.tolist()],
		"reached_goal": [bool(v) for v in reached_goal_array.tolist()],
	}


def candidate_is_better(candidate: Dict[str, Any], incumbent: Dict[str, Any] | None) -> bool:
	"""Select better objective; break ties by higher success and fewer infeasible solves."""
	if incumbent is None:
		return True

	eps = 1e-12
	if candidate["objective"] < incumbent["objective"] - eps:
		return True
	if abs(candidate["objective"] - incumbent["objective"]) <= eps:
		if candidate["success_rate"] > incumbent["success_rate"] + eps:
			return True
		if abs(candidate["success_rate"] - incumbent["success_rate"]) <= eps:
			if candidate["infeasible_total"] < incumbent["infeasible_total"]:
				return True
	return False


def to_builtin(value: Any) -> Any:
	"""Convert numpy/container values into JSON-serializable builtins."""
	if isinstance(value, dict):
		return {k: to_builtin(v) for k, v in value.items()}
	if isinstance(value, (list, tuple)):
		return [to_builtin(v) for v in value]
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, np.generic):
		return value.item()
	return value


def run_grid_search(
	gain_ranges: Dict[str, Iterable[float]],
	num_robots: int,
	fixed_params: Dict[str, float],
	progress_every: int,
	max_combinations: int | None,
) -> Tuple[list[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
	"""Evaluate gain grid over all scenarios and return all records + best summaries."""
	cbf_rm_module = load_cbf_rm_module()

	total_all = count_total_combinations(gain_ranges)
	total = min(total_all, max_combinations) if max_combinations is not None else total_all

	records: list[Dict[str, Any]] = []
	best_by_scenario: Dict[str, Dict[str, Any] | None] = {scenario: None for scenario in SCENARIOS}
	best_overall: Dict[str, Any] | None = None

	start_time = time.time()
	for idx, gains in enumerate(iter_gain_combinations(gain_ranges), start=1):
		if max_combinations is not None and idx > max_combinations:
			break

		scenario_results: Dict[str, Dict[str, Any]] = {}
		for scenario in SCENARIOS:
			scenario_results[scenario] = run_single_scenario(
				cbf_rm_module=cbf_rm_module,
				scenario=scenario,
				num_robots=num_robots,
				gains=gains,
				fixed_params=fixed_params,
			)

		overall_avg_ttg = float(np.mean([scenario_results[s]["mean_ttg"] for s in SCENARIOS]))
		overall_avg_success = float(np.mean([scenario_results[s]["success_rate"] for s in SCENARIOS]))
		overall_infeasible = int(sum(scenario_results[s]["infeasible_total"] for s in SCENARIOS))

		row: Dict[str, Any] = {k: float(v) for k, v in gains.items()}
		for scenario in SCENARIOS:
			row[f"{scenario}_mean_ttg"] = scenario_results[scenario]["mean_ttg"]
			row[f"{scenario}_makespan_ttg"] = scenario_results[scenario]["makespan_ttg"]
			row[f"{scenario}_success_rate"] = scenario_results[scenario]["success_rate"]
			row[f"{scenario}_infeasible_total"] = scenario_results[scenario]["infeasible_total"]
		row["overall_avg_ttg"] = overall_avg_ttg
		row["overall_avg_success_rate"] = overall_avg_success
		row["overall_infeasible_total"] = overall_infeasible
		records.append(row)

		for scenario in SCENARIOS:
			candidate = {
				"objective": scenario_results[scenario]["mean_ttg"],
				"success_rate": scenario_results[scenario]["success_rate"],
				"infeasible_total": scenario_results[scenario]["infeasible_total"],
				"gains": {k: float(v) for k, v in gains.items()},
				"metrics": scenario_results[scenario],
			}
			if candidate_is_better(candidate, best_by_scenario[scenario]):
				best_by_scenario[scenario] = candidate

		overall_candidate = {
			"objective": overall_avg_ttg,
			"success_rate": overall_avg_success,
			"infeasible_total": overall_infeasible,
			"gains": {k: float(v) for k, v in gains.items()},
			"scenario_metrics": scenario_results,
		}
		if candidate_is_better(overall_candidate, best_overall):
			best_overall = overall_candidate

		if idx % max(1, progress_every) == 0 or idx == total:
			elapsed = time.time() - start_time
			print(f"[{idx}/{total}] searched in {elapsed:.1f}s")

	# best_by_scenario entries are guaranteed not None if at least one combo runs.
	finalized_best_by_scenario = {k: v for k, v in best_by_scenario.items() if v is not None}
	if best_overall is None:
		raise RuntimeError("No combinations were evaluated.")

	return records, finalized_best_by_scenario, best_overall


def save_results(
	output_dir: Path,
	gain_ranges: Dict[str, Iterable[float]],
	num_robots: int,
	records: list[Dict[str, Any]],
	best_by_scenario: Dict[str, Dict[str, Any]],
	best_overall: Dict[str, Any],
) -> Tuple[Path, Path]:
	"""Save full grid-search table and summary JSON."""
	output_dir.mkdir(parents=True, exist_ok=True)

	csv_path = output_dir / "qp_gain_grid_search.csv"
	if records:
		with open(csv_path, "w", newline="") as f:
			writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
			writer.writeheader()
			writer.writerows(records)

	summary = {
		"searched_combinations": len(records),
		"scenarios": list(SCENARIOS),
		"default_num_robots": num_robots,
		"objective": "minimize mean TTG (seconds)",
		"qp_gain_ranges": {k: list(v) for k, v in gain_ranges.items()},
		"best_by_scenario": best_by_scenario,
		"best_overall_average": best_overall,
	}

	json_path = output_dir / "qp_gain_summary.json"
	with open(json_path, "w") as f:
		json.dump(to_builtin(summary), f, indent=2)

	return csv_path, json_path


def print_summary(best_by_scenario: Dict[str, Dict[str, Any]], best_overall: Dict[str, Any]):
	"""Print concise best-parameter summary to stdout."""
	print("\nBest gains per scenario (by mean TTG):")
	for scenario in SCENARIOS:
		best = best_by_scenario[scenario]
		print(
			f"  {scenario}: mean_ttg={best['objective']:.3f}s, "
			f"success={best['success_rate']:.3f}, gains={best['gains']}"
		)

	print("\nBest gains by average TTG across all scenarios:")
	print(
		f"  mean_ttg={best_overall['objective']:.3f}s, "
		f"success={best_overall['success_rate']:.3f}, gains={best_overall['gains']}"
	)
	for scenario in SCENARIOS:
		metric = best_overall["scenario_metrics"][scenario]
		print(
			f"    {scenario}: mean_ttg={metric['mean_ttg']:.3f}s, "
			f"success={metric['success_rate']:.3f}"
		)


def parse_args() -> argparse.Namespace:
	"""Parse command-line options."""
	root_dir = Path(__file__).resolve().parents[1]

	parser = argparse.ArgumentParser(
		description=(
			"Grid search over CBF-RM QP gains using default robots/positions "
			"for doorway, hallway, and intersection."
		)
	)
	parser.add_argument(
		"--quick",
		action="store_true",
		help="Use a smaller gain grid (2 values per gain) for faster runs.",
	)
	parser.add_argument(
		"--num-robots",
		type=int,
		default=DEFAULT_NUM_ROBOTS,
		help="Number of robots (default: 2, matching standardized defaults).",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=root_dir / "logs" / "CBF-RM" / "fine_tune",
		help="Directory where tuning CSV/JSON outputs are written.",
	)
	parser.add_argument(
		"--progress-every",
		type=int,
		default=10,
		help="Print progress every N combinations.",
	)
	parser.add_argument(
		"--max-combinations",
		type=int,
		default=None,
		help="Optional cap on number of gain combinations to evaluate.",
	)
	return parser.parse_args()


def main():
	args = parse_args()

	gain_ranges = QUICK_QP_GAIN_RANGES if args.quick else DEFAULT_QP_GAIN_RANGES
	total_combinations = count_total_combinations(gain_ranges)
	if args.max_combinations is not None:
		total_combinations = min(total_combinations, args.max_combinations)

	print("Starting QP gain grid search")
	print(f"Scenarios: {', '.join(SCENARIOS)}")
	print(f"Num robots: {args.num_robots}")
	print(f"Combinations to evaluate: {total_combinations}")

	start = time.time()
	records, best_by_scenario, best_overall = run_grid_search(
		gain_ranges=gain_ranges,
		num_robots=args.num_robots,
		fixed_params=FIXED_PARAMS,
		progress_every=args.progress_every,
		max_combinations=args.max_combinations,
	)
	elapsed = time.time() - start

	csv_path, json_path = save_results(
		output_dir=args.output_dir,
		gain_ranges=gain_ranges,
		num_robots=args.num_robots,
		records=records,
		best_by_scenario=best_by_scenario,
		best_overall=best_overall,
	)

	print_summary(best_by_scenario, best_overall)
	print(f"\nSearch completed in {elapsed:.1f}s")
	print(f"Full results: {csv_path}")
	print(f"Summary: {json_path}")


if __name__ == "__main__":
	main()
