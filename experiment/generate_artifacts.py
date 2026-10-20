from __future__ import annotations

import csv
import json
import math
import shutil
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        return
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_numbers(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def build_synthetic_summary(summary: dict, ortools_rows: list[dict]) -> list[dict]:
    metrics = {
        "cost_rs": "cost_rs",
        "distance_km": "distance_km",
        "penalty_rs": "penalty_rs",
        "ontime_pct": "ontime_pct",
    }
    methods = [
        ("Trajecta best candidate", "candidate", summary["candidate_best_cost_rs"]),
        ("OR-Tools CP-SAT", "ortools_shared_model", summary["ortools_shared_model"]),
        ("ALNS", "metaheuristics", summary["metaheuristics"]["ALNS"]),
        ("Genetic Algorithm", "metaheuristics", summary["metaheuristics"]["Genetic Algorithm"]),
        ("Tabu Search", "metaheuristics", summary["metaheuristics"]["Tabu Search"]),
    ]
    rows = []
    for name, section, values in methods:
        for metric, key in metrics.items():
            stat = values if section == "candidate" else values.get(metric, {})
            rows.append({
                "method": name,
                "metric": metric,
                "mean": stat.get("mean"),
                "sd": stat.get("sd"),
                "n": summary["n_seeds"] if section != "ortools_shared_model" else summary["ortools_shared_model"]["n_solved"],
            })
    runtime_mean, runtime_sd = mean_sd(finite_numbers(ortools_rows, "runtime_s"))
    rows.append({"method": "OR-Tools CP-SAT", "metric": "runtime_s", "mean": runtime_mean,
                 "sd": runtime_sd, "n": len(ortools_rows)})
    return rows


def grouped_summary(rows: list[dict[str, str]], group_key: str, metrics: list[str]) -> list[dict]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(row)
    output = []
    for group, group_rows in groups.items():
        result = {group_key: group}
        for metric in metrics:
            mean, sd = mean_sd(finite_numbers(group_rows, metric))
            result[f"{metric}_mean"] = mean
            result[f"{metric}_sd"] = sd
        result["n"] = len(group_rows)
        output.append(result)
    return output


def select_candidate_rows(rows: list[dict[str, str]]) -> list[dict]:
    by_seed: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        try:
            feasible = math.isfinite(float(row["cost_rs"]))
        except (KeyError, TypeError, ValueError):
            feasible = False
        if feasible:
            by_seed.setdefault(row["seed"], []).append(row)

    selected = []
    for seed, candidates in sorted(by_seed.items(), key=lambda item: int(item[0])):
        for objective, field in (("minimum_cost", "cost_rs"),
                                 ("minimum_distance", "distance_km"),
                                 ("minimum_penalty", "penalty_rs")):
            row = min(candidates, key=lambda candidate: float(candidate[field]))
            selected.append({"selection": objective, **row})
    return selected


def make_figures(summary: dict, candidate_rows: list[dict[str, str]],
                 ablation_rows: list[dict[str, str]], scalability_rows: list[dict[str, str]],
                 benchmark_rows: list[dict[str, str]]) -> None:
    methods = [
        ("Trajecta", summary["candidate_best_cost_rs"], summary["candidate_distance_km"],
         summary["candidate_penalty_rs"], summary["candidate_ontime_pct"]),
        ("OR-Tools", summary["ortools_shared_model"]["cost_rs"], summary["ortools_shared_model"]["distance_km"],
         summary["ortools_shared_model"]["penalty_rs"], summary["ortools_shared_model"]["ontime_pct"]),
        ("ALNS", summary["metaheuristics"]["ALNS"]["cost_rs"], summary["metaheuristics"]["ALNS"]["distance_km"],
         summary["metaheuristics"]["ALNS"]["penalty_rs"], summary["metaheuristics"]["ALNS"]["ontime_pct"]),
        ("Genetic Algorithm", summary["metaheuristics"]["Genetic Algorithm"]["cost_rs"],
         summary["metaheuristics"]["Genetic Algorithm"]["distance_km"],
         summary["metaheuristics"]["Genetic Algorithm"]["penalty_rs"],
         summary["metaheuristics"]["Genetic Algorithm"]["ontime_pct"]),
        ("Tabu Search", summary["metaheuristics"]["Tabu Search"]["cost_rs"],
         summary["metaheuristics"]["Tabu Search"]["distance_km"],
         summary["metaheuristics"]["Tabu Search"]["penalty_rs"],
         summary["metaheuristics"]["Tabu Search"]["ontime_pct"]),
    ]
    metric_indexes = (("Cost (Rs.)", 1), ("Distance (km)", 2),
                      ("Penalty (Rs.)", 3), ("On-time (%)", 4))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (title, index) in zip(axes.flat, metric_indexes):
        values = [method[index].get("mean") for method in methods]
        errors = [method[index].get("sd") for method in methods]
        axis.bar(range(len(methods)), [value if value is not None else 0 for value in values],
                 yerr=[error if error is not None else 0 for error in errors], capsize=3)
        axis.set_title(title)
        axis.set_xticks(range(len(methods)), [method[0] for method in methods], rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Synthetic shared-cost comparison (mean ± SD)")
    figure.tight_layout()
    figure.savefig(FIGURES / "synthetic_comparison.png", dpi=180)
    plt.close(figure)

    valid_candidates = [row for row in candidate_rows
                        if math.isfinite(float(row["cost_rs"])) and math.isfinite(float(row["distance_km"]))]
    figure, axis = plt.subplots(figsize=(8, 5))
    for seed in sorted({row["seed"] for row in valid_candidates}, key=int):
        selected = [row for row in valid_candidates if row["seed"] == seed]
        axis.scatter([float(row["distance_km"]) for row in selected],
                     [float(row["cost_rs"]) for row in selected], label=f"seed {seed}", alpha=0.65, s=20)
    axis.set_xlabel("Batch distance (km)")
    axis.set_ylabel("Total cost (Rs.)")
    axis.set_title("Synthetic candidate routes")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURES / "candidate_cost_distance.png", dpi=180)
    plt.close(figure)

    variant_names = sorted({row["variant"] for row in ablation_rows})
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, key, title in ((axes[0], "best_cost_rs", "Best cost (Rs.)"),
                             (axes[1], "runtime_s", "Runtime (s)")):
        means, errors = [], []
        for name in variant_names:
            values = finite_numbers([row for row in ablation_rows if row["variant"] == name], key)
            mean, sd = mean_sd(values)
            means.append(mean or 0)
            errors.append(sd or 0)
        axis.bar(range(len(variant_names)), means, yerr=errors, capsize=3)
        axis.set_title(title)
        axis.set_xticks(range(len(variant_names)), variant_names, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Ablation results (mean ± SD)")
    figure.tight_layout()
    figure.savefig(FIGURES / "ablation.png", dpi=180)
    plt.close(figure)

    sizes = sorted({int(row["n_orders"]) for row in scalability_rows})
    figure, axis = plt.subplots(figsize=(8, 5))
    for algorithm, label in (("OR-Tools", "OR-Tools"), ("ALNS", "Metaheuristics combined")):
        means, errors = [], []
        for size in sizes:
            values = finite_numbers([row for row in scalability_rows
                                     if int(row["n_orders"]) == size and row["algorithm"] == algorithm], "runtime_s")
            mean, sd = mean_sd(values)
            means.append(mean or 0)
            errors.append(sd or 0)
        axis.errorbar(sizes, means, yerr=errors, marker="o", capsize=3, label=label)
    axis.set_xlabel("Orders")
    axis.set_ylabel("Runtime (s)")
    axis.set_title("Scalability runtime (mean ± SD)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES / "scalability_runtime.png", dpi=180)
    plt.close(figure)

    instances = sorted({row["instance"] for row in benchmark_rows})
    benchmark_methods = ("Trajecta", "OR-Tools", "ALNS", "Genetic Algorithm", "Tabu Search")
    figure, axis = plt.subplots(figsize=(10, 5))
    width = 0.15
    for method_index, method in enumerate(benchmark_methods):
        means = []
        for instance in instances:
            match = next((row for row in benchmark_rows
                          if row["instance"] == instance and row["algorithm"] == method), None)
            value = match.get("mean_distance") if match else ""
            try:
                means.append(float(value) if value else float("nan"))
            except ValueError:
                means.append(float("nan"))
        positions = [index + (method_index - (len(benchmark_methods) - 1) / 2) * width
                     for index in range(len(instances))]
        axis.bar(positions, means, width, label=method)
    axis.set_xticks(range(len(instances)), instances)
    axis.set_ylabel("Mean benchmark distance (source units)")
    axis.set_title("Reduced public-benchmark formulation")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES / "benchmark_distance.png", dpi=180)
    plt.close(figure)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    candidate_rows = read_csv(RESULTS / "candidate_routes.csv")
    ortools_rows = read_csv(RESULTS / "ortools_baseline.csv")
    ablation_rows = read_csv(RESULTS / "ablation_summary.csv")
    scalability_rows = read_csv(RESULTS / "scalability_summary.csv")
    benchmark_rows = read_csv(RESULTS / "benchmark_algorithm_summary.csv")

    write_csv(TABLES / "synthetic_method_summary.csv", build_synthetic_summary(summary, ortools_rows))
    write_csv(TABLES / "selected_candidate_routes.csv", select_candidate_rows(candidate_rows))
    write_csv(TABLES / "ablation_mean_sd.csv",
              grouped_summary(ablation_rows, "variant", ["best_cost_rs", "best_distance_km",
                             "best_penalty_rs", "best_ontime_pct", "runtime_s"]))

    scaling_table = []
    for size in sorted({row["n_orders"] for row in scalability_rows}, key=int):
        for algorithm in sorted({row["algorithm"] for row in scalability_rows if row["n_orders"] == size}):
            matching = [row for row in scalability_rows
                        if row["n_orders"] == size and row["algorithm"] == algorithm]
            result = {"n_orders": size, "algorithm": algorithm}
            for metric in ("cost_rs", "distance_km", "penalty_rs", "ontime_pct", "runtime_s"):
                mean, sd = mean_sd(finite_numbers(matching, metric))
                result[f"{metric}_mean"] = mean
                result[f"{metric}_sd"] = sd
            result["n_runs"] = len(matching)
            scaling_table.append(result)
    write_csv(TABLES / "scalability_mean_sd.csv", scaling_table)
    shutil.copy2(RESULTS / "benchmark_algorithm_summary.csv", TABLES / "benchmark_algorithm_summary.csv")
    shutil.copy2(RESULTS / "benchmark_algorithm_runs.csv", TABLES / "benchmark_algorithm_runs.csv")

    make_figures(summary, candidate_rows, ablation_rows, scalability_rows, benchmark_rows)
    print(f"Generated figures in {FIGURES}")
    print(f"Generated tables in {TABLES}")


if __name__ == "__main__":
    main()
