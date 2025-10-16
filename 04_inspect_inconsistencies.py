#!/usr/bin/env -S uv run --script

import argparse
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd

from plotting import ensure_directory, horizontal_bar

DEFAULT_INPUT = Path("data/pairs.csv")
DEFAULT_FIGURES_DIR = Path("figures")
INTENSITY_REL_ERROR_THRESHOLD = 0.25  # 25%


@dataclass(frozen=True)
class PairOutcome:
    winner: str
    loser: str
    log_ratio: float
    comparisons: int

    @property
    def ratio(self) -> float:
        return math.exp(self.log_ratio)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check DeepFunding comparison inconsistencies."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to pairs.csv (default: data/pairs.csv).",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Directory where charts will be saved (default: figures/).",
    )
    return parser.parse_args(argv)


def load_pairs(path: Path) -> pd.DataFrame:
    if not path.is_file():
        msg = f"Input file {path} does not exist."
        raise FileNotFoundError(msg)
    frame = pd.read_csv(path)
    expected = {"juror", "repo_a", "repo_b", "choice", "multiplier"}
    missing = expected.difference(frame.columns)
    if missing:
        msg = f"Missing expected columns: {', '.join(sorted(missing))}."
        raise ValueError(msg)
    return frame


def prepare_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.dropna(subset=["juror", "repo_a", "repo_b", "choice", "multiplier"])
    clean = clean.copy()
    clean["choice"] = pd.to_numeric(clean["choice"], errors="coerce").astype("Int64")
    clean["multiplier"] = pd.to_numeric(clean["multiplier"], errors="coerce")
    clean = clean[clean["choice"].isin([1, 2]) & (clean["multiplier"] > 0)]
    clean["winner"] = clean["repo_a"].where(clean["choice"] == 1, clean["repo_b"])
    clean["loser"] = clean["repo_b"].where(clean["choice"] == 1, clean["repo_a"])
    clean["side"] = clean["choice"].map({1: "left", 2: "right"})
    clean["log_multiplier"] = clean["multiplier"].map(math.log)
    return clean[
        [
            "juror",
            "winner",
            "loser",
            "multiplier",
            "log_multiplier",
            "side",
        ]
    ]


def aggregate_outcomes(comparisons: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        comparisons.groupby(["juror", "winner", "loser"], dropna=False)
        .agg(
            comparisons=("multiplier", "size"),
            mean_log=("log_multiplier", "mean"),
        )
        .reset_index()
    )
    return grouped


def build_pair_map(outcomes: pd.DataFrame) -> dict[str, dict[frozenset[str], list[PairOutcome]]]:
    pair_map: dict[str, dict[frozenset[str], list[PairOutcome]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in outcomes.to_dict(orient="records"):
        juror = row.get("juror")
        winner = row.get("winner")
        loser = row.get("loser")
        mean_log = row.get("mean_log")
        comparisons = row.get("comparisons")
        if (
            juror is None
            or winner is None
            or loser is None
            or mean_log is None
            or comparisons is None
        ):
            continue
        outcome = PairOutcome(
            winner=str(winner),
            loser=str(loser),
            log_ratio=float(mean_log),
            comparisons=int(comparisons),
        )
        pair_key = frozenset({outcome.winner, outcome.loser})
        pair_map[str(juror)][pair_key].append(outcome)
    return pair_map


def log_ratio_for(outcome: PairOutcome, top: str, bottom: str) -> float:
    if outcome.winner == top and outcome.loser == bottom:
        return outcome.log_ratio
    if outcome.winner == bottom and outcome.loser == top:
        return -outcome.log_ratio
    msg = f"Outcome inconsistent with requested direction: {top} vs {bottom}"
    raise ValueError(msg)


def analyze_triangles(
    pair_map: dict[str, dict[frozenset[str], list[PairOutcome]]]
) -> tuple[pd.DataFrame, pd.DataFrame, Counter, Counter]:
    cycle_rows: list[dict[str, object]] = []
    intensity_rows: list[dict[str, object]] = []
    triangle_totals: Counter[str] = Counter()
    pair_conflicts: Counter[str] = Counter()

    for juror, outcomes in pair_map.items():
        nodes: set[str] = set()
        for entry_list in outcomes.values():
            if len(entry_list) > 1:
                pair_conflicts[juror] += 1
            for entry in entry_list:
                nodes.add(entry.winner)
                nodes.add(entry.loser)
        if len(nodes) < 3:
            continue
        lookup: dict[tuple[str, str], PairOutcome] = {}
        for entry_list in outcomes.values():
            if not entry_list:
                continue
            entry = entry_list[0]
            lookup[(entry.winner, entry.loser)] = entry
        for trio in combinations(sorted(nodes), 3):
            a, b, c = trio
            pair_keys = [
                frozenset({a, b}),
                frozenset({b, c}),
                frozenset({a, c}),
            ]
            if any(key not in outcomes for key in pair_keys):
                continue
            if any(len(outcomes[key]) != 1 for key in pair_keys):
                continue
            edge_ab = outcomes[pair_keys[0]][0]
            edge_bc = outcomes[pair_keys[1]][0]
            edge_ac = outcomes[pair_keys[2]][0]
            edges_present = {
                (edge_ab.winner, edge_ab.loser),
                (edge_bc.winner, edge_bc.loser),
                (edge_ac.winner, edge_ac.loser),
            }
            triangle_totals[juror] += 1
            cycle_found = False
            for order in permutations(trio):
                x, y, z = order
                if (x, y) in edges_present and (y, z) in edges_present and (z, x) in edges_present:
                    cycle_rows.append(
                        {
                            "juror": juror,
                            "sequence": order,
                        }
                    )
                    cycle_found = True
                    break
            if cycle_found:
                continue
            found_order: tuple[str, str, str] | None = None
            for order in permutations(trio):
                x, y, z = order
                if (x, y) in edges_present and (y, z) in edges_present and (x, z) in edges_present:
                    found_order = order
                    break
            if found_order is None:
                continue
            top, middle, bottom = found_order
            outcome_top_middle = lookup[(top, middle)]
            outcome_middle_bottom = lookup[(middle, bottom)]
            outcome_top_bottom = lookup[(top, bottom)]
            log_top_middle = log_ratio_for(outcome_top_middle, top, middle)
            log_middle_bottom = log_ratio_for(outcome_middle_bottom, middle, bottom)
            log_top_bottom = log_ratio_for(outcome_top_bottom, top, bottom)
            predicted_log = log_top_middle + log_middle_bottom
            log_error = log_top_bottom - predicted_log
            abs_log_error = abs(log_error)
            error_factor = math.exp(log_error)
            rel_error = abs(error_factor - 1.0)
            intensity_rows.append(
                {
                    "juror": juror,
                    "top": top,
                    "middle": middle,
                    "bottom": bottom,
                    "predicted_ratio": math.exp(predicted_log),
                    "actual_ratio": math.exp(log_top_bottom),
                    "relative_error": rel_error,
                    "abs_log_error": abs_log_error,
                    "top_middle_ratio": math.exp(log_top_middle),
                    "middle_bottom_ratio": math.exp(log_middle_bottom),
                    "top_bottom_ratio": math.exp(log_top_bottom),
                    "top_middle_count": outcome_top_middle.comparisons,
                    "middle_bottom_count": outcome_middle_bottom.comparisons,
                    "top_bottom_count": outcome_top_bottom.comparisons,
                }
            )
    cycles = (
        pd.DataFrame(cycle_rows, columns=["juror", "sequence"])
        if cycle_rows
        else pd.DataFrame(columns=["juror", "sequence"])
    )
    intensity = (
        pd.DataFrame(
            intensity_rows,
            columns=[
                "juror",
                "top",
                "middle",
                "bottom",
                "predicted_ratio",
                "actual_ratio",
                "relative_error",
                "abs_log_error",
                "top_middle_ratio",
                "middle_bottom_ratio",
                "top_bottom_ratio",
                "top_middle_count",
                "middle_bottom_count",
                "top_bottom_count",
            ],
        )
        if intensity_rows
        else pd.DataFrame(
            columns=[
                "juror",
                "top",
                "middle",
                "bottom",
                "predicted_ratio",
                "actual_ratio",
                "relative_error",
                "abs_log_error",
                "top_middle_ratio",
                "middle_bottom_ratio",
                "top_bottom_ratio",
                "top_middle_count",
                "middle_bottom_count",
                "top_bottom_count",
            ]
        )
    )
    return cycles, intensity, triangle_totals, pair_conflicts


def compute_side_bias(comparisons: pd.DataFrame) -> pd.DataFrame:
    side_counts = (
        comparisons.groupby(["juror", "side"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={"left": "left", "right": "right"})
    )
    if "left" not in side_counts:
        side_counts["left"] = 0
    if "right" not in side_counts:
        side_counts["right"] = 0
    side_counts["total"] = side_counts["left"] + side_counts["right"]
    side_counts = side_counts[side_counts["total"] > 0]
    side_counts["left_share"] = side_counts["left"] / side_counts["total"]
    side_counts["right_share"] = side_counts["right"] / side_counts["total"]
    side_counts["abs_bias"] = (side_counts["left_share"] - 0.5).abs()
    side_counts = side_counts.reset_index()
    return side_counts

def write_charts(
    figures_dir: Path,
    cycle_counts: pd.DataFrame,
    intensity_counts: pd.DataFrame,
    bias_table: pd.DataFrame,
) -> list[Path]:
    outputs: list[Path] = []
    ensure_directory(figures_dir)

    if not cycle_counts.empty:
        cycle_path = figures_dir / "order-cycles-per-juror.png"
        if horizontal_bar(
            cycle_counts,
            label_col="juror",
            value_col="order_cycles",
            title="Order cycles per juror (top 10)",
            destination=cycle_path,
            top=10,
            xlabel="Cycle triangles",
            value_formatter="{:.0f}",
        ):
            outputs.append(cycle_path)

    return outputs


def summarize_counts(
    cycles: pd.DataFrame,
    intensity: pd.DataFrame,
    triangle_totals: Counter[str],
    pair_conflicts: Counter[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if cycles.empty:
        cycle_counts = pd.DataFrame(columns=["juror", "order_cycles"])
    else:
        cycle_counts = (
            cycles.groupby("juror")
            .size()
            .to_frame(name="order_cycles")
            .reset_index()
            .sort_values("order_cycles", ascending=False, ignore_index=True)
        )
    if intensity.empty:
        intensity_loops = pd.DataFrame(columns=intensity.columns)
    else:
        intensity_loops = intensity[intensity["relative_error"] > INTENSITY_REL_ERROR_THRESHOLD]
    if intensity_loops.empty:
        intensity_counts = pd.DataFrame(columns=["juror", "intensity_loops"])
    else:
        intensity_counts = (
            intensity_loops.groupby("juror")
            .size()
            .to_frame(name="intensity_loops")
            .reset_index()
            .sort_values("intensity_loops", ascending=False, ignore_index=True)
        )
    totals = (
        pd.DataFrame(
            {
                "juror": list(triangle_totals.keys()),
                "triangles": list(triangle_totals.values()),
            }
        )
        if triangle_totals
        else pd.DataFrame(columns=["juror", "triangles"])
    )
    if not totals.empty:
        totals = totals.merge(cycle_counts, on="juror", how="left").merge(
            intensity_counts, on="juror", how="left"
        )
        totals["order_cycles"] = np.nan_to_num(
            totals["order_cycles"].to_numpy(dtype=float), nan=0.0
        ).astype(int)
        totals["intensity_loops"] = np.nan_to_num(
            totals["intensity_loops"].to_numpy(dtype=float), nan=0.0
        ).astype(int)
        totals["cycle_rate"] = totals["order_cycles"] / totals["triangles"]
        totals["intensity_rate"] = totals["intensity_loops"] / totals["triangles"]
    if pair_conflicts:
        conflicts = pd.DataFrame(
            {
                "juror": list(pair_conflicts.keys()),
                "pair_conflicts": list(pair_conflicts.values()),
            }
        )
        totals = totals.merge(conflicts, on="juror", how="left")
        totals["pair_conflicts"] = np.nan_to_num(
            totals["pair_conflicts"].to_numpy(dtype=float), nan=0.0
        ).astype(int)
    return cycle_counts, intensity_counts, totals


def print_summary(
    totals: pd.DataFrame,
    cycles: pd.DataFrame,
    intensity: pd.DataFrame,
    intensity_counts: pd.DataFrame,
    pair_conflicts: Counter[str],
    bias_table: pd.DataFrame,
) -> None:
    total_triangles = int(totals["triangles"].sum()) if not totals.empty else 0
    total_cycles = int(cycles.shape[0])
    intensity_loops = int(intensity_counts["intensity_loops"].sum()) if not intensity_counts.empty else 0
    print("Triangle checks")
    print(f"- triangles analyzed: {total_triangles}")
    if total_triangles:
        cycle_rate = total_cycles / total_triangles if total_triangles else 0.0
        print(f"- order cycles: {total_cycles} ({cycle_rate:.1%})")
        intensity_rate = intensity_loops / total_triangles if total_triangles else 0.0
        print(f"- intensity loops (>25% error): {intensity_loops} ({intensity_rate:.1%})")
    else:
        print("- order cycles: 0")
        print("- intensity loops (>25% error): 0")
    if pair_conflicts:
        total_pair_conflicts = sum(pair_conflicts.values())
        print(f"- conflicting pair directions: {total_pair_conflicts}")
    print()
    if not cycles.empty:
        top_cycles = cycles["juror"].value_counts().head(5)
        print("Order consistency issues")
        for juror, count in top_cycles.items():
            print(f"- {juror} has {count} triangle order cycle(s)")
        print()
    if not intensity.empty:
        worst = (
            intensity.assign(relative_pct=lambda df: df["relative_error"] * 100)
            .sort_values("relative_error", ascending=False)
            .head(5)
        )
        print("Intensity inconsistencies")
        for row in worst.to_dict(orient="records"):
            juror = row.get("juror", "")
            top_repo = row.get("top", "")
            middle_repo = row.get("middle", "")
            bottom_repo = row.get("bottom", "")
            top_ratio = float(row.get("top_middle_ratio", 0.0))
            middle_ratio = float(row.get("middle_bottom_ratio", 0.0))
            top_bottom_ratio = float(row.get("top_bottom_ratio", 0.0))
            predicted_ratio = float(row.get("predicted_ratio", 0.0))
            top_middle_count = int(row.get("top_middle_count", 0))
            middle_bottom_count = int(row.get("middle_bottom_count", 0))
            top_bottom_count = int(row.get("top_bottom_count", 0))
            relative_pct = float(row.get("relative_pct", 0.0))
            print(
                f"- {juror}: {top_repo}>{middle_repo} "
                f"(x{top_ratio:.2f}, n={top_middle_count}) and {middle_repo}>{bottom_repo} "
                f"(x{middle_ratio:.2f}, n={middle_bottom_count}) imply "
                f"{top_repo}>{bottom_repo} should be x{predicted_ratio:.2f}, "
                f"but recorded x{top_bottom_ratio:.2f} "
                f"(n={top_bottom_count}) → error {relative_pct:.1f}%"
            )
        print()
    if pair_conflicts:
        top_conflicts = pair_conflicts.most_common(5)
        print("Conflicting pair directions")
        for juror, count in top_conflicts:
            print(f"- {juror} recorded {count} opposing judgments on the same pair")
        print()
    if not bias_table.empty:
        avg_bias = bias_table["abs_bias_percent"].mean()
        print("Side bias")
        print(f"- average absolute bias: {avg_bias:.1f}%")
        biased = bias_table.head(5)
        for row in biased.to_dict(orient="records"):
            left_share = float(row.get("left_share", 0.0))
            right_share = float(row.get("right_share", 0.0))
            left = int(row.get("left", 0))
            right = int(row.get("right", 0))
            total = int(row.get("total", 0))
            bias_pct = float(row.get("abs_bias_percent", 0.0))
            side = "left" if left_share > 0.5 else "right"
            majority_pct = max(left_share, right_share) * 100
            juror = row.get("juror", "")
            print(
                f"- {juror} picked {side} {majority_pct:.1f}% "
                f"({left}L/{right}R of {total}, bias {bias_pct:.1f}%)"
            )
        print()


def compute_bias_table(bias: pd.DataFrame) -> pd.DataFrame:
    if bias.empty:
        return bias
    table = bias.copy()
    table["abs_bias_percent"] = table["abs_bias"] * 100
    table = table.sort_values("abs_bias_percent", ascending=False)
    return table


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    frame = load_pairs(args.csv)
    comparisons = prepare_comparisons(frame)
    outcomes = aggregate_outcomes(comparisons)
    pair_map = build_pair_map(outcomes)
    cycles, intensity, triangle_totals, pair_conflicts = analyze_triangles(pair_map)
    cycle_counts, intensity_counts, totals = summarize_counts(
        cycles, intensity, triangle_totals, pair_conflicts
    )
    bias = compute_side_bias(comparisons)
    bias_table = compute_bias_table(bias)
    figures_dir = ensure_directory(args.figures)
    chart_paths = write_charts(figures_dir, cycle_counts, intensity_counts, bias_table)
    print_summary(
        totals,
        cycles,
        intensity,
        intensity_counts,
        pair_conflicts,
        bias_table,
    )
    if chart_paths:
        print("Charts saved:")
        for path in chart_paths:
            print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
