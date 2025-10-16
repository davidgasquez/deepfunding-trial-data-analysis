#!/usr/bin/env -S uv run --script

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from plotting import small_multiple_bars
from weights import (
    DEFAULT_INPUT,
    DEFAULT_METHOD,
    WEIGHT_METHODS,
    compute_weights as compute_repo_weights,
)

OUTPUT_DIR = Path("data/repo_weights")
COMBINED_OUTPUT = OUTPUT_DIR / "all_methods.csv"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute repository weights from DeepFunding comparison data."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to comparison pairs CSV (default: data/pairs.csv).",
    )
    parser.add_argument(
        "--method",
        choices=WEIGHT_METHODS,
        action="append",
        help=(
            "Weighting method to apply. Can be provided multiple times; defaults to "
            "least-squares unless --all-methods is set."
        ),
    )
    parser.add_argument(
        "--all-methods",
        action="store_true",
        help="Compute weights for every available method.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            f"Optional output CSV path. Defaults to {OUTPUT_DIR}/<method>.csv."
        ),
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        help=(
            "Path for the combined weights CSV when multiple methods are computed "
            f"(default: {COMBINED_OUTPUT})."
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Render a comparison bar chart when multiple methods are computed.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        help="Destination for the comparison bar chart (default: figures/repo-weights-by-method.png).",
    )
    parser.add_argument(
        "--plot-top",
        type=int,
        default=15,
        help="Number of repositories to display in the comparison plot (default: 15).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    methods = _select_methods(args.method, args.all_methods)
    results: list[pd.DataFrame] = []

    for method in methods:
        weights = compute_repo_weights(args.csv, method=method)
        output_path = _resolve_output_path(args.output if len(methods) == 1 else None, method)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        weights.to_csv(output_path, index=False)
        print(f"Wrote weights to {output_path}")
        results.append(weights.assign(method=method))

    if len(results) > 1:
        combined = pd.concat(results, ignore_index=True)
        combined_path = _resolve_combined_output(args.combined_output)
        combined.to_csv(combined_path, index=False)
        print(f"Wrote combined weights to {combined_path}")
        if args.plot:
            pivot = combined.pivot(index="repo", columns="method", values="weight")
            plot_path = _resolve_plot_output(args.plot_output)
            if small_multiple_bars(
                pivot,
                title="Repository weights by method",
                destination=plot_path,
                cols=5,
            ):
                print(f"Wrote comparison plot to {plot_path}")
            else:
                print("Skipping plot: insufficient data for visualization.")
    elif args.plot:
        print("Skipping plot request: need at least two methods for comparison.")
    return 0


def _select_methods(selected: list[str] | None, all_methods: bool) -> list[str]:
    if all_methods or not selected:
        return list(WEIGHT_METHODS) if all_methods else [DEFAULT_METHOD]
    ordered = dict.fromkeys(selected)
    return [method for method in ordered if method in WEIGHT_METHODS]


def _resolve_output_path(path: Path | None, method: str) -> Path:
    if path is not None:
        return path
    sanitized_method = method.replace("/", "-")
    return OUTPUT_DIR / f"{sanitized_method}.csv"


def _resolve_combined_output(path: Path | None) -> Path:
    return path if path is not None else COMBINED_OUTPUT


def _resolve_plot_output(path: Path | None) -> Path:
    return path if path is not None else Path("figures/repo-weights-by-method.png")


if __name__ == "__main__":
    raise SystemExit(main())
