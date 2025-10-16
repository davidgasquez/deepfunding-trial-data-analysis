#!/usr/bin/env -S uv run --script

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from weights import clean_pairs, design_matrix, load_pairs, normalize_logits, solve_logits  # noqa: E402
from plotting import ensure_directory, horizontal_bar

DEFAULT_INPUT = Path("data/pairs.csv")
DEFAULT_FIGURES_DIR = Path("figures")
PAIR_SEPARATOR = " ↔ "


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate leave-one-out robustness of repository weights."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to comparison pairs CSV (default: data/pairs.csv).",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Directory where charts will be saved (default: figures/).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Number of entities to show in each impact plot (default: 15).",
    )
    return parser.parse_args(argv)

def format_repo_label(value: object) -> str:
    if isinstance(value, str) and value.startswith("https://github.com/"):
        return value.removeprefix("https://github.com/")
    return str(value)


def compute_weights(frame: pd.DataFrame) -> pd.Series | None:
    repos_series = pd.concat(
        [frame["repo_a"], frame["repo_b"]], ignore_index=True
    ).dropna()
    repos = pd.Index(pd.unique(repos_series))
    if len(frame) == 0 or len(repos) < 2:
        return None
    design, ratios = design_matrix(frame, repos)
    if ratios.size == 0:
        return None
    logits = solve_logits(design, ratios)
    weights = normalize_logits(logits)
    return pd.Series(weights, index=repos)


def track_impact(
    level: str,
    entity_id: str,
    entity_label: str,
    subset: pd.DataFrame,
    baseline: pd.Series,
    baseline_index: pd.Index,
    summary_rows: list[dict[str, object]],
) -> None:
    weights = compute_weights(subset)
    if weights is None:
        return
    aligned = weights.reindex(baseline_index, fill_value=0.0)
    delta = aligned - baseline
    if np.allclose(delta.to_numpy(), 0.0):
        return
    abs_delta = delta.abs()
    top_repo = abs_delta.idxmax()
    summary_rows.append(
        {
            "level": level,
            "entity_id": entity_id,
            "entity": entity_label,
            "max_abs": float(abs_delta[top_repo]),
            "top_repo": top_repo,
            "top_repo_label": format_repo_label(top_repo),
            "top_delta": float(delta[top_repo]),
            "l1": float(abs_delta.sum()),
        }
    )


def plot_top_impacts(
    summary: pd.DataFrame, level: str, destination: Path, top_n: int
) -> bool:
    subset = summary[summary["level"] == level].copy()
    if subset.empty:
        return False
    subset = subset.sort_values("max_abs", ascending=False).head(top_n)
    subset = subset.assign(entity_label=subset["entity"].astype(str))
    annotations = [
        f"{row.top_repo_label} ({row.top_delta:+.3f})"
        for row in subset.itertuples(index=False)
    ]
    return horizontal_bar(
        subset,
        label_col="entity_label",
        value_col="max_abs",
        title=f"Leave-one-out impact by {level}",
        destination=destination,
        annotations=annotations,
        xlabel="Max |Δ weight|",
        value_formatter="{:.3f}",
    )


def plot_baseline_weights(
    weights: pd.Series, destination: Path, top_n: int
) -> bool:
    if weights.empty:
        return False
    subset = weights.sort_values(ascending=False).head(top_n).rename("weight")
    frame = (
        subset.reset_index()
        .rename(columns={"index": "repository"})
        .assign(repository_label=lambda df: df["repository"].map(format_repo_label))
    )
    annotations = [f"{value:.3f}" for value in frame["weight"]]
    return horizontal_bar(
        frame,
        label_col="repository_label",
        value_col="weight",
        title="Baseline repository weights",
        destination=destination,
        annotations=annotations,
        xlabel="Weight",
        value_formatter="{:.3f}",
    )


def plot_total_redistribution(
    summary: pd.DataFrame, level: str, destination: Path, top_n: int
) -> bool:
    subset = summary[summary["level"] == level].copy()
    if subset.empty:
        return False
    subset = subset.sort_values("l1", ascending=False).head(top_n)
    subset = subset.assign(entity_label=subset["entity"].astype(str))
    return horizontal_bar(
        subset,
        label_col="entity_label",
        value_col="l1",
        title=f"Total redistribution by {level}",
        destination=destination,
        xlabel="Total redistribution (Σ|Δ|)",
        value_formatter="{:.3f}",
    )


def compute_pair_key(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(
        lambda row: tuple(sorted((row["repo_a"], row["repo_b"]))), axis=1
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw = load_pairs(args.csv)
    clean = clean_pairs(raw).copy()
    clean["pair_key"] = compute_pair_key(clean)

    baseline = compute_weights(clean)
    if baseline is None:
        raise ValueError("Unable to compute baseline weights from input data.")
    baseline_index = baseline.index

    summary_rows: list[dict[str, object]] = []

    comparisons = len(clean)
    juror_count = clean["juror"].nunique(dropna=True)
    repo_count = len(baseline)
    pair_count = len(clean["pair_key"].unique())
    print(
        f"Comparisons: {comparisons} | Jurors: {juror_count} | "
        f"Repositories: {repo_count} | Distinct pairs: {pair_count}"
    )

    figures_dir = ensure_directory(args.figures)
    baseline_plot = figures_dir / "baseline-weights.png"
    if plot_baseline_weights(baseline, baseline_plot, args.top):
        print(f"Wrote {baseline_plot}")

    # Juror leave-one-out
    for juror in sorted(clean["juror"].dropna().unique()):
        subset = clean[clean["juror"] != juror]
        track_impact(
            level="juror",
            entity_id=str(juror),
            entity_label=str(juror),
            subset=subset,
            baseline=baseline,
            baseline_index=baseline_index,
            summary_rows=summary_rows,
        )

    # Repository leave-one-out
    for repo in baseline_index:
        subset = clean[
            (clean["repo_a"] != repo) & (clean["repo_b"] != repo)
        ]
        track_impact(
            level="repository",
            entity_id=repo,
            entity_label=format_repo_label(repo),
            subset=subset,
            baseline=baseline,
            baseline_index=baseline_index,
            summary_rows=summary_rows,
        )

    # Pair leave-one-out (unordered repo pairs)
    unique_pairs = clean["pair_key"].dropna().unique().tolist()
    for pair in unique_pairs:
        repo_left, repo_right = pair
        subset = clean[clean["pair_key"] != pair]
        entity_label = (
            f"{format_repo_label(repo_left)}{PAIR_SEPARATOR}"
            f"{format_repo_label(repo_right)}"
        )
        track_impact(
            level="pair",
            entity_id=f"{repo_left}||{repo_right}",
            entity_label=entity_label,
            subset=subset,
            baseline=baseline,
            baseline_index=baseline_index,
            summary_rows=summary_rows,
        )

    summary = pd.DataFrame(summary_rows)

    if summary.empty:
        print("No leave-one-out impacts produced.")
        return 0

    summary = summary.sort_values(["level", "max_abs"], ascending=[True, False])
    top_display = summary.groupby("level").head(5)
    print("Top leave-one-out impacts (by max |Δ|):")
    for level, group in top_display.groupby("level"):
        print(f"\n[{level}]")
        print(
            group[
                [
                    "entity",
                    "max_abs",
                    "top_repo_label",
                    "top_delta",
                    "l1",
                ]
            ].to_string(index=False, justify="left", float_format=lambda v: f"{v:.4f}")
        )

    plots = {
        "juror": figures_dir / "loo-juror-impacts.png",
        "pair": figures_dir / "loo-pair-impacts.png",
    }
    for level, output_path in plots.items():
        created = plot_top_impacts(summary, level, output_path, args.top)
        if created:
            print(f"Wrote {output_path}")

    redistribution_plots = {
        "juror": figures_dir / "loo-juror-redistribution.png",
        "repository": figures_dir / "loo-repository-redistribution.png",
    }
    for level, output_path in redistribution_plots.items():
        created = plot_total_redistribution(summary, level, output_path, args.top)
        if created:
            print(f"Wrote {output_path}")

    # Print baseline leaders for quick context.
    top_baseline = baseline.sort_values(ascending=False).head(5)
    print(
        "\nTop baseline repositories:\n"
        + "\n".join(
            f"  {format_repo_label(repo)}: {weight:.3f}"
            for repo, weight in top_baseline.items()
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
