#!/usr/bin/env -S uv run --script

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from plotting import ensure_directory, histogram, horizontal_bar

DEFAULT_INPUT = Path("data/pairs.csv")
DEFAULT_FIGURES_DIR = Path("figures")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize DeepFunding comparison pairs."
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
    expected = {"juror", "repo_a", "repo_b"}
    missing = expected.difference(frame.columns)
    if missing:
        msg = f"Missing expected columns: {', '.join(sorted(missing))}."
        raise ValueError(msg)
    return frame


def compute_totals(
    frame: pd.DataFrame, *, pair_table: pd.DataFrame | None = None
) -> dict[str, int]:
    totals: dict[str, int] = {"comparisons": int(len(frame))}
    totals["jurors"] = int(frame["juror"].nunique(dropna=True))
    all_repos = pd.Series(
        pd.concat([frame["repo_a"], frame["repo_b"]], ignore_index=True)
    )
    totals["repositories"] = int(all_repos.nunique(dropna=True))
    if pair_table is None:
        pair_table = per_repo_pair(frame)
    totals["repo_pairs"] = int(len(pair_table))
    if "parent" in frame.columns:
        totals["parents"] = int(frame["parent"].nunique(dropna=True))
    if "choice" in frame.columns:
        totals["choices"] = int(frame["choice"].nunique(dropna=True))
    return totals


def per_juror(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame["juror"].value_counts().rename_axis("juror").rename("comparisons")
    return counts.reset_index().sort_values(
        "comparisons", ascending=False, ignore_index=True
    )


def per_repository(frame: pd.DataFrame) -> pd.DataFrame:
    repos = pd.concat(
        [frame["repo_a"], frame["repo_b"]],
        ignore_index=True,
        axis=0,
    )
    counts = repos.value_counts().rename_axis("repository").rename("appearances")
    return counts.reset_index().sort_values(
        "appearances", ascending=False, ignore_index=True
    )


def per_repo_pair(frame: pd.DataFrame) -> pd.DataFrame:
    pair_df = pd.DataFrame({
        "repo_left": frame[["repo_a", "repo_b"]].min(axis=1),
        "repo_right": frame[["repo_a", "repo_b"]].max(axis=1),
    })
    grouped = (
        pair_df.groupby(["repo_left", "repo_right"], sort=False)
        .size()
        .rename("comparisons")
        .reset_index()
        .sort_values("comparisons", ascending=False, ignore_index=True)
    )
    return grouped


def format_repo_label(value: str) -> str:
    if isinstance(value, str) and value.startswith("https://github.com/"):
        return value.removeprefix("https://github.com/")
    return value

def write_charts(
    juror_table: pd.DataFrame,
    repository_table: pd.DataFrame,
    pair_table: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    ensure_directory(figures_dir)
    outputs: list[Path] = []
    juror_path = figures_dir / "comparisons-per-juror.png"
    if horizontal_bar(
        juror_table,
        label_col="juror",
        value_col="comparisons",
        title="Comparisons per juror (top 10)",
        destination=juror_path,
        top=10,
        xlabel="Comparisons",
    ):
        outputs.append(juror_path)

    juror_hist_path = figures_dir / "comparisons-per-juror-hist.png"
    if histogram(
        juror_table["comparisons"],
        title="Comparisons per juror (distribution)",
        destination=juror_hist_path,
        xlabel="Comparisons per juror",
        ylabel="Jurors",
    ):
        outputs.append(juror_hist_path)

    repo_display = repository_table.assign(
        repository_label=repository_table["repository"].map(format_repo_label)
    )
    repo_path = figures_dir / "appearances-per-repository.png"
    if horizontal_bar(
        repo_display,
        label_col="repository_label",
        value_col="appearances",
        title="Repository appearances (top 10)",
        destination=repo_path,
        top=10,
        xlabel="Appearances",
    ):
        outputs.append(repo_path)

    pairs_path = figures_dir / "comparisons-per-repo-pair.png"
    if histogram(
        pair_table["comparisons"],
        title="Comparisons per repository pair (distribution)",
        destination=pairs_path,
        xlabel="Comparisons per pair",
        ylabel="Repository pairs",
        annotate=True,
    ):
        outputs.append(pairs_path)
    return outputs


def print_totals(totals: dict[str, int]) -> None:
    print("Totals")
    for key, value in totals.items():
        print(f"- {key}: {value}")
    print()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    frame = load_pairs(args.csv)
    pair_table = per_repo_pair(frame)
    totals = compute_totals(frame, pair_table=pair_table)
    juror_table = per_juror(frame)
    repository_table = per_repository(frame)
    figures_dir = ensure_directory(args.figures)

    print_totals(totals)
    chart_paths = write_charts(
        juror_table=juror_table,
        repository_table=repository_table,
        pair_table=pair_table,
        figures_dir=figures_dir,
    )
    if chart_paths:
        print("Charts saved:")
        for path in chart_paths:
            print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
