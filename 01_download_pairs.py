#!/usr/bin/env -S uv run --script

import argparse
import io
import sys
import zipfile
from pathlib import Path
from typing import Sequence
from urllib.request import urlopen

import pandas as pd

DATA_URL = "https://pond-open-files.s3.amazonaws.com/frontier/others/BlButrKh/dataset_Oct_9_2025.zip"
DEFAULT_OUTPUT = Path("data/pairs.csv")
TARGET_FILES = {
    "train": "train.csv",
    "raw_private": "raw_private.csv",
    "sample": "sample_submission.csv",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the DeepFunding train data and merge public and private comparisons."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination CSV path (default: data/pairs.csv).",
    )
    return parser.parse_args(argv)


def download_archive(url: str) -> zipfile.ZipFile:
    with urlopen(url) as response:
        payload = io.BytesIO(response.read())
    return zipfile.ZipFile(payload)


def locate_member(archive: zipfile.ZipFile, name: str) -> str:
    for member in archive.namelist():
        if member.endswith(name) and not member.startswith("__MACOSX"):
            return member
    msg = f"Could not find {name} inside archive."
    raise FileNotFoundError(msg)


def load_frames(
    archive: zipfile.ZipFile,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    members = {
        key: locate_member(archive, target) for key, target in TARGET_FILES.items()
    }
    with archive.open(members["train"]) as train_file:
        train = pd.read_csv(train_file)
    with archive.open(members["raw_private"]) as private_file:
        raw_private = pd.read_csv(private_file)
    with archive.open(members["sample"]) as sample_file:
        sample_submission = pd.read_csv(sample_file)
    return train, raw_private, sample_submission


def merge_frames(train: pd.DataFrame, raw_private: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for frame in (train, raw_private):
        if "reasoning" in frame.columns:
            frame = frame.drop(columns="reasoning")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(ignore_index=True)
    if "timestamp" in combined.columns:
        combined = combined.sort_values(
            "timestamp", kind="mergesort", ignore_index=True
        )
    return combined


def filter_by_sample(
    combined: pd.DataFrame, sample_submission: pd.DataFrame
) -> tuple[pd.DataFrame, set[str], int]:
    valid_repos = set(sample_submission["repo"].dropna().unique())
    existing_repos = set(
        combined["repo_a"].dropna().unique()
    ) | set(combined["repo_b"].dropna().unique())
    mask = combined["repo_a"].isin(valid_repos) & combined["repo_b"].isin(valid_repos)
    filtered = combined.loc[mask].reset_index(drop=True)
    removed_pairs = len(combined) - len(filtered)
    removed_repos = existing_repos - valid_repos
    return filtered, removed_repos, removed_pairs


def export_frame(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    with download_archive(DATA_URL) as archive:
        train, raw_private, sample_submission = load_frames(archive)
    combined = merge_frames(train, raw_private)
    combined, removed_repos, removed_pairs = filter_by_sample(
        combined, sample_submission
    )
    export_frame(combined, args.output)
    print(
        f"Filtered out {len(removed_repos)} repos and {removed_pairs} pairs "
        "not present in sample submission.",
        file=sys.stderr,
    )
    print(f"Wrote {len(combined)} rows to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
