#!/usr/bin/env -S uv run --script

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from weights import DEFAULT_INPUT, clean_pairs, load_pairs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score candidate repository weights against DeepFunding pairwise comparisons."
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to the comparison pairs CSV (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        action="append",
        required=True,
        help="Path to a candidate weights CSV. Provide multiple times to score several files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        raw_pairs = load_pairs(args.pairs)
    except (OSError, ValueError) as exc:
        print(f"Failed to read pairs data: {exc}", file=sys.stderr)
        return 1

    clean = clean_pairs(raw_pairs)
    ground_truth = _ground_truth_probabilities(clean)
    truth_weights = _ground_truth_weights(clean)
    baseline_brier = float(np.mean((0.5 - ground_truth) ** 2))

    print(f"Pairs evaluated: {len(clean)}")

    exit_code = 0
    seen: set[Path] = set()
    for weights_path in args.weights:
        resolved = weights_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            weights = _load_candidate_weights(resolved)
            metrics = _score_weights(clean, ground_truth, truth_weights, weights)
        except (OSError, ValueError) as exc:
            print(f"[ERROR] {weights_path}: {exc}", file=sys.stderr)
            exit_code = 1
            continue

        brier = metrics["brier"]
        coverage = metrics["coverage"]
        log_loss = metrics["log_loss"]
        log_odds_rmse = metrics["log_odds_rmse"]
        accuracy = metrics["pairwise_accuracy"]
        weight_tv = metrics["weight_tv_distance"]
        skill = (
            float("nan")
            if baseline_brier <= 0.0
            else 1.0 - (brier / baseline_brier)
        )
        print(
            f"{weights_path}: "
            f"Brier={brier:.6f} "
            f"LogLoss={log_loss:.6f} "
            f"LogOddsRMSE={log_odds_rmse:.6f} "
            f"Accuracy={accuracy:.3f} "
            f"Coverage={coverage:.3f} "
            f"Skill={skill:.4f} "
            f"TV={weight_tv:.3f}"
        )

    return exit_code


def _ground_truth_probabilities(clean: pd.DataFrame) -> np.ndarray:
    choice = clean["choice"].to_numpy(dtype=int)
    multiplier = clean["multiplier"].to_numpy(dtype=float)
    log_ratio = np.log(multiplier)
    log_ratio[choice == 1] *= -1.0
    log_ratio = np.clip(log_ratio, -60.0, 60.0)
    return _sigmoid(log_ratio)


def _ground_truth_weights(clean: pd.DataFrame) -> pd.Series:
    repos = pd.Index(
        pd.unique(
            pd.concat([clean["repo_a"], clean["repo_b"]], ignore_index=True)
        )
    )
    if repos.empty:
        raise ValueError("No repositories available to derive ground truth weights")

    repo_to_index = {repo: idx for idx, repo in enumerate(repos)}
    comparisons = len(clean)
    design = np.zeros((comparisons, len(repos)), dtype=float)
    for row_idx, (repo_a, repo_b) in enumerate(
        clean[["repo_a", "repo_b"]].itertuples(index=False, name=None)
    ):
        design[row_idx, repo_to_index[repo_b]] = 1.0
        design[row_idx, repo_to_index[repo_a]] = -1.0

    log_ratio = np.log(clean["multiplier"].to_numpy(dtype=float))
    choices = clean["choice"].to_numpy(dtype=int)
    log_ratio[choices == 1] *= -1.0
    logits, *_ = np.linalg.lstsq(design, log_ratio, rcond=None)
    logits -= logits.mean()
    shifted = logits - logits.max()
    positive = np.exp(shifted)
    total = positive.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Failed to compute positive ground truth weights")
    distribution = positive / total
    return pd.Series(distribution, index=repos)


def _load_candidate_weights(path: Path) -> pd.Series:
    if not path.is_file():
        msg = f"weights file {path} does not exist"
        raise FileNotFoundError(msg)
    frame = pd.read_csv(path)
    if "repo" not in frame.columns or "weight" not in frame.columns:
        msg = "weights file must include 'repo' and 'weight' columns"
        raise ValueError(msg)

    weights = (
        frame.dropna(subset=["repo", "weight"])
        .assign(weight=lambda df: pd.to_numeric(df["weight"], errors="coerce"))
        .dropna(subset=["weight"])
    )
    if weights.empty:
        raise ValueError("weights file contains no valid rows")

    grouped = weights.groupby("repo", sort=False)["weight"].sum()
    positive = grouped[grouped > 0.0]
    if positive.empty:
        raise ValueError("weights must contain positive mass")

    normalized = positive / positive.sum()
    return normalized


def _score_weights(
    clean: pd.DataFrame,
    truth_probs: np.ndarray,
    truth_weights: pd.Series,
    weights: pd.Series,
) -> dict[str, float]:
    w_a = weights.reindex(clean["repo_a"]).fillna(0.0).to_numpy(dtype=float)
    w_b = weights.reindex(clean["repo_b"]).fillna(0.0).to_numpy(dtype=float)

    total = w_a + w_b
    has_mass = total > 0.0
    raw_preds = np.where(has_mass, w_b / total, 0.5)
    preds = np.clip(raw_preds, 1e-12, 1.0 - 1e-12)
    truth_probs = np.clip(truth_probs, 1e-12, 1.0 - 1e-12)

    brier = float(np.mean((raw_preds - truth_probs) ** 2))
    log_loss = float(
        -np.mean(
            truth_probs * np.log(preds)
            + (1.0 - truth_probs) * np.log(1.0 - preds)
        )
    )
    log_odds_error = _safe_logit(preds) - _safe_logit(truth_probs)
    log_odds_rmse = float(np.sqrt(np.mean(log_odds_error**2)))

    actual_b = clean["choice"].to_numpy(dtype=int) == 2
    ties = np.isclose(raw_preds, 0.5)
    pairwise_correct = (raw_preds > 0.5) == actual_b
    pairwise_scores = pairwise_correct.astype(float)
    pairwise_scores[ties] = 0.5
    pairwise_accuracy = float(pairwise_scores.mean())

    coverage = float(np.mean(has_mass))
    weight_tv_distance = _total_variation_distance(truth_weights, weights)

    return {
        "brier": brier,
        "coverage": coverage,
        "log_loss": log_loss,
        "log_odds_rmse": log_odds_rmse,
        "pairwise_accuracy": pairwise_accuracy,
        "weight_tv_distance": weight_tv_distance,
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _safe_logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-12, 1.0 - 1e-12)
    return np.log(clipped / (1.0 - clipped))


def _total_variation_distance(truth: pd.Series, candidate: pd.Series) -> float:
    union = truth.index.union(candidate.index)
    truth_aligned = truth.reindex(union, fill_value=0.0).to_numpy(dtype=float)
    candidate_aligned = candidate.reindex(union, fill_value=0.0).to_numpy(dtype=float)
    truth_sum = truth_aligned.sum()
    cand_sum = candidate_aligned.sum()
    if truth_sum <= 0.0 or cand_sum <= 0.0:
        return float("nan")
    truth_normalized = truth_aligned / truth_sum
    candidate_normalized = candidate_aligned / cand_sum
    distance = 0.5 * np.sum(np.abs(truth_normalized - candidate_normalized))
    return float(distance)


if __name__ == "__main__":
    raise SystemExit(main())
