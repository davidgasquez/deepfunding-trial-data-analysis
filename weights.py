from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

DEFAULT_INPUT = Path("data/pairs.csv")
EXPECTED_COLUMNS = {"repo_a", "repo_b", "choice", "multiplier"}

WeightMethod = Literal[
    "least-squares",
    "bradley-terry",
    "bradley-terry-regularized",
    "bradley-terry-intensity",
    "huber-log",
    "elo",
    "pagerank",
]

DEFAULT_METHOD: WeightMethod = "least-squares"
WEIGHT_METHODS: tuple[WeightMethod, ...] = (
    "least-squares",
    "bradley-terry",
    "bradley-terry-regularized",
    "bradley-terry-intensity",
    "huber-log",
    "elo",
    "pagerank",
)

_MAD_SCALE = 0.6744897501960817
_RIDGE_EPS = 1e-6
_MAX_IRLS_ITER = 75
_IRLS_TOL = 1e-8
_BT_REGULARIZATION = 1e-2
_ELO_K = 16.0
_PAGERANK_DAMPING = 0.85
_PAGERANK_TOL = 1e-10
_PAGERANK_MAX_ITER = 200


def load_pairs(path: Path) -> pd.DataFrame:
    if not path.is_file():
        msg = f"Input file {path} does not exist."
        raise FileNotFoundError(msg)
    frame = pd.read_csv(path)
    missing = EXPECTED_COLUMNS.difference(frame.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        msg = f"Missing expected columns: {missing_columns}."
        raise ValueError(msg)
    return frame


def clean_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.dropna(subset=list(EXPECTED_COLUMNS)).copy()
    clean["choice"] = pd.to_numeric(clean["choice"], errors="coerce").astype("Int64")
    clean["multiplier"] = pd.to_numeric(clean["multiplier"], errors="coerce")
    clean = clean[clean["choice"].isin([1, 2]) & (clean["multiplier"] > 0)]
    if clean.empty:
        raise ValueError("No valid comparisons available after cleaning.")
    return clean.reset_index(drop=True)


def design_matrix(
    clean: pd.DataFrame, repos: pd.Index
) -> tuple[np.ndarray, np.ndarray]:
    repo_to_index = {repo: idx for idx, repo in enumerate(repos)}
    comparisons = len(clean)
    design = np.zeros((comparisons, len(repos)), dtype=float)
    ratios = np.log(clean["multiplier"].to_numpy(dtype=float))
    choices = clean["choice"].to_numpy(dtype=int)
    ratios[choices == 1] *= -1

    for row_idx, (repo_a, repo_b) in enumerate(
        clean[["repo_a", "repo_b"]].itertuples(index=False, name=None)
    ):
        design[row_idx, repo_to_index[repo_b]] = 1.0
        design[row_idx, repo_to_index[repo_a]] = -1.0

    return design, ratios


def solve_logits(design: np.ndarray, ratios: np.ndarray) -> np.ndarray:
    solution, *_ = np.linalg.lstsq(design, ratios, rcond=None)
    return solution


def normalize_logits(logits: np.ndarray) -> np.ndarray:
    if logits.size == 0:
        return logits
    shifted = logits - logits.max()
    positive = np.exp(shifted)
    total = positive.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Failed to compute finite weights from logits.")
    return positive / total


def compute_weights(
    csv_path: Path = DEFAULT_INPUT, *, method: WeightMethod = DEFAULT_METHOD
) -> pd.DataFrame:
    """Return repository weights derived from pairwise comparisons."""
    frame = load_pairs(csv_path)
    clean = clean_pairs(frame)
    repos = _unique_repos(clean)
    try:
        solver = _METHOD_BUILDERS[method]
    except KeyError as exc:
        expected = ", ".join(WEIGHT_METHODS)
        msg = f"Unknown method {method!r}. Expected one of: {expected}."
        raise ValueError(msg) from exc
    logits = solver(clean, repos)
    return _finalize_weights(repos, logits)


def _least_squares_logits(clean: pd.DataFrame, repos: pd.Index) -> np.ndarray:
    design, ratios = design_matrix(clean, repos)
    logits = solve_logits(design, ratios)
    return logits


def _solve_bradley_terry(
    clean: pd.DataFrame, repos: pd.Index, *, use_intensity: bool, ridge: float
) -> np.ndarray:
    design, _ = design_matrix(clean, repos)
    outcomes = (clean["choice"].to_numpy(dtype=int) == 2).astype(float)
    sample_weights = (
        clean["multiplier"].to_numpy(dtype=float) if use_intensity else np.ones_like(outcomes)
    )
    sample_weights = np.clip(sample_weights, 1e-6, None)
    sample_weights = sample_weights / sample_weights.mean()
    logits = _logistic_irls(design, outcomes, sample_weights, ridge=ridge)
    return logits


def _bradley_terry_logits(clean: pd.DataFrame, repos: pd.Index) -> np.ndarray:
    return _solve_bradley_terry(clean, repos, use_intensity=False, ridge=_RIDGE_EPS)


def _bradley_terry_regularized_logits(
    clean: pd.DataFrame, repos: pd.Index
) -> np.ndarray:
    ridge = max(_BT_REGULARIZATION, _RIDGE_EPS)
    return _solve_bradley_terry(clean, repos, use_intensity=False, ridge=ridge)


def _bradley_terry_intensity_logits(
    clean: pd.DataFrame, repos: pd.Index
) -> np.ndarray:
    ridge = max(_RIDGE_EPS, _BT_REGULARIZATION * 0.5)
    return _solve_bradley_terry(clean, repos, use_intensity=True, ridge=ridge)


def _logistic_irls(
    design: np.ndarray,
    targets: np.ndarray,
    sample_weights: np.ndarray,
    *,
    max_iter: int = _MAX_IRLS_ITER,
    tol: float = _IRLS_TOL,
    ridge: float,
) -> np.ndarray:
    logits = np.zeros(design.shape[1], dtype=float)
    identity = np.eye(design.shape[1], dtype=float)

    for _ in range(max_iter):
        eta = design @ logits
        eta = np.clip(eta, -30.0, 30.0)
        probs = 1.0 / (1.0 + np.exp(-eta))
        probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
        var = probs * (1.0 - probs)
        working = np.maximum(var, 1e-12)
        weights = sample_weights * working
        if np.all(weights <= 1e-12):
            break
        z = eta + (targets - probs) / working
        lhs = design.T @ (weights[:, None] * design) + identity * max(ridge, _RIDGE_EPS)
        rhs = design.T @ (weights * z)
        updated = np.linalg.solve(lhs, rhs)
        updated -= updated.mean()
        if np.linalg.norm(updated - logits) <= tol * (1.0 + np.linalg.norm(logits)):
            logits = updated
            break
        logits = updated
    else:
        logits -= logits.mean()

    return logits


def _huber_llsm_logits(clean: pd.DataFrame, repos: pd.Index) -> np.ndarray:
    design, ratios = design_matrix(clean, repos)
    delta = _huber_delta(ratios)
    logits = np.zeros(design.shape[1], dtype=float)
    identity = np.eye(design.shape[1], dtype=float)

    for _ in range(_MAX_IRLS_ITER):
        residual = ratios - design @ logits
        weights = _huber_weights(residual, delta)
        lhs = design.T @ (weights[:, None] * design) + identity * _RIDGE_EPS
        rhs = design.T @ (weights * ratios)
        updated = np.linalg.solve(lhs, rhs)
        updated -= updated.mean()
        if np.linalg.norm(updated - logits) <= _IRLS_TOL * (1.0 + np.linalg.norm(logits)):
            logits = updated
            break
        logits = updated
    else:
        logits -= logits.mean()

    return logits


def _elo_logits(clean: pd.DataFrame, repos: pd.Index) -> np.ndarray:
    repo_to_index = {repo: idx for idx, repo in enumerate(repos)}
    ratings = np.zeros(len(repos), dtype=float)

    ordered = clean.copy()
    if "timestamp" in ordered.columns:
        ordered = ordered.assign(
            _timestamp=pd.to_datetime(ordered["timestamp"], errors="coerce")
        )
        ordered = ordered.sort_values("_timestamp", kind="stable").drop(columns=["_timestamp"])

    for row in ordered.itertuples(index=False):
        repo_a = getattr(row, "repo_a")
        repo_b = getattr(row, "repo_b")
        choice = int(getattr(row, "choice"))
        multiplier = float(getattr(row, "multiplier"))
        if repo_a not in repo_to_index or repo_b not in repo_to_index:
            continue
        if choice == 1:
            winner_idx, loser_idx = repo_to_index[repo_a], repo_to_index[repo_b]
        else:
            winner_idx, loser_idx = repo_to_index[repo_b], repo_to_index[repo_a]
        winner_rating = ratings[winner_idx]
        loser_rating = ratings[loser_idx]
        diff = winner_rating - loser_rating
        expected = 1.0 / (1.0 + np.exp(-diff))
        scale = _elo_scale(multiplier)
        delta = scale * (1.0 - expected)
        ratings[winner_idx] = winner_rating + delta
        ratings[loser_idx] = loser_rating - delta

    ratings -= ratings.mean()
    scale = np.std(ratings)
    if np.isfinite(scale) and scale > 1.0:
        ratings /= scale
    return ratings


def _elo_scale(multiplier: float) -> float:
    return _ELO_K


def _pagerank_logits(clean: pd.DataFrame, repos: pd.Index) -> np.ndarray:
    repo_to_index = {repo: idx for idx, repo in enumerate(repos)}
    n = len(repos)
    adjacency = np.zeros((n, n), dtype=float)

    for choice, repo_a, repo_b, multiplier in clean[["choice", "repo_a", "repo_b", "multiplier"]].itertuples(index=False, name=None):
        if repo_a not in repo_to_index or repo_b not in repo_to_index:
            continue
        winner_idx: int
        loser_idx: int
        if int(choice) == 1:
            winner_idx = repo_to_index[repo_a]
            loser_idx = repo_to_index[repo_b]
        else:
            winner_idx = repo_to_index[repo_b]
            loser_idx = repo_to_index[repo_a]
        weight = float(multiplier)
        if not np.isfinite(weight) or weight <= 0:
            weight = 1.0
        adjacency[loser_idx, winner_idx] += weight

    row_sums = adjacency.sum(axis=1, keepdims=True)
    transition = np.divide(
        adjacency,
        np.where(row_sums <= 0, 1.0, row_sums),
        out=np.full_like(adjacency, 1.0 / n),
        where=row_sums > 0,
    )
    rank = np.full(n, 1.0 / n, dtype=float)

    teleport = (1.0 - _PAGERANK_DAMPING) / n
    for _ in range(_PAGERANK_MAX_ITER):
        updated = teleport + _PAGERANK_DAMPING * transition.T @ rank
        if np.linalg.norm(updated - rank, ord=1) <= _PAGERANK_TOL:
            rank = updated
            break
        rank = updated

    rank = np.clip(rank, 1e-12, None)
    logits = np.log(rank)
    logits -= logits.mean()
    return logits


def _huber_delta(values: np.ndarray) -> float:
    mad = _median_absolute_deviation(values)
    if not np.isfinite(mad) or mad <= 0:
        mad = np.mean(np.abs(values - np.mean(values))) if values.size else 1.0
    if not np.isfinite(mad) or mad <= 0:
        mad = 1.0
    sigma = mad / _MAD_SCALE
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return float(1.345 * sigma)


def _median_absolute_deviation(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def _huber_weights(residual: np.ndarray, delta: float) -> np.ndarray:
    weights = np.ones_like(residual, dtype=float)
    abs_residual = np.abs(residual)
    mask = abs_residual > delta
    weights[mask] = delta / np.maximum(abs_residual[mask], 1e-12)
    return weights


def _unique_repos(clean: pd.DataFrame) -> pd.Index:
    repos_series = pd.concat(
        [clean["repo_a"], clean["repo_b"]], ignore_index=True
    ).dropna()
    repos = pd.Index(pd.unique(repos_series))
    if repos.empty:
        raise ValueError("No repositories present in the input data.")
    return repos


def _finalize_weights(repos: pd.Index, logits: np.ndarray) -> pd.DataFrame:
    weights = normalize_logits(logits)
    if np.any(~np.isfinite(weights)):
        raise ValueError("Computed weights contain non-finite values.")
    return (
        pd.DataFrame({"repo": repos, "weight": weights})
        .sort_values("weight", ascending=False, ignore_index=True)
    )


_METHOD_BUILDERS: dict[WeightMethod, Callable[[pd.DataFrame, pd.Index], np.ndarray]] = {
    "least-squares": _least_squares_logits,
    "bradley-terry": _bradley_terry_logits,
    "bradley-terry-regularized": _bradley_terry_regularized_logits,
    "bradley-terry-intensity": _bradley_terry_intensity_logits,
    "huber-log": _huber_llsm_logits,
    "elo": _elo_logits,
    "pagerank": _pagerank_logits,
}


__all__ = [
    "DEFAULT_INPUT",
    "DEFAULT_METHOD",
    "EXPECTED_COLUMNS",
    "WEIGHT_METHODS",
    "WeightMethod",
    "clean_pairs",
    "compute_weights",
    "design_matrix",
    "load_pairs",
    "normalize_logits",
    "solve_logits",
]
