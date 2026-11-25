"""Simple endpoint-only warning rules. Threshold calibration is a separate validation step."""

from __future__ import annotations

import math
from collections.abc import Sequence


def threshold_rule(
    values: Sequence[float | None], *, threshold: float, direction: str = "above"
) -> list[bool | None]:
    if not math.isfinite(threshold) or direction not in {"above", "below"}:
        raise ValueError("threshold must be finite and direction must be above/below")
    return [
        None
        if value is None
        else (value >= threshold if direction == "above" else value <= threshold)
        for value in values
    ]


def ewma_scores(
    values: Sequence[float | None], run_ids: Sequence[str], *, alpha: float
) -> list[float | None]:
    """Causal EWMA; resets at run boundaries and missing observations."""
    if len(values) != len(run_ids):
        raise ValueError("values and run_ids must have equal length")
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("EWMA alpha must be finite and in (0,1]")
    scores: list[float | None] = []
    previous_run: str | None = None
    state: float | None = None
    for value, run_id in zip(values, run_ids):
        if run_id != previous_run:
            state = None
            previous_run = run_id
        if value is None:
            state = None
            scores.append(None)
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("EWMA input values must be finite or None")
        state = numeric if state is None else alpha * numeric + (1 - alpha) * state
        scores.append(state)
    return scores


def predict_ewma(
    scores: Sequence[float | None], *, threshold: float, direction: str = "above"
) -> list[bool | None]:
    return threshold_rule(scores, threshold=threshold, direction=direction)
