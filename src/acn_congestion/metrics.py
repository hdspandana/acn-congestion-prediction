"""Binary-window and event-level forecasting metrics with explicit undefined cases."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from itertools import pairwise
from statistics import mean, median
from typing import Any


class MetricError(ValueError):
    """Invalid metric inputs or inconsistent event/forecast timelines."""


def _finite_pairs(
    y_true: Sequence[int | None], scores: Sequence[float | None]
) -> tuple[list[int], list[float], int]:
    if len(y_true) != len(scores):
        raise MetricError("labels and scores must have equal length")
    labels: list[int] = []
    values: list[float] = []
    omitted = 0
    for label, score in zip(y_true, scores):
        if label is None or score is None:
            omitted += 1
            continue
        if label not in (0, 1) or isinstance(label, bool):
            raise MetricError("non-censored labels must be integer 0 or 1")
        value = float(score)
        if not math.isfinite(value):
            raise MetricError("scores must be finite")
        labels.append(label)
        values.append(value)
    return labels, values, omitted


def _average_precision(y: list[int], score: list[float]) -> float | None:
    positives = sum(y)
    if positives == 0 or positives == len(y):
        return None
    ordering = sorted(range(len(y)), key=lambda i: score[i], reverse=True)
    tp = fp = 0
    previous_recall = 0.0
    area = 0.0
    cursor = 0
    while cursor < len(ordering):
        end = cursor + 1
        threshold_score = score[ordering[cursor]]
        while end < len(ordering) and score[ordering[end]] == threshold_score:
            end += 1
        for pos in ordering[cursor:end]:
            tp += y[pos]
            fp += 1 - y[pos]
        recall = tp / positives
        precision = tp / (tp + fp)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        cursor = end
    return area


def _roc_auc(y: list[int], score: list[float]) -> float | None:
    positives = sum(y)
    negatives = len(y) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(score, y), key=lambda pair: pair[0])
    rank_sum = 0.0
    rank = 1
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][0] == ordered[cursor][0]:
            end += 1
        average_rank = (rank + rank + (end - cursor) - 1) / 2
        rank_sum += average_rank * sum(label for _, label in ordered[cursor:end])
        rank += end - cursor
        cursor = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def binary_metrics(
    y_true: Sequence[int | None],
    scores: Sequence[float | None],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Score point labels; censored rows are counted and omitted, never negatives."""
    if not math.isfinite(threshold):
        raise MetricError("threshold must be finite")
    y, values, omitted = _finite_pairs(y_true, scores)
    predicted = [int(value >= threshold) for value in values]
    tp = sum(a == 1 and b == 1 for a, b in zip(y, predicted))
    fp = sum(a == 0 and b == 1 for a, b in zip(y, predicted))
    fn = sum(a == 1 and b == 0 for a, b in zip(y, predicted))
    tn = sum(a == 0 and b == 0 for a, b in zip(y, predicted))
    positives = tp + fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / positives if positives else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "n_scored": len(y),
        "n_censored_or_missing": omitted,
        "class_prevalence": positives / len(y) if y else None,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_precision": _average_precision(y, values),
        "roc_auc": _roc_auc(y, values),
        "threshold": threshold,
        "undefined_conventions": "Precision/recall/AUC are null when their denominators/classes are absent; censored rows are excluded and counted.",
    }


def score_calibration_metrics(
    y_true: Sequence[int | None],
    scores: Sequence[float | None],
    *,
    bin_count: int = 10,
    epsilon: float = 1e-15,
) -> dict[str, Any]:
    """Descriptive probability-quality metrics for scores already in [0, 1].

    This does **not** fit a calibrator and does not convert an arbitrary
    ranking score into a probability.  Reliability-bin summaries are row-level
    and can be highly correlated across overlapping forecast horizons, so the
    caller must report them as descriptive diagnostics rather than proof of
    calibrated operational probabilities.
    """
    if isinstance(bin_count, bool) or not isinstance(bin_count, int) or not 2 <= bin_count <= 50:
        raise MetricError("bin_count must be an integer in [2, 50]")
    if not math.isfinite(epsilon) or not 0 < epsilon < 0.5:
        raise MetricError("epsilon must be finite and in (0, 0.5)")
    y, values, omitted = _finite_pairs(y_true, scores)
    counts = {
        "n_scored": len(y),
        "n_censored_or_missing": omitted,
        "positive_count": sum(y),
        "negative_count": len(y) - sum(y),
    }
    if not y:
        return {
            "estimable": False,
            "reason": "no labeled finite scores",
            **counts,
            "reliability_bins": [],
        }
    if any(value < 0 or value > 1 for value in values):
        return {
            "estimable": False,
            "reason": "scores are not probability-like values in [0, 1]",
            **counts,
            "reliability_bins": [],
        }
    if counts["positive_count"] == 0 or counts["negative_count"] == 0:
        return {
            "estimable": False,
            "reason": "calibration diagnostics require both labeled classes",
            **counts,
            "mean_score": mean(values),
            "observed_positive_rate": mean(y),
            "reliability_bins": [],
        }
    brier = mean((value - label) ** 2 for label, value in zip(y, values))
    log_loss = -mean(
        label * math.log(min(max(value, epsilon), 1 - epsilon))
        + (1 - label) * math.log(min(max(1 - value, epsilon), 1 - epsilon))
        for label, value in zip(y, values)
    )
    bins: list[dict[str, Any]] = []
    expected_calibration_error = 0.0
    maximum_calibration_error = 0.0
    for bin_index in range(bin_count):
        lower = bin_index / bin_count
        upper = (bin_index + 1) / bin_count
        members = [
            (label, value)
            for label, value in zip(y, values)
            if lower <= value < upper or (bin_index == bin_count - 1 and value == 1.0)
        ]
        if not members:
            bins.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "mean_score": None,
                    "observed_positive_rate": None,
                    "absolute_gap": None,
                }
            )
            continue
        mean_score = mean(value for _, value in members)
        observed_rate = mean(label for label, _ in members)
        gap = abs(mean_score - observed_rate)
        expected_calibration_error += len(members) / len(y) * gap
        maximum_calibration_error = max(maximum_calibration_error, gap)
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "mean_score": mean_score,
                "observed_positive_rate": observed_rate,
                "absolute_gap": gap,
            }
        )
    return {
        "estimable": True,
        "reason": None,
        **counts,
        "mean_score": mean(values),
        "observed_positive_rate": mean(y),
        "calibration_in_large_difference": mean(values) - mean(y),
        "brier_score": brier,
        "log_loss": log_loss,
        "log_loss_epsilon": epsilon,
        "expected_calibration_error": expected_calibration_error,
        "maximum_calibration_error": maximum_calibration_error,
        "binning": f"{bin_count} equal-width bins over [0, 1]; final bin includes 1.0",
        "reliability_bins": bins,
        "interpretation": "Descriptive row-level diagnostics only; no calibrator was fitted and scores remain uncalibrated unless a separate validated calibration procedure is recorded.",
    }


def select_threshold_at_max_fpr(
    y_true: Sequence[int | None],
    scores: Sequence[float | None],
    *,
    max_false_positive_rate: float,
) -> dict[str, float | int]:
    """Select the highest-recall validation threshold under a predeclared FPR cap."""
    if not 0 <= max_false_positive_rate <= 1:
        raise MetricError("max_false_positive_rate must be in [0,1]")
    y, values, _ = _finite_pairs(y_true, scores)
    negatives = sum(label == 0 for label in y)
    if not negatives or not any(label == 1 for label in y):
        raise MetricError(
            "threshold selection requires both positive and negative validation examples"
        )
    candidates = sorted(set(values), reverse=True) + [math.nextafter(max(values), math.inf)]
    best: tuple[float, float, int] | None = None
    for threshold in candidates:
        predicted = [score >= threshold for score in values]
        fp = sum(label == 0 and pred for label, pred in zip(y, predicted))
        tp = sum(label == 1 and pred for label, pred in zip(y, predicted))
        fpr, recall = fp / negatives, tp / sum(y)
        if fpr <= max_false_positive_rate:
            candidate = (recall, -fpr, threshold)
            if best is None or candidate > best:
                best = candidate
    assert best is not None
    threshold = best[2]
    return {
        "threshold": threshold,
        "validation_fpr": -best[1],
        "validation_recall": best[0],
        "validation_examples": len(y),
    }


def event_forecast_metrics(
    timestamps_seconds: Sequence[float],
    predictions: Sequence[bool | None],
    event_onsets_seconds: Sequence[float],
    *,
    horizon_seconds: float,
    eligible_exposure_seconds: float,
    alarm_merge_gap_seconds: float,
    event_active: Sequence[bool | None] | None = None,
    episode_breaks: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Match each pre-event alarm episode to at most one future onset.

    Alarms at/after an active-event sample are reported as detections and do not
    earn forecast credit. Episode merging retains the first alarm time. Onsets
    are matched chronologically when ``alarm_time < onset <= alarm_time+H``.
    Exposure is supplied by the caller from eligible non-event observation time.
    """
    n = len(timestamps_seconds)
    if len(predictions) != n or (event_active is not None and len(event_active) != n):
        raise MetricError(
            "timestamps, predictions, and optional event states must have equal length"
        )
    if episode_breaks is not None and len(episode_breaks) != n:
        raise MetricError("episode_breaks must have the same length as timestamps")
    if episode_breaks is not None and any(not isinstance(value, bool) for value in episode_breaks):
        raise MetricError("episode_breaks must contain booleans")
    if not math.isfinite(horizon_seconds) or horizon_seconds <= 0:
        raise MetricError("horizon_seconds must be finite and positive")
    if not math.isfinite(alarm_merge_gap_seconds) or alarm_merge_gap_seconds < 0:
        raise MetricError("alarm_merge_gap_seconds must be finite and non-negative")
    if not math.isfinite(eligible_exposure_seconds) or eligible_exposure_seconds < 0:
        raise MetricError("eligible_exposure_seconds must be finite and non-negative")
    times = [float(t) for t in timestamps_seconds]
    events = [float(t) for t in event_onsets_seconds]
    if any(not math.isfinite(t) for t in times + events):
        raise MetricError("all times must be finite")
    if any(b <= a for a, b in pairwise(times)):
        raise MetricError("sample timestamps must be strictly increasing")
    if any(b <= a for a, b in pairwise(events)):
        raise MetricError("event onset times must be sorted and unique")
    raw_alarm_times: list[tuple[float, int]] = []
    post_onset_detections = 0
    segment = 0
    for index, prediction in enumerate(predictions):
        if episode_breaks is not None and episode_breaks[index]:
            segment += 1
        if prediction is not True and prediction is not False and prediction is not None:
            raise MetricError("predictions must be bool or None")
        if (
            event_active is not None
            and event_active[index] is not True
            and event_active[index] is not False
            and event_active[index] is not None
        ):
            raise MetricError("event_active states must be bool or None")
        if prediction is None or (event_active is not None and event_active[index] is None):
            segment += 1
            continue
        if prediction is True:
            if event_active is not None and event_active[index] is True:
                post_onset_detections += 1
                segment += 1
            else:
                raw_alarm_times.append((times[index], segment))
        elif event_active is not None and event_active[index] is True:
            segment += 1
    alarm_times: list[float] = []
    last_segment: int | None = None
    last_alarm_time: float | None = None
    for time, alarm_segment in raw_alarm_times:
        # The merge gap applies between successive above-threshold samples;
        # retain the first sample as the episode/warning time.
        if (
            last_alarm_time is None
            or alarm_segment != last_segment
            or time - last_alarm_time > alarm_merge_gap_seconds
        ):
            alarm_times.append(time)
        last_segment = alarm_segment
        last_alarm_time = time
    unmatched_events = set(range(len(events)))
    matched: list[tuple[float, float]] = []
    false_alarms = 0
    for alarm in alarm_times:
        candidates = [
            idx for idx in unmatched_events if alarm < events[idx] <= alarm + horizon_seconds
        ]
        if candidates:
            event_index = min(candidates, key=lambda idx: events[idx])
            unmatched_events.remove(event_index)
            matched.append((alarm, events[event_index]))
        else:
            false_alarms += 1
    leads = [event - alarm for alarm, event in matched]
    matched_count = len(matched)
    precision = matched_count / len(alarm_times) if alarm_times else None
    recall = matched_count / len(events) if events else None
    return {
        "event_count": len(events),
        "alarm_episode_count": len(alarm_times),
        "eligible_exposure_seconds": eligible_exposure_seconds,
        "raw_pre_event_alarm_count": len(raw_alarm_times),
        "repeated_warnings_collapsed": len(raw_alarm_times) - len(alarm_times),
        "matched_event_count": matched_count,
        "missed_event_count": len(unmatched_events),
        "false_alarm_episode_count": false_alarms,
        "post_onset_detections_not_credited_as_forecasts": post_onset_detections,
        "event_precision": precision,
        "event_recall": recall,
        "false_alarms_per_minute": (
            false_alarms / (eligible_exposure_seconds / 60)
            if eligible_exposure_seconds > 0
            else None
        ),
        "warning_lead_time_seconds": {
            "count": len(leads),
            "minimum": min(leads) if leads else None,
            "median": median(leads) if leads else None,
            "mean": mean(leads) if leads else None,
            "maximum": max(leads) if leads else None,
        },
        "matched_alarm_event_times": [
            {"alarm_seconds": a, "event_onset_seconds": e, "lead_seconds": e - a}
            for a, e in matched
        ],
        "undefined_conventions": "Precision/recall are null when there are no alarm episodes/events; false alarms per minute is null at zero eligible exposure.",
    }


def run_bootstrap_event_ci(
    per_run: Sequence[dict[str, Any]],
    *,
    metric: str = "event_recall",
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 2026,
    cluster_key: str = "configuration_id",
) -> dict[str, Any]:
    """Bootstrap independent configurations when present, otherwise individual runs."""
    if metric not in {"event_recall", "event_precision", "false_alarms_per_minute"}:
        raise MetricError(f"unsupported bootstrap metric {metric!r}")
    if not per_run:
        return {"estimable": False, "reason": "no independent run summaries", "run_count": 0}
    if all(item.get(cluster_key) is not None for item in per_run):
        grouped: dict[str, dict[str, float]] = {}
        for item in per_run:
            key = str(item[cluster_key])
            aggregate = grouped.setdefault(
                key,
                {
                    "event_count": 0,
                    "matched_event_count": 0,
                    "alarm_episode_count": 0,
                    "false_alarm_episode_count": 0,
                    "eligible_exposure_seconds": 0,
                },
            )
            for field in aggregate:
                aggregate[field] += item[field]
        clusters = list(grouped.values())
        unit = cluster_key.removesuffix("_id")
    else:
        clusters = list(per_run)
        unit = "run"
    if len(clusters) < 2:
        return {
            "estimable": False,
            "reason": f"fewer than two independent {unit} clusters",
            "run_count": len(per_run),
            "cluster_count": len(clusters),
        }
    if resamples < 100 or not 0 < confidence < 1:
        raise MetricError("resamples must be >=100 and confidence must be in (0,1)")
    rng = random.Random(seed)
    estimates: list[float] = []
    alpha = (1 - confidence) / 2
    for _ in range(resamples):
        sample = [rng.choice(clusters) for _ in clusters]
        if metric == "event_recall":
            denominator = sum(item["event_count"] for item in sample)
            numerator = sum(item["matched_event_count"] for item in sample)
            estimate = numerator / denominator if denominator else None
        elif metric == "event_precision":
            denominator = sum(item["alarm_episode_count"] for item in sample)
            numerator = sum(item["matched_event_count"] for item in sample)
            estimate = numerator / denominator if denominator else None
        else:
            exposure = sum(item["eligible_exposure_seconds"] for item in sample)
            numerator = sum(item["false_alarm_episode_count"] for item in sample)
            estimate = numerator / (exposure / 60) if exposure > 0 else None
        if estimate is not None:
            estimates.append(estimate)
    if not estimates:
        return {
            "estimable": False,
            "reason": "metric undefined in all resamples",
            "run_count": len(per_run),
        }
    estimates.sort()
    low = estimates[min(len(estimates) - 1, int(alpha * len(estimates)))]
    high = estimates[min(len(estimates) - 1, int((1 - alpha) * len(estimates)))]
    return {
        "estimable": True,
        "metric": metric,
        "confidence": confidence,
        "lower": low,
        "upper": high,
        "run_count": len(per_run),
        "cluster_count": len(clusters),
        "bootstrap_resamples_used": len(estimates),
        "seed": seed,
        "unit_of_resampling": f"whole {unit}",
    }
