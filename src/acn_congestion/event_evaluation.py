"""Event-matched warning calibration at a validation false-alarm rate."""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import mean, median
from typing import Any

from .metrics import (
    MetricError,
    binary_metrics,
    event_forecast_metrics,
    run_bootstrap_event_ci,
    score_calibration_metrics,
)
from .splits import check_group_isolation


class EventEvaluationError(ValueError):
    """Event evaluation lacks usable truth, exposure, or independent groups."""


def _state(value: Any) -> bool | None:
    if value is True or value is False or value is None:
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
        if normalized == "":
            return None
    raise EventEvaluationError(f"invalid event_active value {value!r}")


def _per_run_event_metrics(
    rows: list[dict[str, Any]],
    indices: list[int],
    scores: Sequence[float | None],
    *,
    threshold: float,
    horizon_seconds: float,
    max_gap_seconds: float,
    alarm_merge_gap_seconds: float,
) -> list[dict[str, Any]]:
    runs: dict[str, list[int]] = {}
    for index in indices:
        run_id = rows[index].get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise EventEvaluationError(f"row {index} has no run_id")
        runs.setdefault(run_id, []).append(index)
    results = []
    for run_id, run_indices in runs.items():
        times: list[float] = []
        states: list[bool | None] = []
        predictions: list[bool | None] = []
        episode_breaks: list[bool] = []
        unscorable_anchors = 0
        for index in run_indices:
            raw_time = rows[index].get("timestamp_seconds")
            if raw_time is None or not math.isfinite(float(raw_time)):
                raise EventEvaluationError(
                    f"run {run_id}: missing/invalid timestamp at row {index}"
                )
            time = float(raw_time)
            gap_break = bool(times and time - times[-1] > max_gap_seconds)
            if times and time <= times[-1]:
                raise EventEvaluationError(f"run {run_id}: timestamps not strictly increasing")
            times.append(time)
            state = _state(rows[index].get("event_active"))
            states.append(state)
            score = scores[index]
            if score is not None and not math.isfinite(float(score)):
                raise EventEvaluationError(f"run {run_id}: non-finite score at row {index}")
            label = rows[index].get("future_event_onset")
            label_known = label in (0, 1) and not isinstance(label, bool)
            if state is True:
                # Report post-onset detections separately; never score them as forecasts.
                predictions.append(None if score is None else float(score) >= threshold)
            elif state is False and label_known:
                predictions.append(None if score is None else float(score) >= threshold)
                if score is None:
                    unscorable_anchors += 1
            else:
                predictions.append(None)
                unscorable_anchors += 1
            episode_breaks.append(
                gap_break or state is None or (state is False and not label_known)
            )
        onsets: list[float] = []
        unidentified_onsets = 0
        exposure = 0.0
        eligible_interval_count = 0
        excluded_intervals: dict[str, int] = {}
        for i in range(1, len(times)):
            gap = times[i] - times[i - 1]
            prior, current = states[i - 1], states[i]
            if gap > max_gap_seconds:
                excluded_intervals["gap_exceeds_maximum"] = (
                    excluded_intervals.get("gap_exceeds_maximum", 0) + 1
                )
                if current is True and prior is not True:
                    unidentified_onsets += 1
                continue
            if prior is False and current is True:
                onsets.append(times[i])
            elif prior is None and current is True:
                unidentified_onsets += 1
            start_label = rows[run_indices[i - 1]].get("future_event_onset")
            start_label_known = start_label in (0, 1) and not isinstance(start_label, bool)
            if prior is not False:
                reason = "event_or_unknown_at_interval_start"
            elif not start_label_known:
                reason = "censored_target_at_interval_start"
            elif scores[run_indices[i - 1]] is None:
                reason = "missing_score_at_interval_start"
            elif current is None:
                reason = "unknown_target_at_interval_end"
            elif current is False and scores[run_indices[i]] is None:
                reason = "missing_score_at_interval_end"
            else:
                exposure += gap
                eligible_interval_count += 1
                continue
            excluded_intervals[reason] = excluded_intervals.get(reason, 0) + 1
        if states and states[0] is True:
            unidentified_onsets += 1
        try:
            result = event_forecast_metrics(
                times,
                predictions,
                onsets,
                horizon_seconds=horizon_seconds,
                eligible_exposure_seconds=exposure,
                alarm_merge_gap_seconds=alarm_merge_gap_seconds,
                event_active=states,
                episode_breaks=episode_breaks,
            )
        except MetricError as exc:
            raise EventEvaluationError(f"run {run_id}: {exc}") from exc
        result["run_id"] = run_id
        configuration_ids = {rows[index].get("configuration_id") for index in run_indices}
        if len(configuration_ids) > 1:
            raise EventEvaluationError(f"run {run_id}: configuration_id changes within run")
        result["configuration_id"] = next(iter(configuration_ids)) if configuration_ids else None
        result["unidentified_event_onset_count"] = unidentified_onsets
        result["unscorable_anchor_count"] = unscorable_anchors
        result["eligible_interval_count"] = eligible_interval_count
        result["excluded_interval_counts"] = excluded_intervals
        results.append(result)
    return results


def _aggregate(run_results: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        events = sum(item["event_count"] for item in items)
        alarms = sum(item["alarm_episode_count"] for item in items)
        matched = sum(item["matched_event_count"] for item in items)
        false = sum(item["false_alarm_episode_count"] for item in items)
        exposure = sum(item["eligible_exposure_seconds"] for item in items)
        leads = [
            pair["lead_seconds"] for item in items for pair in item["matched_alarm_event_times"]
        ]
        return {
            "run_count": len(items),
            "event_count": events,
            "alarm_episode_count": alarms,
            "matched_event_count": matched,
            "missed_event_count": events - matched,
            "false_alarm_episode_count": false,
            "eligible_exposure_seconds": exposure,
            "event_precision": matched / alarms if alarms else None,
            "event_recall": matched / events if events else None,
            "false_alarms_per_minute": false / (exposure / 60) if exposure > 0 else None,
            "warning_lead_time_seconds": {
                "count": len(leads),
                "minimum": min(leads) if leads else None,
                "median": median(leads) if leads else None,
                "mean": mean(leads) if leads else None,
                "maximum": max(leads) if leads else None,
            },
        }

    result = summarize(run_results)
    configurations: dict[str, list[dict[str, Any]]] = {}
    for item in run_results:
        configurations.setdefault(str(item.get("configuration_id")), []).append(item)
    result["configuration_count"] = len(configurations)
    result["unidentified_event_onset_count"] = sum(
        item["unidentified_event_onset_count"] for item in run_results
    )
    result["unscorable_anchor_count"] = sum(item["unscorable_anchor_count"] for item in run_results)
    result["eligible_interval_count"] = sum(
        item.get("eligible_interval_count", 0) for item in run_results
    )
    excluded: dict[str, int] = {}
    for item in run_results:
        for reason, count in item.get("excluded_interval_counts", {}).items():
            excluded[reason] = excluded.get(reason, 0) + count
    result["excluded_interval_counts"] = excluded
    result["per_configuration"] = {
        key: summarize(items) for key, items in sorted(configurations.items())
    }
    result["per_run"] = run_results
    adequacy_warnings = []
    if result["event_count"] == 0:
        adequacy_warnings.append("no_identified_events")
    elif result["event_count"] < 2:
        adequacy_warnings.append("fewer_than_two_events_descriptive_only")
    if result["configuration_count"] < 2:
        adequacy_warnings.append("fewer_than_two_independent_configurations")
    if result["eligible_exposure_seconds"] <= 0:
        adequacy_warnings.append("no_eligible_exposure")
    result["adequacy_warnings"] = adequacy_warnings
    result["adequacy_note"] = (
        "These mechanical warnings identify zero or extremely sparse denominators; absence of a warning is not evidence of scientific sufficiency."
    )
    return result


def _threshold_tradeoff_point(threshold: float, aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "threshold": threshold,
        "event_count": aggregate["event_count"],
        "alarm_episode_count": aggregate["alarm_episode_count"],
        "matched_event_count": aggregate["matched_event_count"],
        "missed_event_count": aggregate["missed_event_count"],
        "false_alarm_episode_count": aggregate["false_alarm_episode_count"],
        "event_precision": aggregate["event_precision"],
        "event_recall": aggregate["event_recall"],
        "false_alarms_per_minute": aggregate["false_alarms_per_minute"],
        "eligible_exposure_seconds": aggregate["eligible_exposure_seconds"],
        "median_warning_lead_seconds": aggregate["warning_lead_time_seconds"]["median"],
    }


def _bounded_tradeoff_points(
    points: list[dict[str, Any]], selected_threshold: float, *, maximum: int = 101
) -> list[dict[str, Any]]:
    if len(points) <= maximum:
        return points
    selected_indices = {
        round(index * (len(points) - 1) / (maximum - 1)) for index in range(maximum)
    }
    selected_indices.update(
        index for index, point in enumerate(points) if point["threshold"] == selected_threshold
    )
    bounded = [points[index] for index in sorted(selected_indices)]
    # Adding the selected threshold can produce maximum+1 points; drop the
    # nearest non-selected interior point to keep artifacts bounded.
    while len(bounded) > maximum:
        removable = next(
            index
            for index in range(1, len(bounded) - 1)
            if bounded[index]["threshold"] != selected_threshold
        )
        del bounded[removable]
    return bounded


def select_threshold_at_max_fapm(
    rows: list[dict[str, Any]],
    indices: list[int],
    scores: Sequence[float | None],
    *,
    max_false_alarms_per_minute: float,
    horizon_seconds: float,
    max_gap_seconds: float,
    alarm_merge_gap_seconds: float,
) -> dict[str, Any]:
    """Choose a score threshold on validation groups under a declared event FAPM cap."""
    if not math.isfinite(max_false_alarms_per_minute) or max_false_alarms_per_minute < 0:
        raise EventEvaluationError("max_false_alarms_per_minute must be finite and non-negative")
    candidates = sorted(
        {
            float(scores[i])
            for i in indices
            if scores[i] is not None
            and _state(rows[i].get("event_active")) is False
            and rows[i].get("future_event_onset") in (0, 1)
            and not isinstance(rows[i].get("future_event_onset"), bool)
        }
    )
    if not candidates:
        raise EventEvaluationError("validation partition has no finite scores")
    candidates.append(math.nextafter(max(candidates), math.inf))
    best: tuple[float, float, float] | None = None
    best_result: dict[str, Any] | None = None
    tradeoff_points: list[dict[str, Any]] = []
    for threshold in candidates:
        per_run = _per_run_event_metrics(
            rows,
            indices,
            scores,
            threshold=threshold,
            horizon_seconds=horizon_seconds,
            max_gap_seconds=max_gap_seconds,
            alarm_merge_gap_seconds=alarm_merge_gap_seconds,
        )
        aggregate = _aggregate(per_run)
        if aggregate["event_count"] == 0 or aggregate["eligible_exposure_seconds"] <= 0:
            continue
        tradeoff_points.append(_threshold_tradeoff_point(threshold, aggregate))
        fapm, recall = aggregate["false_alarms_per_minute"], aggregate["event_recall"]
        if fapm is not None and fapm <= max_false_alarms_per_minute:
            key = (
                recall if recall is not None else -1.0,
                aggregate["event_precision"] if aggregate["event_precision"] is not None else -1.0,
                threshold,
            )
            if best is None or key > best:
                best, best_result = key, aggregate
    if best is None or best_result is None:
        raise EventEvaluationError("no validation threshold satisfies the declared false-alarm cap")
    bounded_tradeoff = _bounded_tradeoff_points(tradeoff_points, best[2])
    return {
        "threshold": best[2],
        "validation_event_metrics": best_result,
        "max_false_alarms_per_minute": max_false_alarms_per_minute,
        "selection_partition": "validation only",
        "selection_criterion": "maximize validation event recall subject to the declared false-alarms-per-minute cap; break ties by event precision, then the higher threshold",
        "threshold_tradeoff": {
            "candidate_threshold_count": len(candidates),
            "estimable_tradeoff_point_count": len(tradeoff_points),
            "reported_point_count": len(bounded_tradeoff),
            "sampling": "all estimable points"
            if len(bounded_tradeoff) == len(tradeoff_points)
            else "deterministically sampled to at most 101 points, always retaining the selected threshold",
            "points": bounded_tradeoff,
        },
    }


def _score_calibration_assessment(
    rows: list[dict[str, Any]],
    *,
    validation_indices: list[int],
    test_indices: list[int],
    scores: Sequence[float | None],
    validation_event_metrics: dict[str, Any],
) -> dict[str, Any]:
    def metrics_for(indices: list[int]) -> dict[str, Any]:
        labels = [rows[index].get("future_event_onset") for index in indices]
        values = [scores[index] for index in indices]
        return score_calibration_metrics(labels, values)

    validation = metrics_for(validation_indices)
    support_warnings: list[str] = []
    if not validation.get("estimable"):
        support_warnings.append("validation_calibration_metrics_not_estimable")
    if validation_event_metrics.get("event_count", 0) < 2:
        support_warnings.append("fewer_than_two_validation_events")
    if validation_event_metrics.get("configuration_count", 0) < 2:
        support_warnings.append("fewer_than_two_validation_configurations")
    if validation.get("positive_count", 0) < 2 or validation.get("negative_count", 0) < 2:
        support_warnings.append("fewer_than_two_rows_in_a_validation_class")
    return {
        "status": "not_fitted",
        "fitted_calibrator": None,
        "score_semantics": "uncalibrated positive-class model score; not a calibrated probability",
        "validation_descriptive_metrics": validation,
        "test_descriptive_metrics_secondary": metrics_for(test_indices),
        "support": {
            "validation_event_count": validation_event_metrics.get("event_count"),
            "validation_configuration_count": validation_event_metrics.get("configuration_count"),
            "validation_run_count": validation_event_metrics.get("run_count"),
            "validation_positive_rows": validation.get("positive_count"),
            "validation_negative_rows": validation.get("negative_count"),
            "warnings": support_warnings,
        },
        "interpretation": "No Platt, isotonic, or other calibrator is fitted in this pipeline. Descriptive Brier/log-loss/reliability values may help diagnose score behavior, but they do not establish calibrated probabilities, especially with few independent events or overlapping forecast horizons.",
    }


def evaluate_event_scores(
    rows: list[dict[str, Any]],
    split: dict[str, Any],
    scores: Sequence[float | None],
    *,
    identity: str,
    max_false_alarms_per_minute: float,
    horizon_seconds: float,
    max_gap_seconds: float,
    alarm_merge_gap_seconds: float,
    feature_columns: list[str] | None = None,
    bootstrap_resamples: int = 2000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 2026,
) -> dict[str, Any]:
    """Calibrate on validation event FAPM and return frozen held-out event metrics."""
    if not isinstance(split, dict):
        raise EventEvaluationError("split manifest must be an object")
    partitions = split.get("row_indices")
    grouping = split.get("grouping")
    if not isinstance(partitions, dict) or set(partitions) != {"train", "validation", "test"}:
        raise EventEvaluationError("split manifest must contain train/validation/test partitions")
    try:
        check_group_isolation(rows, partitions, grouping=grouping)
    except (KeyError, TypeError, ValueError) as exc:
        raise EventEvaluationError(f"invalid/leaky split: {exc}") from exc
    if len(scores) != len(rows):
        raise EventEvaluationError("one score slot is required per source row")
    train_labels = [
        rows[i].get("future_event_onset")
        for i in partitions["train"]
        if rows[i].get("future_event_onset") in (0, 1)
        and not isinstance(rows[i].get("future_event_onset"), bool)
    ]
    if set(train_labels) != {0, 1}:
        raise EventEvaluationError("train partition must contain both label classes")
    calibration = select_threshold_at_max_fapm(
        rows,
        partitions["validation"],
        scores,
        max_false_alarms_per_minute=max_false_alarms_per_minute,
        horizon_seconds=horizon_seconds,
        max_gap_seconds=max_gap_seconds,
        alarm_merge_gap_seconds=alarm_merge_gap_seconds,
    )
    test_per_run = _per_run_event_metrics(
        rows,
        partitions["test"],
        scores,
        threshold=calibration["threshold"],
        horizon_seconds=horizon_seconds,
        max_gap_seconds=max_gap_seconds,
        alarm_merge_gap_seconds=alarm_merge_gap_seconds,
    )
    test_metrics = _aggregate(test_per_run)
    test_labels = [rows[i].get("future_event_onset") for i in partitions["test"]]
    test_scores = [scores[i] for i in partitions["test"]]
    window_metrics = binary_metrics(test_labels, test_scores, threshold=calibration["threshold"])
    calibration_assessment = _score_calibration_assessment(
        rows,
        validation_indices=partitions["validation"],
        test_indices=partitions["test"],
        scores=scores,
        validation_event_metrics=calibration["validation_event_metrics"],
    )
    uncertainty = {
        metric: run_bootstrap_event_ci(
            test_per_run,
            metric=metric,
            cluster_key="configuration_id",
            resamples=bootstrap_resamples,
            confidence=bootstrap_confidence,
            seed=bootstrap_seed,
        )
        for metric in ("event_recall", "event_precision", "false_alarms_per_minute")
    }
    predictions = []
    for i in partitions["test"]:
        row = rows[i]
        score = scores[i]
        predictions.append(
            {
                "source_row_index": i,
                "run_id": row["run_id"],
                "configuration_id": row["configuration_id"],
                "timestamp_seconds": row.get("timestamp_seconds"),
                "label": row.get("future_event_onset"),
                "event_active": _state(row.get("event_active")),
                "score": score,
                "warning": (
                    float(score) >= calibration["threshold"] if score is not None else None
                ),
            }
        )
    return {
        "model_name": identity,
        "feature_columns": feature_columns or [],
        "grouping": grouping,
        "threshold_calibration": calibration,
        "score_calibration": calibration_assessment,
        "test_event_metrics": test_metrics,
        "test_window_metrics_secondary": window_metrics,
        "test_uncertainty": uncertainty,
        "test_predictions": predictions,
        "warning": "Threshold selected on validation event false alarms only. Source-data provenance and scientific validity are not independently verified.",
    }
