"""Predeclared, group-aware exploratory robustness analyses."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any

from .evaluation import EvaluationError, evaluate_baseline_events, evaluate_model_events
from .labels import derive_persistent_events, label_future_onsets_detailed
from .models import (
    ModelError,
    has_observed_endpoint_feature,
    select_training_observed_features,
    validate_feature_columns,
)
from .processing import (
    ProcessingError,
    apply_feature_temporal_policy_to_rows,
    feature_contract_metadata,
    feature_temporal_policy_from_specification,
    validate_feature_specification_object,
)
from .splits import check_group_isolation


class RobustnessError(ValueError):
    """Robustness analysis is invalid or cannot preserve the declared structure."""


_ENDPOINT_FAMILIES = ("rtt_ms", "loss_fraction", "goodput_bps", "tcp_retransmission_delta")


def feature_family(feature_name: str) -> str:
    for family in _ENDPOINT_FAMILIES:
        if feature_name.startswith(family + "_"):
            return family
    raise RobustnessError(f"feature is not a recognized endpoint family: {feature_name!r}")


def _validated_feature_specification(
    feature_specification: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if feature_specification is None:
        return None
    try:
        return validate_feature_specification_object(feature_specification)
    except ProcessingError as exc:
        raise RobustnessError(f"invalid feature specification: {exc}") from exc


def _feature_contract_metadata(
    specification: dict[str, Any] | None,
    classifications: list[dict[str, Any]] | None = None,
    *,
    scope: str = "all_rows",
) -> dict[str, Any]:
    return feature_contract_metadata(specification, classifications, scope=scope)


def _validated_split_partitions(
    rows: list[dict[str, Any]], split: dict[str, Any]
) -> dict[str, list[int]]:
    """Reject a malformed/leaky supplied split before any robustness analysis."""
    if not isinstance(split, dict):
        raise RobustnessError("split must be an object")
    partitions = split.get("row_indices")
    if not isinstance(partitions, dict) or set(partitions) != {"train", "validation", "test"}:
        raise RobustnessError("split must contain train/validation/test row indices")
    try:
        check_group_isolation(rows, partitions, grouping=split.get("grouping"))
    except (KeyError, TypeError, ValueError) as exc:
        raise RobustnessError(f"invalid or leaking split: {exc}") from exc
    return partitions


def _apply_feature_contract_to_indices(
    rows: list[dict[str, Any]],
    indices: Sequence[int],
    feature_columns: Sequence[str],
    specification: dict[str, Any] | None,
    *,
    scope: str,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    selected = [rows[index] for index in indices]
    if specification is None:
        return dict(zip(indices, selected)), _feature_contract_metadata(None, scope=scope)
    try:
        masked, classifications = apply_feature_temporal_policy_to_rows(
            selected, list(feature_columns), specification
        )
    except ProcessingError as exc:
        raise RobustnessError(str(exc)) from exc
    return dict(zip(indices, masked)), _feature_contract_metadata(
        specification, classifications, scope=scope
    )


def circular_shift_training_labels(
    rows: list[dict[str, Any]],
    split: dict[str, Any],
    *,
    minimum_shift_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Circularly shift complete within-run training label sequences; never shuffle rows.

    Validation/test targets remain untouched. The lag is in samples, not seconds; choose
    it from observed cadence and the forecast horizon before inspecting outcomes. A
    circular shift has one wrap discontinuity and is a diagnostic, not a p-value test.
    """
    if (
        isinstance(minimum_shift_samples, bool)
        or not isinstance(minimum_shift_samples, int)
        or minimum_shift_samples < 1
    ):
        raise RobustnessError("minimum_shift_samples must be an integer >= 1")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RobustnessError("seed must be a non-negative integer")
    partitions = _validated_split_partitions(rows, split)
    output = [dict(row) for row in rows]
    training_by_run: dict[str, list[int]] = defaultdict(list)
    for index in partitions["train"]:
        run_id = rows[index].get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise RobustnessError(f"training row {index} has no run_id")
        training_by_run[run_id].append(index)
    rng = random.Random(seed)
    offsets: dict[str, int] = {}
    for run_id, indices in sorted(training_by_run.items()):
        try:
            indices.sort(key=lambda i: float(rows[i]["timestamp_seconds"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise RobustnessError(f"run {run_id}: timestamps are missing/invalid") from exc
        times = [float(rows[index]["timestamp_seconds"]) for index in indices]
        if any(not math.isfinite(t) for t in times) or any(b <= a for a, b in pairwise(times)):
            raise RobustnessError(
                f"run {run_id}: timestamps must be finite and strictly increasing"
            )
        labels = [rows[index].get("future_event_onset") for index in indices]
        if any(label not in (0, 1, None) or isinstance(label, bool) for label in labels):
            raise RobustnessError(f"run {run_id}: labels must be 0, 1, or None")
        candidates = list(range(minimum_shift_samples, len(indices) - minimum_shift_samples + 1))
        if not candidates:
            raise RobustnessError(
                f"run {run_id}: {len(indices)} rows cannot support minimum circular shift of {minimum_shift_samples} samples"
            )
        offset = rng.choice(candidates)
        shifted = labels[-offset:] + labels[:-offset]
        for index, label in zip(indices, shifted):
            output[index]["__negative_control_training_label"] = label
        offsets[run_id] = offset
    for partition in ("validation", "test"):
        for index in partitions[partition]:
            output[index]["__negative_control_training_label"] = rows[index].get(
                "future_event_onset"
            )
    return output, offsets


def evaluate_circular_shift_negative_control(
    rows: list[dict[str, Any]],
    split: dict[str, Any],
    *,
    model_name: str,
    feature_columns: list[str],
    minimum_shift_samples: int,
    random_seed: int,
    max_false_alarms_per_minute: float,
    horizon_seconds: float,
    max_gap_seconds: float,
    alarm_merge_gap_seconds: float,
    class_weight: str | None = None,
    bootstrap_resamples: int = 2000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 2026,
    feature_specification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train on within-run shifted labels; calibrate/evaluate against untouched truth."""
    from .event_evaluation import evaluate_event_scores
    from .models import fit_estimator, predict_scores

    validate_feature_columns(feature_columns)
    specification = _validated_feature_specification(feature_specification)
    shifted_rows, offsets = circular_shift_training_labels(
        rows,
        split,
        minimum_shift_samples=minimum_shift_samples,
        seed=random_seed,
    )
    partitions = split["row_indices"]
    masked_train_by_index, train_temporal_metadata = _apply_feature_contract_to_indices(
        shifted_rows,
        partitions["train"],
        feature_columns,
        specification,
        scope="training_partition",
    )
    train_rows = [
        masked_train_by_index[i]
        for i in partitions["train"]
        if masked_train_by_index[i].get("__negative_control_training_label") in (0, 1)
        and not isinstance(masked_train_by_index[i].get("__negative_control_training_label"), bool)
        and has_observed_endpoint_feature(masked_train_by_index[i], feature_columns)
    ]
    used_features, dropped_features = select_training_observed_features(train_rows, feature_columns)
    if not used_features:
        raise RobustnessError("all endpoint features are missing in shifted-label training rows")
    train_rows = [row for row in train_rows if has_observed_endpoint_feature(row, used_features)]
    train_labels = [int(row["__negative_control_training_label"]) for row in train_rows]
    if set(train_labels) != {0, 1}:
        raise RobustnessError("eligible shifted-label training rows do not retain both classes")
    estimator = fit_estimator(
        model_name,
        train_rows,
        train_labels,
        used_features,
        random_seed=random_seed,
        class_weight=class_weight,
    )
    scores: list[float | None] = [None] * len(rows)
    scoring_temporal_metadata_by_partition: dict[str, Any] = {}
    for partition in ("validation", "test"):
        masked_by_index, partition_metadata = _apply_feature_contract_to_indices(
            rows,
            partitions[partition],
            used_features,
            specification,
            scope=f"{partition}_partition",
        )
        scoring_temporal_metadata_by_partition[partition] = partition_metadata[
            "feature_temporal_eligibility"
        ]
        indices = [
            index
            for index in partitions[partition]
            if has_observed_endpoint_feature(masked_by_index[index], used_features)
        ]
        if not indices:
            continue
        predicted = predict_scores(estimator, [masked_by_index[i] for i in indices], used_features)
        for index, score in zip(indices, predicted, strict=True):
            scores[index] = score
    result = evaluate_event_scores(
        rows,
        split,
        scores,
        identity=f"negative_control_circular_shift:{model_name}",
        max_false_alarms_per_minute=max_false_alarms_per_minute,
        horizon_seconds=horizon_seconds,
        max_gap_seconds=max_gap_seconds,
        alarm_merge_gap_seconds=alarm_merge_gap_seconds,
        feature_columns=used_features,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_confidence=bootstrap_confidence,
        bootstrap_seed=bootstrap_seed,
    )
    result.update(
        {
            "analysis_type": "exploratory negative control, not a formal significance test",
            "control": "training labels circularly shifted within each training run; validation/test labels unchanged",
            "minimum_shift_samples": minimum_shift_samples,
            "run_shift_offsets_samples": offsets,
            "requested_feature_columns": feature_columns,
            "feature_columns": used_features,
            "all_missing_training_feature_columns": dropped_features,
            "random_seed": random_seed,
            "class_weight": class_weight,
            **train_temporal_metadata,
            "feature_temporal_eligibility_scoring": (scoring_temporal_metadata_by_partition),
        }
    )
    return result


def evaluate_feature_ablation(
    rows: list[dict[str, Any]],
    split: dict[str, Any],
    *,
    model_name: str,
    feature_columns: list[str],
    max_false_alarms_per_minute: float,
    horizon_seconds: float,
    max_gap_seconds: float,
    alarm_merge_gap_seconds: float,
    random_seed: int = 2026,
    class_weight: str | None = None,
    bootstrap_resamples: int = 2000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 2026,
    feature_specification: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compare full endpoint features with leave-one-signal-family-out models."""
    validate_feature_columns(feature_columns)
    specification = _validated_feature_specification(feature_specification)
    _validated_split_partitions(rows, split)
    families = {
        family: [col for col in feature_columns if feature_family(col) == family]
        for family in _ENDPOINT_FAMILIES
    }
    families = {family: columns for family, columns in families.items() if columns}
    if len(families) < 2:
        raise RobustnessError(
            "feature ablation requires at least two represented endpoint signal families"
        )
    common = {
        "max_false_alarms_per_minute": max_false_alarms_per_minute,
        "horizon_seconds": horizon_seconds,
        "max_gap_seconds": max_gap_seconds,
        "alarm_merge_gap_seconds": alarm_merge_gap_seconds,
        "random_seed": random_seed,
        "class_weight": class_weight,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_confidence": bootstrap_confidence,
        "bootstrap_seed": bootstrap_seed,
        "feature_specification": specification,
    }
    methods: dict[str, list[str]] = {"full_features": feature_columns}
    for family, family_columns in families.items():
        remaining = [col for col in feature_columns if col not in family_columns]
        if remaining:
            methods[f"omit_{family}"] = remaining
    results: dict[str, dict[str, Any]] = {}
    for identity, columns in methods.items():
        try:
            result = evaluate_model_events(
                rows, split, model_name=model_name, feature_columns=columns, **common
            )
            result["analysis_type"] = "exploratory feature ablation"
            result["ablation_identity"] = identity
            results[identity] = result
        except (EvaluationError, ModelError, ValueError) as exc:
            results[identity] = {
                "status": "failed",
                "failure": f"{type(exc).__name__}: {exc}",
                "analysis_type": "exploratory feature ablation",
                "feature_columns": columns,
                **_feature_contract_metadata(specification),
            }
    return results


def _relabel_target(
    rows: list[dict[str, Any]],
    *,
    threshold_ms: float,
    persistence_samples: int,
    horizon_seconds: float,
    max_gap_seconds: float,
) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    by_run: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_run[str(row.get("run_id"))].append(index)
    for run_id, indices in by_run.items():
        indices.sort(key=lambda i: float(rows[i]["timestamp_seconds"]))
        times = [float(rows[i]["timestamp_seconds"]) for i in indices]
        values = [rows[i].get("qdisc_queue_delay_proxy_ms") for i in indices]
        states = derive_persistent_events(
            values,
            threshold_ms=threshold_ms,
            consecutive_samples=persistence_samples,
            timestamps_seconds=times,
            max_gap_seconds=max_gap_seconds,
        )
        outcomes = label_future_onsets_detailed(
            times,
            states,
            horizon_seconds=horizon_seconds,
            max_gap_seconds=max_gap_seconds,
            run_ids=[run_id] * len(indices),
        )
        for index, state, outcome in zip(indices, states, outcomes):
            output[index]["event_active"] = state
            output[index]["future_event_onset"] = outcome.label
            output[index]["label_reason"] = outcome.reason
    return output


def evaluate_target_sensitivity(
    rows: list[dict[str, Any]],
    split: dict[str, Any],
    *,
    thresholds_ms: Sequence[float],
    horizons_seconds: Sequence[float],
    persistence_samples: int,
    max_gap_seconds: float,
    baseline_feature: str,
    baseline_direction: str,
    eligible_feature_columns: list[str],
    model_names: Sequence[str],
    max_false_alarms_per_minute: float,
    alarm_merge_gap_seconds: float,
    random_seed: int,
    class_weight: str | None = None,
    bootstrap_resamples: int = 2000,
    bootstrap_confidence: float = 0.95,
    feature_specification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run an explicit, bounded target threshold × horizon grid without selecting a winner."""
    specification = _validated_feature_specification(feature_specification)
    thresholds = [float(value) for value in thresholds_ms]
    horizons = [float(value) for value in horizons_seconds]
    if (
        len(thresholds) < 2
        or len(horizons) < 2
        or any(not math.isfinite(value) or value <= 0 for value in thresholds + horizons)
        or len(set(thresholds)) != len(thresholds)
        or len(set(horizons)) != len(horizons)
    ):
        raise RobustnessError(
            "sensitivity needs >=2 unique finite positive thresholds and horizons"
        )
    if len(thresholds) * len(horizons) > 12:
        raise RobustnessError(
            "sensitivity grid exceeds 12 threshold/horizon combinations; predeclare a smaller range"
        )
    if (
        isinstance(persistence_samples, bool)
        or not isinstance(persistence_samples, int)
        or persistence_samples < 1
    ):
        raise RobustnessError("persistence_samples must be an integer >= 1")
    validate_feature_columns(eligible_feature_columns)
    if not model_names or any(not isinstance(name, str) or not name for name in model_names):
        raise RobustnessError("model_names must be a non-empty list of model names")
    if len(set(model_names)) != len(model_names):
        raise RobustnessError("model_names must be unique")
    partitions = _validated_split_partitions(rows, split)
    variants: dict[str, Any] = {}
    for threshold in thresholds:
        for horizon in horizons:
            variant_rows = _relabel_target(
                rows,
                threshold_ms=threshold,
                persistence_samples=persistence_samples,
                horizon_seconds=horizon,
                max_gap_seconds=max_gap_seconds,
            )
            labels = Counter(
                "censored" if row["future_event_onset"] is None else str(row["future_event_onset"])
                for row in variant_rows
            )
            event_count = sum(
                row["event_active"] is True
                and (
                    index == 0
                    or variant_rows[index - 1]["run_id"] != row["run_id"]
                    or variant_rows[index - 1]["event_active"] is not True
                    or row["timestamp_seconds"] - variant_rows[index - 1]["timestamp_seconds"]
                    > max_gap_seconds
                )
                for index, row in enumerate(variant_rows)
            )
            masked_train_by_index, train_temporal_metadata = _apply_feature_contract_to_indices(
                variant_rows,
                partitions["train"],
                eligible_feature_columns,
                specification,
                scope="training_partition",
            )
            train_rows = [
                masked_train_by_index[i]
                for i in partitions["train"]
                if masked_train_by_index[i].get("future_event_onset") in (0, 1)
                and not isinstance(masked_train_by_index[i].get("future_event_onset"), bool)
                and has_observed_endpoint_feature(
                    masked_train_by_index[i], eligible_feature_columns
                )
            ]
            used_features, dropped_features = select_training_observed_features(
                train_rows, eligible_feature_columns
            )
            key = f"threshold_{threshold:g}ms_horizon_{horizon:g}s"
            method_results: dict[str, Any] = {}
            failures: dict[str, str] = {}
            try:
                method_results.update(
                    evaluate_baseline_events(
                        variant_rows,
                        split,
                        feature_column=baseline_feature,
                        direction=baseline_direction,
                        eligible_feature_columns=used_features,
                        max_false_alarms_per_minute=max_false_alarms_per_minute,
                        horizon_seconds=horizon,
                        max_gap_seconds=max_gap_seconds,
                        alarm_merge_gap_seconds=alarm_merge_gap_seconds,
                        bootstrap_resamples=bootstrap_resamples,
                        bootstrap_confidence=bootstrap_confidence,
                        bootstrap_seed=random_seed,
                        feature_specification=specification,
                    )
                )
            except (EvaluationError, ModelError, ValueError) as exc:
                failures["rule_baselines"] = f"{type(exc).__name__}: {exc}"
            for model_name in model_names:
                try:
                    method_results[model_name] = evaluate_model_events(
                        variant_rows,
                        split,
                        model_name=model_name,
                        feature_columns=used_features,
                        max_false_alarms_per_minute=max_false_alarms_per_minute,
                        horizon_seconds=horizon,
                        max_gap_seconds=max_gap_seconds,
                        alarm_merge_gap_seconds=alarm_merge_gap_seconds,
                        random_seed=random_seed,
                        class_weight=class_weight,
                        bootstrap_resamples=bootstrap_resamples,
                        bootstrap_confidence=bootstrap_confidence,
                        bootstrap_seed=random_seed,
                        feature_specification=specification,
                    )
                except (EvaluationError, ModelError, ValueError) as exc:
                    failures[model_name] = f"{type(exc).__name__}: {exc}"
            variants[key] = {
                "threshold_ms": threshold,
                "horizon_seconds": horizon,
                "persistence_samples": persistence_samples,
                "label_counts": dict(labels),
                "observed_event_count": event_count,
                "feature_columns_used": used_features,
                "all_missing_training_feature_columns": dropped_features,
                **train_temporal_metadata,
                "methods": method_results,
                "method_failures": failures,
            }
    return {
        "analysis_type": "exploratory target sensitivity; no best variant selected",
        "grouping": split.get("grouping"),
        "split_seed": split.get("seed"),
        "configuration": {
            "thresholds_ms": thresholds,
            "horizons_seconds": horizons,
            "persistence_samples": persistence_samples,
            "max_gap_seconds": max_gap_seconds,
        },
        "variants": variants,
        **_feature_contract_metadata(specification),
        "warning": "Unless this exact grid and all analysis choices were frozen before outcomes were inspected, treat all variants as exploratory; do not report the best grid point as confirmatory.",
    }


def evaluate_model_seed_stability(
    rows: list[dict[str, Any]],
    split: dict[str, Any],
    *,
    model_name: str,
    feature_columns: list[str],
    seeds: Sequence[int],
    max_false_alarms_per_minute: float,
    horizon_seconds: float,
    max_gap_seconds: float,
    alarm_merge_gap_seconds: float,
    class_weight: str | None = None,
    bootstrap_resamples: int = 2000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 2026,
    feature_specification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Report seed variation on one frozen split; this is not a confidence interval."""
    specification = _validated_feature_specification(feature_specification)
    seed_list = list(seeds)
    if (
        len(seed_list) < 2
        or len(seed_list) > 10
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seed_list
        )
        or len(set(seed_list)) != len(seed_list)
    ):
        raise RobustnessError("model seed stability needs 2–10 unique non-negative integer seeds")
    validate_feature_columns(feature_columns)
    _validated_split_partitions(rows, split)
    results: dict[str, Any] = {}
    for seed in seed_list:
        try:
            results[str(seed)] = evaluate_model_events(
                rows,
                split,
                model_name=model_name,
                feature_columns=feature_columns,
                max_false_alarms_per_minute=max_false_alarms_per_minute,
                horizon_seconds=horizon_seconds,
                max_gap_seconds=max_gap_seconds,
                alarm_merge_gap_seconds=alarm_merge_gap_seconds,
                random_seed=seed,
                class_weight=class_weight,
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_confidence=bootstrap_confidence,
                bootstrap_seed=bootstrap_seed,
                feature_specification=specification,
            )
        except (EvaluationError, ModelError, ValueError) as exc:
            results[str(seed)] = {
                "status": "failed",
                "failure": f"{type(exc).__name__}: {exc}",
                **_feature_contract_metadata(specification),
            }
    metric_summaries: dict[str, Any] = {}
    for metric in ("event_recall", "event_precision", "false_alarms_per_minute"):
        values = [
            item.get("test_event_metrics", {}).get(metric)
            for item in results.values()
            if isinstance(item, dict) and item.get("test_event_metrics", {}).get(metric) is not None
        ]
        metric_summaries[metric] = {
            "estimable_seed_count": len(values),
            "minimum": min(values) if values else None,
            "median": median(values) if values else None,
            "maximum": max(values) if values else None,
            "range": max(values) - min(values) if values else None,
        }
    return {
        "analysis_type": "exploratory model-seed stability on a fixed group split",
        "model_name": model_name,
        "feature_columns": feature_columns,
        "seeds": seed_list,
        "split_grouping": split.get("grouping"),
        "split_seed": split.get("seed"),
        "bootstrap_seed_fixed_across_models": bootstrap_seed,
        "class_weight": class_weight,
        "results_by_seed": results,
        "metric_variation": metric_summaries,
        **_feature_contract_metadata(specification),
        "warning": "Seed ranges are descriptive, not confidence intervals; do not select the best seed. Group and event counts still determine inferential adequacy.",
    }


def _result_feature_contract_mismatch(
    value: Any, *, specification: dict[str, Any], digest: str
) -> bool:
    expected_policy = feature_temporal_policy_from_specification(specification)
    expected_unit = specification.get("tcp_retransmission_unit")
    method_result_keys = {
        "test_event_metrics",
        "test_metrics",
        "threshold_calibration",
        "model_parameters",
        "training_counts",
    }

    def embedded_mismatch(record: dict[str, Any]) -> bool:
        expected = {
            "feature_specification": specification,
            "feature_specification_sha256": digest,
            "feature_specification_status": "supplied_current_resolved",
            "tcp_retransmission_unit": expected_unit,
            "feature_temporal_policy": expected_policy,
        }
        return any(
            key in record and record.get(key) != expected_value
            for key, expected_value in expected.items()
        ) or (
            "feature_temporal_eligibility" in record
            and not isinstance(record.get("feature_temporal_eligibility"), dict)
        )

    def visit(node: Any) -> bool:
        if isinstance(node, dict):
            if embedded_mismatch(node):
                return True
            is_method_result = node.get("status") != "failed" and bool(
                method_result_keys & set(node)
            )
            if is_method_result and (
                node.get("feature_specification") != specification
                or node.get("feature_specification_sha256") != digest
                or node.get("feature_specification_status") != "supplied_current_resolved"
                or node.get("tcp_retransmission_unit") != expected_unit
                or node.get("feature_temporal_policy") != expected_policy
                or not isinstance(node.get("feature_temporal_eligibility"), dict)
            ):
                return True
            return any(visit(child) for child in node.values())
        if isinstance(node, list):
            return any(visit(child) for child in node)
        return False

    return visit(value)


def _contains_failure(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("status") == "failed" or bool(value.get("method_failures")):
            return True
        return any(_contains_failure(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_failure(child) for child in value)
    return False


def _json_dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False)


def write_robustness_artifacts(
    output_dir: Path,
    *,
    result: dict[str, Any],
    config_sha256: str,
    analysis_sha256: str,
    processing_manifest_sha256: str,
    processing_manifest: dict[str, Any],
    specification_sha256: str,
    configuration: dict[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    """Write an exploratory report with hashes and explicit partial-failure status."""
    from .processing import file_sha256

    def valid_hash(value: str) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(c in "0123456789abcdef" for c in value)
        )

    if not all(
        valid_hash(value)
        for value in (
            config_sha256,
            analysis_sha256,
            processing_manifest_sha256,
            specification_sha256,
        )
    ):
        raise RobustnessError("robustness report requires valid SHA-256 input hashes")
    if processing_manifest.get("analysis_sha256") != analysis_sha256:
        raise RobustnessError("robustness processing manifest does not match analysis hash")
    from .config import ConfigError, validate_config
    from .reporting import ReportError, canonical_json_sha256, validate_processing_compatibility

    try:
        if canonical_json_sha256(processing_manifest) != processing_manifest_sha256:
            raise RobustnessError("processing manifest content does not match its declared SHA-256")
        validate_config(configuration, require_analysis_ready=True)
        processing_compatibility = validate_processing_compatibility(
            processing_manifest,
            configuration,
            current_config_sha256=config_sha256,
            config_path=config_path,
        )
        if _result_feature_contract_mismatch(
            result,
            specification=processing_compatibility["feature_specification"],
            digest=processing_compatibility["feature_specification_sha256"],
        ):
            raise RobustnessError(
                "robustness result contains a feature specification or digest that "
                "differs from current frozen prepared provenance"
            )
    except (ConfigError, ReportError) as exc:
        raise RobustnessError(
            f"robustness requires prepared data matching a frozen protocol: {exc}"
        ) from exc
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise RobustnessError(f"refusing to overwrite existing robustness output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    marker = output_dir / "INCOMPLETE"
    marker.write_text("Robustness artifact generation incomplete.\n", encoding="utf-8")
    result_path = output_dir / "robustness.json"
    result_path.write_text(_json_dumps(result) + "\n", encoding="utf-8")
    outcome = "partial_failure" if _contains_failure(result) else "complete"
    lines = [
        "# Exploratory robustness report",
        "",
        f"**Outcome:** `{outcome}`. Treat this analysis as exploratory unless the exact specification and analysis plan were frozen before outcomes were viewed.",
        "",
        f"- Config SHA-256: `{config_sha256}`",
        f"- Prepared analysis SHA-256: `{analysis_sha256}`",
        f"- Processing manifest SHA-256: `{processing_manifest_sha256}`",
        f"- Specification SHA-256: `{specification_sha256}`",
        f"- Feature specification: `{processing_compatibility['feature_specification']['spec_version']}` (SHA-256 `{processing_compatibility['feature_specification_sha256']}`)",
        f"- TCP retransmission unit: `{processing_compatibility['tcp_retransmission_unit']}`",
        f"- Feature temporal policy status: `{processing_compatibility['feature_temporal_policy_status']}`",
        f"- Analysis type: `{result.get('analysis_type', 'unspecified')}`",
        "",
        "Do not select the best threshold, horizon, ablation, or seed from this report and describe it as confirmatory. Failures remain recorded in `robustness.json`.",
        "",
    ]
    variants = result.get("variants")
    if isinstance(variants, dict):
        lines.extend(
            [
                "## Target sensitivity variants",
                "",
                "| Variant | Threshold (ms) | Horizon (s) | Events | Labels | Failures |",
                "|---|---:|---:|---:|---|---:|",
            ]
        )
        for name, variant in variants.items():
            lines.append(
                f"| {name} | {variant.get('threshold_ms')} | {variant.get('horizon_seconds')} | "
                f"{variant.get('observed_event_count')} | {variant.get('label_counts')} | "
                f"{len(variant.get('method_failures', {}))} |"
            )
    elif isinstance(result.get("test_event_metrics"), dict):
        metrics = result["test_event_metrics"]
        lines.extend(
            [
                "## Negative-control held-out event metrics",
                "",
                "| Events | Recall | Precision | False alarms/min |",
                "|---:|---:|---:|---:|",
            ]
        )
        lines.append(
            f"| {metrics.get('event_count')} | {metrics.get('event_recall')} | "
            f"{metrics.get('event_precision')} | {metrics.get('false_alarms_per_minute')} |"
        )
    elif isinstance(result.get("results_by_seed"), dict):
        lines.extend(["## Seed metric ranges", ""])
        for metric, summary in result.get("metric_variation", {}).items():
            lines.append(
                f"- **{metric}:** {summary['minimum']} to {summary['maximum']} "
                f"across {summary['estimable_seed_count']} estimable seeds (descriptive range)."
            )
    elif isinstance(result.get("methods"), dict):
        lines.extend(["## Feature ablation outcomes", ""])
        for name, item in result["methods"].items():
            lines.append(
                f"- **{name}:** {item.get('status', 'evaluated')}; {item.get('failure', '')}"
            )
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts = {
        name: file_sha256(output_dir / name) for name in (result_path.name, report_path.name)
    }
    manifest = {
        "robustness_report_schema_version": 1,
        "status": outcome,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha256,
        "analysis_sha256": analysis_sha256,
        "protocol_status": processing_manifest.get("protocol_status"),
        "protocol_sha256": processing_manifest.get("protocol_sha256"),
        "protocol_freeze": configuration.get("protocol_freeze"),
        "processing_manifest_sha256": processing_manifest_sha256,
        "feature_specification": processing_compatibility["feature_specification"],
        "feature_specification_sha256": processing_compatibility["feature_specification_sha256"],
        "feature_specification_status": processing_compatibility["feature_specification_status"],
        "tcp_retransmission_unit": processing_compatibility["tcp_retransmission_unit"],
        "tcp_retransmission_unit_provenance": processing_compatibility[
            "tcp_retransmission_unit_provenance"
        ],
        "feature_temporal_policy": processing_compatibility["feature_temporal_policy"],
        "processing_compatibility": processing_compatibility,
        "source_manifest_sha256": processing_manifest.get("source_manifest_sha256"),
        "raw_audit": processing_manifest.get("raw_audit"),
        "source_runs": [
            {
                "run_id": record.get("run_id"),
                "configuration_id": record.get("configuration_id"),
                "source_file": record.get("source_file"),
                "source_sha256": record.get("source_sha256"),
            }
            for record in processing_manifest.get("run_records", [])
        ],
        "specification_sha256": specification_sha256,
        "artifact_sha256": artifacts,
        "analysis_type": result.get("analysis_type", "unspecified"),
    }
    (output_dir / "robustness_manifest.json").write_text(
        _json_dumps(manifest) + "\n", encoding="utf-8"
    )
    marker.unlink()
    return {"status": outcome, "output_dir": str(output_dir), "report": str(report_path)}


def verify_robustness_artifacts(
    output_dir: Path, *, processed_dir: Path | None = None, raw_root: Path | None = None
) -> dict[str, Any]:
    from .processing import (
        ProcessingError,
        feature_specification_sha256,
        feature_temporal_policy_from_specification,
        file_sha256,
        resolve_feature_specification,
    )
    from .reporting import ReportError, load_analysis_dataset

    root = Path(output_dir).resolve()
    if (root / "INCOMPLETE").exists():
        raise RobustnessError("robustness report is marked incomplete")
    manifest_path = root / "robustness_manifest.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise RobustnessError("robustness report is missing robustness_manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RobustnessError(f"invalid robustness report manifest: {exc}") from exc
    if manifest.get("robustness_report_schema_version") != 1 or manifest.get("status") not in {
        "complete",
        "partial_failure",
    }:
        raise RobustnessError("robustness report is unsupported or incomplete")
    from .config import ConfigError, _validate_freeze_record

    if manifest.get("protocol_status") != "protocol_frozen" or manifest.get(
        "protocol_sha256"
    ) != manifest.get("config_sha256"):
        raise RobustnessError("robustness report lacks a consistent frozen protocol fingerprint")
    try:
        _validate_freeze_record(manifest.get("protocol_freeze"))
    except ConfigError as exc:
        raise RobustnessError(f"robustness report has invalid freeze metadata: {exc}") from exc
    artifacts = manifest.get("artifact_sha256")
    if not isinstance(artifacts, dict) or set(artifacts) != {"robustness.json", "report.md"}:
        raise RobustnessError("robustness artifact inventory is invalid")
    for relative, expected in artifacts.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RobustnessError(f"artifact path escapes report directory: {relative}") from exc
        if not path.is_file() or file_sha256(path) != expected:
            raise RobustnessError(f"robustness artifact missing or checksum mismatch: {relative}")
    allowed = set(artifacts) | {manifest_path.name}
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")}
    if any(path.is_symlink() or path.is_dir() for path in root.rglob("*")) or actual != allowed:
        raise RobustnessError(f"unexpected or missing report files: {sorted(actual)}")
    if processed_dir is not None:
        try:
            _, processing_manifest = load_analysis_dataset(Path(processed_dir))
        except ReportError as exc:
            raise RobustnessError(str(exc)) from exc
        if file_sha256(Path(processed_dir) / "analysis.csv") != manifest.get("analysis_sha256"):
            raise RobustnessError("prepared analysis differs from robustness report input")
        if file_sha256(Path(processed_dir) / "processing_manifest.json") != manifest.get(
            "processing_manifest_sha256"
        ):
            raise RobustnessError("processing manifest differs from robustness report input")
        if (
            processing_manifest.get("protocol_status") != "protocol_frozen"
            or processing_manifest.get("protocol_sha256") != manifest.get("protocol_sha256")
            or processing_manifest.get("experiment_config_sha256") != manifest.get("config_sha256")
        ):
            raise RobustnessError(
                "robustness report protocol/config fingerprints disagree with prepared data"
            )
        if any(
            key in manifest
            for key in (
                "feature_specification",
                "feature_specification_sha256",
                "tcp_retransmission_unit",
                "feature_temporal_policy",
                "processing_compatibility",
            )
        ):
            try:
                prepared_specification = resolve_feature_specification(processing_manifest)
            except ProcessingError as exc:
                raise RobustnessError(str(exc)) from exc
            prepared_policy = feature_temporal_policy_from_specification(prepared_specification)
            if (
                manifest.get("feature_specification") != prepared_specification
                or manifest.get("feature_specification_sha256")
                != feature_specification_sha256(prepared_specification)
                or manifest.get("tcp_retransmission_unit")
                != prepared_specification.get("tcp_retransmission_unit")
                or manifest.get("feature_temporal_policy") != prepared_policy
                or manifest.get("processing_compatibility", {}).get("feature_specification")
                != prepared_specification
                or manifest.get("processing_compatibility", {}).get("feature_specification_sha256")
                != feature_specification_sha256(prepared_specification)
            ):
                raise RobustnessError(
                    "robustness feature specification, TCP retransmission unit, or "
                    "temporal policy differs from prepared data"
                )
            try:
                result_payload = json.loads((root / "robustness.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RobustnessError(f"invalid robustness result JSON: {exc}") from exc
            if _result_feature_contract_mismatch(
                result_payload,
                specification=prepared_specification,
                digest=feature_specification_sha256(prepared_specification),
            ):
                raise RobustnessError(
                    "a robustness method result does not bind the prepared feature "
                    "specification, TCP unit, temporal policy, and eligibility record"
                )
    if raw_root is not None:
        raw_root = Path(raw_root).resolve()
        for record in manifest.get("source_runs", []):
            source = (raw_root / record["source_file"]).resolve()
            try:
                source.relative_to(raw_root)
            except ValueError as exc:
                raise RobustnessError("raw source path escapes raw root") from exc
            if not source.is_file() or file_sha256(source) != record.get("source_sha256"):
                raise RobustnessError(
                    f"raw source missing or hash mismatch: {record['source_file']}"
                )
    return {
        "status": manifest["status"],
        "output_dir": str(root),
        "verified_artifacts": sorted(artifacts),
        "prepared_data_verified": processed_dir is not None,
        "feature_contract_verified": (
            processed_dir is not None and "feature_specification" in manifest
        ),
        "feature_specification_sha256": manifest.get("feature_specification_sha256"),
        "tcp_retransmission_unit": manifest.get("tcp_retransmission_unit"),
        "feature_temporal_policy": manifest.get("feature_temporal_policy"),
        "raw_sources_verified": raw_root is not None,
    }
