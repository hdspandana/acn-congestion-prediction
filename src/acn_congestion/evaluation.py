"""Grouped train/validation/test evaluation; no row-level fallback is provided."""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import median
from typing import Any

from .baselines import ewma_scores
from .event_evaluation import evaluate_event_scores
from .metrics import binary_metrics, select_threshold_at_max_fpr
from .model_selection import ModelSelectionError, select_model_parameters
from .models import (
    ModelError,
    fit_estimator,
    has_observed_endpoint_feature,
    model_parameter_record,
    predict_scores,
    select_training_observed_features,
    validate_feature_columns,
)
from .processing import (
    ProcessingError,
    apply_feature_temporal_policy_to_rows,
    feature_contract_metadata,
    feature_temporal_eligibility_counts,
    validate_feature_specification_object,
)
from .splits import check_group_isolation


class EvaluationError(ValueError):
    """Evaluation cannot be performed under the requested leakage-safe protocol."""


def _validated_feature_specification(
    feature_specification: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if feature_specification is None:
        return None
    try:
        return validate_feature_specification_object(feature_specification)
    except ProcessingError as exc:
        raise EvaluationError(f"invalid feature specification: {exc}") from exc


def _temporal_metadata(
    specification: dict[str, Any] | None,
    classifications: list[dict[str, Any]] | None = None,
    *,
    scope: str = "all_rows",
) -> dict[str, Any]:
    return feature_contract_metadata(specification, classifications, scope=scope)


def _apply_temporal_policy(
    rows: list[dict[str, Any]],
    feature_columns: Sequence[str],
    specification: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if specification is None:
        return rows, _temporal_metadata(None, scope="all_rows")
    try:
        masked, classifications = apply_feature_temporal_policy_to_rows(
            rows, list(feature_columns), specification
        )
    except ProcessingError as exc:
        raise EvaluationError(str(exc)) from exc
    return masked, _temporal_metadata(specification, classifications, scope="all_rows")


def _apply_temporal_policy_to_indices(
    rows: list[dict[str, Any]],
    indices: Sequence[int],
    feature_columns: Sequence[str],
    specification: dict[str, Any] | None,
    *,
    scope: str,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    selected_rows = [rows[index] for index in indices]
    if specification is None:
        return (
            dict(zip(indices, selected_rows)),
            {},
            _temporal_metadata(None, scope=scope),
        )
    try:
        masked, classifications = apply_feature_temporal_policy_to_rows(
            selected_rows, list(feature_columns), specification
        )
    except ProcessingError as exc:
        raise EvaluationError(str(exc)) from exc
    return (
        dict(zip(indices, masked)),
        dict(zip(indices, classifications, strict=True)),
        _temporal_metadata(specification, classifications, scope=scope),
    )


def _attach_temporal_metadata(result: dict[str, Any], metadata: dict[str, Any]) -> None:
    result.update(metadata)


def _validate_split(
    rows: list[dict[str, Any]], split: dict[str, Any]
) -> tuple[dict[str, list[int]], str]:
    if not isinstance(split, dict):
        raise EvaluationError("split manifest must be an object")
    partitions = split.get("row_indices")
    grouping = split.get("grouping")
    if not isinstance(partitions, dict) or set(partitions) != {"train", "validation", "test"}:
        raise EvaluationError("split manifest must contain train, validation, and test row indices")
    try:
        check_group_isolation(rows, partitions, grouping=grouping)
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationError(f"invalid/leaky split manifest: {exc}") from exc
    return partitions, grouping


def _partition_scored(
    rows: list[dict[str, Any]], indices: list[int], scores: Sequence[float | None]
):
    kept_rows, labels, kept_indices, kept_scores = [], [], [], []
    for index in indices:
        label = rows[index].get("future_event_onset")
        score = scores[index]
        if label in (0, 1) and not isinstance(label, bool) and score is not None:
            if not math.isfinite(float(score)):
                raise EvaluationError(f"non-finite score for row {index}")
            kept_rows.append(rows[index])
            labels.append(int(label))
            kept_indices.append(index)
            kept_scores.append(float(score))
    return kept_rows, labels, kept_indices, kept_scores


def evaluate_scores(
    rows: list[dict[str, Any]],
    split: dict[str, Any],
    scores: Sequence[float | None],
    *,
    identity: str,
    max_validation_fpr: float = 0.05,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Point-wise secondary evaluation; threshold calibration is validation-only."""
    if len(scores) != len(rows):
        raise EvaluationError("one score slot is required per source row")
    partitions, grouping = _validate_split(rows, split)
    train_labels = [
        rows[i].get("future_event_onset")
        for i in partitions["train"]
        if rows[i].get("future_event_onset") in (0, 1)
        and not isinstance(rows[i].get("future_event_onset"), bool)
    ]
    validation = _partition_scored(rows, partitions["validation"], scores)
    test = _partition_scored(rows, partitions["test"], scores)
    for split_name, labels in (
        ("train", train_labels),
        ("validation", validation[1]),
        ("test", test[1]),
    ):
        if set(labels) != {0, 1}:
            class_counts = {0: labels.count(0), 1: labels.count(1)}
            raise EvaluationError(
                f"{split_name} partition must contain both labeled classes; found {class_counts}"
            )
    calibration = select_threshold_at_max_fpr(
        validation[1],
        validation[3],
        max_false_positive_rate=max_validation_fpr,
    )
    metrics = binary_metrics(test[1], test[3], threshold=calibration["threshold"])
    predictions = []
    for index in partitions["test"]:
        row, score = rows[index], scores[index]
        predictions.append(
            {
                "source_row_index": index,
                "run_id": row["run_id"],
                "configuration_id": row["configuration_id"],
                "timestamp_seconds": row.get("timestamp_seconds"),
                "label": row.get("future_event_onset"),
                "score": score,
                "warning": (
                    float(score) >= calibration["threshold"] if score is not None else None
                ),
                "event_active": row.get("event_active"),
            }
        )
    return {
        "model_name": identity,
        "feature_columns": feature_columns or [],
        "grouping": grouping,
        "counts": {
            "train_labeled_rows": len(train_labels),
            "validation_rows_scored": len(validation[1]),
            "test_rows_scored": len(test[1]),
            "train_run_count": len({rows[i]["run_id"] for i in partitions["train"]}),
            "validation_run_count": len({rows[i]["run_id"] for i in partitions["validation"]}),
            "test_run_count": len({rows[i]["run_id"] for i in partitions["test"]}),
            "train_configuration_count": len(
                {rows[i]["configuration_id"] for i in partitions["train"]}
            ),
            "validation_configuration_count": len(
                {rows[i]["configuration_id"] for i in partitions["validation"]}
            ),
            "test_configuration_count": len(
                {rows[i]["configuration_id"] for i in partitions["test"]}
            ),
        },
        "validation_threshold_calibration": calibration,
        "test_metrics": metrics,
        "test_predictions": predictions,
        "warning": "Point-wise secondary software evaluation only; event false-alarm calibration is implemented separately. Source provenance is not verified.",
    }


def evaluate_model(
    rows: list[dict[str, Any]],
    split: dict[str, Any],
    *,
    model_name: str,
    feature_columns: list[str],
    max_validation_fpr: float = 0.05,
    random_seed: int = 2026,
    class_weight: str | None = None,
    model_parameters: dict[str, Any] | None = None,
    feature_specification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit train-only Pipeline, calibrate on validation, and score the held-out test."""
    validate_feature_columns(feature_columns)
    specification = _validated_feature_specification(feature_specification)
    partitions, _ = _validate_split(rows, split)
    masked_by_index, _classifications, temporal_metadata = _apply_temporal_policy_to_indices(
        rows,
        [*partitions["train"], *partitions["validation"], *partitions["test"]],
        feature_columns,
        specification,
        scope="train_validation_test_partitions",
    )
    model_rows = [masked_by_index.get(index, row) for index, row in enumerate(rows)]
    train_rows = [
        model_rows[i]
        for i in partitions["train"]
        if model_rows[i].get("future_event_onset") in (0, 1)
        and not isinstance(model_rows[i].get("future_event_onset"), bool)
        and _has_any_feature(model_rows[i], feature_columns)
    ]
    observed_features, dropped_training_features = select_training_observed_features(
        train_rows, feature_columns
    )
    if not observed_features:
        raise EvaluationError(
            "all requested endpoint features are missing in labeled training rows"
        )
    train_rows = [row for row in train_rows if _has_any_feature(row, observed_features)]
    train_labels = [int(row["future_event_onset"]) for row in train_rows]
    if set(train_labels) != {0, 1}:
        raise EvaluationError("eligible train partition must contain both labeled classes")
    estimator = fit_estimator(
        model_name,
        train_rows,
        train_labels,
        observed_features,
        random_seed=random_seed,
        class_weight=class_weight,
        parameters=model_parameters,
    )
    score_vector: list[float | None] = [None] * len(rows)
    for partition in ("validation", "test"):
        indices = [
            index
            for index in partitions[partition]
            if _has_any_feature(model_rows[index], observed_features)
        ]
        if not indices:
            continue
        scores = predict_scores(estimator, [model_rows[i] for i in indices], observed_features)
        for index, score in zip(indices, scores, strict=True):
            score_vector[index] = score
    result = evaluate_scores(
        model_rows,
        split,
        score_vector,
        identity=model_name,
        max_validation_fpr=max_validation_fpr,
        feature_columns=observed_features,
    )
    _attach_temporal_metadata(result, temporal_metadata)
    result["requested_feature_columns"] = feature_columns
    result["all_missing_training_feature_columns"] = dropped_training_features
    result["counts"]["train_labeled_rows"] = len(train_rows)
    result["counts"]["train_positive_rows_used"] = sum(label == 1 for label in train_labels)
    result["counts"]["train_negative_rows_used"] = sum(label == 0 for label in train_labels)
    result["random_seed"] = random_seed
    result["class_weight"] = class_weight
    result["model_parameter_overrides"] = dict(model_parameters or {})
    result["model_parameters"] = model_parameter_record(
        model_name,
        random_seed=random_seed,
        class_weight=class_weight,
        parameters=model_parameters,
    )
    result["preprocessing_fit_partition"] = "train only (scikit-learn Pipeline)"
    return result


def _has_any_feature(row: dict[str, Any], feature_columns: Sequence[str]) -> bool:
    """Eligibility requires an observed measurement summary, not coverage alone."""
    return has_observed_endpoint_feature(row, feature_columns)


def _baseline_signal_scores(
    rows: list[dict[str, Any]],
    train_indices: list[int],
    *,
    feature_column: str,
    direction: str,
    eligible_feature_columns: Sequence[str] | None = None,
    feature_specification: dict[str, Any] | None = None,
) -> tuple[list[float | None], dict[str, Any]]:
    """Return complete baseline signals using a training-only median for missing inputs."""
    if direction not in {"above", "below"}:
        raise EvaluationError("direction must be above or below")
    endpoint_fields = {"rtt_ms", "loss_fraction", "goodput_bps", "tcp_retransmission_delta"}
    if feature_column not in endpoint_fields:
        try:
            validate_feature_columns([feature_column])
        except ModelError as exc:
            raise EvaluationError(
                f"baseline feature is not an endpoint-only signal: {exc}"
            ) from exc
    if any(feature_column not in row for row in rows):
        raise EvaluationError(f"baseline feature {feature_column!r} is absent")
    specification = _validated_feature_specification(feature_specification)
    temporal_metadata = _temporal_metadata(specification, scope="all_rows")
    if specification is not None:
        if eligible_feature_columns is not None:
            temporal_columns = list(eligible_feature_columns)
        elif feature_column in specification["feature_columns"]:
            temporal_columns = [feature_column]
        else:
            temporal_columns = list(specification["feature_columns"])
        if (
            feature_column in specification["feature_columns"]
            and feature_column not in temporal_columns
        ):
            temporal_columns.append(feature_column)
        rows, temporal_metadata = _apply_temporal_policy(rows, temporal_columns, specification)

    def value_at(index: int) -> float | None:
        raw = rows[index].get(feature_column)
        if raw is None or raw == "":
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise EvaluationError(
                f"invalid baseline feature {feature_column!r} at row {index}: {raw!r}"
            ) from exc
        if not math.isfinite(value):
            raise EvaluationError(f"non-finite baseline feature at row {index}")
        return value

    dropped_eligibility_features: list[str] = []
    if eligible_feature_columns is not None:
        try:
            requested_eligibility = list(eligible_feature_columns)
            validate_feature_columns(requested_eligibility)
        except ModelError as exc:
            raise EvaluationError(
                f"baseline eligibility features are not canonical: {exc}"
            ) from exc
        labeled_train_indices = [
            index
            for index in train_indices
            if rows[index].get("future_event_onset") in (0, 1)
            and not isinstance(rows[index].get("future_event_onset"), bool)
            and _has_any_feature(rows[index], requested_eligibility)
        ]
        eligibility, dropped_eligibility_features = select_training_observed_features(
            [rows[index] for index in labeled_train_indices],
            requested_eligibility,
        )
        if not eligibility:
            raise EvaluationError(
                "all configured baseline-eligibility features are missing in labeled training rows"
            )
    else:
        eligibility = [feature_column]
    train_indices = [
        index
        for index in train_indices
        if rows[index].get("future_event_onset") in (0, 1)
        and not isinstance(rows[index].get("future_event_onset"), bool)
        and _has_any_feature(rows[index], eligibility)
    ]
    training_values = [value for index in train_indices if (value := value_at(index)) is not None]
    if not training_values:
        raise EvaluationError("baseline training partition has no observed feature values")
    fill_value = median(training_values)
    sign = 1.0 if direction == "above" else -1.0
    raw_values = [value_at(index) for index in range(len(rows))]
    eligible_rows = [_has_any_feature(row, eligibility) for row in rows]
    missing_count = sum(
        value is None and eligible for value, eligible in zip(raw_values, eligible_rows)
    )
    scores = [
        None if not eligible else sign * (fill_value if value is None else value)
        for value, eligible in zip(raw_values, eligible_rows)
    ]
    return scores, {
        "feature_column": feature_column,
        "direction": direction,
        "missing_value_imputation": "median fitted on labeled training rows only",
        "training_median": fill_value,
        "training_rows_with_feature": len(training_values),
        "imputed_row_count": missing_count,
        "unscorable_no_endpoint_feature_count": eligible_rows.count(False),
        "all_missing_training_eligibility_features": dropped_eligibility_features,
        "eligibility_feature_columns_used": eligibility,
        **temporal_metadata,
    }


def evaluate_baselines(
    rows: list[dict[str, Any]],
    split: dict[str, Any],
    *,
    feature_column: str,
    max_validation_fpr: float = 0.05,
    ewma_alpha: float = 0.3,
    direction: str = "above",
    eligible_feature_columns: Sequence[str] | None = None,
    feature_specification: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Point-wise static/EWMA rules calibrated on the same validation FPR cap."""
    partitions, _ = _validate_split(rows, split)
    raw_scores, imputation = _baseline_signal_scores(
        rows,
        partitions["train"],
        feature_column=feature_column,
        direction=direction,
        eligible_feature_columns=eligible_feature_columns,
        feature_specification=feature_specification,
    )
    ewma = ewma_scores(raw_scores, [row.get("run_id") for row in rows], alpha=ewma_alpha)
    results = {
        "static_threshold": evaluate_scores(
            rows,
            split,
            raw_scores,
            identity=f"static_threshold:{feature_column}:{direction}",
            max_validation_fpr=max_validation_fpr,
            feature_columns=[feature_column],
        ),
        "ewma": evaluate_scores(
            rows,
            split,
            ewma,
            identity=f"ewma:{feature_column}:alpha={ewma_alpha:g}:{direction}",
            max_validation_fpr=max_validation_fpr,
            feature_columns=[feature_column],
        ),
    }
    for result in results.values():
        result["baseline_input_handling"] = imputation
        _attach_temporal_metadata(
            result,
            {
                key: imputation[key]
                for key in (
                    "feature_specification",
                    "feature_specification_sha256",
                    "feature_specification_status",
                    "tcp_retransmission_unit",
                    "feature_temporal_policy",
                    "feature_temporal_eligibility",
                )
            },
        )
        result["ewma_alpha"] = ewma_alpha
    return results


def fit_model_on_training_split(
    rows: list[dict[str, Any]],
    split: dict[str, Any],
    *,
    model_name: str,
    feature_columns: list[str],
    random_seed: int = 2026,
    class_weight: str | None = None,
    model_parameters: dict[str, Any] | None = None,
    feature_specification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit the exact train-only endpoint Pipeline used by grouped event evaluation."""
    validate_feature_columns(feature_columns)
    specification = _validated_feature_specification(feature_specification)
    partitions, grouping = _validate_split(rows, split)
    masked_by_index, _classifications, temporal_metadata = _apply_temporal_policy_to_indices(
        rows,
        partitions["train"],
        feature_columns,
        specification,
        scope="train_partition",
    )
    train_rows = [
        masked_by_index[i]
        for i in partitions["train"]
        if masked_by_index[i].get("future_event_onset") in (0, 1)
        and not isinstance(masked_by_index[i].get("future_event_onset"), bool)
        and _has_any_feature(masked_by_index[i], feature_columns)
    ]
    observed_features, dropped_training_features = select_training_observed_features(
        train_rows, feature_columns
    )
    if not observed_features:
        raise EvaluationError(
            "all requested endpoint features are missing in labeled training rows"
        )
    train_rows = [row for row in train_rows if _has_any_feature(row, observed_features)]
    train_labels = [int(row["future_event_onset"]) for row in train_rows]
    if set(train_labels) != {0, 1}:
        raise EvaluationError("eligible train partition must contain both labeled classes")
    estimator = fit_estimator(
        model_name,
        train_rows,
        train_labels,
        observed_features,
        random_seed=random_seed,
        class_weight=class_weight,
        parameters=model_parameters,
    )
    return {
        "estimator": estimator,
        "partitions": partitions,
        "grouping": grouping,
        "feature_columns": observed_features,
        "model_parameter_overrides": dict(model_parameters or {}),
        "model_parameters": model_parameter_record(
            model_name,
            random_seed=random_seed,
            class_weight=class_weight,
            parameters=model_parameters,
        ),
        "dropped_training_features": dropped_training_features,
        "training_counts": {
            "labeled_rows_used": len(train_rows),
            "positive_rows_used": sum(label == 1 for label in train_labels),
            "negative_rows_used": sum(label == 0 for label in train_labels),
        },
        "preprocessing_fit_partition": "train only (scikit-learn Pipeline)",
        **temporal_metadata,
    }


def evaluate_model_events(
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
    model_parameters: dict[str, Any] | None = None,
    model_selection: dict[str, Any] | None = None,
    bootstrap_resamples: int = 2000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 2026,
    feature_specification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit on training groups and calibrate the event FAPM cap on validation only."""
    specification = _validated_feature_specification(feature_specification)
    selection_result: dict[str, Any] | None = None
    selected_parameters = model_parameters
    if model_selection is not None:
        if model_parameters is not None:
            raise EvaluationError(
                "specify either fixed model_parameters or model_selection, not both"
            )
        try:
            selection_result = select_model_parameters(
                rows,
                split,
                model_name=model_name,
                feature_columns=feature_columns,
                selection_config=model_selection,
                max_false_alarms_per_minute=max_false_alarms_per_minute,
                horizon_seconds=horizon_seconds,
                max_gap_seconds=max_gap_seconds,
                alarm_merge_gap_seconds=alarm_merge_gap_seconds,
                random_seed=random_seed,
                class_weight=class_weight,
                feature_specification=specification,
            )
        except ModelSelectionError as exc:
            raise EvaluationError(str(exc)) from exc
        selected_parameters = selection_result.get("selected_parameters") or {}
    fitted = fit_model_on_training_split(
        rows,
        split,
        model_name=model_name,
        feature_columns=feature_columns,
        random_seed=random_seed,
        class_weight=class_weight,
        model_parameters=selected_parameters,
        feature_specification=specification,
    )
    partitions = fitted["partitions"]
    observed_features = fitted["feature_columns"]
    estimator = fitted["estimator"]
    dropped_training_features = fitted["dropped_training_features"]
    train_counts = fitted["training_counts"]
    scoring_rows, scoring_classifications, scoring_temporal_metadata = (
        _apply_temporal_policy_to_indices(
            rows,
            [*partitions["validation"], *partitions["test"]],
            observed_features,
            specification,
            scope="validation_test_partitions",
        )
    )
    score_vector: list[float | None] = [None] * len(rows)
    for partition in ("validation", "test"):
        indices = [
            index
            for index in partitions[partition]
            if _has_any_feature(scoring_rows[index], observed_features)
        ]
        if not indices:
            continue
        values = predict_scores(estimator, [scoring_rows[i] for i in indices], observed_features)
        for index, score in zip(indices, values, strict=True):
            score_vector[index] = score
    result = evaluate_event_scores(
        rows,
        split,
        score_vector,
        identity=model_name,
        max_false_alarms_per_minute=max_false_alarms_per_minute,
        horizon_seconds=horizon_seconds,
        max_gap_seconds=max_gap_seconds,
        alarm_merge_gap_seconds=alarm_merge_gap_seconds,
        feature_columns=observed_features,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_confidence=bootstrap_confidence,
        bootstrap_seed=bootstrap_seed,
    )
    result["random_seed"] = random_seed
    result["class_weight"] = class_weight
    result["model_parameter_overrides"] = fitted["model_parameter_overrides"]
    result["model_parameters"] = fitted["model_parameters"]
    result["model_selection"] = selection_result or {
        "status": "not_requested",
        "selected_parameters": dict(model_parameters or {}),
        "selection_partition": "not used; predeclared fixed parameters",
        "outer_test_used": False,
        "outer_validation_used_for_hyperparameters": False,
        **{
            key: fitted[key]
            for key in (
                "feature_specification",
                "feature_specification_sha256",
                "feature_specification_status",
                "tcp_retransmission_unit",
                "feature_temporal_policy",
                "feature_temporal_eligibility",
            )
        },
    }
    result["training_counts"] = train_counts
    result["requested_feature_columns"] = feature_columns
    result["all_missing_training_feature_columns"] = dropped_training_features
    result["preprocessing_fit_partition"] = "train only (scikit-learn Pipeline)"
    _attach_temporal_metadata(
        result,
        {
            key: fitted[key]
            for key in (
                "feature_specification",
                "feature_specification_sha256",
                "feature_specification_status",
                "tcp_retransmission_unit",
                "feature_temporal_policy",
                "feature_temporal_eligibility",
            )
        },
    )
    result["feature_temporal_eligibility_scoring"] = scoring_temporal_metadata[
        "feature_temporal_eligibility"
    ]
    combined_scoring_eligibility = result["feature_temporal_eligibility_scoring"]
    for partition in ("validation", "test"):
        scope = f"{partition}_partition"
        if "anchor_count" in combined_scoring_eligibility:
            partition_counts = {
                **feature_temporal_eligibility_counts(
                    [scoring_classifications[index] for index in partitions[partition]]
                ),
                "scope": scope,
            }
        else:
            partition_counts = {**combined_scoring_eligibility, "scope": scope}
        result[f"feature_temporal_eligibility_{partition}"] = partition_counts
    return result


def evaluate_baseline_events(
    rows: list[dict[str, Any]],
    split: dict[str, Any],
    *,
    feature_column: str,
    max_false_alarms_per_minute: float,
    horizon_seconds: float,
    max_gap_seconds: float,
    alarm_merge_gap_seconds: float,
    ewma_alpha: float = 0.3,
    direction: str = "above",
    eligible_feature_columns: Sequence[str] | None = None,
    bootstrap_resamples: int = 2000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 2026,
    feature_specification: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Event-matched threshold and EWMA baselines at a validation event-FAPM cap."""
    partitions, _ = _validate_split(rows, split)
    raw_scores, imputation = _baseline_signal_scores(
        rows,
        partitions["train"],
        feature_column=feature_column,
        direction=direction,
        eligible_feature_columns=eligible_feature_columns,
        feature_specification=feature_specification,
    )
    ewma = ewma_scores(raw_scores, [row.get("run_id") for row in rows], alpha=ewma_alpha)
    common = {
        "max_false_alarms_per_minute": max_false_alarms_per_minute,
        "horizon_seconds": horizon_seconds,
        "max_gap_seconds": max_gap_seconds,
        "alarm_merge_gap_seconds": alarm_merge_gap_seconds,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_confidence": bootstrap_confidence,
        "bootstrap_seed": bootstrap_seed,
    }
    results = {
        "static_threshold": evaluate_event_scores(
            rows,
            split,
            raw_scores,
            identity=f"static_threshold:{feature_column}:{direction}",
            feature_columns=[feature_column],
            **common,
        ),
        "ewma": evaluate_event_scores(
            rows,
            split,
            ewma,
            identity=f"ewma:{feature_column}:alpha={ewma_alpha:g}:{direction}",
            feature_columns=[feature_column],
            **common,
        ),
    }
    for result in results.values():
        result["baseline_input_handling"] = imputation
        _attach_temporal_metadata(
            result,
            {
                key: imputation[key]
                for key in (
                    "feature_specification",
                    "feature_specification_sha256",
                    "feature_specification_status",
                    "tcp_retransmission_unit",
                    "feature_temporal_policy",
                    "feature_temporal_eligibility",
                )
            },
        )
        result["ewma_alpha"] = ewma_alpha
    return results
