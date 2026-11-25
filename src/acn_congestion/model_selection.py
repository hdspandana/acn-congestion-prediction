"""Small, group-aware hyperparameter selection inside outer training groups.

The outer train/validation/test protocol remains authoritative: this module
uses only outer-training groups to choose among a bounded, predeclared set of
hyperparameters.  The operating threshold is still chosen afterwards on the
outer validation partition, and outer test labels are never read here.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from statistics import mean, median
from typing import Any

from .event_evaluation import EventEvaluationError, select_threshold_at_max_fapm
from .metrics import binary_metrics
from .models import (
    ModelError,
    default_parameter_candidates,
    fit_estimator,
    has_observed_endpoint_feature,
    model_parameter_record,
    predict_scores,
    resolve_model_parameters,
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
from .splits import SplitError, check_group_isolation, make_grouped_folds

MAX_SELECTION_FITS = 30
_SUPPORTED_SELECTION_KEYS = {
    "enabled",
    "inner_folds",
    "inner_repeats",
    "seed",
    "candidate_parameters",
    "unsupported_policy",
}


class ModelSelectionError(ValueError):
    """Grouped inner model selection is invalid or unsupported by the groups."""


def _temporal_metadata(
    specification: dict[str, Any] | None,
    classifications: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return feature_contract_metadata(
        specification, classifications, scope="outer_training_partition"
    )


def _candidate_list(raw: Any, model_name: str) -> list[dict[str, Any]]:
    if raw is None:
        return default_parameter_candidates(model_name)
    if isinstance(raw, Mapping):
        if model_name not in raw:
            raise ModelSelectionError(
                f"model_selection.candidate_parameters has no candidate list for {model_name!r}"
            )
        raw = raw[model_name]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ModelSelectionError("model selection candidates must be a non-empty list")
    if len(raw) > 8:
        raise ModelSelectionError("model selection supports at most 8 candidate parameter sets")
    candidates: list[dict[str, Any]] = []
    for candidate in raw:
        if not isinstance(candidate, Mapping):
            raise ModelSelectionError("each model selection candidate must be an object")
        candidate_dict = dict(candidate)
        try:
            resolve_model_parameters(model_name, candidate_dict)
        except ModelError as exc:
            raise ModelSelectionError(str(exc)) from exc
        candidates.append(candidate_dict)
    return candidates


def normalize_model_selection_config(
    value: Any,
    *,
    model_name: str,
    default_seed: int,
) -> dict[str, Any] | None:
    """Validate the optional frozen-config model-selection object for one method."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ModelSelectionError("evaluation.model_selection must be an object or null")
    unknown = sorted(set(value) - _SUPPORTED_SELECTION_KEYS)
    if unknown:
        raise ModelSelectionError(f"unsupported model_selection key(s): {unknown}")
    enabled = value.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ModelSelectionError("model_selection.enabled must be a boolean")
    if not enabled:
        return {"enabled": False}
    inner_folds = value.get("inner_folds", 3)
    inner_repeats = value.get("inner_repeats", 1)
    seed = value.get("seed", default_seed)
    unsupported_policy = value.get("unsupported_policy", "fail")
    if (
        isinstance(inner_folds, bool)
        or not isinstance(inner_folds, int)
        or not 2 <= inner_folds <= 5
    ):
        raise ModelSelectionError("model_selection.inner_folds must be an integer in [2, 5]")
    if (
        isinstance(inner_repeats, bool)
        or not isinstance(inner_repeats, int)
        or not 1 <= inner_repeats <= 3
    ):
        raise ModelSelectionError("model_selection.inner_repeats must be an integer in [1, 3]")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ModelSelectionError("model_selection.seed must be a non-negative integer")
    if unsupported_policy not in {"fail", "use_predeclared_default"}:
        raise ModelSelectionError(
            "model_selection.unsupported_policy must be 'fail' or 'use_predeclared_default'"
        )
    candidates = _candidate_list(value.get("candidate_parameters"), model_name)
    total_fits = len(candidates) * inner_folds * inner_repeats
    if total_fits > MAX_SELECTION_FITS:
        raise ModelSelectionError(
            f"model selection would require {total_fits} inner fits; the bounded limit is {MAX_SELECTION_FITS}"
        )
    return {
        "enabled": True,
        "inner_folds": inner_folds,
        "inner_repeats": inner_repeats,
        "seed": seed,
        "candidate_parameters": candidates,
        "unsupported_policy": unsupported_policy,
        "max_inner_fits": total_fits,
    }


def _is_label(row: dict[str, Any], value: int) -> bool:
    label = row.get("future_event_onset")
    return label == value and not isinstance(label, bool)


def _eligible_training_rows(
    rows: list[dict[str, Any]],
    indices: Sequence[int],
    feature_columns: Sequence[str],
) -> tuple[list[dict[str, Any]], list[int], list[str], list[str]]:
    initial = [
        rows[index]
        for index in indices
        if any(_is_label(rows[index], value) for value in (0, 1))
        and has_observed_endpoint_feature(rows[index], feature_columns)
    ]
    observed, dropped = select_training_observed_features(initial, feature_columns)
    if not observed:
        raise ModelSelectionError("inner training fold has no observed endpoint features")
    eligible = [row for row in initial if has_observed_endpoint_feature(row, observed)]
    labels = [int(row["future_event_onset"]) for row in eligible]
    if set(labels) != {0, 1}:
        raise ModelSelectionError("inner training fold does not contain both labeled classes")
    return eligible, labels, observed, dropped


def _fold_summary(
    fold: dict[str, Any],
    classifications_by_index: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    summary = {
        "fold_id": fold["fold_id"],
        "repeat": fold["repeat"],
        "fold": fold["fold"],
        "train_groups": fold["train_groups"],
        "validation_groups": fold["validation_groups"],
        "train_row_count": len(fold["train_indices"]),
        "validation_row_count": len(fold["validation_indices"]),
    }
    if classifications_by_index:
        summary["train_feature_temporal_eligibility"] = {
            **feature_temporal_eligibility_counts(
                [classifications_by_index[index] for index in fold["train_indices"]]
            ),
            "scope": "inner_train_fold",
        }
        summary["validation_feature_temporal_eligibility"] = {
            **feature_temporal_eligibility_counts(
                [classifications_by_index[index] for index in fold["validation_indices"]]
            ),
            "scope": "inner_validation_fold",
        }
    return summary


def select_model_parameters(
    rows: list[dict[str, Any]],
    outer_split: dict[str, Any],
    *,
    model_name: str,
    feature_columns: list[str],
    selection_config: Mapping[str, Any],
    max_false_alarms_per_minute: float,
    horizon_seconds: float,
    max_gap_seconds: float,
    alarm_merge_gap_seconds: float,
    random_seed: int = 2026,
    class_weight: str | None = None,
    feature_specification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Choose hyperparameters by repeated grouped folds within outer training data."""
    validate_feature_columns(feature_columns)
    try:
        specification = (
            None
            if feature_specification is None
            else validate_feature_specification_object(feature_specification)
        )
    except ProcessingError as exc:
        raise ModelSelectionError(f"invalid feature specification: {exc}") from exc
    temporal_metadata = _temporal_metadata(specification)
    normalized = normalize_model_selection_config(
        selection_config, model_name=model_name, default_seed=random_seed
    )
    if normalized is None or not normalized.get("enabled"):
        return {
            "status": "disabled",
            "selected_parameters": {},
            "selection_partition": "not used; predeclared fixed parameters",
            "outer_test_used": False,
            **temporal_metadata,
        }
    if not isinstance(outer_split, dict):
        raise ModelSelectionError("outer split must be an object")
    partitions = outer_split.get("row_indices")
    grouping = outer_split.get("grouping")
    if not isinstance(partitions, dict) or set(partitions) != {"train", "validation", "test"}:
        raise ModelSelectionError("outer split must contain train/validation/test row indices")
    try:
        check_group_isolation(rows, partitions, grouping=grouping)
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelSelectionError(f"outer split is invalid or leaking: {exc}") from exc
    try:
        fold_manifest = make_grouped_folds(
            rows,
            indices=partitions["train"],
            grouping=grouping,
            n_splits=normalized["inner_folds"],
            repeats=normalized["inner_repeats"],
            seed=normalized["seed"],
        )
    except (KeyError, SplitError, ValueError) as exc:
        if normalized["unsupported_policy"] == "use_predeclared_default":
            return {
                "status": "unsupported_fallback_to_predeclared_default",
                "selected_candidate_index": None,
                "selected_parameters": {},
                "selection_partition": "outer training groups only (grouped inner folds)",
                "outer_test_used": False,
                "grouping": grouping,
                "configuration": normalized,
                "unsupported_reason": str(exc),
                "candidates": [],
                **temporal_metadata,
            }
        raise ModelSelectionError(f"grouped inner selection is unsupported: {exc}") from exc

    effective_rows = list(rows)
    classifications_by_index: dict[int, dict[str, Any]] = {}
    if specification is not None:
        try:
            masked_train_rows, classifications = apply_feature_temporal_policy_to_rows(
                [rows[index] for index in partitions["train"]],
                list(feature_columns),
                specification,
            )
        except ProcessingError as exc:
            raise ModelSelectionError(str(exc)) from exc
        for index, masked in zip(partitions["train"], masked_train_rows, strict=True):
            effective_rows[index] = masked
        classifications_by_index = dict(zip(partitions["train"], classifications, strict=True))
        temporal_metadata = _temporal_metadata(specification, classifications)

    candidate_results: list[dict[str, Any]] = []
    selection_started = time.perf_counter()
    for candidate_index, parameters in enumerate(normalized["candidate_parameters"]):
        candidate_started = time.perf_counter()
        fold_summaries: list[dict[str, Any]] = []
        pooled = {
            "event_count": 0,
            "alarm_episode_count": 0,
            "matched_event_count": 0,
            "false_alarm_episode_count": 0,
            "eligible_exposure_seconds": 0.0,
        }
        lead_times: list[float] = []
        average_precisions: list[float] = []
        feasible_folds = 0
        for fold in fold_manifest["folds"]:
            fold_started = time.perf_counter()
            summary = _fold_summary(fold, classifications_by_index)
            try:
                train_rows, train_labels, observed_features, dropped = _eligible_training_rows(
                    effective_rows, fold["train_indices"], feature_columns
                )
                estimator = fit_estimator(
                    model_name,
                    train_rows,
                    train_labels,
                    observed_features,
                    random_seed=random_seed,
                    class_weight=class_weight,
                    parameters=parameters,
                )
                score_vector: list[float | None] = [None] * len(effective_rows)
                validation_indices = list(fold["validation_indices"])
                scorable_indices = [
                    index
                    for index in validation_indices
                    if has_observed_endpoint_feature(effective_rows[index], observed_features)
                ]
                if scorable_indices:
                    raw_scores = predict_scores(
                        estimator,
                        [effective_rows[index] for index in scorable_indices],
                        observed_features,
                    )
                    for index, score in zip(scorable_indices, raw_scores, strict=True):
                        score_vector[index] = score
                threshold_result = select_threshold_at_max_fapm(
                    effective_rows,
                    validation_indices,
                    score_vector,
                    max_false_alarms_per_minute=max_false_alarms_per_minute,
                    horizon_seconds=horizon_seconds,
                    max_gap_seconds=max_gap_seconds,
                    alarm_merge_gap_seconds=alarm_merge_gap_seconds,
                )
                aggregate = threshold_result["validation_event_metrics"]
                for key in pooled:
                    pooled[key] += aggregate[key]
                lead_times.extend(
                    pair["lead_seconds"]
                    for run in aggregate["per_run"]
                    for pair in run["matched_alarm_event_times"]
                )
                labels = [
                    effective_rows[index].get("future_event_onset") for index in validation_indices
                ]
                values = [score_vector[index] for index in validation_indices]
                average_precision = binary_metrics(
                    labels, values, threshold=threshold_result["threshold"]
                )["average_precision"]
                if average_precision is not None:
                    average_precisions.append(float(average_precision))
                feasible_folds += 1
                summary.update(
                    {
                        "status": "feasible",
                        "feature_columns": observed_features,
                        "all_missing_inner_training_feature_columns": dropped,
                        "training_rows_used": len(train_rows),
                        "training_positive_rows": sum(label == 1 for label in train_labels),
                        "training_negative_rows": sum(label == 0 for label in train_labels),
                        "selected_inner_threshold": threshold_result["threshold"],
                        "event_count": aggregate["event_count"],
                        "event_precision": aggregate["event_precision"],
                        "event_recall": aggregate["event_recall"],
                        "false_alarms_per_minute": aggregate["false_alarms_per_minute"],
                        "average_precision_secondary": average_precision,
                        "runtime_seconds": time.perf_counter() - fold_started,
                    }
                )
            except (
                EventEvaluationError,
                ModelError,
                ModelSelectionError,
                ValueError,
                ImportError,
            ) as exc:
                summary.update(
                    {
                        "status": "unsupported",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "runtime_seconds": time.perf_counter() - fold_started,
                    }
                )
            fold_summaries.append(summary)
        pooled_precision = (
            pooled["matched_event_count"] / pooled["alarm_episode_count"]
            if pooled["alarm_episode_count"]
            else None
        )
        pooled_recall = (
            pooled["matched_event_count"] / pooled["event_count"] if pooled["event_count"] else None
        )
        pooled_fapm = (
            pooled["false_alarm_episode_count"] / (pooled["eligible_exposure_seconds"] / 60)
            if pooled["eligible_exposure_seconds"] > 0
            else None
        )
        candidate_results.append(
            {
                "candidate_index": candidate_index,
                "parameters": parameters,
                "resolved_model_parameters": model_parameter_record(
                    model_name,
                    random_seed=random_seed,
                    class_weight=class_weight,
                    parameters=parameters,
                ),
                "feasible_fold_count": feasible_folds,
                "unsupported_fold_count": len(fold_summaries) - feasible_folds,
                "pooled_inner_event_metrics": {
                    **pooled,
                    "event_precision": pooled_precision,
                    "event_recall": pooled_recall,
                    "false_alarms_per_minute": pooled_fapm,
                    "median_warning_lead_seconds": median(lead_times) if lead_times else None,
                },
                "mean_average_precision_secondary": (
                    mean(average_precisions) if average_precisions else None
                ),
                "runtime_seconds": time.perf_counter() - candidate_started,
                "folds": fold_summaries,
            }
        )

    feasible = [
        candidate for candidate in candidate_results if candidate["feasible_fold_count"] > 0
    ]
    if not feasible:
        if normalized["unsupported_policy"] == "use_predeclared_default":
            return {
                "status": "unsupported_fallback_to_predeclared_default",
                "selected_candidate_index": None,
                "selected_parameters": {},
                "selection_partition": "outer training groups only (grouped inner folds)",
                "outer_test_used": False,
                "grouping": grouping,
                "fold_manifest": {
                    key: fold_manifest[key]
                    for key in (
                        "grouping",
                        "n_splits",
                        "repeats",
                        "seed",
                        "group_count",
                        "row_count",
                        "assignment",
                    )
                },
                "configuration": normalized,
                "candidates": candidate_results,
                "runtime_seconds": time.perf_counter() - selection_started,
                **temporal_metadata,
            }
        raise ModelSelectionError(
            "no hyperparameter candidate was feasible on any grouped inner fold; see candidate fold reasons"
        )

    def selection_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, float, int]:
        pooled = candidate["pooled_inner_event_metrics"]
        return (
            float(candidate["feasible_fold_count"]),
            pooled["event_recall"] if pooled["event_recall"] is not None else -1.0,
            pooled["event_precision"] if pooled["event_precision"] is not None else -1.0,
            candidate["mean_average_precision_secondary"]
            if candidate["mean_average_precision_secondary"] is not None
            else -1.0,
            pooled["median_warning_lead_seconds"]
            if pooled["median_warning_lead_seconds"] is not None
            else -1.0,
            -int(candidate["candidate_index"]),
        )

    selected = max(feasible, key=selection_key)
    return {
        "status": "selected",
        "selected_candidate_index": selected["candidate_index"],
        "selected_parameters": selected["parameters"],
        "selection_partition": "outer training groups only (grouped inner folds)",
        "outer_test_used": False,
        "outer_validation_used_for_hyperparameters": False,
        "grouping": grouping,
        "selection_criterion": [
            "feasible grouped inner folds",
            "pooled inner event recall at thresholds satisfying the declared false-alarms-per-minute cap",
            "pooled inner event precision",
            "mean secondary average precision",
            "median warning lead time",
            "earlier predeclared candidate on ties",
        ],
        "fold_manifest": {
            key: fold_manifest[key]
            for key in (
                "grouping",
                "n_splits",
                "repeats",
                "seed",
                "group_count",
                "row_count",
                "assignment",
            )
        },
        "configuration": normalized,
        "candidates": candidate_results,
        "runtime_seconds": time.perf_counter() - selection_started,
        **temporal_metadata,
        "limitations": "Inner-fold thresholds are selected and assessed within each inner validation fold for candidate comparison; final operating-threshold selection remains on the untouched outer validation partition. Sparse independent events can make this selection unstable.",
    }
