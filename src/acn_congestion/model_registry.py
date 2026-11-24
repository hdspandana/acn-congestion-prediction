"""Safely train, register, and resolve evaluation-linked inference candidates.

Only artifacts written by this application are loadable; no model pickle is
accepted from an upload or caller-controlled path. Eligibility is rechecked
against current protocol, report, processed-data, audit, and source fingerprints.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ConfigError, load_config, verify_protocol_freeze_evidence
from .dashboard_data import (
    DashboardDataError,
    Workspace,
    discover_report_directories,
    file_sha256,
    read_json,
    safe_child,
    verify_evaluation_bundle,
    verify_processed_artifacts,
    verify_workspace_lineage_sources,
)
from .evaluation import fit_model_on_training_split
from .models import ModelError, model_parameter_record, validate_feature_columns
from .processing import (
    ProcessingError,
    feature_specification_operational_status,
    feature_specification_sha256,
    feature_temporal_policy_from_specification,
    normalize_feature_temporal_policy,
    validate_feature_specification_object,
)
from .reporting import ReportError, load_analysis_dataset
from .splits import check_group_isolation

MODEL_ID_PATTERN = re.compile(r"^model_[0-9]{8}T[0-9]{6}Z_[a-f0-9]{8}$")
SUPPORTED_METHODS = {"logistic_regression", "random_forest", "hist_gradient_boosting"}
MAX_MODEL_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_ANALYSIS_BYTES_FOR_TRAINING = 64 * 1024 * 1024
MAX_ANALYSIS_ROWS_FOR_TRAINING = 100_000


class ModelRegistryError(ValueError):
    """Candidate training, registry integrity, or eligibility failure."""


def model_view(
    manifest: dict[str, Any], *, eligible: bool, reason: str | None = None
) -> dict[str, Any]:
    """Return a schema-safe public summary without inventing contract fields.

    Registry manifests are local JSON artifacts and may be incomplete or
    hand-edited.  Eligibility is decided by ``ModelRegistry`` before this
    presentation helper is used; malformed values are represented as absent
    or explicitly ``not_recorded`` here so listing an ineligible artifact
    cannot fail with an attribute/type error or look like a valid contract.
    """

    def optional_string(value: Any) -> str | None:
        return value if isinstance(value, str) and value else None

    def optional_number(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        return number if math.isfinite(number) else None

    specification = manifest.get("feature_specification")
    specification = specification if isinstance(specification, dict) else {}
    temporal_policy = manifest.get("feature_temporal_policy")
    temporal_policy = temporal_policy if isinstance(temporal_policy, dict) else None
    feature_columns = manifest.get("feature_columns")
    feature_columns = (
        list(feature_columns)
        if isinstance(feature_columns, (list, tuple))
        and all(isinstance(column, str) for column in feature_columns)
        else []
    )
    return {
        "model_id": optional_string(manifest.get("model_id")),
        "method": optional_string(manifest.get("method")),
        "created_at_utc": optional_string(manifest.get("created_at_utc")),
        "eligible": bool(eligible),
        "status": "eligible" if eligible else "ineligible",
        "reason": reason if isinstance(reason, str) and reason else None,
        "forecast_horizon_seconds": optional_number(manifest.get("forecast_horizon_seconds")),
        "score_threshold": optional_number(manifest.get("score_threshold")),
        "score_semantics": optional_string(manifest.get("score_semantics")) or "not_recorded",
        "calibration_status": (
            optional_string(manifest.get("calibration_status")) or "not_recorded"
        ),
        "feature_columns": feature_columns,
        "feature_spec_version": optional_string(specification.get("spec_version")),
        "feature_specification_sha256": optional_string(
            manifest.get("feature_specification_sha256")
        ),
        "feature_specification_status": optional_string(
            manifest.get("feature_specification_status")
        ),
        "tcp_retransmission_unit": optional_string(manifest.get("tcp_retransmission_unit")),
        "feature_temporal_policy": temporal_policy,
        "feature_temporal_policy_status": optional_string((temporal_policy or {}).get("status")),
        "model_parameter_overrides": (
            manifest["model_parameter_overrides"]
            if isinstance(manifest.get("model_parameter_overrides"), dict)
            else None
        ),
        "evaluation_id": optional_string(manifest.get("evaluation_id")),
        "protocol_sha256": optional_string(manifest.get("protocol_sha256")),
        "analysis_sha256": optional_string(manifest.get("analysis_sha256")),
    }


class ModelRegistry:
    def __init__(self, workspace: Workspace, model_root: Path, state_root: Path):
        self.workspace = workspace
        self.model_root = model_root
        self.state_root = state_root
        # Eligibility and artifact hashes are rechecked before every resolve;
        # this bounded cache only avoids repeatedly deserializing the exact
        # verified artifact bytes for every inference request.
        # Bind reuse to both the verified estimator bytes and the exact
        # verified model-manifest bytes/feature contract. Eligibility is still
        # re-evaluated before every cache lookup; this only prevents conflating
        # two different verified manifests that happen to share estimator bytes.
        self._estimator_cache: OrderedDict[tuple[str, str, str, str], Any] = OrderedDict()
        self._estimator_cache_lock = threading.Lock()

    @property
    def active_file(self) -> Path:
        return self.state_root / "active_model.json"

    def _read_active_id(self) -> str | None:
        if self.active_file.is_symlink() or not self.active_file.is_file():
            return None
        try:
            value = read_json(self.active_file).get("model_id")
            return value if isinstance(value, str) and MODEL_ID_PATTERN.fullmatch(value) else None
        except (DashboardDataError, OSError, ValueError):
            return None

    def list_models(self) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        if self.model_root.is_symlink() or not self.model_root.is_dir():
            return {"models": [], "active_model_id": None}
        candidates = sorted(
            (p for p in self.model_root.iterdir() if p.is_dir() and not p.is_symlink()),
            key=lambda p: p.name,
            reverse=True,
        )[:100]
        try:
            source_proof = self._current_source_proof()
        except (ModelRegistryError, DashboardDataError, OSError, ValueError) as exc:
            source_proof = (None, {"status": "unverified", "error": str(exc)})
        for directory in candidates:
            if not MODEL_ID_PATTERN.fullmatch(directory.name):
                continue
            manifest_path = directory / "model_manifest.json"
            try:
                manifest = read_json(manifest_path)
            except (DashboardDataError, OSError, ValueError) as exc:
                entries.append(
                    model_view(
                        {"model_id": directory.name},
                        eligible=False,
                        reason=f"malformed or unavailable model manifest: {exc}",
                    )
                )
                continue
            if manifest.get("model_id") != directory.name or manifest.get("status") != "candidate":
                entries.append(
                    model_view(
                        {**manifest, "model_id": directory.name},
                        eligible=False,
                        reason="model manifest identity or candidate status is invalid",
                    )
                )
                continue
            try:
                reason = self._eligibility_reason(directory, manifest, source_proof=source_proof)
                entries.append(model_view(manifest, eligible=reason is None, reason=reason))
            except (
                DashboardDataError,
                OSError,
                ValueError,
                ModelRegistryError,
                ProcessingError,
                ReportError,
                AttributeError,
                TypeError,
                KeyError,
                IndexError,
            ) as exc:
                entries.append(
                    model_view(
                        manifest,
                        eligible=False,
                        reason=(
                            "malformed model manifest or linked artifact: "
                            f"{exc or type(exc).__name__}"
                        ),
                    )
                )
        active = self._read_active_id()
        if active not in {item["model_id"] for item in entries if item["eligible"]}:
            active = None
        return {"models": entries, "active_model_id": active}

    def _evaluation_directory(self, evaluation_id: str) -> Path:
        if not isinstance(evaluation_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]{1,120}", evaluation_id
        ):
            raise ModelRegistryError("invalid evaluation identifier")
        matches = [
            path
            for path in discover_report_directories(
                self.workspace.reports_root,
                "evaluation_manifest.json",
                max_reports=100,
            )
            if path.name == evaluation_id
        ]
        if len(matches) != 1 or matches[0].is_symlink():
            raise ModelRegistryError("evaluation report is missing, ambiguous, or unsafe")
        return matches[0]

    def _current_source_proof(self) -> tuple[dict[str, Any], dict[str, Any]]:
        processed = verify_processed_artifacts(self.workspace.processed_root)
        if processed.get("status") != "verified":
            raise ModelRegistryError("processed dataset is missing or its checksums do not verify")
        from .dashboard_data import verify_audit_inventory

        audit = verify_audit_inventory(self.workspace.raw_root, self.workspace.audit_report_path)
        shared = verify_workspace_lineage_sources(self.workspace, processed, audit)
        return processed, shared

    @staticmethod
    def _evaluation_model_contract(method_metrics: dict[str, Any], method: str) -> dict[str, Any]:
        """Recover the exact fitted parameter contract from an evaluation record."""
        record = method_metrics.get("model_parameters")
        if not isinstance(record, dict):
            raise ModelRegistryError("evaluation lacks a resolved model parameter record")
        random_seed = method_metrics.get("random_seed")
        if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
            raise ModelRegistryError("evaluation lacks a valid model random seed")
        class_weight = method_metrics.get("class_weight")
        if class_weight not in (None, "balanced"):
            raise ModelRegistryError("evaluation has an unsupported class-weight policy")
        if "model_parameter_overrides" not in method_metrics:
            raise ModelRegistryError(
                "evaluation lacks an explicit model_parameter_overrides object"
            )
        try:
            supplied_overrides = method_metrics.get("model_parameter_overrides")
            if not isinstance(supplied_overrides, dict):
                raise ModelError("model_parameter_overrides must be an object")
            parameters = supplied_overrides
            resolved = model_parameter_record(
                method,
                random_seed=random_seed,
                class_weight=class_weight,
                parameters=parameters,
            )
        except ModelError as exc:
            raise ModelRegistryError(f"evaluation model parameters are invalid: {exc}") from exc
        if record != resolved:
            raise ModelRegistryError(
                "evaluation model parameter record is incomplete or stale; rerun evaluation with the current resolved parameter contract"
            )
        selection = method_metrics.get("model_selection")
        if not isinstance(selection, dict):
            raise ModelRegistryError("evaluation lacks a model-selection record")
        selection_summary: dict[str, Any] | None = None
        if selection is not None:
            status = selection.get("status")
            if status not in {
                "selected",
                "unsupported_fallback_to_predeclared_default",
                "disabled",
                "not_requested",
            }:
                raise ModelRegistryError("evaluation model_selection status is unsupported")
            selected_parameters = selection.get("selected_parameters")
            if not isinstance(selected_parameters, dict):
                raise ModelRegistryError("evaluation model selection lacks selected parameters")
            if selected_parameters != parameters:
                raise ModelRegistryError(
                    "evaluation fitted parameter overrides do not match its grouped "
                    "model-selection result"
                )
            try:
                selected_resolved = model_parameter_record(
                    method,
                    random_seed=random_seed,
                    class_weight=class_weight,
                    parameters=selected_parameters,
                )
            except ModelError as exc:
                raise ModelRegistryError(
                    f"evaluation selected model parameters are invalid: {exc}"
                ) from exc
            if selected_resolved != resolved:
                raise ModelRegistryError(
                    "evaluation fitted parameters do not match its grouped model-selection result"
                )
            expected_partition = (
                "not used; predeclared fixed parameters"
                if status in {"disabled", "not_requested"}
                else "outer training groups only (grouped inner folds)"
            )
            if (
                selection.get("selection_partition") != expected_partition
                or selection.get("outer_test_used") is not False
            ):
                raise ModelRegistryError(
                    "evaluation model selection does not document train-only selection "
                    "with outer test data unused"
                )
            if (
                "outer_validation_used_for_hyperparameters" in selection
                and selection.get("outer_validation_used_for_hyperparameters") is not False
            ):
                raise ModelRegistryError(
                    "evaluation model selection does not document that outer validation "
                    "was unused for hyperparameters"
                )
            if status == "selected":
                candidate_index = selection.get("selected_candidate_index")
                candidates = selection.get("candidates")
                fold_manifest = selection.get("fold_manifest")
                selection_criterion = selection.get("selection_criterion")
                selection_configuration = selection.get("configuration")
                if (
                    selection.get("outer_validation_used_for_hyperparameters") is not False
                    or isinstance(candidate_index, bool)
                    or not isinstance(candidate_index, int)
                    or candidate_index < 0
                    or not isinstance(candidates, list)
                    or candidate_index >= len(candidates)
                    or not isinstance(candidates[candidate_index], dict)
                    or candidates[candidate_index].get("candidate_index") != candidate_index
                    or candidates[candidate_index].get("parameters") != selected_parameters
                    or candidates[candidate_index].get("resolved_model_parameters")
                    != selected_resolved
                    or selection.get("grouping") != "configuration_id"
                    or not isinstance(fold_manifest, dict)
                    or fold_manifest.get("grouping") != "configuration_id"
                    or not isinstance(selection_criterion, list)
                    or not selection_criterion
                    or not all(isinstance(item, str) and item for item in selection_criterion)
                    or not isinstance(selection_configuration, dict)
                    or selection_configuration.get("enabled") is not True
                    or not isinstance(selection.get("limitations"), str)
                    or not selection.get("limitations")
                ):
                    raise ModelRegistryError(
                        "evaluation grouped model-selection summary is incomplete"
                    )
            elif status == "unsupported_fallback_to_predeclared_default":
                if (
                    selection.get("selected_candidate_index") is not None
                    or selected_parameters != {}
                    or selection.get("grouping") != "configuration_id"
                    or not isinstance(selection.get("configuration"), dict)
                    or selection["configuration"].get("enabled") is not True
                    or not (
                        isinstance(selection.get("unsupported_reason"), str)
                        and selection.get("unsupported_reason")
                        or isinstance(selection.get("candidates"), list)
                    )
                ):
                    raise ModelRegistryError(
                        "evaluation unsupported model-selection fallback is incomplete"
                    )
            elif status == "disabled" and selected_parameters != {}:
                raise ModelRegistryError(
                    "disabled model selection must retain predeclared default parameters"
                )
            elif (
                status == "not_requested"
                and selection.get("outer_validation_used_for_hyperparameters") is not False
            ):
                raise ModelRegistryError(
                    "fixed-parameter evaluation does not document that outer validation "
                    "was unused for hyperparameters"
                )
            selection_summary = {
                key: selection.get(key)
                for key in (
                    "status",
                    "selected_candidate_index",
                    "selection_partition",
                    "outer_test_used",
                    "outer_validation_used_for_hyperparameters",
                    "grouping",
                    "selection_criterion",
                    "fold_manifest",
                    "configuration",
                    "unsupported_reason",
                    "limitations",
                    "feature_specification",
                    "feature_specification_sha256",
                    "feature_specification_status",
                    "tcp_retransmission_unit",
                    "feature_temporal_policy",
                    "feature_temporal_eligibility",
                )
                if key in selection
            }
        try:
            evaluated_specification = validate_feature_specification_object(
                method_metrics.get("feature_specification")
            )
        except ProcessingError as exc:
            raise ModelRegistryError(
                f"evaluation lacks a valid current feature specification: {exc}"
            ) from exc
        evaluated_digest = feature_specification_sha256(evaluated_specification)
        evaluated_policy = feature_temporal_policy_from_specification(evaluated_specification)
        if (
            method_metrics.get("feature_specification_sha256") != evaluated_digest
            or method_metrics.get("feature_specification_status") != "supplied_current_resolved"
            or method_metrics.get("tcp_retransmission_unit")
            != evaluated_specification.get("tcp_retransmission_unit")
            or method_metrics.get("feature_temporal_policy") != evaluated_policy
            or evaluated_policy.get("status") != "resolved"
            or not isinstance(method_metrics.get("feature_temporal_eligibility"), dict)
        ):
            raise ModelRegistryError(
                "evaluation feature specification, TCP retransmission unit, temporal policy, "
                "or training temporal-eligibility record is stale or unresolved"
            )
        if selection is not None and (
            selection.get("feature_specification") != evaluated_specification
            or selection.get("feature_specification_sha256") != evaluated_digest
            or selection.get("feature_specification_status") != "supplied_current_resolved"
            or selection.get("tcp_retransmission_unit")
            != evaluated_specification.get("tcp_retransmission_unit")
            or selection.get("feature_temporal_policy") != evaluated_policy
        ):
            raise ModelRegistryError(
                "evaluation grouped model selection used a different feature specification, "
                "TCP retransmission unit, or temporal policy"
            )
        if selection is not None:
            selection_eligibility = selection.get("feature_temporal_eligibility")
            evaluated_eligibility = method_metrics.get("feature_temporal_eligibility")
            eligibility_keys = (
                "anchor_count",
                "eligible_anchor_count",
                "not_eligible_anchor_count",
                "descriptive_only_anchor_count",
                "reason_counts",
                "status_counts",
            )
            counts_recorded = isinstance(selection_eligibility, dict) and (
                "anchor_count" in selection_eligibility
            )
            if (selection.get("status") == "selected" and not counts_recorded) or (
                counts_recorded
                and (
                    not isinstance(evaluated_eligibility, dict)
                    or any(
                        selection_eligibility.get(key) != evaluated_eligibility.get(key)
                        for key in eligibility_keys
                    )
                )
            ):
                raise ModelRegistryError(
                    "evaluation grouped model selection temporal-eligibility counts "
                    "differ from the fitted outer-training record"
                )
        calibration = method_metrics.get("score_calibration")
        if not isinstance(calibration, dict) or calibration.get("status") != "not_fitted":
            raise ModelRegistryError(
                "evaluation score calibration status is unsupported; no calibrated probability claim is accepted"
            )
        if calibration.get("fitted_calibrator") is not None or not str(
            calibration.get("score_semantics", "")
        ).startswith("uncalibrated"):
            raise ModelRegistryError(
                "evaluation score semantics do not establish an uncalibrated-score contract"
            )
        return {
            "parameters": dict(parameters),
            "resolved_record": resolved,
            "random_seed": random_seed,
            "class_weight": class_weight,
            "selection_summary": selection_summary,
            "feature_specification": evaluated_specification,
            "feature_specification_sha256": evaluated_digest,
            "tcp_retransmission_unit": evaluated_specification.get("tcp_retransmission_unit"),
            "feature_temporal_policy": evaluated_policy,
        }

    @staticmethod
    def _require_estimable_heldout_evidence(test_metrics: Any) -> None:
        if not isinstance(test_metrics, dict):
            raise ModelRegistryError("evaluation lacks held-out event metrics")
        event_count = test_metrics.get("event_count")
        exposure = test_metrics.get("eligible_exposure_seconds")
        fapm = test_metrics.get("false_alarms_per_minute")
        recall = test_metrics.get("event_recall")
        if (
            not isinstance(event_count, int)
            or isinstance(event_count, bool)
            or event_count <= 0
            or not isinstance(exposure, (int, float))
            or isinstance(exposure, bool)
            or not math.isfinite(float(exposure))
            or float(exposure) <= 0
            or not isinstance(fapm, (int, float))
            or isinstance(fapm, bool)
            or not math.isfinite(float(fapm))
            or not isinstance(recall, (int, float))
            or isinstance(recall, bool)
            or not math.isfinite(float(recall))
            or not 0 <= float(recall) <= 1
        ):
            raise ModelRegistryError(
                "held-out event evidence is not estimable (requires positive identified events and eligible exposure)"
            )

    @staticmethod
    def _processing_feature_specification(
        processing_manifest: dict[str, Any],
        config: dict[str, Any],
    ) -> tuple[dict[str, Any], str, str]:
        status_record = feature_specification_operational_status(
            processing_manifest,
            require_run_record_unit_proof=True,
        )
        specification = status_record.get("specification")
        if (
            not status_record.get("operationally_validated")
            or not isinstance(specification, dict)
            or status_record.get("status") != "current_contract_complete"
        ):
            reasons = status_record.get("reasons") or [
                "prepared feature contract is not operationally validated"
            ]
            raise ModelRegistryError("; ".join(str(reason) for reason in reasons))
        if (
            "tcp_retransmission_unit" in processing_manifest
            and processing_manifest.get("tcp_retransmission_unit")
            != specification.get("tcp_retransmission_unit")
        ) or (
            "feature_temporal_policy" in processing_manifest
            and processing_manifest.get("feature_temporal_policy")
            != feature_temporal_policy_from_specification(specification)
        ):
            raise ModelRegistryError(
                "processing-manifest TCP unit or temporal policy differs from its feature specification"
            )
        try:
            expected_policy = normalize_feature_temporal_policy(
                (config.get("processing") or {}).get("feature_temporal_policy"),
                require_resolved=True,
            )
        except ProcessingError as exc:
            raise ModelRegistryError(str(exc)) from exc
        if feature_temporal_policy_from_specification(specification) != expected_policy:
            raise ModelRegistryError(
                "prepared feature temporal policy differs from the frozen processing policy"
            )
        provenance = status_record.get("tcp_unit_provenance") or {}
        if specification.get("tcp_retransmission_unit_status") == "declared" and (
            provenance.get("status") != "declared"
            or provenance.get("unit") != specification.get("tcp_retransmission_unit")
        ):
            raise ModelRegistryError(
                "prepared TCP retransmission unit differs from declared run provenance"
            )
        return (
            specification,
            feature_specification_sha256(specification),
            str(status_record["status"]),
        )

    def _load_eligible_evaluation(self, evaluation_id: str, method: str):
        if method not in SUPPORTED_METHODS:
            raise ModelRegistryError("method is not a supported endpoint-telemetry model")
        processed, shared = self._current_source_proof()
        analysis_path = self.workspace.processed_root / "analysis.csv"
        if analysis_path.stat().st_size > MAX_ANALYSIS_BYTES_FOR_TRAINING:
            raise ModelRegistryError(
                "prepared analysis exceeds the web trainer's 64 MiB input limit"
            )
        row_count = processed.get("manifest", {}).get("analysis_row_count")
        if (
            not isinstance(row_count, int)
            or isinstance(row_count, bool)
            or row_count > MAX_ANALYSIS_ROWS_FOR_TRAINING
        ):
            raise ModelRegistryError(
                "prepared analysis exceeds the web trainer's 100,000-row compute limit"
            )
        directory = self._evaluation_directory(evaluation_id)
        checked = verify_evaluation_bundle(
            directory,
            self.workspace,
            processed_artifacts=processed,
            shared_sources=shared,
        )
        if checked.get("status") != "complete_validated_provenance":
            reason = "; ".join(checked.get("errors", [])) or "evaluation provenance is not verified"
            raise ModelRegistryError(f"evaluation is not inference-eligible: {reason}")
        report = checked["report"]
        if report.get("split_grouping") != "configuration_id":
            raise ModelRegistryError(
                "inference eligibility requires a configuration-held-out evaluation"
            )
        if method not in report.get("successful_methods", []):
            raise ModelRegistryError(
                "selected method did not complete successfully in this evaluation"
            )
        metrics = read_json(safe_child(directory, "metrics.json"))
        if method in metrics.get("method_failures", {}) or not isinstance(
            metrics.get(method), dict
        ):
            raise ModelRegistryError(
                "selected method is missing or failed in the evaluation metrics"
            )
        method_metrics = metrics[method]
        calibration = method_metrics.get("threshold_calibration")
        test_metrics = method_metrics.get("test_event_metrics")
        threshold = calibration.get("threshold") if isinstance(calibration, dict) else None
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not math.isfinite(float(threshold))
        ):
            raise ModelRegistryError(
                "evaluation has no finite validation-calibrated warning threshold"
            )
        self._require_estimable_heldout_evidence(test_metrics)
        method_contract = self._evaluation_model_contract(method_metrics, method)
        if (
            report.get("feature_specification") != method_contract["feature_specification"]
            or report.get("feature_specification_sha256")
            != method_contract["feature_specification_sha256"]
            or report.get("tcp_retransmission_unit") != method_contract["tcp_retransmission_unit"]
            or report.get("feature_temporal_policy") != method_contract["feature_temporal_policy"]
            or (
                report.get("processing_compatibility", {}).get("feature_specification")
                != method_contract["feature_specification"]
            )
            or (
                report.get("processing_compatibility", {}).get("feature_specification_sha256")
                != method_contract["feature_specification_sha256"]
            )
        ):
            raise ModelRegistryError(
                "evaluation manifest feature specification, TCP retransmission unit, or "
                "temporal policy differs from the evaluated model"
            )
        split = read_json(safe_child(directory, "splits.json"))
        split_groups = split.get("groups")
        if (
            split.get("grouping") != "configuration_id"
            or not isinstance(split_groups, dict)
            or not split_groups.get("test")
        ):
            raise ModelRegistryError(
                "evaluation lacks a non-empty configuration-held-out test partition"
            )
        config = load_config(self.workspace.config_path)
        if config.get("status") != "protocol_frozen":
            raise ModelRegistryError("current protocol is not frozen")
        verify_protocol_freeze_evidence(config, self.workspace.config_path)
        feature_columns = method_metrics.get("requested_feature_columns") or method_metrics.get(
            "feature_columns"
        )
        try:
            validate_feature_columns(feature_columns)
        except ModelError as exc:
            raise ModelRegistryError(
                f"evaluation features are not valid endpoint predictors: {exc}"
            ) from exc
        if (
            method_metrics.get("preprocessing_fit_partition")
            != "train only (scikit-learn Pipeline)"
        ):
            raise ModelRegistryError("evaluation does not document train-only preprocessing")
        threshold_cap = float(config["evaluation"]["max_false_alarms_per_minute"])
        val_fapm = calibration.get("validation_event_metrics", {}).get("false_alarms_per_minute")
        if val_fapm is None or float(val_fapm) > threshold_cap + 1e-9:
            raise ModelRegistryError(
                "validation threshold does not meet the frozen false-alarm policy"
            )
        rows, processing = load_analysis_dataset(self.workspace.processed_root)
        if processing.get("analysis_sha256") != report.get("analysis_sha256"):
            raise ModelRegistryError(
                "loaded prepared rows do not match evaluation analysis fingerprint"
            )
        try:
            check_group_isolation(rows, split["row_indices"], grouping="configuration_id")
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelRegistryError(f"evaluation split is not isolated: {exc}") from exc
        feature_specification, feature_specification_digest, operational_status = (
            self._processing_feature_specification(processed.get("manifest", {}), config)
        )
        if (
            method_contract["feature_specification"] != feature_specification
            or method_contract["feature_specification_sha256"] != feature_specification_digest
        ):
            raise ModelRegistryError(
                "evaluation feature specification differs from validated prepared provenance"
            )
        feature_contract = {
            "feature_specification": feature_specification,
            "feature_specification_sha256": feature_specification_digest,
            "feature_specification_status": operational_status,
            "tcp_retransmission_unit": feature_specification.get("tcp_retransmission_unit"),
            "feature_temporal_policy": feature_temporal_policy_from_specification(
                feature_specification
            ),
        }
        return (
            directory,
            report,
            method_metrics,
            split,
            rows,
            processing,
            config,
            feature_contract,
        )

    def train_candidate(self, evaluation_id: str, method: str) -> dict[str, Any]:
        try:
            from joblib import dump
        except ImportError as exc:
            raise ModelRegistryError(
                "model registration requires scikit-learn/joblib; install the web and ml extras"
            ) from exc
        (
            directory,
            report,
            method_metrics,
            split,
            rows,
            processing,
            config,
            feature_contract,
        ) = self._load_eligible_evaluation(evaluation_id, method)
        feature_columns = method_metrics.get("requested_feature_columns") or method_metrics.get(
            "feature_columns"
        )
        contract = self._evaluation_model_contract(method_metrics, method)
        feature_specification = feature_contract["feature_specification"]
        feature_specification_digest = feature_contract["feature_specification_sha256"]
        if (
            contract["feature_specification"] != feature_specification
            or contract["feature_specification_sha256"] != feature_specification_digest
        ):
            raise ModelRegistryError(
                "evaluation feature specification differs from validated prepared provenance"
            )
        if not set(feature_columns).issubset(feature_specification["feature_columns"]):
            raise ModelRegistryError(
                "evaluation requested features are outside the current processing feature specification"
            )
        fit = fit_model_on_training_split(
            rows,
            split,
            model_name=method,
            feature_columns=feature_columns,
            random_seed=contract["random_seed"],
            class_weight=contract["class_weight"],
            model_parameters=contract["parameters"],
            feature_specification=feature_specification,
        )
        if (
            fit["feature_columns"] != method_metrics.get("feature_columns")
            or fit["training_counts"] != method_metrics.get("training_counts")
            or fit["model_parameters"] != contract["resolved_record"]
            or fit.get("model_parameter_overrides") != contract["parameters"]
            or fit.get("feature_specification") != feature_specification
            or fit.get("feature_specification_sha256") != feature_specification_digest
            or fit.get("feature_specification_status") != "supplied_current_resolved"
            or fit.get("tcp_retransmission_unit")
            != feature_specification.get("tcp_retransmission_unit")
            or fit.get("feature_temporal_policy") != feature_contract["feature_temporal_policy"]
            or fit.get("feature_temporal_eligibility")
            != method_metrics.get("feature_temporal_eligibility")
        ):
            raise ModelRegistryError(
                "re-fitted train partition, resolved parameters, feature specification, "
                "TCP retransmission unit, temporal policy, or temporal eligibility differs "
                "from the validated evaluation record"
            )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        model_id = f"model_{stamp}_{uuid.uuid4().hex[:8]}"
        if self.model_root.is_symlink():
            raise ModelRegistryError("model registry root must not be a symlink")
        self.model_root.mkdir(parents=True, exist_ok=True)
        destination = self.model_root / model_id
        destination.mkdir(exist_ok=False)
        marker = destination / "INCOMPLETE"
        marker.write_text(
            "Model registration did not reach its verified manifest step.\n", encoding="utf-8"
        )
        artifact = destination / "estimator.joblib"
        dump(fit["estimator"], artifact, compress=3)
        if artifact.stat().st_size > MAX_MODEL_ARTIFACT_BYTES:
            raise ModelRegistryError("fitted estimator exceeds the 256 MiB model-artifact limit")
        evaluation_manifest_path = directory / "evaluation_manifest.json"
        config_hash = file_sha256(self.workspace.config_path)
        manifest = {
            "model_registry_schema_version": 1,
            "status": "candidate",
            "model_id": model_id,
            "method": method,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_file": "estimator.joblib",
            "artifact_sha256": file_sha256(artifact),
            "evaluation_id": evaluation_id,
            "evaluation_manifest_sha256": file_sha256(evaluation_manifest_path),
            "analysis_sha256": report.get("analysis_sha256"),
            "processing_manifest_sha256": report.get("processing_manifest_sha256"),
            "source_manifest_sha256": report.get("source_manifest_sha256"),
            "raw_audit_sha256": (report.get("raw_audit") or {}).get("report_sha256"),
            "config_sha256": config_hash,
            "protocol_sha256": report.get("protocol_sha256"),
            "split_grouping": report.get("split_grouping"),
            "split_seed": report.get("split_seed"),
            "training_partition": "train only",
            "training_counts": fit["training_counts"],
            "preprocessing_fit_partition": fit["preprocessing_fit_partition"],
            "method_parameters": fit["model_parameters"],
            "model_parameter_overrides": fit["model_parameter_overrides"],
            "model_selection_summary": contract["selection_summary"],
            "feature_columns": fit["feature_columns"],
            "endpoint_feature_fields": processing.get("endpoint_feature_fields"),
            "feature_windows_seconds": processing.get("feature_windows_seconds"),
            "feature_specification": feature_specification,
            "feature_specification_sha256": feature_specification_digest,
            "feature_specification_status": feature_contract["feature_specification_status"],
            "tcp_retransmission_unit": feature_specification.get("tcp_retransmission_unit"),
            "feature_temporal_policy": feature_contract["feature_temporal_policy"],
            "feature_temporal_eligibility_training": fit["feature_temporal_eligibility"],
            "score_threshold": float(method_metrics["threshold_calibration"]["threshold"]),
            "score_semantics": "uncalibrated positive-class model score; not a calibrated probability",
            "calibration_status": "not_fitted",
            "forecast_horizon_seconds": float(config["labeling"]["prediction_horizon_ms"]) / 1000,
            "validation_false_alarms_per_minute": method_metrics["threshold_calibration"][
                "validation_event_metrics"
            ]["false_alarms_per_minute"],
            "evidence_status": "evaluated supplied data; empirical external validity is not established",
        }
        manifest_path = destination / "model_manifest.json"
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        # Recheck the evaluation and current sources before publishing the
        # candidate. Keep INCOMPLETE in place throughout the check so a
        # failed or interrupted registration cannot look complete.
        self._load_eligible_evaluation(evaluation_id, method)
        reason = self._eligibility_reason(destination, manifest, allow_registration_marker=True)
        if reason:
            raise ModelRegistryError(
                f"candidate did not pass post-registration eligibility: {reason}"
            )
        marker.unlink()
        return {
            "status": "registered",
            "model": model_view(manifest, eligible=True),
            "message": "Candidate registered from the exact train-only split and passed current provenance eligibility checks.",
        }

    def _eligibility_reason(
        self,
        directory: Path,
        manifest: dict[str, Any],
        *,
        source_proof: tuple[dict[str, Any] | None, dict[str, Any]] | None = None,
        allow_registration_marker: bool = False,
    ) -> str | None:
        try:
            if (directory / "INCOMPLETE").exists() and not allow_registration_marker:
                return "model registration is marked incomplete"
            if (
                manifest.get("model_registry_schema_version") != 1
                or manifest.get("status") != "candidate"
            ):
                return "unsupported or incomplete model manifest"
            method = manifest.get("method")
            if not isinstance(method, str) or method not in SUPPORTED_METHODS:
                return "unsupported model method"
            artifact = safe_child(directory, "estimator.joblib")
            if artifact.stat().st_size > MAX_MODEL_ARTIFACT_BYTES:
                return "model artifact exceeds the safe loading limit"
            if file_sha256(artifact) != manifest.get("artifact_sha256"):
                return "model artifact checksum is stale"
            evaluation_id = manifest.get("evaluation_id")
            evaluation_dir = self._evaluation_directory(evaluation_id)
            eval_manifest_path = safe_child(evaluation_dir, "evaluation_manifest.json")
            if file_sha256(eval_manifest_path) != manifest.get("evaluation_manifest_sha256"):
                return "linked evaluation report has changed"
            if source_proof is None:
                source_proof = self._current_source_proof()
            processed, shared = source_proof
            if processed is None or shared.get("status") != "verified":
                return shared.get("error", "current dataset/source provenance does not verify")
            evaluation = verify_evaluation_bundle(
                evaluation_dir,
                self.workspace,
                processed_artifacts=processed,
                shared_sources=shared,
            )
            if evaluation.get("status") != "complete_validated_provenance":
                return "linked evaluation or current raw/data provenance no longer verifies"
            report = evaluation["report"]
            if (
                manifest.get("evaluation_id") != evaluation_dir.name
                or report.get("analysis_sha256") != manifest.get("analysis_sha256")
                or report.get("processing_manifest_sha256")
                != manifest.get("processing_manifest_sha256")
                or report.get("source_manifest_sha256") != manifest.get("source_manifest_sha256")
                or report.get("protocol_sha256") != manifest.get("protocol_sha256")
                or report.get("split_grouping") != "configuration_id"
                or manifest.get("split_grouping") != "configuration_id"
            ):
                return "linked dataset/protocol fingerprints or held-out grouping changed"
            config = load_config(self.workspace.config_path)
            if (
                file_sha256(self.workspace.config_path) != manifest.get("config_sha256")
                or config.get("status") != "protocol_frozen"
            ):
                return "current frozen configuration is stale or unavailable"
            verify_protocol_freeze_evidence(config, self.workspace.config_path)
            processing = processed.get("manifest", {})
            if manifest.get("endpoint_feature_fields") != processing.get(
                "endpoint_feature_fields"
            ) or manifest.get("feature_windows_seconds") != processing.get(
                "feature_windows_seconds"
            ):
                return "model endpoint fields or trailing-window contract differs from current preparation"
            (
                feature_specification,
                feature_specification_digest,
                operational_status,
            ) = self._processing_feature_specification(processing, config)
            feature_policy = feature_temporal_policy_from_specification(feature_specification)
            if manifest.get("feature_specification") != feature_specification:
                return (
                    "model feature specification differs from current preparation or "
                    "cannot prove the exact current TCP retransmission unit"
                )
            if manifest.get("feature_specification_sha256") != feature_specification_digest:
                return "model feature specification digest is stale"
            if (
                manifest.get("feature_specification_status") != operational_status
                or manifest.get("tcp_retransmission_unit")
                != feature_specification.get("tcp_retransmission_unit")
                or manifest.get("feature_temporal_policy") != feature_policy
            ):
                return (
                    "model feature-specification status, TCP retransmission unit, or "
                    "temporal policy differs from current preparation"
                )
            if (
                report.get("feature_specification") != feature_specification
                or report.get("feature_specification_sha256") != feature_specification_digest
                or report.get("tcp_retransmission_unit")
                != feature_specification.get("tcp_retransmission_unit")
                or report.get("feature_temporal_policy") != feature_policy
                or report.get("processing_compatibility", {}).get("feature_specification")
                != feature_specification
                or report.get("processing_compatibility", {}).get("feature_specification_sha256")
                != feature_specification_digest
            ):
                return (
                    "evaluation manifest feature specification, TCP retransmission unit, "
                    "or temporal policy differs from current preparation"
                )
            if not set(manifest.get("feature_columns") or []).issubset(
                feature_specification["feature_columns"]
            ):
                return "model features are outside the current feature specification"
            expected_horizon = float(config["labeling"]["prediction_horizon_ms"]) / 1000.0
            if not math.isclose(
                float(manifest.get("forecast_horizon_seconds")),
                expected_horizon,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                return "model forecast horizon differs from the frozen protocol"
            metrics = read_json(safe_child(evaluation_dir, "metrics.json"))
            method_result = metrics.get(manifest.get("method"))
            if not isinstance(method_result, dict) or manifest.get("method") in metrics.get(
                "method_failures", {}
            ):
                return "linked model method did not successfully complete evaluation"
            calibration = method_result.get("threshold_calibration")
            if not isinstance(calibration, dict):
                return "linked evaluation lacks validation threshold calibration"
            contract = self._evaluation_model_contract(method_result, str(manifest.get("method")))
            if (
                contract["feature_specification"] != feature_specification
                or contract["feature_specification_sha256"] != feature_specification_digest
                or contract["tcp_retransmission_unit"]
                != feature_specification.get("tcp_retransmission_unit")
                or contract["feature_temporal_policy"] != feature_policy
                or manifest.get("feature_temporal_eligibility_training")
                != method_result.get("feature_temporal_eligibility")
            ):
                return (
                    "model or evaluated feature specification, TCP retransmission unit, "
                    "temporal policy, or training temporal eligibility differs"
                )
            self._require_estimable_heldout_evidence(method_result.get("test_event_metrics"))
            if manifest.get("method_parameters") != contract["resolved_record"]:
                return "model resolved parameters differ from the evaluated fitted pipeline"
            manifest_overrides = manifest.get("model_parameter_overrides")
            if not isinstance(manifest_overrides, dict):
                return "model manifest lacks an explicit model_parameter_overrides object"
            if manifest_overrides != contract["parameters"]:
                return "model parameter overrides differ from the evaluated fitted pipeline"
            manifest_record = model_parameter_record(
                str(manifest.get("method")),
                random_seed=contract["random_seed"],
                class_weight=contract["class_weight"],
                parameters=manifest_overrides,
            )
            if manifest_record != contract["resolved_record"]:
                return "model parameter overrides differ from the evaluated fitted pipeline"
            if manifest.get("model_selection_summary") != contract["selection_summary"]:
                return "model selection summary differs from the linked evaluation"
            if manifest.get("calibration_status") != "not_fitted" or not str(
                manifest.get("score_semantics", "")
            ).startswith("uncalibrated"):
                return "model score semantics do not preserve the uncalibrated-score contract"
            if (
                calibration.get("threshold") != manifest.get("score_threshold")
                or method_result.get("preprocessing_fit_partition")
                != "train only (scikit-learn Pipeline)"
            ):
                return "model threshold or training preprocessing does not match the evaluation"
            if manifest.get("feature_columns") != method_result.get("feature_columns"):
                return "model feature columns differ from the evaluated fitted pipeline"
            if manifest.get("training_counts") != method_result.get("training_counts"):
                return "model training counts differ from the evaluated train partition"
            validation_metrics = calibration.get("validation_event_metrics")
            false_alarm_cap = float(config["evaluation"]["max_false_alarms_per_minute"])
            validation_fapm = (
                validation_metrics.get("false_alarms_per_minute")
                if isinstance(validation_metrics, dict)
                else None
            )
            if (
                not isinstance(validation_fapm, (int, float))
                or isinstance(validation_fapm, bool)
                or not math.isfinite(float(validation_fapm))
                or float(validation_fapm) > false_alarm_cap + 1e-9
            ):
                return "validation threshold no longer satisfies the frozen false-alarm policy"
            validate_feature_columns(manifest.get("feature_columns"))
            if manifest.get("training_partition") != "train only":
                return "candidate was not fitted exclusively on training groups"
            return None
        except (
            DashboardDataError,
            ModelRegistryError,
            ModelError,
            ReportError,
            ProcessingError,
            ConfigError,
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            IndexError,
        ) as exc:
            return str(exc) or "malformed model manifest or linked evaluation artifact"

    def resolve_eligible(self, model_id: str) -> tuple[dict[str, Any], Any]:
        if not isinstance(model_id, str) or not MODEL_ID_PATTERN.fullmatch(model_id):
            raise ModelRegistryError("invalid model identifier")
        directory = safe_child(self.model_root, model_id, must_exist=False)
        if directory.is_symlink() or not directory.is_dir():
            raise ModelRegistryError("model artifact is unavailable")
        try:
            manifest = read_json(directory / "model_manifest.json")
        except (DashboardDataError, OSError) as exc:
            raise ModelRegistryError(f"model manifest is unavailable: {exc}") from exc
        reason = self._eligibility_reason(directory, manifest)
        if reason:
            raise ModelRegistryError(f"model is not currently eligible: {reason}")
        artifact_sha = manifest.get("artifact_sha256")
        feature_specification_sha = manifest.get("feature_specification_sha256")
        if not isinstance(artifact_sha, str):
            raise ModelRegistryError("verified model manifest lacks an artifact checksum")
        if not isinstance(feature_specification_sha, str):
            raise ModelRegistryError(
                "verified model manifest lacks a feature-specification checksum"
            )
        try:
            model_manifest_sha = file_sha256(directory / "model_manifest.json")
        except (DashboardDataError, OSError) as exc:
            raise ModelRegistryError(
                f"verified model manifest could not be fingerprinted: {exc}"
            ) from exc
        cache_key = (
            model_id,
            artifact_sha,
            model_manifest_sha,
            feature_specification_sha,
        )
        with self._estimator_cache_lock:
            estimator = self._estimator_cache.get(cache_key)
            if estimator is not None:
                self._estimator_cache.move_to_end(cache_key)
                return manifest, estimator
        try:
            from joblib import load

            estimator = load(safe_child(directory, "estimator.joblib"))
        except Exception as exc:
            raise ModelRegistryError(f"verified model artifact could not be loaded: {exc}") from exc
        with self._estimator_cache_lock:
            self._estimator_cache[cache_key] = estimator
            self._estimator_cache.move_to_end(cache_key)
            while len(self._estimator_cache) > 2:
                self._estimator_cache.popitem(last=False)
        return manifest, estimator

    def set_active(self, model_id: str) -> dict[str, Any]:
        manifest, _ = self.resolve_eligible(model_id)
        self.state_root.mkdir(parents=True, exist_ok=True)
        if self.state_root.is_symlink() or self.active_file.is_symlink():
            raise ModelRegistryError("active-model state path must not be a symlink")
        temp = self.state_root / f".active-{uuid.uuid4().hex}.tmp"
        try:
            with temp.open("x", encoding="utf-8") as stream:
                json.dump(
                    {
                        "model_id": model_id,
                        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    stream,
                )
                stream.write("\n")
            os.replace(temp, self.active_file)
        finally:
            temp.unlink(missing_ok=True)
        return {
            "active_model_id": manifest["model_id"],
            "message": "Selected model passed current eligibility checks.",
        }
