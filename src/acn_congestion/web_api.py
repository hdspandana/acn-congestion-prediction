"""Loopback-first FastAPI service for the ACN React operations application.

The API reuses the existing Python research pipeline. Request data never
contains filesystem paths or shell commands, and all artifact identifiers are
resolved under operator-configured roots.
"""

from __future__ import annotations

import contextlib
import io
import math
import csv
import os
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .config import ConfigError, load_config, verify_protocol_freeze_evidence
from .dashboard_data import (
    MAX_DASHBOARD_AUDIT_BYTES,
    DashboardDataError,
    Workspace,
    discover_report_directories,
    read_json,
    resolve_workspace,
    run_audit_safely,
    safe_child,
    scan_csv_inventory,
    verify_audit_inventory,
    verify_evaluation_bundle,
    verify_ground_truth_bundle,
    verify_processed_artifacts,
    verify_robustness_bundle,
    verify_workspace_lineage_sources,
)
from .ground_truth import (
    GroundTruthReportError,
    build_ground_truth_report,
    verify_ground_truth_report,
)
from .model_registry import ModelRegistry, ModelRegistryError
from .prediction_history import PredictionHistory, PredictionHistoryError
from .processing import ProcessingError, verify_alignment_review
from .reporting import ReportError, load_analysis_dataset, verify_evaluation_artifacts
from .robustness import RobustnessError, verify_robustness_artifacts
from .telemetry import (
    MAX_TELEMETRY_UPLOAD_BYTES,
    TelemetryInputError,
    inspect_endpoint_csv,
    predict_with_model,
)

API_PREFIX = "/api/v1"
REPORT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,120}$")
MAX_WEB_EVALUATION_BYTES = 64 * 1024 * 1024
MAX_WEB_EVALUATION_ROWS = 100_000
MAX_WEB_TIMELINE_POINTS = 1_200
MAX_WEB_DETAIL_BYTES = 16 * 1024 * 1024
MAX_MULTIPART_OVERHEAD_BYTES = 64 * 1024


class APIProblem(Exception):
    """Expected API error with a stable HTTP status and machine-readable code."""

    def __init__(self, status_code: int, code: str, message: str):
        self.status_code = status_code
        self.code = code
        self.message = message


class TrainRequest(BaseModel):
    evaluation_id: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_.-]+$")
    method: str = Field(min_length=1, max_length=64)


class SelectModelRequest(BaseModel):
    model_id: str = Field(
        min_length=1, max_length=80, pattern=r"^model_[0-9]{8}T[0-9]{6}Z_[a-f0-9]{8}$"
    )


class ConfirmRequest(BaseModel):
    confirmed: bool = Field(description="Explicitly confirm a bounded read-only audit.")


class ApplicationInfo(BaseModel):
    name: str
    version: str
    mode: str


class StatusResponse(BaseModel):
    status: str
    application: str
    version: str
    mode: str


class PredictionRecordResponse(BaseModel):
    """Actual forecast record; a forecast is not a verified congestion event."""

    prediction_id: str
    predicted_at_utc: str
    model_id: str
    model_method: str
    input_name: str
    input_sha256: str
    row_count: int
    forecast_horizon_seconds: float = Field(
        description="Supported horizon from the registered model contract."
    )
    score: float | None = Field(
        description="Uncalibrated model output, not a probability unless separately verified."
    )
    score_threshold: float
    risk: Literal["warning", "below_threshold", "not_scored"]
    outcome_state: Literal["unresolved", "verified_event", "false_alarm"]
    timestamp_seconds: float
    model_artifact_sha256: str | None = None
    feature_spec_version: str | None = None
    feature_specification_sha256: str | None = None
    feature_specification_status: str | None = None
    tcp_retransmission_unit: str | None = None
    input_declared_tcp_retransmission_unit: str | None = None
    feature_temporal_policy_status: str | None = None
    feature_temporal_eligibility_status: str | None = None
    feature_temporal_eligibility_reason: str | None = None
    evaluation_id: str | None = None
    protocol_sha256: str | None = None
    analysis_sha256: str | None = None


class ModelRecordResponse(BaseModel):
    """Registered artifact and current provenance eligibility summary."""

    model_id: str
    method: str | None = None
    created_at_utc: str | None = None
    eligible: bool
    status: str
    reason: str | None = None
    forecast_horizon_seconds: float | None = None
    score_threshold: float | None = None
    score_semantics: str
    calibration_status: str
    feature_columns: list[str]
    feature_spec_version: str | None = None
    feature_specification_sha256: str | None = None
    feature_specification_status: str | None = None
    tcp_retransmission_unit: str | None = None
    feature_temporal_policy: dict[str, Any] | None = None
    feature_temporal_policy_status: str | None = None
    model_parameter_overrides: dict[str, Any] | None = None
    evaluation_id: str | None = None
    protocol_sha256: str | None = None
    analysis_sha256: str | None = None


class ModelContractResponse(BaseModel):
    model_id: str
    method: str | None = None
    forecast_horizon_seconds: float
    score_threshold: float
    feature_columns: list[str]
    evaluation_id: str | None = None
    protocol_sha256: str | None = None
    analysis_sha256: str | None = None
    score_semantics: str
    calibration_status: str
    feature_spec_version: str | None = None
    feature_specification_sha256: str | None = None
    feature_specification_status: str | None = None
    tcp_retransmission_unit: str | None = None
    tcp_retransmission_unit_status: str | None = None
    input_declared_tcp_retransmission_unit: str | None = None
    feature_temporal_policy: dict[str, Any] | None = None
    feature_temporal_eligibility: dict[str, Any] | None = None


class TelemetryQualityResponse(BaseModel):
    status: str
    missing_fraction: dict[str, float]
    gap_count: int | None
    max_gap_seconds: float | None
    timestamp_span_seconds: float | None
    wall_clock_freshness: str
    notes: list[str]


class TelemetryBatchResponse(BaseModel):
    """Bounded batch inspection; run-relative time cannot establish freshness."""

    status: str
    source_kind: Literal["uploaded_batch"]
    received_at_utc: str
    input_name: str
    input_sha256: str
    row_count: int
    fields: list[str]
    points: list[dict[str, float | None]]
    quality: TelemetryQualityResponse
    warnings: list[PredictionRecordResponse] = Field(default_factory=list)
    model: ModelContractResponse | None = None


class TelemetrySourceResponse(BaseModel):
    status: str
    message: str
    csv_count: int
    checked_at_utc: str


class ModelStatusResponse(BaseModel):
    status: str
    eligible_count: int
    active_model_id: str | None
    message: str


class FreshnessResponse(BaseModel):
    status: str
    message: str


class OverviewResponse(BaseModel):
    application: ApplicationInfo
    telemetry_source: TelemetrySourceResponse
    model_status: ModelStatusResponse
    latest_warning: PredictionRecordResponse | None
    history_count: int
    data_freshness: FreshnessResponse
    blockers: list[str]


class RawDatasetResponse(BaseModel):
    status: str
    csv_count: int
    message: str


class ProcessedDatasetResponse(BaseModel):
    status: str
    row_count: int | None = None
    run_count: int | None = None
    protocol_status: str | None = None
    message: str


class AuditStatusResponse(BaseModel):
    status: str
    message: str


class DatasetResponse(BaseModel):
    raw: RawDatasetResponse
    processed: ProcessedDatasetResponse
    audit: AuditStatusResponse


class ModelListResponse(BaseModel):
    models: list[ModelRecordResponse]
    active_model_id: str | None


class HistoryResponse(BaseModel):
    predictions: list[PredictionRecordResponse]


class PredictionResponse(BaseModel):
    predictions: list[PredictionRecordResponse]
    input: TelemetryBatchResponse
    model: ModelContractResponse


class ResearchEvaluationResponse(BaseModel):
    evaluation_id: str
    status: str
    provenance_status: str
    grouping: str | None = None
    created_at_utc: str | None = None
    methods: list[str]
    blockers: list[str]


class ResearchReportResponse(BaseModel):
    report_id: str
    kind: str
    status: str
    created_at_utc: str | None = None


class ProtocolResponse(BaseModel):
    status: str
    message: str
    config_sha256: str | None = None


class AlignmentResponse(BaseModel):
    status: str
    message: str


class ResearchOperationsResponse(BaseModel):
    audit_available: bool
    evaluation_available: bool
    message: str
    prepared_status: str = "unknown"
    source_lineage_status: str = "unknown"
    feature_specification_status: str = "unknown"
    tcp_retransmission_unit: str | None = None
    feature_temporal_policy_status: str = "unknown"
    blockers: list[str] = Field(default_factory=list)


class ResearchResponse(BaseModel):
    protocol: ProtocolResponse
    alignment: AlignmentResponse
    evaluations: list[ResearchEvaluationResponse]
    reports: list[ResearchReportResponse]
    operations: ResearchOperationsResponse


class ReportListResponse(BaseModel):
    reports: list[ResearchReportResponse]


class ArtifactDetailResponse(BaseModel):
    report_id: str
    kind: str
    status: str
    integrity_status: str
    provenance_status: str
    created_at_utc: str | None = None
    protocol_sha256: str | None = None
    analysis_sha256: str | None = None
    processing_manifest_sha256: str | None = None
    source_manifest_sha256: str | None = None
    audit_sha256: str | None = None
    method: str | None = None
    grouping: str | None = None
    blockers: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


class GroundTruthRunResponse(BaseModel):
    run_id: str
    configuration_id: str | None = None
    row_count: int
    event_count: int | None = None
    event_boundaries_available: bool


class GroundTruthIndexResponse(BaseModel):
    status: str
    message: str
    blockers: list[str] = Field(default_factory=list)
    analysis_sha256: str | None = None
    protocol_sha256: str | None = None
    threshold_ms: float | None = None
    max_gap_seconds: float | None = None
    alignment_status: str
    alignment_message: str
    report_id: str | None = None
    runs: list[GroundTruthRunResponse] = Field(default_factory=list)


class GroundTruthPointResponse(BaseModel):
    timestamp_seconds: float
    queue_backlog_bytes: float | None = None
    qdisc_queue_delay_proxy_ms: float | None = None
    event_active: bool | None = None
    future_event_onset: bool | None = None
    segment_start: bool


class GroundTruthEventResponse(BaseModel):
    observed_start_seconds: float | None = None
    observed_end_seconds: float | None = None
    onset_interval_lower_exclusive_seconds: float | None = None
    onset_interval_upper_inclusive_seconds: float | None = None
    onset_interval_identified: bool
    left_censored_or_gap_ambiguous: bool
    right_censored_or_gap_ambiguous: bool
    sample_count: int | None = None


class GroundTruthSeriesResponse(BaseModel):
    status: str
    message: str
    run_id: str
    analysis_sha256: str | None = None
    report_id: str | None = None
    event_boundaries_available: bool
    threshold_ms: float | None = None
    max_gap_seconds: float | None = None
    original_point_count: int
    displayed_point_count: int
    sampling_note: str
    points: list[GroundTruthPointResponse]
    events: list[GroundTruthEventResponse]
    limitations: list[str] = Field(default_factory=list)


class GroundTruthBuildResponse(BaseModel):
    status: str
    report_id: str
    message: str


class ModelMutationResponse(BaseModel):
    status: str | None = None
    active_model_id: str | None = None
    model: ModelRecordResponse | None = None
    message: str


class AuditResponse(BaseModel):
    status: str
    output_id: str
    message: str


class EvaluationRunResponse(BaseModel):
    status: str
    evaluation_id: str
    message: str


@dataclass(frozen=True)
class WebSettings:
    workspace: Workspace
    model_root: Path
    state_root: Path


def _configured_storage_path(
    value: str | None, default: Path, workspace: Workspace, label: str
) -> Path:
    candidate = Path(value).expanduser() if value else default
    if not candidate.is_absolute():
        candidate = workspace.project_root / candidate
    probe = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        probe = probe / part
        if probe.is_symlink():
            raise DashboardDataError(f"{label} must not traverse a symlink")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(workspace.raw_root.resolve())
    except ValueError:
        pass
    else:
        raise DashboardDataError(f"{label} must remain outside the raw source tree")
    return resolved


def resolve_web_settings(
    environ: dict[str, str] | None = None, project_root: Path | None = None
) -> WebSettings:
    env = os.environ if environ is None else environ
    workspace = resolve_workspace(env, project_root=project_root)
    model_root = _configured_storage_path(
        env.get("ACN_MODEL_ROOT"),
        workspace.project_root / "models" / "registry",
        workspace,
        "model registry root",
    )
    state_root = _configured_storage_path(
        env.get("ACN_APP_STATE_DIR"),
        workspace.reports_root / "app-state",
        workspace,
        "application state root",
    )
    if (
        model_root == state_root
        or model_root in state_root.parents
        or state_root in model_root.parents
    ):
        raise DashboardDataError("model registry and application state roots must be distinct")
    return WebSettings(workspace, model_root, state_root)


class WebService:
    def __init__(self, settings: WebSettings):
        self.settings = settings
        self.workspace = settings.workspace
        self.registry = ModelRegistry(self.workspace, settings.model_root, settings.state_root)
        self.history = PredictionHistory(settings.state_root / "prediction_history.sqlite3")

    def operational_overview(self) -> dict[str, Any]:
        inventory = scan_csv_inventory(self.workspace.raw_root)
        models = self.registry.list_models()
        count = self.history.count()
        latest = self.history.latest_warning()
        config_status = "not_checked"
        if self.workspace.config_path.is_file():
            try:
                config = load_config(self.workspace.config_path)
                config_status = str(config.get("status", "not_declared"))
                if config_status == "protocol_frozen":
                    verify_protocol_freeze_evidence(config, self.workspace.config_path)
            except (ConfigError, OSError, ValueError):
                config_status = "invalid_or_unavailable"
        try:
            manifest = read_json(self.workspace.processed_root / "processing_manifest.json")
            processed_exists = (
                manifest.get("status") == "complete"
                and not (self.workspace.processed_root / "PREPARATION_INCOMPLETE").exists()
            )
        except DashboardDataError:
            processed_exists = False
        eligible = [item for item in models["models"] if item.get("eligible")]
        raw_count = int(inventory.get("csv_count", 0))
        blockers: list[str] = []
        if raw_count == 0:
            blockers.append(
                "No raw telemetry files are present in the configured workspace; the collector is not connected."
            )
        elif inventory.get("truncated"):
            blockers.append(
                "The raw file inventory exceeds the bounded dashboard scan and is incomplete."
            )
        if not processed_exists:
            blockers.append(
                "No completed prepared dataset is available in the configured workspace."
            )
        elif not eligible:
            blockers.append(
                "Prepared artifacts are not eligible until checksums, source lineage, and the frozen protocol verify."
            )
        if config_status != "protocol_frozen":
            blockers.append(
                "The approved evaluation protocol is not verified as frozen; evaluation and model registration are blocked."
            )
        if not eligible:
            blockers.append(
                "No model has passed current grouped-evaluation and artifact-provenance eligibility checks."
            )
        return {
            "application": {
                "name": "ACN Network Intelligence",
                "version": __version__,
                "mode": "local",
            },
            "telemetry_source": {
                "status": "batch_sources_present" if raw_count else "not_configured",
                "message": (
                    f"{raw_count:,} source CSV files are discoverable; no live collector is configured."
                    if raw_count
                    else "No source CSV files are discoverable in the configured raw root."
                ),
                "csv_count": raw_count,
                "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            "model_status": {
                "status": "eligible" if eligible else "unavailable",
                "eligible_count": len(eligible),
                "active_model_id": models.get("active_model_id"),
                "message": "Current eligible artifacts are available."
                if eligible
                else "No model is registered and currently eligible for inference.",
            },
            "latest_warning": latest,
            "history_count": count,
            "data_freshness": {
                "status": "unknown",
                "message": "No wall-clock collector timestamp is configured; uploaded run-relative time cannot establish source freshness.",
            },
            "blockers": blockers,
        }

    def datasets(self) -> dict[str, Any]:
        inventory = scan_csv_inventory(self.workspace.raw_root)
        processed_manifest: dict[str, Any] | None = None
        processed_status = "missing"
        if self.workspace.processed_root.is_symlink():
            processed_status = "unsafe_path"
        elif self.workspace.processed_root.is_dir():
            manifest_path = self.workspace.processed_root / "processing_manifest.json"
            if not manifest_path.is_file() or manifest_path.is_symlink():
                processed_status = "missing"
            else:
                try:
                    processed_manifest = read_json(manifest_path)
                    processed_status = (
                        "available_unverified"
                        if processed_manifest.get("status") == "complete"
                        and not (self.workspace.processed_root / "PREPARATION_INCOMPLETE").exists()
                        else "incomplete"
                    )
                except DashboardDataError:
                    processed_status = "invalid"
        csv_count = int(inventory.get("csv_count", 0))
        raw_status = (
            "missing"
            if not csv_count
            else "review_required"
            if inventory.get("truncated") or inventory.get("symlink_count_not_followed", 0)
            else "available"
        )
        audit = (
            {
                "status": "configured_unverified",
                "message": "An audit report is configured; explicit integrity refresh is required before its hashes are trusted.",
            }
            if self.workspace.audit_report_path
            else {"status": "not_configured", "message": "No audit report is configured."}
        )
        return {
            "raw": {
                "status": raw_status,
                "csv_count": csv_count,
                "message": inventory.get(
                    "message", "Filename inventory only; contents and meanings are not verified."
                ),
            },
            "processed": {
                "status": processed_status,
                "row_count": processed_manifest.get("analysis_row_count")
                if processed_manifest
                else None,
                "run_count": processed_manifest.get("run_count") if processed_manifest else None,
                "protocol_status": processed_manifest.get("protocol_status")
                if processed_manifest
                else None,
                "message": "Manifest metadata only; checksums are checked on explicit evaluation/training actions."
                if processed_manifest
                else "No processing manifest is available.",
            },
            "audit": audit,
        }

    def research(self) -> dict[str, Any]:
        protocol_status = "missing"
        config_hash = None
        protocol_message = "Experiment configuration is not available."
        try:
            config = load_config(self.workspace.config_path)
            protocol_status = str(config.get("status", "not_declared"))
            import hashlib

            config_hash = hashlib.sha256(self.workspace.config_path.read_bytes()).hexdigest()
            if protocol_status == "protocol_frozen":
                verify_protocol_freeze_evidence(config, self.workspace.config_path)
                protocol_message = "Frozen decision record and current config fingerprint verify. This is not scientific validation."
            else:
                protocol_message = "Protocol remains draft or exploratory; confirmatory evaluation and inference eligibility are blocked."
        except (ConfigError, OSError, ValueError) as exc:
            protocol_status = "invalid_or_unavailable"
            protocol_message = str(exc)
        alignment = {
            "status": "not_required",
            "message": "No separate-stream alignment manifest is linked to the current processed data.",
        }
        if self.workspace.manifest_path and self.workspace.manifest_path.is_file():
            try:
                manifest = read_json(self.workspace.manifest_path)
                if "alignment_status" in manifest:
                    if manifest.get("alignment_status") == "approved":
                        try:
                            verify_alignment_review(
                                self.workspace.raw_root.resolve(),
                                manifest,
                                self.workspace.manifest_path,
                            )
                            alignment = {
                                "status": "approved",
                                "message": "Hash-linked alignment decision verifies; this does not prove clock or scientific validity.",
                            }
                        except (ProcessingError, OSError, ValueError) as exc:
                            alignment = {
                                "status": "invalid",
                                "message": f"Alignment approval evidence failed verification: {exc}",
                            }
                    else:
                        alignment = {
                            "status": str(manifest.get("alignment_status")),
                            "message": "Alignment remains review-gated; no approval is inferred by the web application.",
                        }
            except DashboardDataError as exc:
                alignment = {"status": "invalid", "message": str(exc)}
        processed = verify_processed_artifacts(self.workspace.processed_root)
        shared: dict[str, Any] = {
            "status": "unverified",
            "error": "processed data are not currently verified",
        }
        if processed.get("status") == "verified":
            try:
                audit = verify_audit_inventory(
                    self.workspace.raw_root, self.workspace.audit_report_path
                )
                shared = verify_workspace_lineage_sources(self.workspace, processed, audit)
            except (DashboardDataError, OSError, ValueError) as exc:
                shared = {"status": "unverified", "error": str(exc)}
        evaluations: list[dict[str, Any]] = []
        reports: list[dict[str, Any]] = []
        for directory in discover_report_directories(
            self.workspace.reports_root, "evaluation_manifest.json", max_reports=20
        )[:5]:
            try:
                result = verify_evaluation_bundle(
                    directory, self.workspace, processed_artifacts=processed, shared_sources=shared
                )
                report = result.get("report") or read_json(
                    safe_child(directory, "evaluation_manifest.json")
                )
                metrics = read_json(safe_child(directory, "metrics.json"))
                evaluations.append(
                    {
                        "evaluation_id": directory.name,
                        "status": report.get("status", "unknown"),
                        "provenance_status": "verified"
                        if result.get("status") == "complete_validated_provenance"
                        else "unverified",
                        "grouping": report.get("split_grouping"),
                        "created_at_utc": report.get("created_at_utc"),
                        "methods": [
                            m
                            for m in report.get("successful_methods", [])
                            if m in metrics and m != "method_failures"
                        ],
                        "blockers": result.get("errors", []),
                    }
                )
            except (DashboardDataError, OSError, ValueError):
                continue
        for manifest_name, kind, verifier in (
            ("ground_truth_manifest.json", "ground_truth", verify_ground_truth_bundle),
            ("robustness_manifest.json", "robustness", verify_robustness_bundle),
        ):
            for directory in discover_report_directories(
                self.workspace.reports_root, manifest_name, max_reports=100
            )[:10]:
                try:
                    item = read_json(safe_child(directory, manifest_name))
                    checked = verifier(
                        directory,
                        self.workspace,
                        processed_artifacts=processed,
                        shared_sources=shared,
                    )
                    reports.append(
                        {
                            "report_id": directory.name,
                            "kind": kind,
                            "status": "integrity verified"
                            if checked.get("status") == "verified"
                            else "partial_failure · integrity verified"
                            if checked.get("status") == "partial_failure"
                            else str(item.get("status", "unverified")),
                            "created_at_utc": item.get("created_at_utc"),
                        }
                    )
                except (DashboardDataError, OSError, ValueError):
                    continue
        frozen = protocol_status == "protocol_frozen"
        prepared = processed.get("status") == "verified"
        feature_contract_validated = (
            processed.get("feature_specification_operationally_validated") is True
        )
        feature_contract_reasons = [
            str(item)
            for item in processed.get("feature_contract_reasons", [])
            if isinstance(item, str) and item
        ]
        sources_verified = shared.get("status") == "verified"
        processed_errors = [
            str(item) for item in processed.get("errors", []) if isinstance(item, str) and item
        ]
        return {
            "protocol": {
                "status": protocol_status,
                "message": protocol_message,
                "config_sha256": config_hash,
            },
            "alignment": alignment,
            "evaluations": evaluations,
            "reports": reports,
            "operations": {
                "audit_available": self.workspace.raw_root.is_dir(),
                "evaluation_available": (
                    frozen and prepared and sources_verified and feature_contract_validated
                ),
                "message": "Evaluation uses the current frozen config and grouped evaluator; this application does not modify protocol parameters.",
                "prepared_status": str(processed.get("status", "unknown")),
                "source_lineage_status": str(shared.get("status", "unknown")),
                "feature_specification_status": str(
                    processed.get("feature_specification_status", "unknown")
                ),
                "tcp_retransmission_unit": processed.get("tcp_retransmission_unit"),
                "feature_temporal_policy_status": str(
                    (processed.get("feature_temporal_policy") or {}).get("status", "unknown")
                ),
                "blockers": ([protocol_message] if not frozen else [])
                + (
                    processed_errors or ["Prepared artifact checks did not pass."]
                    if not prepared
                    else []
                )
                + (
                    feature_contract_reasons
                    or ["Prepared feature contract is not operationally validated."]
                    if prepared and not feature_contract_validated
                    else []
                )
                + (
                    [str(shared.get("error", "Current source lineage checks did not pass."))]
                    if prepared and not sources_verified
                    else []
                ),
            },
        }

    def reports(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for manifest_name, kind in (
            ("evaluation_manifest.json", "evaluation"),
            ("ground_truth_manifest.json", "ground_truth"),
            ("robustness_manifest.json", "robustness"),
        ):
            for directory in discover_report_directories(
                self.workspace.reports_root, manifest_name, max_reports=250
            )[:20]:
                try:
                    payload = read_json(safe_child(directory, manifest_name))
                except (DashboardDataError, OSError):
                    continue
                items.append(
                    {
                        "report_id": directory.name,
                        "kind": kind,
                        "status": payload.get("status", "unknown"),
                        "created_at_utc": payload.get("created_at_utc"),
                    }
                )
        items.sort(key=lambda item: item.get("created_at_utc") or "", reverse=True)
        return items[:100]

    def _current_research_inputs(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        processed = verify_processed_artifacts(self.workspace.processed_root)
        if processed.get("status") != "verified":
            errors = processed.get("errors", [])
            message = (
                str(errors[0])
                if errors
                else "Current prepared data do not pass integrity verification."
            )
            code = (
                "processed_data_missing"
                if processed.get("status") == "missing"
                else "processed_data_unverified"
            )
            raise APIProblem(409, code, message)
        try:
            audit = verify_audit_inventory(
                self.workspace.raw_root, self.workspace.audit_report_path
            )
            shared = verify_workspace_lineage_sources(self.workspace, processed, audit)
        except (DashboardDataError, OSError, ValueError) as exc:
            raise APIProblem(
                409, "lineage_unverified", f"Current source lineage is not verified: {exc}"
            ) from exc
        if shared.get("status") != "verified":
            raise APIProblem(
                409,
                "lineage_unverified",
                str(shared.get("error", "Current source lineage is not verified.")),
            )
        analysis_path = self.workspace.processed_root / "analysis.csv"
        row_count = processed.get("manifest", {}).get("analysis_row_count")
        if analysis_path.stat().st_size > MAX_WEB_EVALUATION_BYTES:
            raise APIProblem(
                413,
                "dataset_too_large",
                "Prepared analysis exceeds the 64 MiB workspace display limit.",
            )
        if (
            not isinstance(row_count, int)
            or isinstance(row_count, bool)
            or row_count > MAX_WEB_EVALUATION_ROWS
        ):
            raise APIProblem(
                413,
                "dataset_too_large",
                "Prepared analysis exceeds the 100,000-row workspace display limit.",
            )
        try:
            rows, processing_manifest = load_analysis_dataset(self.workspace.processed_root)
        except ReportError as exc:
            raise APIProblem(409, "analysis_invalid", str(exc)) from exc
        return processed, shared, {"rows": rows, "manifest": processing_manifest}

    def ground_truth_index(self) -> dict[str, Any]:
        try:
            processed, shared, data = self._current_research_inputs()
        except APIProblem as exc:
            return {
                "status": "unavailable",
                "message": exc.message,
                "blockers": [exc.message],
                "alignment_status": "unverified",
                "alignment_message": "Alignment state is unavailable until the current prepared data and lineage verify.",
                "runs": [],
            }
        manifest = data["manifest"]
        rows = data["rows"]
        run_rows: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            run_rows.setdefault(str(row["run_id"]), []).append(row)
        run_summaries: dict[str, dict[str, Any]] = {}
        report_id: str | None = None
        report_data: dict[str, Any] | None = None
        verified_reports: list[tuple[str, str, dict[str, Any]]] = []
        for directory in discover_report_directories(
            self.workspace.reports_root, "ground_truth_manifest.json", max_reports=100
        ):
            try:
                check = verify_ground_truth_bundle(
                    directory, self.workspace, processed_artifacts=processed, shared_sources=shared
                )
                payload = read_json(safe_child(directory, "ground_truth_report.json"))
                report_manifest = read_json(safe_child(directory, "ground_truth_manifest.json"))
                if check.get("status") == "verified":
                    verified_reports.append(
                        (str(report_manifest.get("created_at_utc", "")), directory.name, payload)
                    )
            except (APIProblem, DashboardDataError, GroundTruthReportError, OSError, ValueError):
                continue
        if verified_reports:
            _, report_id, report_data = max(verified_reports, key=lambda item: item[0])
            run_summaries = {
                str(item.get("run_id")): item
                for item in report_data.get("per_run", [])
                if isinstance(item, dict) and isinstance(item.get("run_id"), str)
            }
        records = manifest.get("run_records", [])
        record_by_id = {
            str(record.get("run_id")): record for record in records if isinstance(record, dict)
        }
        runs = []
        for run_id, run in sorted(run_rows.items()):
            summary = run_summaries.get(run_id, {})
            labels = summary.get("labels", {}) if isinstance(summary, dict) else {}
            events = labels.get("events") if isinstance(labels, dict) else None
            runs.append(
                {
                    "run_id": run_id,
                    "configuration_id": record_by_id.get(run_id, {}).get("configuration_id"),
                    "row_count": len(run),
                    "event_count": labels.get("event_count_observed_in_trace")
                    if isinstance(labels, dict)
                    else None,
                    "event_boundaries_available": isinstance(events, list),
                }
            )
        alignment_status, alignment_message = self._alignment_status()
        params = manifest.get("label_parameters", {})
        max_gap_ms = params.get("max_ground_truth_gap_ms")
        try:
            max_gap_seconds = float(max_gap_ms) / 1000 if max_gap_ms is not None else None
        except (TypeError, ValueError):
            max_gap_seconds = None
        has_report = report_data is not None
        return {
            "status": "ready" if rows else "unavailable",
            "message": "Verified prepared batch data are available. Event boundaries are overlaid only from a current checksum-verified ground-truth report."
            if has_report
            else "Verified prepared batch data are available, but no current integrity-linked ground-truth report is available; event boundaries and event counts are not inferred.",
            "blockers": []
            if has_report
            else [
                "No current verified ground-truth report; generate a descriptive report to expose event intervals."
            ],
            "analysis_sha256": manifest.get("analysis_sha256"),
            "protocol_sha256": manifest.get("protocol_sha256"),
            "threshold_ms": params.get("queue_delay_threshold_ms"),
            "max_gap_seconds": max_gap_seconds,
            "alignment_status": alignment_status,
            "alignment_message": alignment_message,
            "report_id": report_id,
            "runs": runs,
        }

    def _alignment_status(self) -> tuple[str, str]:
        path = self.workspace.manifest_path
        if not path or not path.is_file():
            return (
                "not_required",
                "No separate-stream alignment manifest is linked to the current processed data.",
            )
        try:
            manifest = read_json(path)
            if "alignment_status" not in manifest:
                return (
                    "not_required",
                    "No separate-stream alignment manifest is linked to the current processed data.",
                )
            if manifest.get("alignment_status") == "approved":
                verify_alignment_review(self.workspace.raw_root.resolve(), manifest, path)
                return (
                    "approved",
                    "Hash-linked alignment decision verifies; this does not prove clock or scientific validity.",
                )
            return str(
                manifest.get("alignment_status")
            ), "Alignment remains review-gated; the application does not infer approval."
        except (DashboardDataError, ProcessingError, OSError, ValueError) as exc:
            return "invalid", f"Alignment approval evidence failed verification: {exc}"

    def ground_truth_series(self, run_id: str) -> dict[str, Any]:
        if not run_id or len(run_id) > 160:
            raise APIProblem(422, "invalid_run_id", "Run identifier is invalid.")
        index = self.ground_truth_index()
        if index.get("status") != "ready":
            raise APIProblem(409, "ground_truth_unavailable", str(index.get("message")))
        try:
            _processed, _shared, data = self._current_research_inputs()
        except APIProblem as exc:
            raise APIProblem(409, "ground_truth_unavailable", exc.message) from exc
        run_rows = [row for row in data["rows"] if str(row.get("run_id")) == run_id]
        if not run_rows:
            raise APIProblem(
                404,
                "run_not_found",
                "Run identifier is not in the current verified prepared dataset.",
            )
        report = None
        if index.get("report_id"):
            report_dir = self._report_directory(
                str(index["report_id"]), "ground_truth_manifest.json"
            )
            report = read_json(safe_child(report_dir, "ground_truth_report.json"))
        run_summary = next(
            (item for item in (report or {}).get("per_run", []) if item.get("run_id") == run_id),
            None,
        )
        events_raw = run_summary.get("labels", {}).get("events", []) if run_summary else []
        events = [
            {
                "observed_start_seconds": event.get("observed_start_seconds"),
                "observed_end_seconds": event.get("observed_end_seconds"),
                "onset_interval_lower_exclusive_seconds": event.get(
                    "onset_interval_lower_exclusive_seconds"
                ),
                "onset_interval_upper_inclusive_seconds": event.get(
                    "onset_interval_upper_inclusive_seconds"
                ),
                "onset_interval_identified": bool(event.get("onset_interval_identified", False)),
                "left_censored_or_gap_ambiguous": bool(
                    event.get("left_censored_or_gap_ambiguous", False)
                ),
                "right_censored_or_gap_ambiguous": bool(
                    event.get("right_censored_or_gap_ambiguous", False)
                ),
                "sample_count": event.get("sample_count"),
            }
            for event in events_raw
            if isinstance(event, dict)
        ]
        params = data["manifest"].get("label_parameters", {})
        max_gap_raw = params.get("max_ground_truth_gap_ms")
        max_gap = float(max_gap_raw) / 1000 if isinstance(max_gap_raw, (int, float)) else None
        important = {0, len(run_rows) - 1}
        gap_starts: set[int] = set()
        for i, row in enumerate(run_rows):
            current = row.get("event_active")
            previous = run_rows[i - 1].get("event_active") if i else None
            if i == 0 or current != previous:
                important.add(i)
            if i and max_gap is not None:
                try:
                    if (
                        float(row["timestamp_seconds"])
                        - float(run_rows[i - 1]["timestamp_seconds"])
                        > max_gap
                    ):
                        important.update((i - 1, i))
                        gap_starts.add(i)
                except (KeyError, TypeError, ValueError):
                    important.add(i)
        if len(run_rows) <= MAX_WEB_TIMELINE_POINTS:
            selected = set(range(len(run_rows)))
            sampling_note = "All validated samples are shown."
        else:
            step = (len(run_rows) - 1) / (MAX_WEB_TIMELINE_POINTS - 1)
            selected = {round(i * step) for i in range(MAX_WEB_TIMELINE_POINTS)} | important
            if len(selected) > MAX_WEB_TIMELINE_POINTS * 2:
                selected = set(
                    sorted(selected)[: MAX_WEB_TIMELINE_POINTS * 2 - 1] + [len(run_rows) - 1]
                )
            sampling_note = f"{len(run_rows):,} validated rows reduced to {len(selected):,} display points; endpoints, label transitions, and detected long-gap edges are retained when within the display cap."
        points = []
        for i in sorted(selected):
            row = run_rows[i]

            def finite_number(value: Any) -> float | None:
                if value is None or isinstance(value, bool):
                    return None
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    return None
                return number if math.isfinite(number) else None

            try:
                timestamp = float(row["timestamp_seconds"])
            except (KeyError, TypeError, ValueError) as exc:
                raise APIProblem(
                    409, "analysis_invalid", "Validated analysis contains a non-numeric timestamp."
                ) from exc
            if not math.isfinite(timestamp):
                raise APIProblem(
                    409, "analysis_invalid", "Validated analysis contains a non-finite timestamp."
                )
            segment_start = i == 0 or i in gap_starts
            points.append(
                {
                    "timestamp_seconds": timestamp,
                    "queue_backlog_bytes": finite_number(row.get("queue_backlog_bytes")),
                    "qdisc_queue_delay_proxy_ms": finite_number(
                        row.get("qdisc_queue_delay_proxy_ms")
                    ),
                    "event_active": row.get("event_active")
                    if isinstance(row.get("event_active"), bool)
                    else None,
                    "future_event_onset": bool(row["future_event_onset"])
                    if isinstance(row.get("future_event_onset"), (bool, int))
                    and row.get("future_event_onset") in (False, True, 0, 1)
                    else None,
                    "segment_start": segment_start,
                }
            )
        limitations = list((report or {}).get("limitations", []))
        if not report:
            limitations.append(
                "No current ground-truth report verifies event intervals; the plot contains only validated prepared values and labels, with no event boundaries or event count asserted."
            )
        limitations.append(
            "qdisc_queue_delay_proxy_ms is backlog_bytes × 8 ÷ declared bottleneck rate: a serialization-delay proxy, not direct packet queueing-delay measurement."
        )
        limitations.append(
            "Run-relative timestamps do not establish wall-clock freshness. Gaps are not silently interpolated; lines break at detected sampling gaps, while intermediate points may be omitted for display."
        )
        return {
            "status": "verified_prepared_batch",
            "message": "Timeline values are read from the current integrity- and lineage-verified prepared batch.",
            "run_id": run_id,
            "analysis_sha256": data["manifest"].get("analysis_sha256"),
            "report_id": index.get("report_id"),
            "event_boundaries_available": report is not None and run_summary is not None,
            "threshold_ms": params.get("queue_delay_threshold_ms"),
            "max_gap_seconds": max_gap,
            "original_point_count": len(run_rows),
            "displayed_point_count": len(points),
            "sampling_note": sampling_note,
            "points": points,
            "events": events,
            "limitations": limitations,
        }

    def build_ground_truth(self) -> dict[str, Any]:
        _processed, _shared, _data = self._current_research_inputs()
        parent = self.workspace.reports_root / "ground_truth"
        _ensure_safe_output_parent(parent, self.workspace.raw_root)
        report_id = (
            "ground_truth_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "_"
            + uuid.uuid4().hex[:8]
        )
        output = parent / report_id
        try:
            result = build_ground_truth_report(self.workspace.processed_root, output)
        except (GroundTruthReportError, ReportError, OSError, ValueError) as exc:
            raise APIProblem(422, "ground_truth_report_failed", str(exc)) from exc
        return {
            "status": result.get("status", "complete"),
            "report_id": report_id,
            "message": f"Descriptive report created for {result.get('run_count', 0)} verified prepared runs and {result.get('event_count', 0)} observed target intervals. This does not validate the collector or establish scientific validity.",
        }

    def _report_directory(self, report_id: str, manifest_name: str | None = None) -> Path:
        if not REPORT_ID_PATTERN.fullmatch(report_id):
            raise APIProblem(422, "invalid_report_id", "Report identifier is invalid.")
        names = (
            [manifest_name]
            if manifest_name
            else [
                "evaluation_manifest.json",
                "ground_truth_manifest.json",
                "robustness_manifest.json",
            ]
        )
        matches = [
            directory
            for name in names
            if name
            for directory in discover_report_directories(
                self.workspace.reports_root, name, max_reports=250
            )
            if directory.name == report_id
        ]
        if len(matches) != 1:
            raise APIProblem(
                404,
                "report_not_found",
                "Report identifier is not in the configured report inventory.",
            )
        return matches[0]

    def evaluation_detail(self, evaluation_id: str) -> dict[str, Any]:
        directory = self._report_directory(evaluation_id, "evaluation_manifest.json")
        try:
            for name in ("evaluation_manifest.json", "metrics.json", "splits.json"):
                _check_report_metadata_size(directory, name)
            internal = verify_evaluation_artifacts(directory)
            manifest = read_json(safe_child(directory, "evaluation_manifest.json"))
            metrics = read_json(safe_child(directory, "metrics.json"))
            splits = read_json(safe_child(directory, "splits.json"))
        except (ReportError, DashboardDataError, OSError, ValueError) as exc:
            raise APIProblem(
                409, "evaluation_artifacts_invalid", f"Evaluation artifacts did not verify: {exc}"
            ) from exc
        try:
            current = verify_evaluation_bundle(directory, self.workspace)
            provenance = (
                "verified"
                if current.get("status") == "complete_validated_provenance"
                else "unverified"
            )
            blockers = list(current.get("errors", []))
        except (DashboardDataError, OSError, ValueError) as exc:
            provenance = "unverified"
            blockers = [f"Current lineage verification failed: {exc}"]
        if manifest.get("status") != "complete":
            provenance = "unverified"
            blockers.append(
                "This evaluation is partial or failed; its method results are not eligible for registration."
            )
        groups = splits.get("groups", {}) if isinstance(splits, dict) else {}
        held_out = groups.get("test", []) if isinstance(groups, dict) else []
        if not isinstance(held_out, list):
            held_out = []
        methods = []
        for method in manifest.get("successful_methods", []):
            result = metrics.get(method)
            if not isinstance(result, dict):
                continue
            event = result.get("test_event_metrics")
            event = event if isinstance(event, dict) else {}
            secondary = result.get("test_window_metrics_secondary")
            secondary = secondary if isinstance(secondary, dict) else {}
            calibration = result.get("threshold_calibration")
            calibration = calibration if isinstance(calibration, dict) else {}
            validation_events = calibration.get("validation_event_metrics")
            validation_events = validation_events if isinstance(validation_events, dict) else {}
            lead_time = event.get("warning_lead_time_seconds")
            lead_time = lead_time if isinstance(lead_time, dict) else {}
            raw_uncertainty = result.get("test_uncertainty")
            raw_uncertainty = raw_uncertainty if isinstance(raw_uncertainty, dict) else {}
            uncertainty = {}
            for metric_name, interval in raw_uncertainty.items():
                if not isinstance(interval, dict):
                    continue
                lower, upper = interval.get("lower"), interval.get("upper")
                numeric_interval = (
                    [lower, upper]
                    if isinstance(lower, (int, float))
                    and not isinstance(lower, bool)
                    and isinstance(upper, (int, float))
                    and not isinstance(upper, bool)
                    and math.isfinite(float(lower))
                    and math.isfinite(float(upper))
                    else None
                )
                uncertainty[str(metric_name)] = {
                    "estimable": interval.get("estimable") is True,
                    "interval": numeric_interval,
                    "cluster_count": interval.get("cluster_count"),
                    "reason": interval.get("reason"),
                    "warning": interval.get("warning"),
                    "confidence": interval.get("confidence"),
                    "unit_of_resampling": interval.get("unit_of_resampling"),
                    "bootstrap_resamples_used": interval.get("bootstrap_resamples_used"),
                }
            methods.append(
                {
                    "method": str(method),
                    "event_count": event.get("event_count"),
                    "event_precision": event.get("event_precision"),
                    "event_recall": event.get("event_recall"),
                    "false_alarms_per_minute": event.get("false_alarms_per_minute"),
                    "lead_time_median_seconds": lead_time.get("median"),
                    "lead_time_event_count": lead_time.get("count"),
                    "average_precision": secondary.get("average_precision"),
                    "roc_auc": secondary.get("roc_auc"),
                    "scored_anchor_count": secondary.get("n_scored"),
                    "threshold": calibration.get("threshold"),
                    "validation_false_alarms_per_minute": validation_events.get(
                        "false_alarms_per_minute"
                    ),
                    "uncertainty": uncertainty,
                    "adequacy_warnings": event.get("adequacy_warnings", [])
                    if isinstance(event.get("adequacy_warnings", []), list)
                    else [],
                }
            )
        artifacts = ["report.md", "metrics.json", "splits.json"]
        for record in manifest.get("prediction_files", {}).values():
            if isinstance(record, dict) and isinstance(record.get("path"), str):
                artifacts.append(record["path"])
        return {
            "report_id": evaluation_id,
            "kind": "evaluation",
            "status": manifest.get("status", "unknown"),
            "integrity_status": "verified",
            "provenance_status": provenance,
            "created_at_utc": manifest.get("created_at_utc"),
            "protocol_sha256": manifest.get("protocol_sha256"),
            "analysis_sha256": manifest.get("analysis_sha256"),
            "processing_manifest_sha256": manifest.get("processing_manifest_sha256"),
            "source_manifest_sha256": manifest.get("source_manifest_sha256"),
            "audit_sha256": (manifest.get("raw_audit") or {}).get("report_sha256"),
            "grouping": manifest.get("split_grouping"),
            "blockers": blockers,
            "artifacts": artifacts,
            "summary": {
                "methods": methods,
                "method_failures": manifest.get("method_failures", {}),
                "requested_methods": manifest.get("requested_methods", []),
                "split_seed": manifest.get("split_seed"),
                "held_out_group_count": len(held_out),
                "held_out_groups": held_out[:500],
                "held_out_groups_truncated": len(held_out) > 500,
                "split_row_counts": splits.get("row_counts", {}),
                "split_event_counts": splits.get("event_counts", {}),
                "split_group_counts": splits.get("group_counts", {}),
                "threshold_calibration_split": manifest.get("threshold_calibration_split"),
                "threshold_calibration_rule": manifest.get("threshold_calibration_rule"),
                "verified_artifacts": internal.get("verified_artifacts", []),
            },
        }

    def report_detail(self, report_id: str) -> dict[str, Any]:
        directory = self._report_directory(report_id)
        manifest_name = next(
            (
                name
                for name in (
                    "evaluation_manifest.json",
                    "ground_truth_manifest.json",
                    "robustness_manifest.json",
                )
                if (directory / name).is_file()
            ),
            None,
        )
        if manifest_name == "evaluation_manifest.json":
            return self.evaluation_detail(report_id)
        if manifest_name == "ground_truth_manifest.json":
            try:
                _check_report_metadata_size(directory, "ground_truth_manifest.json")
                _check_report_metadata_size(directory, "ground_truth_report.json")
                verified = verify_ground_truth_report(directory)
                manifest = read_json(safe_child(directory, manifest_name))
                report = read_json(safe_child(directory, "ground_truth_report.json"))
            except (GroundTruthReportError, DashboardDataError, OSError, ValueError) as exc:
                raise APIProblem(
                    409, "report_artifacts_invalid", f"Ground-truth report did not verify: {exc}"
                ) from exc
            provenance = "integrity_only"
            blockers = [
                "Report checksums verify, but current source lineage was not established for this detail request."
            ]
            try:
                processed, shared, _ = self._current_research_inputs()
                current = verify_ground_truth_bundle(
                    directory, self.workspace, processed_artifacts=processed, shared_sources=shared
                )
                if current.get("status") == "verified":
                    provenance = "current_lineage_verified"
                    blockers = []
            except APIProblem:
                pass
            summary = {
                "evidence_status": report.get("evidence_status"),
                "protocol_status": report.get("protocol_status"),
                "run_count": report.get("run_count"),
                "configuration_count": report.get("configuration_count"),
                "event_count_observed_in_trace": report.get("event_count_observed_in_trace"),
                "events_with_identified_onset_interval": report.get(
                    "events_with_identified_onset_interval"
                ),
                "label_parameters": report.get("label_parameters"),
                "limitations": report.get("limitations", []),
                "per_run": report.get("per_run", [])[:500],
            }
            return {
                "report_id": report_id,
                "kind": "ground_truth",
                "status": manifest.get("status", "unknown"),
                "integrity_status": "verified",
                "provenance_status": provenance,
                "created_at_utc": manifest.get("created_at_utc"),
                "protocol_sha256": report.get("protocol_sha256"),
                "analysis_sha256": manifest.get("analysis_sha256"),
                "processing_manifest_sha256": manifest.get("processing_manifest_sha256"),
                "source_manifest_sha256": manifest.get("source_manifest_sha256"),
                "audit_sha256": (report.get("raw_audit") or {}).get("report_sha256"),
                "blockers": blockers,
                "artifacts": verified.get("verified_artifacts", []),
                "summary": summary,
            }
        try:
            _check_report_metadata_size(directory, "robustness_manifest.json")
            _check_report_metadata_size(directory, "robustness.json")
            verified = verify_robustness_artifacts(directory)
            manifest = read_json(safe_child(directory, manifest_name or "robustness_manifest.json"))
            payload = read_json(safe_child(directory, "robustness.json"))
        except (RobustnessError, DashboardDataError, OSError, ValueError) as exc:
            raise APIProblem(
                409, "report_artifacts_invalid", f"Robustness report did not verify: {exc}"
            ) from exc
        provenance = "integrity_only"
        blockers = [
            "Report checksums verify, but current source lineage was not established for this detail request."
        ]
        try:
            processed, _shared, _ = self._current_research_inputs()
            current = verify_robustness_bundle(
                directory, self.workspace, processed_artifacts=processed
            )
            if current.get("status") in {"complete_validated_provenance", "verified"}:
                provenance, blockers = "current_lineage_verified", []
            elif current.get("status") == "partial_failure":
                provenance = "current_lineage_verified"
                blockers = [
                    (
                        "Robustness outcome is partial_failure; verified lineage does not "
                        "convert failed methods into completed results."
                    )
                ]
        except (APIProblem, DashboardDataError, OSError, ValueError):
            pass
        return {
            "report_id": report_id,
            "kind": "robustness",
            "status": manifest.get("status", "unknown"),
            "integrity_status": "verified",
            "provenance_status": provenance,
            "created_at_utc": manifest.get("created_at_utc"),
            "protocol_sha256": manifest.get("protocol_sha256"),
            "analysis_sha256": manifest.get("analysis_sha256"),
            "processing_manifest_sha256": manifest.get("processing_manifest_sha256"),
            "source_manifest_sha256": manifest.get("source_manifest_sha256"),
            "audit_sha256": (manifest.get("raw_audit") or {}).get("report_sha256"),
            "blockers": blockers,
            "artifacts": verified.get("verified_artifacts", []),
            "summary": {
                "scenario_count": payload.get("scenario_count"),
                "status": payload.get("status"),
                "results": payload.get("results", [])[:200],
            },
        }

    def run_evaluation(self) -> dict[str, Any]:
        if not self.workspace.config_path.is_file():
            raise APIProblem(
                409, "config_missing", "Frozen experiment configuration is not available."
            )
        processed = verify_processed_artifacts(self.workspace.processed_root)
        if processed.get("status") != "verified":
            raise APIProblem(
                409,
                "processed_data_unverified",
                "A checksum-verified processed dataset is required.",
            )
        try:
            audit = verify_audit_inventory(
                self.workspace.raw_root, self.workspace.audit_report_path
            )
            verify_workspace_lineage_sources(self.workspace, processed, audit)
            config = load_config(self.workspace.config_path)
            verify_protocol_freeze_evidence(config, self.workspace.config_path)
        except (DashboardDataError, ConfigError, OSError, ValueError) as exc:
            raise APIProblem(409, "research_gates_blocked", str(exc)) from exc
        analysis_path = self.workspace.processed_root / "analysis.csv"
        row_count = processed.get("manifest", {}).get("analysis_row_count")
        if analysis_path.stat().st_size > MAX_WEB_EVALUATION_BYTES:
            raise APIProblem(
                413,
                "dataset_too_large",
                "Prepared analysis exceeds the web evaluator's 64 MiB input limit.",
            )
        if (
            not isinstance(row_count, int)
            or isinstance(row_count, bool)
            or row_count > MAX_WEB_EVALUATION_ROWS
        ):
            raise APIProblem(
                413,
                "dataset_too_large",
                "Prepared analysis exceeds the web evaluator's 100,000-row compute limit.",
            )
        parent = self.workspace.reports_root / "evaluations"
        _ensure_safe_output_parent(parent, self.workspace.raw_root)
        run_id = (
            "evaluation_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "_"
            + uuid.uuid4().hex[:8]
        )
        output = parent / run_id
        from .cli import _evaluate

        args = SimpleNamespace(
            config=self.workspace.config_path,
            processed_dir=self.workspace.processed_root,
            output=output,
            grouping=None,
            seed=None,
        )
        capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(capture):
                exit_code = _evaluate(args)
        except Exception as exc:
            print(f"ACN grouped evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            raise APIProblem(
                422,
                "evaluation_failed",
                f"Existing grouped evaluator failed ({type(exc).__name__}); inspect local logs and any partial report.",
            ) from exc
        if not output.is_dir():
            raise APIProblem(
                500,
                "evaluation_output_missing",
                "Evaluator returned without creating its report directory.",
            )
        if exit_code != 0:
            raise APIProblem(
                422,
                "evaluation_incomplete",
                "Evaluation wrote a failed or partial report; it is not eligible for model registration.",
            )
        return {
            "status": "complete",
            "evaluation_id": run_id,
            "message": "Grouped evaluation completed. Current artifact checks are required before model registration.",
        }


def _ensure_safe_output_parent(path: Path, raw_root: Path) -> None:
    if path.is_symlink():
        raise APIProblem(409, "unsafe_output", "Output directory must not be a symlink.")
    try:
        path.resolve().relative_to(raw_root.resolve())
    except ValueError:
        pass
    else:
        raise APIProblem(
            409, "unsafe_output", "Report outputs must remain outside raw source files."
        )
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if probe.is_symlink():
        raise APIProblem(409, "unsafe_output", "Output parent may not traverse a symlink.")


def _check_report_metadata_size(directory: Path, filename: str) -> Path:
    try:
        path = safe_child(directory, filename)
    except DashboardDataError as exc:
        raise APIProblem(404, "artifact_not_found", str(exc)) from exc
    if path.stat().st_size > MAX_WEB_DETAIL_BYTES:
        raise APIProblem(
            413,
            "artifact_too_large",
            "In-app report detail JSON is limited to 16 MiB per file.",
        )
    return path


def _safe_report_file(service: WebService, report_id: str, filename: str) -> Path:
    if not REPORT_ID_PATTERN.fullmatch(report_id):
        raise APIProblem(422, "invalid_report_id", "Report identifier is invalid.")
    if (
        not filename
        or filename in {".", ".."}
        or Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
    ):
        raise APIProblem(404, "artifact_not_found", "Requested report artifact is not allowlisted.")
    directory = service._report_directory(report_id)
    try:
        if (directory / "evaluation_manifest.json").is_file():
            for name in ("evaluation_manifest.json", "metrics.json", "splits.json"):
                _check_report_metadata_size(directory, name)
            verify_evaluation_artifacts(directory)
            manifest = read_json(safe_child(directory, "evaluation_manifest.json"))
            allowed = {"report.md", "metrics.json", "splits.json"}
            for record in manifest.get("prediction_files", {}).values():
                if isinstance(record, dict) and isinstance(record.get("path"), str):
                    candidate = record["path"]
                    if (
                        re.fullmatch(r"predictions_[A-Za-z0-9_.-]{1,80}\.csv", candidate)
                        and Path(candidate).name == candidate
                    ):
                        allowed.add(candidate)
            if filename not in allowed:
                raise APIProblem(
                    404,
                    "artifact_not_found",
                    "Requested evaluation artifact is not in its verified manifest allowlist.",
                )
        elif (directory / "ground_truth_manifest.json").is_file():
            _check_report_metadata_size(directory, "ground_truth_manifest.json")
            _check_report_metadata_size(directory, "ground_truth_report.json")
            verify_ground_truth_report(directory)
            manifest = read_json(safe_child(directory, "ground_truth_manifest.json"))
            allowed = set(manifest.get("artifact_sha256", {}))
            if filename not in allowed:
                raise APIProblem(
                    404,
                    "artifact_not_found",
                    "Requested ground-truth artifact is not in its verified manifest allowlist.",
                )
        else:
            _check_report_metadata_size(directory, "robustness_manifest.json")
            _check_report_metadata_size(directory, "robustness.json")
            verify_robustness_artifacts(directory)
            manifest = read_json(safe_child(directory, "robustness_manifest.json"))
            allowed = set(manifest.get("artifact_sha256", {}))
            if filename not in allowed:
                raise APIProblem(
                    404,
                    "artifact_not_found",
                    "Requested robustness artifact is not in its verified manifest allowlist.",
                )
        path = safe_child(directory, filename)
    except APIProblem:
        raise
    except (
        DashboardDataError,
        ReportError,
        GroundTruthReportError,
        RobustnessError,
        OSError,
        ValueError,
    ) as exc:
        raise APIProblem(
            409, "report_artifacts_invalid", f"Report artifacts did not verify: {exc}"
        ) from exc
    if path.stat().st_size > MAX_WEB_DETAIL_BYTES:
        raise APIProblem(
            413, "artifact_too_large", "In-app report downloads are limited to 16 MiB."
        )
    return path


def _check_upload_request_size(request: Request) -> None:
    raw_length = request.headers.get("content-length")
    if raw_length is None:
        return
    try:
        content_length = int(raw_length)
    except ValueError as exc:
        raise APIProblem(
            400, "invalid_content_length", "Upload Content-Length must be an integer."
        ) from exc
    if content_length < 0:
        raise APIProblem(400, "invalid_content_length", "Upload Content-Length cannot be negative.")
    if content_length > MAX_TELEMETRY_UPLOAD_BYTES + MAX_MULTIPART_OVERHEAD_BYTES:
        raise APIProblem(
            413, "upload_too_large", "Multipart request exceeds the telemetry upload limit."
        )


def create_app(service: WebService | None = None) -> FastAPI:
    service = service or WebService(resolve_web_settings())
    app = FastAPI(
        title="ACN Network Intelligence API",
        version=__version__,
        description="Local API for validated endpoint telemetry, forecasts, model lifecycle, and evaluation artifacts.",
    )
    app.state.service = service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(APIProblem)
    async def api_problem_handler(_request: Request, exc: APIProblem):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_request: Request, exc: RequestValidationError):
        messages = [
            f"{'.'.join(str(p) for p in error['loc'])}: {error['msg']}" for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_request", "message": "; ".join(messages)}},
        )

    @app.exception_handler(Exception)
    async def unexpected_handler(_request: Request, exc: Exception):
        print(f"ACN API internal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "The local service could not complete the request.",
                }
            },
        )

    @app.get(f"{API_PREFIX}/status", response_model=StatusResponse, summary="Service status")
    def status():
        return {
            "status": "ready",
            "application": "ACN Network Intelligence",
            "version": __version__,
            "mode": "local",
        }

    @app.get(
        f"{API_PREFIX}/overview",
        response_model=OverviewResponse,
        summary="Evidence-backed operational overview",
    )
    def overview():
        return service.operational_overview()

    @app.get(
        f"{API_PREFIX}/datasets",
        response_model=DatasetResponse,
        summary="Configured source and dataset inventory",
    )
    def datasets():
        return service.datasets()

    @app.get(
        f"{API_PREFIX}/models",
        response_model=ModelListResponse,
        summary="Current model registry eligibility",
    )
    def models():
        return service.registry.list_models()

    @app.put(
        f"{API_PREFIX}/models/active",
        response_model=ModelMutationResponse,
        summary="Select an eligible inference model",
    )
    def select_model(body: SelectModelRequest):
        try:
            return service.registry.set_active(body.model_id)
        except ModelRegistryError as exc:
            raise APIProblem(409, "model_ineligible", str(exc)) from exc

    @app.post(
        f"{API_PREFIX}/models/train",
        response_model=ModelMutationResponse,
        summary="Train a provenance-linked candidate",
    )
    def train_model(body: TrainRequest):
        try:
            return service.registry.train_candidate(body.evaluation_id, body.method)
        except ModelRegistryError as exc:
            raise APIProblem(409, "training_blocked", str(exc)) from exc
        except (
            DashboardDataError,
            ReportError,
            ProcessingError,
            ConfigError,
            OSError,
            ValueError,
        ) as exc:
            raise APIProblem(422, "training_failed", str(exc)) from exc

    @app.get(
        f"{API_PREFIX}/history",
        response_model=HistoryResponse,
        summary="List actual inference records",
    )
    def history(limit: int = 200):
        if not 1 <= limit <= 500:
            raise APIProblem(422, "invalid_limit", "History limit must be in [1, 500].")
        return {"predictions": service.history.list(limit=limit)}

    @app.post(
        f"{API_PREFIX}/telemetry/inspect",
        response_model=TelemetryBatchResponse,
        summary="Validate and inspect a telemetry batch",
    )
    async def inspect_telemetry(request: Request, file: Annotated[UploadFile, File()]):
        _check_upload_request_size(request)
        if file.filename and Path(file.filename).suffix.lower() != ".csv":
            raise APIProblem(415, "unsupported_file", "Only CSV telemetry batches are accepted.")
        payload = await file.read(MAX_TELEMETRY_UPLOAD_BYTES + 1)
        if len(payload) > MAX_TELEMETRY_UPLOAD_BYTES:
            raise APIProblem(
                413,
                "upload_too_large",
                f"Upload exceeds the {MAX_TELEMETRY_UPLOAD_BYTES // (1024 * 1024)} MiB limit.",
            )
        try:
            inspected = inspect_endpoint_csv(payload, file.filename)
            inspected["warnings"] = [
                record
                for record in service.history.list(
                    limit=500, input_sha256=inspected["input_sha256"]
                )
                if record.get("risk") == "warning"
            ]
            return inspected
        except TelemetryInputError as exc:
            raise APIProblem(422, "invalid_telemetry", str(exc)) from exc

    @app.post(
        f"{API_PREFIX}/predictions",
        response_model=PredictionResponse,
        summary="Run real inference using a currently eligible model",
    )
    async def predict(
        request: Request,
        model_id: Annotated[str, Form()],
        file: Annotated[UploadFile, File()],
        tcp_retransmission_unit: Annotated[str | None, Form()] = None,
    ):
        _check_upload_request_size(request)
        if file.filename and Path(file.filename).suffix.lower() != ".csv":
            raise APIProblem(415, "unsupported_file", "Only CSV endpoint telemetry is accepted.")
        payload = await file.read(MAX_TELEMETRY_UPLOAD_BYTES + 1)
        if len(payload) > MAX_TELEMETRY_UPLOAD_BYTES:
            raise APIProblem(
                413,
                "upload_too_large",
                f"Upload exceeds the {MAX_TELEMETRY_UPLOAD_BYTES // (1024 * 1024)} MiB limit.",
            )
        try:
            model_manifest, estimator = service.registry.resolve_eligible(model_id)
            predictions, input_data = predict_with_model(
                payload,
                file.filename,
                model_manifest,
                estimator,
                declared_tcp_retransmission_unit=tcp_retransmission_unit,
            )
            service.history.append_many(predictions)
            input_data["warnings"] = [
                record for record in predictions if record.get("risk") == "warning"
            ]
            return {"predictions": predictions, "input": input_data, "model": input_data["model"]}
        except ModelRegistryError as exc:
            raise APIProblem(409, "model_ineligible", str(exc)) from exc
        except TelemetryInputError as exc:
            raise APIProblem(422, "invalid_telemetry", str(exc)) from exc
        except PredictionHistoryError as exc:
            raise APIProblem(507, "history_capacity", str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise APIProblem(422, "prediction_failed", str(exc)) from exc

    @app.post(f"{API_PREFIX}/predictions/pilot", summary="Score RTT with the exploratory Kali pilot")
    async def pilot_prediction(request: Request, file: Annotated[UploadFile, File()], short_window: bool = False, candidate: Literal["expanded", "relative"] | None = None):
        from starlette.concurrency import run_in_threadpool

        _check_upload_request_size(request)
        if file.filename and Path(file.filename).suffix.lower() != ".csv":
            raise APIProblem(415, "unsupported_file", "Only CSV telemetry is accepted.")
        payload = await file.read(MAX_TELEMETRY_UPLOAD_BYTES + 1)
        if len(payload) > MAX_TELEMETRY_UPLOAD_BYTES:
            raise APIProblem(413, "upload_too_large", "Pilot upload exceeds the size limit.")
        try:
            from .pilot_prediction import predict_pilot

            return await run_in_threadpool(predict_pilot, service.workspace.project_root, payload, short_window=short_window, candidate=candidate)
        except (ImportError, OSError) as exc:
            raise APIProblem(503, "pilot_unavailable", "Pilot training files or ML dependencies are unavailable.") from exc
        except (ValueError, csv.Error) as exc:
            raise APIProblem(422, "pilot_input_error", str(exc)) from exc

    @app.get(
        f"{API_PREFIX}/research",
        response_model=ResearchResponse,
        summary="Model quality, alignment and evaluation state",
    )
    def research():
        return service.research()

    @app.get(
        f"{API_PREFIX}/research/ground-truth",
        response_model=GroundTruthIndexResponse,
        summary="List current verified prepared runs and ground-truth report state",
    )
    def ground_truth_index():
        return service.ground_truth_index()

    @app.get(
        f"{API_PREFIX}/research/ground-truth/series",
        response_model=GroundTruthSeriesResponse,
        summary="Read a bounded verified batch timeline and linked event intervals",
    )
    def ground_truth_series(run_id: str = Query(min_length=1, max_length=160)):
        return service.ground_truth_series(run_id)

    @app.post(
        f"{API_PREFIX}/research/ground-truth/run",
        response_model=GroundTruthBuildResponse,
        summary="Build a descriptive report from current verified prepared data",
    )
    def build_ground_truth():
        return service.build_ground_truth()

    @app.get(
        f"{API_PREFIX}/evaluations/{{evaluation_id}}",
        response_model=ArtifactDetailResponse,
        summary="Read verified grouped-evaluation metrics and provenance",
    )
    def evaluation_detail(evaluation_id: str):
        return service.evaluation_detail(evaluation_id)

    @app.get(
        f"{API_PREFIX}/reports/{{report_id}}",
        response_model=ArtifactDetailResponse,
        summary="Read safe report details and verified artifact inventory",
    )
    def report_detail(report_id: str):
        return service.report_detail(report_id)

    @app.get(
        f"{API_PREFIX}/reports",
        response_model=ReportListResponse,
        summary="List configured report artifacts",
    )
    def reports():
        return {"reports": service.reports()}

    @app.get(f"{API_PREFIX}/reports/{{report_id}}/files/{{filename}}")
    def report_file(report_id: str, filename: str):
        path = _safe_report_file(service, report_id, filename)
        if path.stat().st_size > 16 * 1024 * 1024:
            raise APIProblem(
                413, "artifact_too_large", "In-app report downloads are limited to 16 MiB."
            )
        return FileResponse(path, filename=path.name)

    @app.post(
        f"{API_PREFIX}/audit", response_model=AuditResponse, summary="Run a bounded read-only audit"
    )
    def audit(body: ConfirmRequest):
        if not body.confirmed:
            raise APIProblem(
                400,
                "confirmation_required",
                "Explicit confirmation is required to start a read-only audit.",
            )
        inventory = scan_csv_inventory(service.workspace.raw_root)
        total_bytes = 0
        for relative in inventory.get("csv_paths", []):
            total_bytes += safe_child(service.workspace.raw_root, relative).stat().st_size
            if total_bytes > MAX_DASHBOARD_AUDIT_BYTES:
                raise APIProblem(
                    413,
                    "audit_too_large",
                    "Configured CSV inputs exceed the 64 MiB in-app audit limit.",
                )
        try:
            result = run_audit_safely(service.workspace.raw_root, service.workspace.reports_root)
        except DashboardDataError as exc:
            raise APIProblem(422, "audit_failed", str(exc)) from exc
        return {
            "status": result["status"],
            "output_id": result["output_dir"].name,
            "message": "New audit report created outside raw inputs. It is not automatically approved or selected by the current workspace configuration.",
        }

    @app.post(
        f"{API_PREFIX}/evaluations/run",
        response_model=EvaluationRunResponse,
        summary="Run the frozen grouped evaluation",
    )
    def run_evaluation():
        return service.run_evaluation()

    static_root = Path(__file__).resolve().parent / "web_static"
    if (static_root / "index.html").is_file():
        app.mount("/", StaticFiles(directory=static_root, html=True), name="web")
    return app


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "Install the local web dependencies: python3 -m pip install -e '.[web]'"
        ) from exc
    host = os.environ.get("ACN_WEB_HOST", "127.0.0.1")
    if (
        host not in {"127.0.0.1", "localhost", "::1"}
        and os.environ.get("ACN_ALLOW_REMOTE_WEB") != "1"
    ):
        raise SystemExit(
            "ACN web API binds to loopback by default; remote exposure requires authentication and network controls"
        )
    port = int(os.environ.get("ACN_WEB_PORT", "8000"))
    uvicorn.run("acn_congestion.web_api:app", host=host, port=port, reload=False, log_level="info")


app = create_app()
