"""Configuration loading and fail-fast validation (JSON syntax, including .yaml files)."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """An invalid, incomplete, or unsupported project configuration."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON explicitly; JSON is a YAML 1.2 subset but no YAML parser is used."""
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file does not exist: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"configuration must use JSON syntax (line {exc.lineno}, column {exc.colno}): {exc.msg}"
        ) from exc
    except OSError as exc:
        raise ConfigError(f"cannot read configuration {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError("configuration root must be a JSON object")
    validate_config(value)
    return value


def validate_config(config: dict[str, Any], *, require_analysis_ready: bool = False) -> None:
    if config.get("schema_version") != 1:
        raise ConfigError("schema_version must be the integer 1")
    if config.get("status") not in {"protocol_draft_not_frozen", "protocol_frozen"}:
        raise ConfigError("top-level status must be protocol_draft_not_frozen or protocol_frozen")
    if config.get("status") == "protocol_frozen":
        _validate_freeze_record(config.get("protocol_freeze"))
    elif config.get("protocol_freeze") not in (None, {}):
        raise ConfigError("draft config cannot contain a protocol_freeze approval record")
    for section in ("topology", "collection", "labeling", "evaluation"):
        if not isinstance(config.get(section), dict):
            raise ConfigError(f"missing or invalid configuration section: {section}")
    topology = config["topology"]
    topology_strings = (
        "client_namespace",
        "router_namespace",
        "server_namespace",
        "client_interface",
        "router_client_interface",
        "router_server_interface",
        "server_interface",
        "client_address",
        "router_client_address",
        "router_server_address",
        "server_address",
        "bottleneck_egress",
    )
    missing_topology = [
        key
        for key in topology_strings
        if not isinstance(topology.get(key), str) or not topology[key]
    ]
    if missing_topology:
        raise ConfigError(
            f"topology missing non-empty required fields: {', '.join(missing_topology)}"
        )
    for key in ("bottleneck_rate_bps", "bottleneck_burst_bytes", "bottleneck_latency_ms"):
        value = topology.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ConfigError(f"topology.{key} must be finite and positive")
    try:
        addresses = {
            key: ipaddress.ip_interface(topology[key])
            for key in (
                "client_address",
                "router_client_address",
                "router_server_address",
                "server_address",
            )
        }
    except ValueError as exc:
        raise ConfigError(f"topology contains invalid IP interface address: {exc}") from exc
    if any(interface.version != 4 for interface in addresses.values()):
        raise ConfigError("topology addresses must currently be IPv4 interfaces")
    if addresses["client_address"].network != addresses["router_client_address"].network:
        raise ConfigError("client and router client-side addresses must share a subnet")
    if addresses["server_address"].network != addresses["router_server_address"].network:
        raise ConfigError("server and router server-side addresses must share a subnet")
    if addresses["client_address"].network == addresses["server_address"].network:
        raise ConfigError("client and server links must use distinct subnets")

    evaluation = config["evaluation"]
    seed = evaluation.get("random_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError("evaluation.random_seed must be a non-negative integer")
    if evaluation.get("random_row_split_allowed_as_primary") is not False:
        raise ConfigError("random row splits must remain disabled as the primary evaluation")
    if evaluation.get("class_weight") not in (None, "balanced"):
        raise ConfigError("evaluation.class_weight must be null or 'balanced'")
    allowed = evaluation.get("endpoint_feature_allowlist")
    forbidden = evaluation.get("forbidden_predictors")
    if not isinstance(allowed, list) or not all(isinstance(v, str) for v in allowed):
        raise ConfigError("evaluation.endpoint_feature_allowlist must be a list of strings")
    supported_endpoint_fields = {
        "rtt_ms",
        "loss_fraction",
        "goodput_bps",
        "tcp_retransmission_delta",
    }
    if len(set(allowed)) != len(allowed):
        raise ConfigError("evaluation.endpoint_feature_allowlist must be unique")
    if set(allowed) - supported_endpoint_fields:
        raise ConfigError(
            f"evaluation.endpoint_feature_allowlist contains unsupported fields: {sorted(set(allowed) - supported_endpoint_fields)}"
        )
    if not isinstance(forbidden, list) or not all(isinstance(v, str) for v in forbidden):
        raise ConfigError("evaluation.forbidden_predictors must be a list of strings")
    overlap = sorted(set(allowed) & set(forbidden))
    if overlap:
        raise ConfigError(f"predictors cannot be both allowed and forbidden: {', '.join(overlap)}")
    leakage_fragments = (
        "queue",
        "qdisc",
        "backlog",
        "label",
        "phase",
        "run_id",
        "configuration_id",
        "event_active",
    )
    contaminated = [
        name for name in allowed if any(token in name.lower() for token in leakage_fragments)
    ]
    if contaminated:
        raise ConfigError(
            f"endpoint feature allowlist contains leakage-prone fields: {contaminated}"
        )
    processing_section = config.get("processing")
    if processing_section is not None:
        if not isinstance(processing_section, dict):
            raise ConfigError("processing must be an object when present")
        if "feature_temporal_policy" in processing_section:
            _validate_feature_temporal_policy(processing_section["feature_temporal_policy"])

    if require_analysis_ready:
        if config.get("status") != "protocol_frozen":
            raise ConfigError(
                "analysis is blocked until the protocol has been separately reviewed and frozen; "
                "set top-level status to 'protocol_frozen' only after recording the freeze decision"
            )
        labeling = config["labeling"]
        required = (
            "queue_delay_threshold_ms",
            "consecutive_samples_required",
            "prediction_horizon_ms",
            "observation_window_ms",
            "max_ground_truth_gap_ms",
        )
        missing = [key for key in required if labeling.get(key) is None]
        processing = config.get("processing")
        if not isinstance(processing, dict):
            missing.append("processing section")
        else:
            for key in ("manifest", "feature_windows_seconds", "raw_root", "output_root"):
                if processing.get(key) is None or processing.get(key) == "":
                    missing.append(f"processing.{key}")
            temporal_policy = _validate_feature_temporal_policy(
                processing.get("feature_temporal_policy")
            )
            if temporal_policy["status"] != "resolved":
                missing.append(
                    "processing.feature_temporal_policy (resolved max_gap_seconds, "
                    "minimum_valid_fraction, and minimum_temporal_coverage_fraction)"
                )
        for key in (
            "max_false_alarms_per_minute",
            "alarm_merge_gap_ms",
            "baseline_feature",
            "primary_grouping",
            "output_root",
        ):
            if evaluation.get(key) is None:
                missing.append(f"evaluation.{key}")
        if missing:
            raise ConfigError(
                "analysis is blocked until scientifically/configurationally required values are set: "
                + ", ".join(missing)
            )
        if evaluation["primary_grouping"] not in {"configuration_id", "run_id"}:
            raise ConfigError("evaluation.primary_grouping must be configuration_id or run_id")
        if (
            not isinstance(evaluation["baseline_feature"], str)
            or evaluation["baseline_feature"] not in supported_endpoint_fields
        ):
            raise ConfigError(
                "evaluation.baseline_feature must be one of the canonical endpoint measurements"
            )
        if not isinstance(evaluation.get("baseline_feature_direction"), str) or evaluation.get(
            "baseline_feature_direction"
        ) not in {"above", "below"}:
            raise ConfigError("evaluation.baseline_feature_direction must be 'above' or 'below'")
        if evaluation["baseline_feature"] not in allowed:
            raise ConfigError(
                "evaluation.baseline_feature must also be in endpoint_feature_allowlist"
            )
        if not all(
            isinstance(processing.get(key), str) and processing.get(key)
            for key in ("manifest", "raw_root", "output_root")
        ):
            raise ConfigError(
                "processing manifest/raw_root/output_root must be non-empty path strings"
            )
        _positive_number(labeling, "prediction_horizon_ms")
        _positive_number(labeling, "queue_delay_threshold_ms")
        _positive_number(labeling, "observation_window_ms")
        _positive_number(labeling, "max_ground_truth_gap_ms")
        persistence = labeling.get("consecutive_samples_required")
        if isinstance(persistence, bool) or not isinstance(persistence, int) or persistence < 1:
            raise ConfigError("labeling.consecutive_samples_required must be an integer >= 1")
        windows = processing["feature_windows_seconds"]
        if (
            not isinstance(windows, list)
            or not windows
            or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or v <= 0
                for v in windows
            )
        ):
            raise ConfigError("processing.feature_windows_seconds must contain positive numbers")
        if sorted(set(windows)) != windows:
            raise ConfigError("processing.feature_windows_seconds must be sorted and unique")
        if max(windows) > labeling["observation_window_ms"] / 1000:
            raise ConfigError(
                "processing feature windows cannot exceed labeling.observation_window_ms"
            )
        _nonnegative_number(evaluation, "max_false_alarms_per_minute")
        _nonnegative_number(evaluation, "alarm_merge_gap_ms")
        alpha = evaluation.get("ewma_alpha")
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha)
            or not 0 < alpha <= 1
        ):
            raise ConfigError("evaluation.ewma_alpha must be finite and in (0,1]")
        train_fraction = evaluation.get("train_fraction")
        validation_fraction = evaluation.get("validation_fraction")
        if (
            any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or v <= 0
                for v in (train_fraction, validation_fraction)
            )
            or train_fraction + validation_fraction >= 1
        ):
            raise ConfigError(
                "evaluation.train_fraction and validation_fraction must be positive and sum to less than one"
            )
        models = evaluation.get("models_to_evaluate")
        allowed_models = {"logistic_regression", "random_forest", "hist_gradient_boosting"}
        if (
            not isinstance(models, list)
            or not models
            or any(not isinstance(model, str) for model in models)
            or set(models) - allowed_models
        ):
            raise ConfigError(
                f"evaluation.models_to_evaluate must be a non-empty list drawn from {sorted(allowed_models)}"
            )
        if len(set(models)) != len(models):
            raise ConfigError("evaluation.models_to_evaluate must not contain duplicates")
        model_selection = evaluation.get("model_selection")
        if model_selection is not None:
            from .model_selection import ModelSelectionError, normalize_model_selection_config

            try:
                for model_name in models:
                    normalize_model_selection_config(
                        model_selection,
                        model_name=model_name,
                        default_seed=int(seed),
                    )
            except ModelSelectionError as exc:
                raise ConfigError(f"invalid evaluation.model_selection: {exc}") from exc
        resamples = evaluation.get("bootstrap_resamples")
        confidence = evaluation.get("bootstrap_confidence")
        if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 100:
            raise ConfigError("evaluation.bootstrap_resamples must be an integer >= 100")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 < confidence < 1
        ):
            raise ConfigError("evaluation.bootstrap_confidence must be in (0,1)")


def _validate_feature_temporal_policy(value: Any) -> dict[str, Any]:
    from .processing import ProcessingError, normalize_feature_temporal_policy

    try:
        return normalize_feature_temporal_policy(value)
    except ProcessingError as exc:
        raise ConfigError(f"invalid processing.feature_temporal_policy: {exc}") from exc


def _validate_freeze_record(value: Any) -> None:
    if not isinstance(value, dict):
        raise ConfigError("protocol_frozen requires a protocol_freeze decision record")
    required = {
        "protocol_id",
        "version",
        "frozen_at_utc",
        "approvers",
        "decision_record_path",
        "decision_record_sha256",
    }
    missing = sorted(key for key in required if value.get(key) is None or value.get(key) == "")
    if missing:
        raise ConfigError(f"protocol_freeze decision record is missing: {', '.join(missing)}")
    for key in ("protocol_id", "version", "decision_record_path"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ConfigError(f"protocol_freeze.{key} must be a non-empty string")
    try:
        timestamp = datetime.fromisoformat(str(value["frozen_at_utc"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ConfigError(
            "protocol_freeze.frozen_at_utc must be an ISO-8601 timestamp with timezone"
        ) from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ConfigError("protocol_freeze.frozen_at_utc must include a timezone")
    approvers = value.get("approvers")
    if (
        not isinstance(approvers, list)
        or not approvers
        or any(not isinstance(name, str) or not name.strip() for name in approvers)
        or len(set(approvers)) != len(approvers)
    ):
        raise ConfigError("protocol_freeze.approvers must be a non-empty list of unique names")
    if not isinstance(value.get("decision_record_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", value["decision_record_sha256"]
    ):
        raise ConfigError("protocol_freeze.decision_record_sha256 must be a lowercase SHA-256")


def verify_protocol_freeze_evidence(config: dict[str, Any], config_path: str | Path) -> None:
    """Check the frozen decision record file against its declared SHA-256."""
    if config.get("status") != "protocol_frozen":
        raise ConfigError("protocol freeze evidence is unavailable because config is not frozen")
    _validate_freeze_record(config.get("protocol_freeze"))
    record_path = Path(config["protocol_freeze"]["decision_record_path"])
    if not record_path.is_absolute():
        record_path = Path(config_path).resolve().parent / record_path
    record_path = record_path.resolve()
    try:
        digest = hashlib.sha256(record_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ConfigError(
            f"protocol freeze decision record is missing/unreadable: {record_path}"
        ) from exc
    if digest != config["protocol_freeze"]["decision_record_sha256"]:
        raise ConfigError(f"protocol freeze decision record SHA-256 mismatch: {record_path}")


def _positive_number(section: dict[str, Any], key: str) -> float:
    value = section.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ConfigError(f"{key} must be a finite positive number")
    return float(value)


def _nonnegative_number(section: dict[str, Any], key: str) -> float:
    value = section.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ConfigError(f"{key} must be a finite non-negative number")
    return float(value)
