"""Bounded, canonical endpoint-telemetry CSV validation and model inference."""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import math
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ConfigError
from .dashboard_data import DashboardDataError
from .models import (
    ModelError,
    has_observed_endpoint_feature,
    predict_scores,
    validate_feature_columns,
)
from .processing import (
    ENDPOINT_FIELDS,
    FEATURE_SPEC_VERSION,
    TCP_RETRANSMISSION_UNITS,
    ProcessingError,
    apply_feature_temporal_policy_to_rows,
    build_feature_specification,
    build_trailing_window_features,
    feature_specification_sha256,
    feature_temporal_eligibility_counts,
    feature_temporal_policy_from_specification,
    validate_feature_specification_object,
)

MAX_TELEMETRY_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_TELEMETRY_ROWS = 2_500
MAX_TELEMETRY_CHART_POINTS = 1_200


class TelemetryInputError(ValueError):
    """Telemetry upload is malformed, too large, or incompatible with the model."""


def _safe_filename(name: str | None) -> str:
    raw = Path(name or "telemetry.csv").name
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    return (safe or "telemetry.csv")[:120]


def parse_endpoint_csv(payload: bytes, *, required_fields: Sequence[str] = ()) -> dict[str, Any]:
    """Accept a narrow normalized CSV contract; never infer collector headers or units."""
    if not isinstance(payload, bytes) or not payload:
        raise TelemetryInputError("uploaded CSV is empty")
    if len(payload) > MAX_TELEMETRY_UPLOAD_BYTES:
        raise TelemetryInputError(
            f"upload exceeds the {MAX_TELEMETRY_UPLOAD_BYTES // (1024 * 1024)} MiB limit"
        )
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise TelemetryInputError("CSV must be UTF-8 text") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    headers = reader.fieldnames
    if not headers or len(headers) != len(set(headers)):
        raise TelemetryInputError("CSV header is empty or contains duplicate column names")
    if "timestamp_seconds" not in headers:
        raise TelemetryInputError(
            "CSV must contain canonical timestamp_seconds (seconds since run origin)"
        )
    unknown = set(headers) - {"timestamp_seconds", *ENDPOINT_FIELDS}
    if unknown:
        raise TelemetryInputError(
            f"unsupported columns are not accepted for inference: {sorted(unknown)}"
        )
    fields = [name for name in ENDPOINT_FIELDS if name in headers]
    if not fields:
        raise TelemetryInputError(
            "CSV must contain at least one canonical endpoint measurement column"
        )
    missing = sorted(set(required_fields) - set(fields))
    if missing:
        raise TelemetryInputError(
            f"selected model requires endpoint field(s) absent from CSV: {missing}"
        )
    rows: list[dict[str, Any]] = []
    missing_counts = {field: 0 for field in fields}
    timestamps: list[float] = []
    try:
        for row_number, raw in enumerate(reader, start=2):
            if row_number > MAX_TELEMETRY_ROWS + 1:
                raise TelemetryInputError(
                    f"CSV exceeds the {MAX_TELEMETRY_ROWS:,}-row in-app inference limit"
                )
            if None in raw or any(value is None for value in raw.values()):
                raise TelemetryInputError(f"wrong field count at CSV row {row_number}")
            time_value = raw.get("timestamp_seconds", "").strip()
            try:
                timestamp = float(time_value)
            except (TypeError, ValueError) as exc:
                raise TelemetryInputError(
                    f"invalid timestamp_seconds at CSV row {row_number}"
                ) from exc
            if not math.isfinite(timestamp):
                raise TelemetryInputError(f"non-finite timestamp_seconds at CSV row {row_number}")
            if timestamps and timestamp <= timestamps[-1]:
                raise TelemetryInputError(
                    f"timestamps must be strictly increasing (row {row_number})"
                )
            item: dict[str, Any] = {"timestamp_seconds": timestamp}
            for field in fields:
                raw_value = raw.get(field, "").strip()
                if raw_value == "":
                    item[field] = None
                    missing_counts[field] += 1
                    continue
                try:
                    value = float(raw_value)
                except ValueError as exc:
                    raise TelemetryInputError(
                        f"non-numeric {field} at CSV row {row_number}"
                    ) from exc
                if not math.isfinite(value):
                    raise TelemetryInputError(f"non-finite {field} at CSV row {row_number}")
                if field in {"rtt_ms", "goodput_bps", "tcp_retransmission_delta"} and value < 0:
                    raise TelemetryInputError(f"{field} must be non-negative (row {row_number})")
                if field == "loss_fraction" and not 0 <= value <= 1:
                    raise TelemetryInputError(
                        f"loss_fraction must be in [0, 1] (row {row_number}); percent conversion is not inferred"
                    )
                if field == "tcp_retransmission_delta" and not value.is_integer():
                    raise TelemetryInputError(
                        "tcp_retransmission_delta must contain integer count increments "
                        f"per sample (row {row_number}), matching the training feature contract"
                    )
                item[field] = value
            timestamps.append(timestamp)
            rows.append(item)
    except (csv.Error, UnicodeError) as exc:
        raise TelemetryInputError(f"malformed CSV: {exc}") from exc
    if not rows:
        raise TelemetryInputError("CSV contains a header but no telemetry rows")
    if not any(item[field] is not None for item in rows for field in fields):
        raise TelemetryInputError("all endpoint measurements are missing")
    intervals = [right - left for left, right in itertools.pairwise(timestamps)]
    max_gap = max(intervals) if intervals else None
    median_interval = sorted(intervals)[len(intervals) // 2] if intervals else None
    max_gap_ms = None
    try:
        from .dashboard_data import resolve_workspace

        workspace = resolve_workspace()
        if workspace.config_path.is_file():
            from .config import load_config

            config = load_config(workspace.config_path)
            value = config.get("labeling", {}).get("max_ground_truth_gap_ms")
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value > 0
            ):
                max_gap_ms = float(value)
    except (ConfigError, DashboardDataError, ImportError, OSError, TypeError, ValueError):
        max_gap_ms = None
    gap_count = sum(dt * 1000 > max_gap_ms for dt in intervals) if max_gap_ms is not None else None
    input_sha = hashlib.sha256(payload).hexdigest()
    return {
        "rows": rows,
        "fields": fields,
        "timestamps": timestamps,
        "input_sha256": input_sha,
        "input_name": "telemetry.csv",
        "row_count": len(rows),
        "missing_counts": missing_counts,
        "missing_fraction": {field: missing_counts[field] / len(rows) for field in fields},
        "max_gap_seconds": max_gap,
        "median_interval_seconds": median_interval,
        "timestamp_span_seconds": timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0,
        "gap_count": gap_count,
        "gap_threshold_ms": max_gap_ms,
        "received_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "structurally_valid",
    }


def _inspection_from_parsed(parsed: dict[str, Any], filename: str | None) -> dict[str, Any]:
    rows = parsed["rows"]
    stride = max(1, math.ceil(len(rows) / MAX_TELEMETRY_CHART_POINTS))
    selected = rows[::stride]
    if selected[-1] is not rows[-1]:
        selected.append(rows[-1])
    points = [
        {
            "timestamp_seconds": row["timestamp_seconds"],
            **{field: row[field] for field in parsed["fields"]},
        }
        for row in selected
    ]
    notes = [
        "Uploaded file is a batch snapshot; this endpoint has no live collector connection.",
        "Canonical units are required: RTT ms, goodput bit/s, loss fraction, and—when a model uses tcp_retransmission_delta—the exact declared packets-or-bytes increment unit; inspection alone does not declare that unit.",
        "Timestamps are relative to the run origin; wall-clock source freshness cannot be established.",
    ]
    if parsed["gap_count"] is None:
        notes.append(
            "No frozen protocol gap threshold is available; no interval was classified as a protocol gap."
        )
    quality_status = (
        "complete"
        if not any(parsed["missing_counts"].values()) and parsed["gap_count"] in (None, 0)
        else "review_required"
    )
    return {
        "status": parsed["status"],
        "source_kind": "uploaded_batch",
        "received_at_utc": parsed["received_at_utc"],
        "input_name": _safe_filename(filename),
        "input_sha256": parsed["input_sha256"],
        "row_count": parsed["row_count"],
        "fields": ["timestamp_seconds", *parsed["fields"]],
        "points": points,
        "quality": {
            "status": quality_status,
            "missing_fraction": parsed["missing_fraction"],
            "gap_count": parsed["gap_count"],
            "max_gap_seconds": parsed["max_gap_seconds"],
            "timestamp_span_seconds": parsed["timestamp_span_seconds"],
            "wall_clock_freshness": "Unknown · run-relative timestamps are not wall-clock timestamps",
            "notes": notes,
        },
    }


def inspect_endpoint_csv(payload: bytes, filename: str | None) -> dict[str, Any]:
    return _inspection_from_parsed(parse_endpoint_csv(payload), filename)


def predict_with_model(
    payload: bytes,
    filename: str | None,
    manifest: dict[str, Any],
    estimator: Any,
    *,
    predicted_at_utc: str | None = None,
    declared_tcp_retransmission_unit: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute the same trailing features and call the verified fitted sklearn Pipeline."""
    feature_columns = manifest.get("feature_columns")
    try:
        validate_feature_columns(feature_columns)
    except ModelError as exc:
        raise TelemetryInputError(f"registered model feature schema is invalid: {exc}") from exc
    raw_fields = [
        field
        for field in ENDPOINT_FIELDS
        if any(name.startswith(field + "_") for name in feature_columns)
    ]
    windows = manifest.get("feature_windows_seconds")
    if not isinstance(windows, list) or not windows:
        raise TelemetryInputError("registered model lacks its feature-window contract")
    try:
        windows = tuple(float(window) for window in windows)
    except (TypeError, ValueError) as exc:
        raise TelemetryInputError(
            "registered model has an invalid feature-window contract"
        ) from exc
    if any(not math.isfinite(window) or window <= 0 for window in windows):
        raise TelemetryInputError("registered model has an invalid feature-window contract")
    feature_specification_raw = manifest.get("feature_specification")
    feature_specification_digest = manifest.get("feature_specification_sha256")
    try:
        feature_specification = validate_feature_specification_object(feature_specification_raw)
        feature_temporal_policy = feature_temporal_policy_from_specification(feature_specification)
        expected_specification = build_feature_specification(
            tuple(feature_specification["endpoint_fields"]),
            tuple(float(w) for w in feature_specification["feature_windows_seconds"]),
            tcp_retransmission_unit=feature_specification.get("tcp_retransmission_unit"),
            feature_temporal_policy=feature_temporal_policy,
        )
    except (KeyError, ProcessingError, TypeError, ValueError) as exc:
        raise TelemetryInputError(
            f"registered model feature specification is invalid: {exc}"
        ) from exc
    expected_tcp_unit = feature_specification.get("tcp_retransmission_unit")
    if feature_specification["spec_version"] != FEATURE_SPEC_VERSION:
        raise TelemetryInputError(
            "registered model uses a legacy feature specification that cannot prove "
            "the exact TCP retransmission unit or a resolved feature temporal policy"
        )
    if feature_temporal_policy["status"] != "resolved":
        raise TelemetryInputError(
            "registered model feature temporal policy is unresolved; inference is blocked"
        )
    if (
        feature_specification != expected_specification
        or feature_specification_digest != feature_specification_sha256(expected_specification)
        or manifest.get("feature_specification_status") != "current_contract_complete"
        or manifest.get("tcp_retransmission_unit") != expected_tcp_unit
        or manifest.get("feature_temporal_policy") != feature_temporal_policy
        or tuple(expected_specification["feature_windows_seconds"]) != windows
        or manifest.get("endpoint_feature_fields") != expected_specification["endpoint_fields"]
        or not set(feature_columns).issubset(expected_specification["feature_columns"])
    ):
        raise TelemetryInputError(
            "registered model feature specification, TCP retransmission unit, or "
            "feature temporal policy does not match its fitted feature contract"
        )
    score_semantics = manifest.get("score_semantics")
    calibration_status = manifest.get("calibration_status")
    if (
        calibration_status != "not_fitted"
        or not isinstance(score_semantics, str)
        or not score_semantics.startswith("uncalibrated")
    ):
        raise TelemetryInputError(
            "registered model lacks an explicit uncalibrated score contract "
            "(score_semantics and calibration_status='not_fitted')"
        )
    declared_unit_status = feature_specification.get("tcp_retransmission_unit_status")
    tcp_feature_used = any(
        column.startswith("tcp_retransmission_delta_") for column in feature_columns
    )
    if tcp_feature_used and declared_unit_status == "declared":
        if expected_tcp_unit not in TCP_RETRANSMISSION_UNITS:
            raise TelemetryInputError(
                "registered model lacks an exact declared TCP retransmission unit"
            )
        if declared_tcp_retransmission_unit is None:
            raise TelemetryInputError(
                "declare the input tcp_retransmission_delta unit as 'packets' or "
                f"'bytes'; the selected model requires exactly {expected_tcp_unit!r}"
            )
        if declared_tcp_retransmission_unit not in TCP_RETRANSMISSION_UNITS:
            raise TelemetryInputError(
                "declared tcp_retransmission_delta unit must be exactly 'packets' or "
                f"'bytes'; the selected model requires {expected_tcp_unit!r}"
            )
        if declared_tcp_retransmission_unit != expected_tcp_unit:
            raise TelemetryInputError(
                "declared tcp_retransmission_delta unit "
                f"{declared_tcp_retransmission_unit!r} differs from the selected "
                f"model's unit {expected_tcp_unit!r}; packets and bytes are not "
                "interchangeable"
            )
    elif not tcp_feature_used and declared_unit_status in {"declared", "not_applicable"}:
        if declared_tcp_retransmission_unit not in (
            None,
            "not_applicable",
            *TCP_RETRANSMISSION_UNITS,
        ):
            raise TelemetryInputError(
                "declared tcp_retransmission_delta unit must be exactly 'packets', "
                "'bytes', or 'not_applicable'"
            )
        if declared_tcp_retransmission_unit is None:
            declared_tcp_retransmission_unit = "not_applicable"
    else:
        raise TelemetryInputError(
            "registered model uses tcp_retransmission_delta without an exact declared unit"
            if tcp_feature_used
            else "registered model has an unsupported TCP retransmission unit status"
        )
    parsed = parse_endpoint_csv(payload, required_fields=raw_fields)
    if not set(raw_fields).issubset(manifest.get("endpoint_feature_fields", [])):
        raise TelemetryInputError("model raw endpoint fields do not match its processing manifest")
    rows = parsed["rows"]
    try:
        feature_rows = build_trailing_window_features(rows, windows, tuple(raw_fields))
    except (ProcessingError, ValueError) as exc:
        raise TelemetryInputError(f"could not construct model input features: {exc}") from exc
    absent = sorted(set(feature_columns) - set(feature_rows[0]))
    if absent:
        raise TelemetryInputError(
            f"uploaded canonical fields cannot construct model features: {absent}"
        )
    try:
        effective_feature_rows, temporal_classifications = apply_feature_temporal_policy_to_rows(
            feature_rows,
            list(feature_columns),
            feature_specification,
        )
    except ProcessingError as exc:
        raise TelemetryInputError(
            f"could not apply the model's feature temporal policy: {exc}"
        ) from exc
    temporal_counts = feature_temporal_eligibility_counts(temporal_classifications)
    scorable_indices = [
        index
        for index, feature_row in enumerate(effective_feature_rows)
        if temporal_classifications[index]["status"] == "eligible"
        and has_observed_endpoint_feature(feature_row, feature_columns)
    ]
    scores_by_index: dict[int, float] = {}
    try:
        if scorable_indices:
            selected_scores = predict_scores(
                estimator,
                [effective_feature_rows[index] for index in scorable_indices],
                feature_columns,
            )
            scores_by_index = dict(zip(scorable_indices, selected_scores, strict=True))
    except Exception as exc:
        raise TelemetryInputError(
            f"model preprocessing/inference rejected the input: {exc}"
        ) from exc
    threshold = manifest.get("score_threshold")
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(float(threshold))
    ):
        raise TelemetryInputError("registered model lacks its validated warning threshold")
    stamp = predicted_at_utc or datetime.now(timezone.utc).isoformat()
    model_view = {
        "model_id": manifest.get("model_id"),
        "method": manifest.get("method"),
        "forecast_horizon_seconds": manifest.get("forecast_horizon_seconds"),
        "score_threshold": float(threshold),
        "feature_columns": feature_columns,
        "evaluation_id": manifest.get("evaluation_id"),
        "protocol_sha256": manifest.get("protocol_sha256"),
        "analysis_sha256": manifest.get("analysis_sha256"),
        "score_semantics": score_semantics,
        "calibration_status": calibration_status,
        "feature_spec_version": feature_specification.get("spec_version"),
        "feature_specification_sha256": feature_specification_digest,
        "feature_specification_status": manifest.get("feature_specification_status"),
        "tcp_retransmission_unit": expected_tcp_unit,
        "tcp_retransmission_unit_status": declared_unit_status,
        "input_declared_tcp_retransmission_unit": declared_tcp_retransmission_unit,
        "feature_temporal_policy": feature_temporal_policy,
        "feature_temporal_eligibility": temporal_counts,
    }
    predictions: list[dict[str, Any]] = []
    request_token = f"{parsed['input_sha256']}:{manifest.get('model_id')}:{stamp}"
    for idx, row in enumerate(rows):
        temporal_classification = temporal_classifications[idx]
        score = scores_by_index.get(idx)
        value = float(score) if score is not None and math.isfinite(float(score)) else None
        risk = (
            "not_scored"
            if value is None
            else "warning"
            if value >= threshold
            else "below_threshold"
        )
        predictions.append(
            {
                "prediction_id": f"pred_{uuid_token(request_token, idx)}",
                "predicted_at_utc": stamp,
                "model_id": manifest["model_id"],
                "model_method": manifest["method"],
                "model_artifact_sha256": manifest.get("artifact_sha256"),
                "feature_spec_version": feature_specification.get("spec_version"),
                "feature_specification_sha256": feature_specification_digest,
                "feature_specification_status": manifest.get("feature_specification_status"),
                "tcp_retransmission_unit": expected_tcp_unit,
                "input_declared_tcp_retransmission_unit": (declared_tcp_retransmission_unit),
                "feature_temporal_policy_status": feature_temporal_policy["status"],
                "feature_temporal_eligibility_status": temporal_classification["status"],
                "feature_temporal_eligibility_reason": temporal_classification["reason"],
                "evaluation_id": manifest.get("evaluation_id"),
                "protocol_sha256": manifest.get("protocol_sha256"),
                "analysis_sha256": manifest.get("analysis_sha256"),
                "input_name": _safe_filename(filename),
                "input_sha256": parsed["input_sha256"],
                "row_count": parsed["row_count"],
                "forecast_horizon_seconds": float(manifest["forecast_horizon_seconds"]),
                "score": value,
                "score_threshold": float(threshold),
                "risk": risk,
                "outcome_state": "unresolved",
                "timestamp_seconds": row["timestamp_seconds"],
            }
        )
    inspected = _inspection_from_parsed(parsed, filename)
    inspected["feature_temporal_eligibility"] = temporal_counts
    unscored_count = len(predictions) - len(scores_by_index)
    if unscored_count:
        inspected["quality"]["status"] = "review_required"
        inspected["quality"]["notes"].append(
            "The selected model's resolved feature temporal policy and observed-"
            f"measurement gate left {unscored_count} of {len(predictions)} anchors "
            f"unscored; eligibility reasons: {temporal_counts['reason_counts']}. "
            "Unscored rows are not negative predictions."
        )
    return predictions, {**inspected, "model": model_view}


def uuid_token(digest: str, index: int) -> str:
    """Stable per-input row identifier; distinct uploaded inputs have distinct hashes."""
    return hashlib.sha256(f"{digest}:{index}".encode("ascii")).hexdigest()[:16]
