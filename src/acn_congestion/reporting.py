"""Typed loaders and machine/human report writers for derived evaluation outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from . import __version__
from .config import (
    ConfigError,
    _validate_freeze_record,
    load_config,
    validate_config,
    verify_protocol_freeze_evidence,
)
from .models import validate_feature_columns
from .processing import (
    FEATURE_SPEC_VERSION,
    ProcessingError,
    build_feature_specification,
    declared_tcp_retransmission_units_from_run_records,
    feature_specification_operational_status,
    feature_specification_sha256,
    feature_temporal_policy_from_specification,
    resolve_feature_specification,
)


class ReportError(ValueError):
    """A derived dataset or report output violates its declared contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash the exact pretty/sorted JSON serialization used for processing manifests."""
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _numeric(value: str, column: str, row_number: int) -> float | None:
    if value == "":
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise ReportError(
            f"invalid numeric value in {column!r} at row {row_number}: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise ReportError(f"non-finite numeric value in {column!r} at row {row_number}")
    return number


def _event_state(value: str, row_number: int) -> bool | None:
    if value == "":
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    raise ReportError(f"invalid event_active value at row {row_number}: {value!r}")


def load_analysis_dataset(processed_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and verify the processing output without changing any files."""
    manifest_path = processed_dir / "processing_manifest.json"
    data_path = processed_dir / "analysis.csv"
    exclusion_path = processed_dir / "excluded_rows.csv"
    summary_path = processed_dir / "summary.json"
    dictionary_path = processed_dir / "DATA_DICTIONARY.md"
    if (processed_dir / "PREPARATION_INCOMPLETE").exists():
        raise ReportError("processed directory is marked as an incomplete preparation")
    if not all(
        path.is_file()
        for path in (manifest_path, data_path, exclusion_path, summary_path, dictionary_path)
    ):
        raise ReportError(
            "processed directory must contain analysis.csv, excluded_rows.csv, summary.json, DATA_DICTIONARY.md, and processing_manifest.json"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"invalid processing manifest: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("processing_schema_version") != 2
        or manifest.get("status") != "complete"
    ):
        raise ReportError("unsupported or incomplete processing manifest")
    if manifest.get("analysis_sha256") != sha256_file(data_path):
        raise ReportError("analysis.csv checksum does not match processing_manifest.json")
    if manifest.get("exclusions_sha256") != sha256_file(exclusion_path):
        raise ReportError("excluded_rows.csv checksum does not match processing_manifest.json")
    if manifest.get("summary_sha256") != sha256_file(summary_path):
        raise ReportError("summary.json checksum does not match processing_manifest.json")
    if manifest.get("data_dictionary_sha256") != sha256_file(dictionary_path):
        raise ReportError("DATA_DICTIONARY.md checksum does not match processing_manifest.json")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"invalid processing summary: {exc}") from exc
    if not isinstance(summary, dict):
        raise ReportError("processing summary root must be an object")
    features = manifest.get("feature_columns")
    if not isinstance(features, list):
        raise ReportError("processing manifest has no feature_columns list")
    validate_feature_columns(features)
    if (
        manifest.get("feature_specification") is not None
        or manifest.get("feature_specification_sha256") is not None
        or (
            manifest.get("endpoint_feature_fields") is not None
            and manifest.get("feature_windows_seconds") is not None
        )
    ):
        try:
            specification = resolve_feature_specification(manifest)
        except ProcessingError as exc:
            raise ReportError(str(exc)) from exc
        specification_digest = feature_specification_sha256(specification)
        if summary.get("feature_specification_sha256") not in (None, specification_digest):
            raise ReportError(
                "summary feature specification digest disagrees with processing manifest"
            )
        if summary.get("feature_spec_version") not in (None, specification["spec_version"]):
            raise ReportError(
                "summary feature specification version disagrees with processing manifest"
            )
        expected_tcp_unit = specification.get("tcp_retransmission_unit")
        expected_temporal_policy = feature_temporal_policy_from_specification(specification)
        for owner_name, owner in (("processing manifest", manifest), ("summary", summary)):
            if (
                "tcp_retransmission_unit" in owner
                and owner["tcp_retransmission_unit"] != expected_tcp_unit
            ):
                raise ReportError(
                    f"{owner_name} TCP retransmission unit disagrees with its feature specification"
                )
            if (
                "feature_temporal_policy" in owner
                and owner["feature_temporal_policy"] != expected_temporal_policy
            ):
                raise ReportError(
                    f"{owner_name} feature temporal policy disagrees with its feature specification"
                )
        if specification["spec_version"] == FEATURE_SPEC_VERSION:
            if summary.get("tcp_retransmission_unit") != expected_tcp_unit:
                raise ReportError("summary is missing its v2 TCP retransmission unit")
            if summary.get("feature_temporal_policy") != expected_temporal_policy:
                raise ReportError("summary is missing its v2 feature temporal policy")
            if manifest.get("feature_temporal_eligibility_counts") != summary.get(
                "feature_temporal_eligibility_counts"
            ):
                raise ReportError(
                    "processing manifest and summary disagree on feature temporal eligibility counts"
                )
    try:
        with data_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ReportError("analysis.csv has no header")
            missing = sorted(
                (
                    set(features)
                    | {
                        "run_id",
                        "configuration_id",
                        "source_row",
                        "timestamp_seconds",
                        "event_active",
                        "future_event_onset",
                        "label_reason",
                    }
                )
                - set(reader.fieldnames)
            )
            if missing:
                raise ReportError(f"analysis.csv is missing required column(s): {missing}")
            rows: list[dict[str, Any]] = []
            numeric_columns = set(features) | {
                "timestamp_seconds",
                "source_row",
                "rtt_ms",
                "loss_fraction",
                "goodput_bps",
                "tcp_retransmission_delta",
                "queue_backlog_bytes",
                "qdisc_queue_delay_proxy_ms",
                "qdisc_drops_delta",
                "qdisc_overlimits_delta",
                "event_onset_source_row",
                "label_window_ms",
            }
            numeric_columns |= {
                field
                for field in reader.fieldnames
                if field.startswith("window_")
                and field.endswith(("_sample_count", "_observed_span_seconds", "_max_gap_seconds"))
            }
            for row_number, raw in enumerate(reader, start=2):
                if None in raw or any(
                    value is None for key, value in raw.items() if key is not None
                ):
                    raise ReportError(f"wrong field count in analysis.csv at row {row_number}")
                item: dict[str, Any] = dict(raw)
                for column in numeric_columns:
                    if column in item:
                        item[column] = _numeric(item[column], column, row_number)
                integer_columns = {
                    "source_row",
                    "event_onset_source_row",
                    "qdisc_drops_delta",
                    "qdisc_overlimits_delta",
                }
                integer_columns |= {
                    field for field in numeric_columns if field.endswith("_sample_count")
                }
                for integer_column in integer_columns:
                    value = item.get(integer_column)
                    if value is not None:
                        if not float(value).is_integer():
                            raise ReportError(f"non-integral {integer_column} at row {row_number}")
                        item[integer_column] = int(value)
                item["event_active"] = _event_state(raw.get("event_active", ""), row_number)
                label = raw.get("future_event_onset", "")
                if label == "":
                    item["future_event_onset"] = None
                elif label in {"0", "1"}:
                    item["future_event_onset"] = int(label)
                else:
                    raise ReportError(f"invalid future_event_onset at row {row_number}: {label!r}")
                if not item.get("run_id") or not item.get("configuration_id"):
                    raise ReportError(f"missing run/configuration identifier at row {row_number}")
                rows.append(item)
    except OSError as exc:
        raise ReportError(f"cannot read analysis.csv: {exc}") from exc
    if not rows:
        raise ReportError("analysis.csv contains no data rows")
    if manifest.get("analysis_row_count") != len(rows) or summary.get("analysis_row_count") != len(
        rows
    ):
        raise ReportError("analysis row count disagrees with processing manifest/summary")
    for field in (
        "data_quality_log_entry_count",
        "distinct_source_rows_with_log_entries",
        "rows_excluded_for_invalid_timestamp",
        "retained_rows_with_measurement_issue_entries",
        "censored_label_anchor_count",
    ):
        if manifest.get(field) != summary.get(field):
            raise ReportError(f"processing manifest and summary disagree on {field}")
    if manifest.get("data_quality_log_entry_count") != manifest.get("excluded_record_count"):
        raise ReportError(
            "legacy exclusion count does not match explicit data-quality log entry count"
        )
    if manifest.get("censored_label_anchor_count") != manifest.get("label_counts", {}).get(
        "censored"
    ):
        raise ReportError("censored label anchor count disagrees with label counts")
    if summary.get("excluded_or_censored_count") != manifest.get("data_quality_log_entry_count"):
        raise ReportError("summary data-quality log count disagrees with processing manifest")
    if summary.get("feature_columns") != features:
        raise ReportError("summary feature_columns disagree with processing manifest")
    observed_labels: dict[str, int] = {}
    for row in rows:
        key = (
            str(row["future_event_onset"]) if row["future_event_onset"] is not None else "censored"
        )
        observed_labels[key] = observed_labels.get(key, 0) + 1
    if (
        manifest.get("label_counts") != observed_labels
        or summary.get("label_counts") != observed_labels
    ):
        raise ReportError("label counts disagree with processing manifest/summary")
    temporal_counts = manifest.get("feature_temporal_eligibility_counts")
    if temporal_counts is not None:
        if not isinstance(temporal_counts, dict):
            raise ReportError("feature temporal eligibility counts must be an object")
        if not all(
            "feature_temporal_eligibility_status" in row
            and "feature_temporal_eligibility_reason" in row
            for row in rows
        ):
            raise ReportError(
                "analysis.csv lacks the feature temporal eligibility columns recorded in its manifest"
            )
        observed_status_counts = Counter(
            str(row["feature_temporal_eligibility_status"]) for row in rows
        )
        observed_reason_counts = Counter(
            str(row["feature_temporal_eligibility_reason"]) for row in rows
        )
        if (
            temporal_counts.get("anchor_count") != len(rows)
            or temporal_counts.get("eligible_anchor_count")
            != observed_status_counts.get("eligible", 0)
            or temporal_counts.get("not_eligible_anchor_count")
            != observed_status_counts.get("not_eligible", 0)
            or temporal_counts.get("descriptive_only_anchor_count")
            != observed_status_counts.get("descriptive_only", 0)
            or temporal_counts.get("status_counts") != dict(sorted(observed_status_counts.items()))
            or temporal_counts.get("reason_counts") != dict(sorted(observed_reason_counts.items()))
        ):
            raise ReportError("feature temporal eligibility counts disagree with analysis.csv")
    try:
        with exclusion_path.open("r", encoding="utf-8-sig", newline="") as stream:
            exclusion_reader = csv.DictReader(stream)
            if exclusion_reader.fieldnames != ["run_id", "source_row", "reason", "detail"]:
                raise ReportError("excluded_rows.csv has unexpected columns")
            excluded_records = list(exclusion_reader)
            if any(
                None in record or any(value is None for value in record.values())
                for record in excluded_records
            ):
                raise ReportError("excluded_rows.csv contains malformed field counts")
    except OSError as exc:
        raise ReportError(f"cannot read exclusions: {exc}") from exc
    if len(excluded_records) != manifest.get("excluded_record_count"):
        raise ReportError("excluded_rows.csv count disagrees with processing manifest")
    run_records = manifest.get("run_records")
    if not isinstance(run_records, list):
        raise ReportError("processing manifest has no run_records list")
    if any(
        not isinstance(record, dict)
        or not isinstance(record.get("run_id"), str)
        or not record.get("run_id")
        or not isinstance(record.get("configuration_id"), str)
        or not record.get("configuration_id")
        for record in run_records
    ):
        raise ReportError("processing manifest run_records are malformed or have empty identifiers")
    run_config = {record["run_id"]: record["configuration_id"] for record in run_records}
    if len(run_config) != len(run_records):
        raise ReportError("processing manifest run_records contain duplicate run IDs")
    run_records_by_id = {record["run_id"]: record for record in run_records}
    rows_by_source: dict[tuple[str, int], dict[str, Any]] = {}
    retained_by_run: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        if row["run_id"] not in run_config or row["configuration_id"] != run_config[row["run_id"]]:
            raise ReportError(
                f"analysis row run/configuration does not match processing manifest: {row['run_id']!r}"
            )
        source_row = row.get("source_row")
        if not isinstance(source_row, int) or source_row < 2:
            raise ReportError(f"analysis row has invalid source_row for run {row['run_id']!r}")
        key = (row["run_id"], source_row)
        if key in rows_by_source:
            raise ReportError(f"source row appears more than once in analysis.csv: {key}")
        rows_by_source[key] = row
        retained_by_run[row["run_id"]].add(source_row)

    reason_counts: Counter[str] = Counter()
    logged_source_rows: set[tuple[str, int]] = set()
    invalid_timestamp_rows: dict[str, set[int]] = defaultdict(set)
    censor_log_rows: set[tuple[str, int]] = set()
    measurement_issue_count = 0
    for record in excluded_records:
        run_id, reason = record.get("run_id"), record.get("reason")
        try:
            source_row_value = int(record.get("source_row", ""))
        except (TypeError, ValueError) as exc:
            raise ReportError("excluded_rows.csv contains an invalid source_row") from exc
        if (
            run_id not in run_records_by_id
            or str(source_row_value) != record.get("source_row")
            or source_row_value < 2
            or not isinstance(reason, str)
            or not reason
        ):
            raise ReportError("excluded_rows.csv contains an invalid run/source-row/reason record")
        run_record = run_records_by_id[run_id]
        source_count = run_record.get("source_row_count")
        if (
            isinstance(source_count, bool)
            or not isinstance(source_count, int)
            or source_row_value > source_count + 1
        ):
            raise ReportError(
                f"excluded source row is outside recorded source row count: {(run_id, source_row_value)}"
            )
        reason_counts[reason] += 1
        key = (run_id, source_row_value)
        logged_source_rows.add(key)
        if reason == "invalid_timestamp":
            if source_row_value in invalid_timestamp_rows[run_id]:
                raise ReportError(f"duplicate invalid-timestamp exclusion for {key}")
            if key in rows_by_source:
                raise ReportError(f"invalid-timestamp row was retained in analysis.csv: {key}")
            invalid_timestamp_rows[run_id].add(source_row_value)
        else:
            retained = rows_by_source.get(key)
            if retained is None:
                raise ReportError(
                    f"excluded/measurement row is absent from analysis.csv without invalid_timestamp reason: {key}"
                )
            if reason in {"invalid_endpoint_measurement", "invalid_qdisc_measurement"}:
                measurement_issue_count += 1
            if (
                retained.get("future_event_onset") is None
                and retained.get("label_reason") == reason
            ):
                if key in censor_log_rows:
                    raise ReportError(f"duplicate censor record for {key}")
                censor_log_rows.add(key)
    for run_id, record in run_records_by_id.items():
        source_count = record.get("source_row_count")
        retained = retained_by_run.get(run_id, set())
        invalid = invalid_timestamp_rows.get(run_id, set())
        if isinstance(source_count, bool) or not isinstance(source_count, int) or source_count < 1:
            raise ReportError(
                f"processing manifest has invalid source_row_count for run {run_id!r}"
            )
        if record.get("retained_timestamp_rows") != len(retained):
            raise ReportError(f"retained row count disagrees with analysis.csv for run {run_id!r}")
        accounted = retained | invalid
        expected_source_rows = set(range(2, source_count + 2))
        if accounted != expected_source_rows:
            missing_rows = sorted(expected_source_rows - accounted)
            unexpected_rows = sorted(accounted - expected_source_rows)
            raise ReportError(
                f"source rows are not fully accounted for in run {run_id!r}: "
                f"unaccounted={missing_rows[:10]}, unexpected={unexpected_rows[:10]}"
            )
    if dict(reason_counts) != manifest.get("exclusion_reason_counts"):
        raise ReportError("excluded_rows.csv reason counts disagree with processing manifest")
    if len(logged_source_rows) != manifest.get("distinct_source_rows_with_log_entries"):
        raise ReportError("distinct logged source-row count disagrees with processing manifest")
    if len(invalid_timestamp_rows) and sum(
        map(len, invalid_timestamp_rows.values())
    ) != manifest.get("rows_excluded_for_invalid_timestamp"):
        raise ReportError("invalid timestamp exclusion count disagrees with processing manifest")
    if measurement_issue_count != manifest.get("retained_rows_with_measurement_issue_entries"):
        raise ReportError("measurement issue entries disagree with processing manifest")
    if len(censor_log_rows) != manifest.get("censored_label_anchor_count"):
        raise ReportError("censored labels do not have exactly one matching exclusion-log entry")
    if any(
        row.get("future_event_onset") is None
        and (row["run_id"], row["source_row"]) not in censor_log_rows
        for row in rows
    ):
        raise ReportError("a censored analysis anchor is missing its exclusion-log record")
    # Verify time order within each contiguous run and that runs do not reappear later.
    finished: set[str] = set()
    previous: str | None = None
    last_time: dict[str, float] = {}
    run_config: dict[str, str] = {}
    for index, row in enumerate(rows):
        run_id = row["run_id"]
        if run_id != previous:
            if run_id in finished:
                raise ReportError(f"run {run_id!r} occurs in multiple non-contiguous blocks")
            if previous is not None:
                finished.add(previous)
            previous = run_id
        time = row["timestamp_seconds"]
        if time is None:
            raise ReportError(f"missing timestamp at analysis row {index + 2}")
        if run_id in last_time and time <= last_time[run_id]:
            raise ReportError(f"non-increasing timestamp in run {run_id!r}")
        last_time[run_id] = time
        config = row["configuration_id"]
        if run_id in run_config and run_config[run_id] != config:
            raise ReportError(f"run {run_id!r} maps to multiple configurations")
        run_config[run_id] = config
    return rows, manifest


def validate_processing_compatibility(
    processing_manifest: dict[str, Any],
    configuration: dict[str, Any],
    *,
    current_config_sha256: str,
    config_path: Path,
) -> dict[str, Any]:
    """Require frozen approval metadata and an exact whole-config fingerprint match."""
    if not isinstance(config_path, Path) or not config_path.is_file():
        raise ReportError(
            "evaluation requires the frozen config file to verify its decision record"
        )
    if (
        not isinstance(current_config_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", current_config_sha256) is None
    ):
        raise ReportError("current frozen config SHA-256 is missing or malformed")
    if sha256_file(config_path) != current_config_sha256:
        raise ReportError("current config file SHA-256 does not match the supplied fingerprint")
    try:
        loaded_configuration = load_config(config_path)
        validate_config(configuration, require_analysis_ready=True)
        verify_protocol_freeze_evidence(configuration, config_path)
    except ConfigError as exc:
        raise ReportError(f"evaluation config is not analysis-ready: {exc}") from exc
    if loaded_configuration != configuration:
        raise ReportError("supplied evaluation configuration does not match the frozen config file")
    if processing_manifest.get("protocol_status") != "protocol_frozen":
        raise ReportError(
            "evaluation refuses exploratory or unmarked processing; re-prepare after protocol freeze"
        )
    if processing_manifest.get("protocol_sha256") != current_config_sha256:
        raise ReportError(
            "prepared protocol fingerprint differs from current frozen config; re-prepare all artifacts"
        )
    if processing_manifest.get("experiment_config_sha256") != current_config_sha256:
        raise ReportError(
            "prepared full-config fingerprint differs from current config; re-prepare before evaluation"
        )
    raw_audit = processing_manifest.get("raw_audit")
    if not isinstance(raw_audit, dict) or raw_audit.get("status") != "verified":
        raise ReportError(
            "evaluation requires a processing manifest linked to a hash-matched completed raw audit"
        )
    labeling = configuration["labeling"]
    expected_labels = {
        "queue_delay_threshold_ms": float(labeling["queue_delay_threshold_ms"]),
        "consecutive_samples_required": int(labeling["consecutive_samples_required"]),
        "prediction_horizon_ms": float(labeling["prediction_horizon_ms"]),
        "observation_window_ms": float(labeling["observation_window_ms"]),
        "max_ground_truth_gap_ms": float(labeling["max_ground_truth_gap_ms"]),
    }
    actual_labels = processing_manifest.get("label_parameters")
    if not isinstance(actual_labels, dict):
        raise ReportError("processing manifest has no label_parameters object")
    mismatches = {
        key: {"prepared": actual_labels.get(key), "requested": expected}
        for key, expected in expected_labels.items()
        if actual_labels.get(key) != expected
    }
    if mismatches:
        raise ReportError(
            f"processed labels were generated with different parameters: {mismatches}"
        )

    processing = configuration["processing"]
    expected_windows = [float(value) for value in processing["feature_windows_seconds"]]
    actual_windows = processing_manifest.get("feature_windows_seconds")
    if actual_windows != expected_windows:
        raise ReportError(
            f"processed feature windows differ from current configuration: prepared={actual_windows}, requested={expected_windows}"
        )
    expected_endpoint_fields = list(configuration["evaluation"]["endpoint_feature_allowlist"])
    actual_endpoint_fields = processing_manifest.get("endpoint_feature_fields")
    if actual_endpoint_fields != expected_endpoint_fields:
        raise ReportError(
            "processed endpoint feature allowlist differs from current configuration: "
            f"prepared={actual_endpoint_fields}, requested={expected_endpoint_fields}"
        )
    tcp_provenance = declared_tcp_retransmission_units_from_run_records(
        processing_manifest.get("run_records"), tuple(expected_endpoint_fields)
    )
    expected_tcp_unit = None
    if "tcp_retransmission_delta" in expected_endpoint_fields:
        if tcp_provenance["status"] == "conflicting":
            details = ", ".join(
                f"{run_id}={unit!r}"
                for run_id, unit in sorted(tcp_provenance["units_by_run"].items())
            )
            raise ReportError(
                "processing manifest has inconsistent tcp_retransmission_delta "
                f"units across runs ({details}); re-prepare after correcting the run manifest"
            )
        if tcp_provenance["status"] != "declared":
            raise ReportError(
                "processing manifest cannot prove the exact tcp_retransmission_delta "
                "unit from every run using that feature; re-prepare with the current "
                "explicit run-manifest contract"
            )
        expected_tcp_unit = tcp_provenance["unit"]
    try:
        expected_specification = build_feature_specification(
            tuple(expected_endpoint_fields),
            tuple(expected_windows),
            tcp_retransmission_unit=expected_tcp_unit,
            feature_temporal_policy=processing.get("feature_temporal_policy"),
        )
    except ProcessingError as exc:
        raise ReportError(str(exc)) from exc
    expected_features = expected_specification["feature_columns"]
    actual_features = processing_manifest.get("feature_columns")
    if actual_features != expected_features:
        raise ReportError(
            "processing manifest feature_columns do not match its declared features/windows"
        )
    try:
        resolved_specification = resolve_feature_specification(processing_manifest)
    except ProcessingError as exc:
        raise ReportError(str(exc)) from exc
    operational_status = feature_specification_operational_status(
        processing_manifest, require_run_record_unit_proof=True
    )
    if not operational_status["operationally_validated"]:
        raise ReportError(
            "processing manifest feature contract is descriptive only and cannot be "
            "used for confirmatory evaluation: "
            + "; ".join(operational_status["reasons"])
            + "; re-prepare under the current frozen feature specification"
        )
    if resolved_specification != expected_specification:
        raise ReportError(
            "processing manifest feature specification differs from the frozen configuration"
        )
    if (
        processing_manifest.get("tcp_retransmission_unit") != expected_tcp_unit
        or processing_manifest.get("feature_temporal_policy")
        != expected_specification["feature_temporal_policy"]
    ):
        raise ReportError(
            "processing manifest TCP unit or feature temporal policy disagrees with its feature specification"
        )
    return {
        "label_parameters_match": True,
        "feature_windows_match": True,
        "endpoint_allowlist_match": True,
        "feature_specification": expected_specification,
        "feature_specification_sha256": feature_specification_sha256(expected_specification),
        "feature_specification_status": operational_status["status"],
        "feature_specification_operationally_validated": True,
        "tcp_retransmission_unit": expected_tcp_unit,
        "tcp_retransmission_unit_provenance": tcp_provenance,
        "feature_temporal_policy": expected_specification["feature_temporal_policy"],
        "feature_temporal_policy_status": expected_specification["feature_temporal_policy"][
            "status"
        ],
        "experiment_config_hash_matches": processing_manifest.get("experiment_config_sha256")
        == current_config_sha256,
        "protocol_fingerprint_matches": processing_manifest.get("protocol_sha256")
        == current_config_sha256,
        "note": "The exact frozen configuration fingerprint must match; any config edit requires fresh preparation before confirmatory evaluation.",
    }


def verify_evaluation_artifacts(
    output_dir: Path,
    *,
    processed_dir: Path | None = None,
    raw_root: Path | None = None,
    audit_report_path: Path | None = None,
    run_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Verify a complete report marker and every artifact checksum without modifying files."""
    root = output_dir.resolve()
    marker = root / "INCOMPLETE"
    manifest_path = root / "evaluation_manifest.json"
    if marker.exists():
        raise ReportError(f"evaluation artifact directory is marked incomplete: {marker}")
    if not root.is_dir() or not manifest_path.is_file():
        raise ReportError("report directory must contain a completed evaluation_manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"invalid evaluation manifest: {exc}") from exc
    if manifest.get("report_schema_version") != 1 or manifest.get("status") not in {
        "complete",
        "partial_failure",
        "failed",
    }:
        raise ReportError("evaluation manifest is unsupported or incomplete")
    if (
        manifest.get("configuration_status") != "protocol_frozen"
        or manifest.get("protocol_status") != "protocol_frozen"
        or manifest.get("protocol_sha256") != manifest.get("config_sha256")
        or not isinstance(manifest.get("protocol_freeze"), dict)
    ):
        raise ReportError(
            "evaluation report lacks a consistent frozen-protocol fingerprint and decision record"
        )
    try:
        _validate_freeze_record(manifest["protocol_freeze"])
    except ConfigError as exc:
        raise ReportError(f"evaluation report has invalid freeze metadata: {exc}") from exc
    expected = {
        "metrics.json": manifest.get("metrics_sha256"),
        "splits.json": manifest.get("split_manifest_sha256"),
        "report.md": manifest.get("report_sha256"),
    }
    for record in manifest.get("prediction_files", {}).values():
        if not isinstance(record, dict):
            raise ReportError("invalid prediction file record")
        expected[record.get("path")] = record.get("sha256")
    prediction_paths = [
        record.get("path")
        for record in manifest.get("prediction_files", {}).values()
        if isinstance(record, dict)
    ]
    if len(prediction_paths) != len(set(prediction_paths)):
        raise ReportError("multiple methods point to the same prediction artifact")
    verified: list[str] = []
    for relative, digest in expected.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ReportError("evaluation manifest contains an incomplete artifact checksum record")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ReportError(
                f"evaluation artifact path escapes report directory: {relative!r}"
            ) from exc
        if not path.is_file() or sha256_file(path) != digest:
            raise ReportError(f"evaluation artifact missing or checksum mismatch: {relative}")
        verified.append(relative)
    allowed_files = set(expected) | {"evaluation_manifest.json"}
    actual_paths = {path.relative_to(root).as_posix() for path in root.rglob("*")}
    if (
        any(path.is_symlink() or path.is_dir() for path in root.rglob("*"))
        or actual_paths != allowed_files
    ):
        raise ReportError(
            f"unexpected or missing evaluation artifacts: expected {sorted(allowed_files)}, found {sorted(actual_paths)}"
        )
    try:
        metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        split_data = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"invalid metrics/split artifacts: {exc}") from exc
    if not isinstance(metrics, dict) or metrics.get("method_failures") != manifest.get(
        "method_failures"
    ):
        raise ReportError("metrics failure records disagree with evaluation manifest")
    successful = sorted(key for key in metrics if key != "method_failures")
    if successful != manifest.get("successful_methods"):
        raise ReportError("successful method list disagrees with metrics artifact")
    if sorted(metrics.get("method_failures", {})) != manifest.get("failed_methods"):
        raise ReportError("failed method list disagrees with metrics artifact")
    if set(manifest.get("requested_methods", [])) != set(
        manifest.get("successful_methods", [])
    ) | set(manifest.get("failed_methods", [])):
        raise ReportError("requested methods are not fully accounted for as successful or failed")
    if manifest.get("status") == "partial_failure" and not manifest.get("failed_methods"):
        raise ReportError("evaluation marked partial_failure without recorded failures")
    if manifest.get("status") == "complete" and manifest.get("failed_methods"):
        raise ReportError("evaluation marked complete despite recorded method failures")
    if manifest.get("status") == "failed" and manifest.get("successful_methods"):
        raise ReportError("evaluation marked failed despite successful methods")
    if (
        not isinstance(split_data, dict)
        or split_data.get("seed") != manifest.get("split_seed")
        or split_data.get("grouping") != manifest.get("split_grouping")
    ):
        raise ReportError("split artifact seed/grouping disagrees with evaluation manifest")
    if processed_dir is not None:
        processed_dir = processed_dir.resolve()
        source_analysis = processed_dir / "analysis.csv"
        source_manifest = processed_dir / "processing_manifest.json"
        if not source_analysis.is_file() or not source_manifest.is_file():
            raise ReportError(
                "processed directory must contain analysis.csv and processing_manifest.json"
            )
        if sha256_file(source_analysis) != manifest.get("analysis_sha256"):
            raise ReportError("processed analysis.csv does not match the evaluation input checksum")
        if sha256_file(source_manifest) != manifest.get("processing_manifest_sha256"):
            raise ReportError("processing manifest does not match the evaluation input checksum")
        _, linked_processing_manifest = load_analysis_dataset(processed_dir)
        if (
            linked_processing_manifest.get("protocol_status") != "protocol_frozen"
            or linked_processing_manifest.get("protocol_sha256") != manifest.get("protocol_sha256")
            or linked_processing_manifest.get("experiment_config_sha256")
            != manifest.get("config_sha256")
        ):
            raise ReportError(
                "evaluation report protocol/config fingerprints disagree with processed data"
            )
        feature_contract = feature_specification_operational_status(
            linked_processing_manifest, require_run_record_unit_proof=True
        )
        expected_specification = feature_contract.get("specification")
        if feature_contract.get("status") != "current_contract_complete" or not isinstance(
            expected_specification, dict
        ):
            reasons = feature_contract.get("reasons") or [
                "prepared feature contract is not current and complete"
            ]
            raise ReportError(
                "linked processed feature contract is not operationally complete: "
                + "; ".join(str(reason) for reason in reasons)
            )
        expected_digest = feature_contract["feature_specification_sha256"]
        expected_policy = feature_contract["feature_temporal_policy"]
        expected_unit = expected_specification.get("tcp_retransmission_unit")
        compatibility = manifest.get("processing_compatibility")
        if (
            manifest.get("feature_specification") != expected_specification
            or manifest.get("feature_specification_sha256") != expected_digest
            or manifest.get("tcp_retransmission_unit") != expected_unit
            or manifest.get("feature_temporal_policy") != expected_policy
            or not isinstance(compatibility, dict)
            or compatibility.get("feature_specification") != expected_specification
            or compatibility.get("feature_specification_sha256") != expected_digest
            or compatibility.get("tcp_retransmission_unit") != expected_unit
            or compatibility.get("feature_temporal_policy") != expected_policy
        ):
            raise ReportError(
                "evaluation manifest feature specification, TCP retransmission unit, "
                "or temporal policy disagrees with linked processed data"
            )
        for method_name in successful:
            method_result = metrics.get(method_name)
            if (
                not isinstance(method_result, dict)
                or method_result.get("feature_specification") != expected_specification
                or method_result.get("feature_specification_sha256") != expected_digest
                or method_result.get("feature_specification_status") != "supplied_current_resolved"
                or method_result.get("tcp_retransmission_unit") != expected_unit
                or method_result.get("feature_temporal_policy") != expected_policy
                or not isinstance(method_result.get("feature_temporal_eligibility"), dict)
            ):
                raise ReportError(
                    f"evaluation metrics for {method_name!r} do not bind the linked "
                    "processed feature specification, TCP unit, temporal policy, "
                    "and temporal-eligibility record"
                )
    audit_report_verified = False
    if audit_report_path is not None:
        audit_path = Path(audit_report_path).resolve()
        if not audit_path.is_file() or sha256_file(audit_path) != manifest.get("raw_audit", {}).get(
            "report_sha256"
        ):
            raise ReportError(
                "raw audit report is missing or its SHA-256 differs from evaluation provenance"
            )
        try:
            audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReportError(f"invalid raw audit report: {exc}") from exc
        recorded = manifest.get("raw_audit", {}).get("files", [])
        audited = audit_payload.get("files", []) if isinstance(audit_payload, dict) else []
        if len(recorded) != len(audited) or any(
            not any(
                item.get("path") == source.get("path")
                and item.get("sha256") == source.get("sha256")
                for item in audited
                if isinstance(item, dict)
            )
            for source in recorded
            if isinstance(source, dict)
        ):
            raise ReportError("raw audit inventory disagrees with evaluation provenance")
        audit_report_verified = True
    run_manifest_verified = False
    if run_manifest_path is not None:
        source_manifest_path = Path(run_manifest_path).resolve()
        if not source_manifest_path.is_file() or sha256_file(source_manifest_path) != manifest.get(
            "source_manifest_sha256"
        ):
            raise ReportError(
                "run manifest is missing or its SHA-256 differs from evaluation provenance"
            )
        run_manifest_verified = True
    raw_sources_verified = False
    if raw_root is not None:
        raw_root = raw_root.resolve()
        for record in manifest.get("source_runs", []):
            relative = record.get("source_file")
            if not isinstance(relative, str):
                raise ReportError("evaluation manifest has an invalid raw source path")
            source = (raw_root / relative).resolve()
            try:
                source.relative_to(raw_root)
            except ValueError as exc:
                raise ReportError(f"raw source path escapes raw_root: {relative}") from exc
            if not source.is_file() or sha256_file(source) != record.get("source_sha256"):
                raise ReportError(f"raw source missing or hash mismatch: {relative}")
        raw_sources_verified = True
    return {
        "status": manifest["status"],
        "report_dir": str(root),
        "verified_artifacts": sorted(verified),
        "processed_data_verified": processed_dir is not None,
        "feature_contract_verified": processed_dir is not None,
        "feature_specification_sha256": manifest.get("feature_specification_sha256"),
        "tcp_retransmission_unit": manifest.get("tcp_retransmission_unit"),
        "feature_temporal_policy": manifest.get("feature_temporal_policy"),
        "raw_sources_verified": raw_sources_verified,
        "audit_report_verified": audit_report_verified,
        "run_manifest_verified": run_manifest_verified,
        "method_failures": manifest.get("method_failures", {}),
    }


def _distribution_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def write_evaluation_artifacts(
    output_dir: Path,
    *,
    config_hash: str,
    data_hash: str,
    split: dict[str, Any],
    results: dict[str, dict[str, Any]],
    configuration: dict[str, Any],
    failures: dict[str, str] | None = None,
    processing_manifest: dict[str, Any] | None = None,
    processing_manifest_sha256: str | None = None,
    compatibility: dict[str, Any] | None = None,
    config_path: Path,
) -> dict[str, Any]:
    """Write new-only split, metrics, per-method predictions, manifest, and Markdown report."""
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ReportError(f"refusing to overwrite existing evaluation directory: {output_dir}")
    if not results and not failures:
        raise ReportError(
            "refusing to write an empty successful evaluation with no results or recorded failures"
        )
    configured_methods = configuration.get("evaluation", {}).get("models_to_evaluate")
    if isinstance(configured_methods, list):
        requested_methods = set(configured_methods) | {"static_threshold", "ewma"}
        reported_methods = set(results) | set(failures or {})
        if requested_methods != reported_methods:
            raise ReportError(
                f"requested methods must be explicitly successful or failed; missing={sorted(requested_methods - reported_methods)}, "
                f"unexpected={sorted(reported_methods - requested_methods)}"
            )
    else:
        requested_methods = set(results) | set(failures or {})
    valid_sha = lambda value: (
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )
    if (
        not valid_sha(config_hash)
        or not valid_sha(data_hash)
        or not valid_sha(processing_manifest_sha256)
    ):
        raise ReportError(
            "evaluation requires valid SHA-256 hashes for config, analysis, and processing manifest"
        )
    if (
        not isinstance(processing_manifest, dict)
        or processing_manifest.get("analysis_sha256") != data_hash
    ):
        raise ReportError("evaluation requires the matching verified processing manifest")
    try:
        canonical_processing_sha = canonical_json_sha256(processing_manifest)
    except (TypeError, ValueError) as exc:
        raise ReportError(f"processing manifest is not canonically serializable: {exc}") from exc
    if canonical_processing_sha != processing_manifest_sha256:
        raise ReportError("processing manifest content does not match its declared SHA-256")
    if configuration.get("status") != "protocol_frozen":
        raise ReportError("evaluation report writing requires a protocol_frozen configuration")
    if not isinstance(config_path, Path) or not config_path.is_file():
        raise ReportError(
            "evaluation requires the frozen config file to verify its decision record"
        )
    if not isinstance(config_hash, str) or re.fullmatch(r"[0-9a-f]{64}", config_hash) is None:
        raise ReportError("frozen config SHA-256 is missing or malformed")
    if sha256_file(config_path) != config_hash:
        raise ReportError("config file SHA-256 does not match the supplied fingerprint")
    try:
        loaded_configuration = load_config(config_path)
        validate_config(configuration, require_analysis_ready=True)
        verify_protocol_freeze_evidence(configuration, config_path)
    except ConfigError as exc:
        raise ReportError(f"evaluation config is not analysis-ready: {exc}") from exc
    if loaded_configuration != configuration:
        raise ReportError("supplied evaluation configuration does not match the frozen config file")
    if processing_manifest.get("protocol_status") != "protocol_frozen":
        raise ReportError(
            "evaluation artifacts cannot be written from exploratory or unmarked preparation"
        )
    if (
        processing_manifest.get("experiment_config_sha256") != config_hash
        or processing_manifest.get("protocol_sha256") != config_hash
    ):
        raise ReportError(
            "prepared protocol/config fingerprints differ from evaluation config; re-prepare"
        )
    if (
        not isinstance(split, dict)
        or split.get("grouping") != configuration["evaluation"].get("primary_grouping")
        or split.get("seed") != configuration["evaluation"].get("random_seed")
    ):
        raise ReportError("evaluation split does not match frozen primary grouping and seed")
    calculated_compatibility = validate_processing_compatibility(
        processing_manifest,
        configuration,
        current_config_sha256=config_hash,
        config_path=config_path,
    )
    if (
        not isinstance(compatibility, dict)
        or not all(
            compatibility.get(key) is True
            for key in (
                "label_parameters_match",
                "feature_windows_match",
                "endpoint_allowlist_match",
                "experiment_config_hash_matches",
                "protocol_fingerprint_matches",
            )
        )
        or calculated_compatibility.get("protocol_fingerprint_matches") is not True
        or compatibility != calculated_compatibility
    ):
        raise ReportError("evaluation requires exact frozen protocol and feature compatibility")
    expected_specification = calculated_compatibility["feature_specification"]
    expected_specification_digest = calculated_compatibility["feature_specification_sha256"]
    expected_temporal_policy = calculated_compatibility["feature_temporal_policy"]
    for method_name, method_result in results.items():
        if (
            not isinstance(method_result, dict)
            or method_result.get("feature_specification") != expected_specification
            or method_result.get("feature_specification_sha256") != expected_specification_digest
            or method_result.get("feature_specification_status") != "supplied_current_resolved"
            or method_result.get("tcp_retransmission_unit")
            != expected_specification.get("tcp_retransmission_unit")
            or method_result.get("feature_temporal_policy") != expected_temporal_policy
            or not isinstance(method_result.get("feature_temporal_eligibility"), dict)
        ):
            raise ReportError(
                f"evaluation result for {method_name!r} does not bind the exact prepared "
                "feature specification, TCP retransmission unit, resolved temporal "
                "policy, and temporal-eligibility record"
            )
    raw_audit = processing_manifest.get("raw_audit")
    if (
        not isinstance(raw_audit, dict)
        or raw_audit.get("status") != "verified"
        or not valid_sha(raw_audit.get("report_sha256"))
    ):
        raise ReportError("evaluation requires a hash-matched completed raw audit")
    if not isinstance(compatibility, dict) or not all(
        compatibility.get(key) is True
        for key in ("label_parameters_match", "feature_windows_match", "endpoint_allowlist_match")
    ):
        raise ReportError(
            "evaluation requires verified label/feature compatibility with current configuration"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    incomplete_marker = output_dir / "INCOMPLETE"
    with incomplete_marker.open("x", encoding="utf-8") as stream:
        stream.write("Artifact generation did not reach its final manifest step.\n")
    (output_dir / "splits.json").write_text(
        json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    failures = failures or {}
    evaluation_status = "failed" if not results else "partial_failure" if failures else "complete"
    metrics: dict[str, Any] = {"method_failures": failures}
    prediction_paths: dict[str, str] = {}
    for name, result in results.items():
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        predictions = result.get("test_predictions", [])
        metrics[name] = {key: value for key, value in result.items() if key != "test_predictions"}
        pred_path = output_dir / f"predictions_{safe_name}.csv"
        fields = [
            "source_row_index",
            "run_id",
            "configuration_id",
            "timestamp_seconds",
            "label",
            "event_active",
            "score",
            "warning",
        ]
        with pred_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(predictions)
        prediction_paths[name] = pred_path.name
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    audit_info = (processing_manifest or {}).get("raw_audit", {})
    report_lines = [
        "# ACN held-out evaluation report",
        "",
        f"**Evaluation outcome:** `{evaluation_status}`. Requested-method failures are explicit below; partial or failed runs are not a successful complete evaluation.",
        "**Evidence status:** software evaluation of supplied data. Source measurement provenance, testbed state, and final scientific protocol are not independently verified by this report generator.",
        "",
        f"- Data SHA-256: `{data_hash}`",
        f"- Configuration SHA-256: `{config_hash}`",
        f"- Processing manifest SHA-256: `{processing_manifest_sha256 or 'not supplied'}`",
        f"- Raw audit: `{audit_info.get('status', 'not supplied')}`; report SHA-256: `{audit_info.get('report_sha256', 'not supplied')}`",
        f"- Split grouping: `{split['grouping']}`; seed: {split['seed']}",
        f"- Test groups: {len(split['groups']['test'])}",
        "",
        "Only validation groups select warning thresholds. Censored targets do not become negative labels. Event warnings are merged according to the configured gap, reset at missing/active/gap boundaries, and matched to at most one future event. Censored warning anchors are not scored as forecasts.",
        "",
        "## Event-level held-out results",
        "",
        "| Method | Events | Event recall | Event precision | False alarms/min | Median lead (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        m = result.get("test_event_metrics", {})
        lead = m.get("warning_lead_time_seconds", {}).get("median")
        show = lambda value: "—" if value is None else f"{value:.5g}"
        report_lines.append(
            f"| {name} | {m.get('event_count', 0)} | {show(m.get('event_recall'))} | "
            f"{show(m.get('event_precision'))} | {show(m.get('false_alarms_per_minute'))} | {show(lead)} |"
        )
    report_lines.extend(
        [
            "",
            "## Per-configuration held-out event metrics",
            "",
            "| Method | Configuration | Runs | Events | Alarm episodes | Recall | Precision | False alarms/min |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, result in results.items():
        metrics_by_config = result.get("test_event_metrics", {}).get("per_configuration", {})
        for config_id, m in sorted(metrics_by_config.items()):
            show = lambda value: "—" if value is None else f"{value:.5g}"
            report_lines.append(
                f"| {name} | {config_id} | {m.get('run_count', 0)} | {m.get('event_count', 0)} | "
                f"{m.get('alarm_episode_count', 0)} | {show(m.get('event_recall'))} | "
                f"{show(m.get('event_precision'))} | {show(m.get('false_alarms_per_minute'))} |"
            )
    report_lines.extend(
        [
            "",
            "## Per-run held-out event metrics",
            "",
            "| Method | Run | Configuration | Events | Alarm episodes | Matched | Missed | False alarms | Exposure (s) | Recall | FAPM |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, result in results.items():
        metrics_by_run = result.get("test_event_metrics", {}).get("per_run", [])
        for m in metrics_by_run:
            show = lambda value: "—" if value is None else f"{value:.5g}"
            report_lines.append(
                f"| {name} | {m.get('run_id')} | {m.get('configuration_id')} | "
                f"{m.get('event_count', 0)} | {m.get('alarm_episode_count', 0)} | "
                f"{m.get('matched_event_count', 0)} | {m.get('missed_event_count', 0)} | "
                f"{m.get('false_alarm_episode_count', 0)} | {show(m.get('eligible_exposure_seconds'))} | "
                f"{show(m.get('event_recall'))} | {show(m.get('false_alarms_per_minute'))} |"
            )
    report_lines.extend(
        [
            "",
            "## Secondary row/window-level ranking metrics",
            "",
            "Average precision is the step-wise area under the precision-recall curve; it is secondary to event metrics and is null when either class is absent. Row-level observations are not independent events.",
            "",
            "| Method | Scored anchors | Positive prevalence | Average precision | ROC-AUC |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, result in results.items():
        m = result.get("test_window_metrics_secondary", result.get("test_metrics", {}))
        show = lambda value: "—" if value is None else f"{value:.5g}"
        report_lines.append(
            f"| {name} | {m.get('n_scored', 0)} | {show(m.get('class_prevalence'))} | "
            f"{show(m.get('average_precision'))} | {show(m.get('roc_auc'))} |"
        )
    if failures:
        report_lines.extend(["", "## Methods not evaluated", ""])
        report_lines.extend(
            f"- **{name}:** {message}" for name, message in sorted(failures.items())
        )
    report_lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "These numbers are not evidence that the original Kali pilot data are valid. Inspect the audit, manifests, collector failures, qdisc timing, phase alignment, and independent group counts before making any performance claim. A single held-out configuration does not establish generalization; configuration-cluster uncertainty is reported as unavailable when there are too few held-out configurations.",
            "",
            "The report contains no dashboard/live prediction claim. It is a deterministic summary of the saved split and evaluation outputs.",
            "",
        ]
    )
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    source_run_records = []
    if processing_manifest is not None:
        source_run_records = [
            {
                key: record.get(key)
                for key in (
                    "run_id",
                    "configuration_id",
                    "source_file",
                    "source_sha256",
                    "source_row_count",
                    "retained_timestamp_rows",
                    "timestamp_unit",
                    "timestamp_source_column",
                    "bottleneck_rate_bps",
                    "column_map",
                    "column_units",
                    "measurement_parse_error_counts",
                    "missing_measurement_counts",
                )
            }
            for record in processing_manifest.get("run_records", [])
        ]
    run_manifest = {
        "report_schema_version": 1,
        "status": evaluation_status,
        "requested_methods": sorted(requested_methods),
        "successful_methods": sorted(results),
        "failed_methods": sorted(failures),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_hash,
        "analysis_sha256": data_hash,
        "processing_manifest_sha256": processing_manifest_sha256,
        "source_manifest_sha256": (processing_manifest or {}).get("source_manifest_sha256"),
        "source_manifest_path": (processing_manifest or {}).get("source_manifest_path"),
        "raw_audit": (processing_manifest or {}).get("raw_audit"),
        "processing_compatibility": calculated_compatibility,
        "feature_specification": calculated_compatibility["feature_specification"],
        "feature_specification_sha256": calculated_compatibility["feature_specification_sha256"],
        "tcp_retransmission_unit": calculated_compatibility["tcp_retransmission_unit"],
        "feature_temporal_policy": calculated_compatibility["feature_temporal_policy"],
        "source_runs": source_run_records,
        "analysis_row_count": (processing_manifest or {}).get("analysis_row_count"),
        "label_counts": (processing_manifest or {}).get("label_counts"),
        "excluded_record_count": (processing_manifest or {}).get("excluded_record_count"),
        "exclusion_reason_counts": (processing_manifest or {}).get("exclusion_reason_counts"),
        "split_counts": split.get("counts"),
        "software": {
            "acn_congestion_prediction": __version__,
            "python": sys.version.split()[0],
            "dependencies": {
                name: _distribution_version(name) for name in ("scikit-learn", "numpy", "scipy")
            },
        },
        "configuration_schema_version": configuration.get("schema_version"),
        "configuration_status": configuration.get("status"),
        "protocol_status": processing_manifest.get("protocol_status"),
        "protocol_sha256": processing_manifest.get("protocol_sha256"),
        "protocol_freeze": configuration.get("protocol_freeze"),
        "split_seed": split["seed"],
        "split_grouping": split["grouping"],
        "evaluation_parameters": configuration["evaluation"],
        "labeling_parameters": configuration["labeling"],
        "method_failures": failures,
        "metrics_sha256": sha256_file(metrics_path),
        "split_manifest_sha256": sha256_file(output_dir / "splits.json"),
        "prediction_files": {
            name: {"path": path, "sha256": sha256_file(output_dir / path)}
            for name, path in prediction_paths.items()
        },
        "report_sha256": sha256_file(report_path),
        "evidence_status": "source data supplied but independently unverified",
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    incomplete_marker.unlink()
    return {
        "status": evaluation_status,
        "output_dir": str(output_dir),
        "method_count": len(results),
        "report": str(report_path),
    }
