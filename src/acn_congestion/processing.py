"""Configurable, provenance-preserving preparation for an explicit aligned CSV contract."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from .labels import derive_persistent_events, label_future_onsets_detailed, normalize_timestamps
from .parsing import parse_backlog_bytes, parse_finite_number

ENDPOINT_FIELDS = ("rtt_ms", "loss_fraction", "goodput_bps", "tcp_retransmission_delta")
OPTIONAL_QDISC_COUNT_FIELDS = ("qdisc_drops_delta", "qdisc_overlimits_delta")
FEATURE_SPEC_VERSION = "acn-trailing-window-features-v2"
LEGACY_FEATURE_SPEC_VERSION = "acn-trailing-window-features-v1"
FEATURE_TEMPORAL_POLICY_VERSION = "acn-feature-temporal-policy-v1"
TCP_RETRANSMISSION_UNITS = ("packets", "bytes")
FEATURE_STATISTICS = ("latest", "mean", "std", "slope_per_s", "valid_fraction")
ENDPOINT_CANONICAL_UNITS = {
    "rtt_ms": "milliseconds",
    "loss_fraction": "fraction in [0, 1]",
    "goodput_bps": "bits per second",
    # This ambiguous wording is retained only to reconstruct the legacy v1
    # specification exactly.  The current v2 specification replaces it with
    # the one packets-or-bytes unit declared by every run using the field;
    # those meanings are never treated as interchangeable.
    "tcp_retransmission_delta": "non-negative integer count increment per sample in one source-declared packets-or-bytes unit, unchanged across runs",
}
ENDPOINT_UNIT_FACTORS = {
    "rtt_ms": {"ms": 1.0, "s": 1000.0, "us": 0.001},
    "loss_fraction": {"fraction": 1.0, "percent": 0.01},
    "goodput_bps": {"bps": 1.0, "kbps": 1000.0, "Mbps": 1_000_000.0, "Gbps": 1_000_000_000.0},
    "tcp_retransmission_delta": {"packets": 1.0, "bytes": 1.0},
}
REQUIRED_RUN_FIELDS = {"timestamp", "queue_backlog_bytes"}
_ALLOWED_MAP_FIELDS = (
    set(ENDPOINT_FIELDS)
    | set(OPTIONAL_QDISC_COUNT_FIELDS)
    | REQUIRED_RUN_FIELDS
    | {"phase_observed", "phase_commanded"}
)
_FORBIDDEN_FEATURE_TOKENS = (
    "queue",
    "qdisc",
    "backlog",
    "label",
    "phase",
    "run_id",
    "configuration_id",
)
_HIGH_RISK_SOURCE_TOKENS = {
    "queue",
    "queued",
    "qdisc",
    "backlog",
    "label",
    "target",
    "future",
    "groundtruth",
    "phase",
    "eventactive",
    "eventonset",
    "overlimit",
    "dropcount",
    "runid",
    "configurationid",
    "congestion",
    "congested",
    "serialization",
    "proxy",
    "prediction",
    "predicted",
    "lookahead",
    "lead",
    "next",
    "postevent",
    "afterevent",
}
_HIGH_RISK_COMPACT_FRAGMENTS = (
    "queuedelay",
    "queueingdelay",
    "qdelay",
    "groundtruth",
    "eventactive",
    "eventonset",
    "overlimit",
    "dropcount",
    "futurelabel",
    "targetlabel",
    "lookahead",
    "leadtime",
    "postevent",
    "afterevent",
    "nextsample",
    "nextstate",
    "predicted",
    "prediction",
    "serializationdelay",
)


def _source_header_risk(source_column: str) -> tuple[list[str], list[str]]:
    """Return high-risk and ambiguous semantic tokens from a header, not a proof of safety."""
    camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", source_column)
    tokens = re.findall(r"[a-z0-9]+", camel_split.lower())
    compact = "".join(tokens)
    high_risk = sorted(_HIGH_RISK_SOURCE_TOKENS.intersection(tokens))
    high_risk.extend(
        fragment
        for fragment in _HIGH_RISK_COMPACT_FRAGMENTS
        if fragment in compact and fragment not in high_risk
    )
    ambiguous = (
        ["event"]
        if "event" in tokens and not any(token in {"eventactive", "eventonset"} for token in tokens)
        else []
    )
    return sorted(set(high_risk)), ambiguous


class ProcessingError(ValueError):
    """Input data does not meet the declared processing contract."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _alignment_binding_digest(manifest: dict[str, Any]) -> str:
    source_root = manifest.get("alignment_source_root")
    spec_file = manifest.get("alignment_spec_file")
    spec_hash = manifest.get("alignment_spec_sha256")
    policy = manifest.get("alignment_policy")
    runs = manifest.get("runs")
    if (
        not isinstance(source_root, str)
        or not Path(source_root).is_absolute()
        or not isinstance(spec_file, str)
        or not Path(spec_file).is_absolute()
        or not isinstance(spec_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", spec_hash)
        or not isinstance(policy, dict)
        or not isinstance(runs, list)
        or not runs
    ):
        raise ProcessingError(
            "approved alignment manifest lacks a valid spec hash, policy, or run binding"
        )
    basis = {
        "alignment_source_root": source_root,
        "alignment_spec_file": spec_file,
        "alignment_spec_sha256": spec_hash,
        "alignment_policy": policy,
        "runs": runs,
    }
    try:
        encoded = json.dumps(basis, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise ProcessingError(
            "alignment approval binding contains non-canonical JSON values"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _verify_aligned_artifact(root: Path, relative: Any, expected_hash: Any, label: str) -> None:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not isinstance(expected_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
    ):
        raise ProcessingError(f"approved alignment has an invalid {label} path or SHA-256")
    probe = root
    for part in Path(relative).parts:
        probe = probe / part
        if probe.is_symlink():
            raise ProcessingError(f"approved alignment {label} must not traverse symlinks")
    path = root / relative
    if not path.is_file() or file_sha256(path) != expected_hash:
        raise ProcessingError(f"approved alignment {label} is missing or its SHA-256 is stale")


def verify_alignment_review(
    root: Path,
    manifest: dict[str, Any],
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Require a fingerprint-bound human decision and intact derived artifacts."""
    review = manifest.get("alignment_review")
    if review is None:
        if "alignment_status" in manifest:
            raise ProcessingError("alignment_status is present but alignment_review is missing")
        return {"status": "not_applicable"}
    if not isinstance(review, dict):
        raise ProcessingError("alignment_review must be an object")
    if review.get("status") != "approved" or manifest.get("alignment_status") != "approved":
        raise ProcessingError(
            "stream-aligned manifest is review_required; add a hash-linked alignment approval only after inspecting real collector evidence"
        )
    required = {
        "status",
        "decision_record_file",
        "decision_record_sha256",
        "alignment_binding_sha256",
    }
    if set(review) != required:
        raise ProcessingError(
            "approved alignment_review must contain exactly status, decision_record_file, decision_record_sha256, and alignment_binding_sha256"
        )
    binding_hash = _alignment_binding_digest(manifest)
    declared_binding_hash = manifest.get("alignment_binding_sha256")
    if (
        not isinstance(declared_binding_hash, str)
        or declared_binding_hash != binding_hash
        or review.get("alignment_binding_sha256") != binding_hash
    ):
        raise ProcessingError(
            "alignment approval is stale or does not bind this manifest's sources, mappings, policy, and outputs"
        )
    relative = review.get("decision_record_file")
    expected = review.get("decision_record_sha256")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not isinstance(expected, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected)
    ):
        raise ProcessingError(
            "alignment approval requires a local relative decision-record path and lowercase SHA-256"
        )
    base = manifest_path.resolve().parent if manifest_path is not None else root.resolve()
    record = (base / relative).resolve()
    try:
        record.relative_to(base)
    except ValueError as exc:
        raise ProcessingError(
            "alignment approval decision record escapes its manifest directory"
        ) from exc
    probe = base
    for part in Path(relative).parts:
        probe = probe / part
        if probe.is_symlink():
            raise ProcessingError("alignment approval decision record must not traverse symlinks")
    if not record.is_file():
        raise ProcessingError(f"alignment approval decision record is missing: {relative!r}")
    actual = file_sha256(record)
    if actual != expected:
        raise ProcessingError("alignment approval decision-record SHA-256 mismatch")
    try:
        decision = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProcessingError("alignment decision record must be valid UTF-8 JSON") from exc
    if not isinstance(decision, dict) or set(decision) != {
        "schema_version",
        "decision",
        "alignment_binding_sha256",
        "reviewer",
        "reviewed_at_utc",
        "rationale",
    }:
        raise ProcessingError("alignment decision record has an invalid schema")
    if (
        decision.get("schema_version") != 1
        or decision.get("decision") != "approve_alignment"
        or decision.get("alignment_binding_sha256") != binding_hash
        or not isinstance(decision.get("reviewer"), str)
        or not decision["reviewer"].strip()
        or not isinstance(decision.get("rationale"), str)
        or not decision["rationale"].strip()
    ):
        raise ProcessingError(
            "alignment decision record does not explicitly approve this alignment binding"
        )
    reviewed_at = decision.get("reviewed_at_utc")
    try:
        parsed_time = (
            datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
            if isinstance(reviewed_at, str)
            else None
        )
    except ValueError as exc:
        raise ProcessingError(
            "alignment decision timestamp must be timezone-aware ISO-8601"
        ) from exc
    if parsed_time is None or parsed_time.tzinfo is None or parsed_time.utcoffset() is None:
        raise ProcessingError("alignment decision timestamp must be timezone-aware ISO-8601")
    if parsed_time.utcoffset().total_seconds() != 0:
        raise ProcessingError("alignment decision timestamp must be recorded in UTC")
    source_root = Path(manifest["alignment_source_root"])
    if source_root.is_symlink() or not source_root.is_dir():
        raise ProcessingError("approved alignment original source root is missing or unsafe")
    spec_file = Path(manifest["alignment_spec_file"])
    _verify_aligned_artifact(
        spec_file.parent,
        spec_file.name,
        manifest["alignment_spec_sha256"],
        "alignment specification",
    )
    if not isinstance(manifest.get("runs"), list):
        raise ProcessingError("approved alignment runs must be a list")
    for run in manifest["runs"]:
        if not isinstance(run, dict) or not isinstance(run.get("metadata"), dict):
            raise ProcessingError("approved alignment run metadata is malformed")
        metadata = run["metadata"]
        if metadata.get("alignment_spec_sha256") != manifest.get("alignment_spec_sha256"):
            raise ProcessingError("approved alignment run spec hash is inconsistent")
        _verify_aligned_artifact(
            root, run.get("file"), metadata.get("aligned_data_sha256"), "aligned data"
        )
        _verify_aligned_artifact(
            root,
            metadata.get("alignment_provenance_file"),
            metadata.get("alignment_provenance_sha256"),
            "provenance",
        )
        _verify_aligned_artifact(
            root,
            metadata.get("alignment_exclusions_file"),
            metadata.get("alignment_exclusions_sha256"),
            "exclusions",
        )
        sources = metadata.get("source_streams")
        if not isinstance(sources, dict):
            raise ProcessingError("approved alignment lacks original source-stream hashes")
        for stream_name in ("endpoint", "router"):
            source = sources.get(stream_name)
            if not isinstance(source, dict):
                raise ProcessingError(
                    f"approved alignment lacks original {stream_name} source metadata"
                )
            _verify_aligned_artifact(
                source_root,
                source.get("file"),
                source.get("sha256"),
                f"original {stream_name} source",
            )
        clock = metadata.get("clock_review")
        if not isinstance(clock, dict):
            raise ProcessingError("approved alignment clock review metadata is malformed")
        _verify_aligned_artifact(
            source_root,
            clock.get("evidence_file"),
            clock.get("evidence_sha256"),
            "clock-review evidence",
        )
    return {
        "status": "approved",
        "decision_record_file": relative,
        "decision_record_sha256": actual,
        "alignment_binding_sha256": binding_hash,
    }


def _safe_child(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ProcessingError(f"input path escapes raw root: {relative!r}") from exc
    if not path.is_file():
        raise ProcessingError(f"input file does not exist: {relative!r}")
    return path


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str | None]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ProcessingError(f"CSV has no header: {path}")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ProcessingError(f"CSV has duplicate header names: {path}")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ProcessingError(f"cannot read CSV {path}: {exc}") from exc
    for index, row in enumerate(rows, 2):
        if None in row:
            raise ProcessingError(f"extra fields in {path} at CSV line {index}")
        if any(value is None for value in row.values()):
            raise ProcessingError(f"short row in {path} at CSV line {index}")
    return list(reader.fieldnames), rows


def _timestamp_suffix_unit(column: str) -> str | None:
    lowered = column.strip().lower()
    return (
        "nanoseconds"
        if lowered.endswith("_ns")
        else "microseconds"
        if lowered.endswith("_us")
        else "milliseconds"
        if lowered.endswith("_ms")
        else None
    )


def _audit_timestamp_unit_compatible(
    audit: dict[str, Any], item: dict[str, Any], source_column: str, declared_unit: str
) -> None:
    audited_column = item.get("timestamp_column")
    if not isinstance(audited_column, str) or audited_column.strip() != source_column.strip():
        raise ProcessingError(
            f"raw audit selected timestamp column {audited_column!r}, but manifest maps {source_column!r}; "
            "rerun audit with the mapped column explicitly selected"
        )
    kinds = item.get("timestamp_timezone_kinds", {})
    if not isinstance(kinds, dict):
        raise ProcessingError("raw audit has invalid timestamp timezone/unit metadata")
    numeric_kinds = [kind for kind in kinds if kind.startswith("numeric_")]
    if "naive_assumed_utc" in kinds:
        raise ProcessingError(
            "timezone-naive ISO timestamps have unverified timezone semantics; record an explicit UTC offset "
            "or convert from a documented source timezone without altering the original raw files"
        )
    if kinds and not numeric_kinds and declared_unit != "seconds":
        raise ProcessingError(
            "ISO-8601 timestamps are normalized as seconds; manifest timestamp_unit must be 'seconds'"
        )
    if "numeric_unit_conflict" in numeric_kinds:
        raise ProcessingError("raw audit reports a timestamp unit/suffix conflict")
    if numeric_kinds:
        override = audit.get("numeric_timestamp_unit_override")
        inferred_suffix = _timestamp_suffix_unit(source_column)
        effective = override or inferred_suffix
        if effective != declared_unit:
            raise ProcessingError(
                f"numeric timestamp unit is not corroborated by the audit: manifest={declared_unit!r}, "
                f"audit override={override!r}, source suffix={inferred_suffix!r}; rerun audit with an explicit matching --timestamp-unit"
            )


def _validate_feature_contract(
    endpoint_fields: tuple[str, ...], feature_windows_seconds: tuple[float, ...]
) -> None:
    if (
        not endpoint_fields
        or len(set(endpoint_fields)) != len(endpoint_fields)
        or set(endpoint_fields) - set(ENDPOINT_FIELDS)
    ):
        raise ProcessingError(
            f"endpoint_feature_fields must be a non-empty unique subset of {ENDPOINT_FIELDS}"
        )
    if not feature_windows_seconds or any(
        not math.isfinite(float(window)) or float(window) <= 0 for window in feature_windows_seconds
    ):
        raise ProcessingError("feature_windows_seconds must contain finite positive values")
    if sorted(set(feature_windows_seconds)) != list(feature_windows_seconds):
        raise ProcessingError("feature_windows_seconds must be sorted and unique")
    window_suffixes = [f"{float(window):g}" for window in feature_windows_seconds]
    if len(set(window_suffixes)) != len(window_suffixes):
        raise ProcessingError(
            "feature_windows_seconds must remain distinct under canonical ':g' feature "
            f"naming; colliding suffixes: {window_suffixes}"
        )


def expected_feature_columns(
    endpoint_fields: tuple[str, ...], feature_windows_seconds: tuple[float, ...]
) -> list[str]:
    """Return the canonical sorted predictor names for a field/window contract."""
    _validate_feature_contract(endpoint_fields, feature_windows_seconds)
    columns: list[str] = []
    for window in feature_windows_seconds:
        suffix = f"{float(window):g}s"
        for field in endpoint_fields:
            stem = f"{field}_{suffix}"
            columns.extend(f"{stem}_{statistic}" for statistic in FEATURE_STATISTICS)
            if field == "tcp_retransmission_delta":
                columns.append(f"{stem}_sum")
    return sorted(columns)


_FEATURE_TEMPORAL_POLICY_CORE_KEYS = {
    "status",
    "max_gap_seconds",
    "minimum_valid_fraction",
    "minimum_temporal_coverage_fraction",
}
_FEATURE_TEMPORAL_POLICY_CANONICAL_KEYS = _FEATURE_TEMPORAL_POLICY_CORE_KEYS | {
    "policy_version",
    "application",
    "window_gap_rule",
    "temporal_coverage_rule",
    "measurement_coverage_rule",
    "valid_fraction_handling",
    "scientific_status",
}


def normalize_feature_temporal_policy(
    value: Any = None, *, require_resolved: bool = False
) -> dict[str, Any]:
    """Return the canonical feature-gap/coverage policy without inventing values.

    ``None`` and an explicit ``status: unresolved`` object both mean that the
    scientifically consequential thresholds have not been approved.  A
    resolved policy must explicitly supply all three dimensions.  Software
    validates only their types/ranges; it does not claim that a supplied
    threshold is scientifically appropriate for any collector or cadence.
    """
    raw: dict[str, Any] = {}
    if value is not None:
        if not isinstance(value, dict):
            raise ProcessingError("feature temporal policy must be an object or null")
        unknown = sorted(set(value) - _FEATURE_TEMPORAL_POLICY_CANONICAL_KEYS)
        if unknown:
            raise ProcessingError(f"unsupported feature temporal policy key(s): {unknown}")
        raw = value
    if (
        raw.get("policy_version") is not None
        and raw.get("policy_version") != FEATURE_TEMPORAL_POLICY_VERSION
    ):
        raise ProcessingError("unsupported feature temporal policy version")

    values = {
        "max_gap_seconds": raw.get("max_gap_seconds"),
        "minimum_valid_fraction": raw.get("minimum_valid_fraction"),
        "minimum_temporal_coverage_fraction": raw.get("minimum_temporal_coverage_fraction"),
    }
    status = raw.get("status")
    if status is None:
        status = "resolved" if all(item is not None for item in values.values()) else "unresolved"
    if status not in {"resolved", "unresolved"}:
        raise ProcessingError("feature temporal policy status must be 'resolved' or 'unresolved'")

    normalized_values: dict[str, float | None]
    if status == "unresolved":
        supplied = [key for key, item in values.items() if item is not None]
        if supplied:
            raise ProcessingError(
                "an unresolved feature temporal policy cannot carry threshold values; "
                f"remove {supplied} or mark the policy resolved only after the values are approved"
            )
        normalized_values = {key: None for key in values}
    else:
        normalized_values = {}
        for key, item in values.items():
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ProcessingError(f"resolved feature temporal policy requires {key}")
            number = float(item)
            if not math.isfinite(number):
                raise ProcessingError(f"feature temporal policy {key} must be finite")
            if key == "max_gap_seconds" and number <= 0:
                raise ProcessingError("feature temporal policy max_gap_seconds must be positive")
            if key != "max_gap_seconds" and not 0 <= number <= 1:
                raise ProcessingError(f"feature temporal policy {key} must be in [0, 1]")
            normalized_values[key] = number

    if require_resolved and status != "resolved":
        raise ProcessingError(
            "feature temporal policy is unresolved; maximum feature gap and minimum "
            "observed coverage must be explicitly approved before confirmatory use"
        )
    return {
        "policy_version": FEATURE_TEMPORAL_POLICY_VERSION,
        "status": status,
        **normalized_values,
        "application": "model_input_eligibility_gate",
        "window_gap_rule": (
            "mask measurement summaries for a trailing window when its maximum "
            "successive timestamp gap exceeds max_gap_seconds"
        ),
        "temporal_coverage_rule": (
            "observed_span_seconds divided by the nominal window must be at least "
            "minimum_temporal_coverage_fraction"
        ),
        "measurement_coverage_rule": (
            "each endpoint field/window valid_fraction must be at least minimum_valid_fraction"
        ),
        "valid_fraction_handling": (
            "retained as a coverage feature; coverage alone never makes an anchor scorable"
        ),
        "scientific_status": (
            "explicitly configured; software does not validate the scientific choice"
            if status == "resolved"
            else "unresolved; descriptive/exploratory use only and confirmatory use is blocked"
        ),
    }


def _tcp_retransmission_unit_for_fields(fields: tuple[str, ...], value: Any) -> str | None:
    if "tcp_retransmission_delta" not in fields:
        if value not in (None, "not_applicable"):
            raise ProcessingError(
                "tcp_retransmission_unit must be null when tcp_retransmission_delta is not selected"
            )
        return None
    if value not in TCP_RETRANSMISSION_UNITS:
        raise ProcessingError(
            "tcp_retransmission_delta requires an exact declared unit of 'packets' or "
            "'bytes'; the two units are not interchangeable"
        )
    return str(value)


def _canonical_units_for_fields(
    fields: tuple[str, ...], tcp_retransmission_unit: str | None
) -> dict[str, str]:
    units = {field: ENDPOINT_CANONICAL_UNITS[field] for field in fields}
    if tcp_retransmission_unit is not None:
        units["tcp_retransmission_delta"] = (
            f"non-negative integer {tcp_retransmission_unit} increment per sample"
        )
    return units


def _feature_groups(
    fields: tuple[str, ...],
    windows: tuple[float, ...],
    feature_columns: list[str],
) -> dict[str, Any]:
    by_signal = {
        field: [column for column in feature_columns if column.startswith(field + "_")]
        for field in fields
    }
    by_statistic: dict[str, list[str]] = {}
    for statistic in (*FEATURE_STATISTICS, "sum"):
        selected = [column for column in feature_columns if column.endswith(f"_{statistic}")]
        if selected:
            by_statistic[statistic] = selected
    by_window = {
        f"{window:g}s": [column for column in feature_columns if f"_{window:g}s_" in column]
        for window in windows
    }
    return {
        "by_endpoint_signal": by_signal,
        "by_statistic": by_statistic,
        "by_window": by_window,
    }


def _build_legacy_feature_specification(
    endpoint_feature_fields: tuple[str, ...],
    feature_windows_seconds: tuple[float, ...],
) -> dict[str, Any]:
    """Reconstruct v1 exactly for descriptive inspection of older artifacts.

    V1 does not encode the chosen TCP retransmission unit or a resolved
    feature-gap/coverage policy.  It is therefore never silently upgraded to
    the current contract and cannot by itself validate an operational model.
    """
    fields = tuple(endpoint_feature_fields)
    windows = tuple(float(window) for window in feature_windows_seconds)
    _validate_feature_contract(fields, windows)
    feature_columns = expected_feature_columns(fields, windows)
    return {
        "spec_version": LEGACY_FEATURE_SPEC_VERSION,
        "endpoint_fields": list(fields),
        "canonical_units": {field: ENDPOINT_CANONICAL_UNITS[field] for field in fields},
        "feature_windows_seconds": list(windows),
        "statistics_by_endpoint_field": {
            field: [
                *FEATURE_STATISTICS,
                *(["sum"] if field == "tcp_retransmission_delta" else []),
            ]
            for field in fields
        },
        "feature_columns": feature_columns,
        "feature_order": "lexicographically sorted canonical feature names",
        "feature_groups": _feature_groups(fields, windows, feature_columns),
        "temporal_policy": {
            "window_bounds": "[anchor_timestamp - window_seconds, anchor_timestamp]",
            "future_observations": "excluded",
            "run_boundaries": "features are constructed separately for each run or uploaded sequence",
            "partial_window_policy": "anchors near the sequence start use the available past observations; window diagnostics record coverage",
            "latest": "most recent observed value in the trailing window",
            "mean": "arithmetic mean of observed values in the trailing window",
            "std": "population standard deviation of observed values; zero for one observed value",
            "slope_per_s": "ordinary least-squares slope per second over observed timestamp/value pairs",
            "valid_fraction": "observed samples divided by all samples in the trailing window; coverage alone does not make an anchor scorable",
            "sum": "sum of observed TCP retransmission increments in the trailing window",
            "missing_values": "measurement summaries remain missing; train-fitted preprocessing may impute partial vectors, but anchors with no observed measurement summary are unscored",
        },
        "diagnostics_not_predictors": [
            "window_*_sample_count",
            "window_*_observed_span_seconds",
            "window_*_max_gap_seconds",
        ],
    }


def build_feature_specification(
    endpoint_feature_fields: tuple[str, ...],
    feature_windows_seconds: tuple[float, ...],
    *,
    tcp_retransmission_unit: str | None = None,
    feature_temporal_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the current v2 past-only endpoint feature specification.

    V2 binds three things V1 could not prove: the exact TCP retransmission
    packets-or-bytes unit (when that field is selected), the canonical
    feature-gap/coverage policy object (resolved or explicitly unresolved),
    and the model-input masking semantics for that policy.  Preparation,
    confirmatory compatibility checks, evaluation, registry refits, and
    inference resolve this same object and digest.
    """
    fields = tuple(endpoint_feature_fields)
    try:
        windows = tuple(float(window) for window in feature_windows_seconds)
    except (TypeError, ValueError) as exc:
        raise ProcessingError(
            "feature_windows_seconds must contain finite positive values"
        ) from exc
    _validate_feature_contract(fields, windows)
    tcp_unit = _tcp_retransmission_unit_for_fields(fields, tcp_retransmission_unit)
    temporal_policy = normalize_feature_temporal_policy(feature_temporal_policy)
    feature_columns = expected_feature_columns(fields, windows)
    partial_window_policy = (
        "descriptive trailing summaries use the available past observations; "
        "the resolved feature temporal policy masks measurement summaries from "
        "windows with excessive gaps, inadequate temporal coverage, or inadequate "
        "field measurement coverage before model fitting/scoring"
        if temporal_policy["status"] == "resolved"
        else "descriptive trailing summaries use the available past observations; "
        "the feature-gap/coverage policy is unresolved, so these artifacts are "
        "exploratory only and cannot validate confirmatory evaluation or an "
        "operational model"
    )
    return {
        "spec_version": FEATURE_SPEC_VERSION,
        "endpoint_fields": list(fields),
        "canonical_units": _canonical_units_for_fields(fields, tcp_unit),
        "tcp_retransmission_unit": tcp_unit,
        "tcp_retransmission_unit_status": (
            "declared" if tcp_unit is not None else "not_applicable"
        ),
        "feature_windows_seconds": list(windows),
        "statistics_by_endpoint_field": {
            field: [
                *FEATURE_STATISTICS,
                *(["sum"] if field == "tcp_retransmission_delta" else []),
            ]
            for field in fields
        },
        "feature_columns": feature_columns,
        "feature_order": "lexicographically sorted canonical feature names",
        "feature_groups": _feature_groups(fields, windows, feature_columns),
        "feature_temporal_policy": temporal_policy,
        "temporal_policy": {
            "window_bounds": "[anchor_timestamp - window_seconds, anchor_timestamp]",
            "future_observations": "excluded",
            "run_boundaries": "features are constructed separately for each run or uploaded sequence",
            "partial_window_policy": partial_window_policy,
            "latest": "most recent observed value in the trailing window",
            "mean": "arithmetic mean of observed values in the trailing window",
            "std": "population standard deviation of observed values; zero for one observed value",
            "slope_per_s": "ordinary least-squares slope per second over observed timestamp/value pairs",
            "valid_fraction": "observed samples divided by all samples in the trailing window; coverage alone does not make an anchor scorable",
            "sum": "sum of observed TCP retransmission increments in the trailing window, in the exact declared unit",
            "missing_values": "measurement summaries remain missing; train-fitted preprocessing may impute partial vectors, but anchors with no policy-eligible observed measurement summary are unscored",
        },
        "diagnostics_not_predictors": [
            "window_*_sample_count",
            "window_*_observed_span_seconds",
            "window_*_max_gap_seconds",
            "feature_temporal_eligibility_status",
            "feature_temporal_eligibility_reason",
        ],
        "compatibility": {
            "supersedes": LEGACY_FEATURE_SPEC_VERSION,
            "legacy_artifacts": (
                "may be inspected descriptively, but are not silently reinterpreted "
                "as v2 and cannot validate an operational model when their exact TCP "
                "unit or resolved temporal policy cannot be proven"
            ),
        },
    }


def feature_specification_sha256(specification: dict[str, Any]) -> str:
    payload = json.dumps(
        specification, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_feature_specification_object(specification: Any) -> dict[str, Any]:
    """Validate and canonicalize an embedded v1 or v2 feature specification."""
    if not isinstance(specification, dict):
        raise ProcessingError("feature_specification must be an object")
    fields = specification.get("endpoint_fields")
    windows = specification.get("feature_windows_seconds")
    if (
        not isinstance(fields, list)
        or not all(isinstance(field, str) for field in fields)
        or not isinstance(windows, list)
    ):
        raise ProcessingError(
            "feature specification lacks valid endpoint_fields/feature_windows_seconds"
        )
    try:
        field_tuple = tuple(fields)
        window_tuple = tuple(
            float(window)
            for window in windows
            if not isinstance(window, bool) and isinstance(window, (int, float))
        )
        if len(window_tuple) != len(windows):
            raise ProcessingError("feature specification windows must be numbers")
        version = specification.get("spec_version")
        if version == LEGACY_FEATURE_SPEC_VERSION:
            expected = _build_legacy_feature_specification(field_tuple, window_tuple)
        elif version == FEATURE_SPEC_VERSION:
            expected = build_feature_specification(
                field_tuple,
                window_tuple,
                tcp_retransmission_unit=specification.get("tcp_retransmission_unit"),
                feature_temporal_policy=specification.get("feature_temporal_policy"),
            )
        else:
            raise ProcessingError(f"unsupported feature specification version: {version!r}")
    except ProcessingError:
        raise
    except (TypeError, ValueError) as exc:
        raise ProcessingError(f"invalid feature specification contract: {exc}") from exc
    if specification != expected:
        raise ProcessingError(
            "embedded feature_specification is stale or noncanonical for its fields, "
            "windows, TCP unit, and feature temporal policy"
        )
    return expected


def feature_temporal_policy_from_specification(
    specification: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a specification's temporal policy, treating v1 as unresolved."""
    if specification is None:
        return normalize_feature_temporal_policy(None)
    validated = validate_feature_specification_object(specification)
    if validated["spec_version"] == LEGACY_FEATURE_SPEC_VERSION:
        return normalize_feature_temporal_policy(None)
    return normalize_feature_temporal_policy(validated["feature_temporal_policy"])


def declared_tcp_retransmission_units_from_run_records(
    run_records: Any, endpoint_fields: tuple[str, ...]
) -> dict[str, Any]:
    """Summarize per-run TCP-unit declarations without exposing source paths."""
    result: dict[str, Any] = {
        "status": "not_applicable",
        "unit": None,
        "units_by_run": {},
        "runs_using_feature": [],
        "missing_run_ids": [],
        "invalid_run_ids": [],
    }
    if "tcp_retransmission_delta" not in endpoint_fields:
        return result
    if not isinstance(run_records, list) or not run_records:
        result["status"] = "not_recorded"
        return result
    units_by_run: dict[str, Any] = {}
    for record in run_records:
        if not isinstance(record, dict):
            result["invalid_run_ids"].append("<malformed run record>")
            continue
        run_id = record.get("run_id")
        label = run_id if isinstance(run_id, str) and run_id else "<unknown run>"
        mapping = record.get("column_map")
        units = record.get("column_units")
        mapping = mapping if isinstance(mapping, dict) else {}
        units = units if isinstance(units, dict) else {}
        mapped = "tcp_retransmission_delta" in mapping or (
            not mapping and "tcp_retransmission_delta" in units
        )
        if not mapped:
            continue
        result["runs_using_feature"].append(label)
        unit = units.get("tcp_retransmission_delta")
        units_by_run[label] = unit
        if unit is None:
            result["missing_run_ids"].append(label)
        elif unit not in TCP_RETRANSMISSION_UNITS:
            result["invalid_run_ids"].append(label)
    result["units_by_run"] = units_by_run
    if result["invalid_run_ids"]:
        result["status"] = "invalid"
    elif not units_by_run or result["missing_run_ids"]:
        result["status"] = "missing"
    else:
        observed = set(units_by_run.values())
        if len(observed) > 1:
            result["status"] = "conflicting"
        else:
            result["status"] = "declared"
            result["unit"] = next(iter(observed))
    return result


def _tcp_unit_conflict_message(units_by_run: dict[str, Any]) -> str:
    details = ", ".join(f"{run_id}={unit!r}" for run_id, unit in sorted(units_by_run.items()))
    return (
        "inconsistent tcp_retransmission_delta units across runs "
        f"({details}); packets and bytes are not interchangeable"
    )


def _validate_run_record_tcp_consistency(
    manifest: dict[str, Any], specification: dict[str, Any]
) -> None:
    if "run_records" not in manifest:
        return
    fields = tuple(specification["endpoint_fields"])
    declared = declared_tcp_retransmission_units_from_run_records(
        manifest.get("run_records"), fields
    )
    if "tcp_retransmission_delta" not in fields:
        return
    if declared["status"] == "conflicting":
        raise ProcessingError(_tcp_unit_conflict_message(declared["units_by_run"]))
    if declared["status"] == "invalid":
        raise ProcessingError(
            "tcp_retransmission_delta has an unsupported unit declaration for run(s): "
            + ", ".join(declared["invalid_run_ids"])
        )
    if specification["spec_version"] != FEATURE_SPEC_VERSION:
        return
    if declared["status"] != "declared":
        raise ProcessingError(
            "processing/model manifest cannot prove the exact tcp_retransmission_delta "
            "unit from its run records"
        )
    if declared["unit"] != specification.get("tcp_retransmission_unit"):
        raise ProcessingError(
            "feature specification tcp_retransmission_delta unit does not match the "
            f"unit declared by its run records ({declared['units_by_run']})"
        )


def resolve_feature_specification(manifest: dict[str, Any]) -> dict[str, Any]:
    """Resolve a manifest's specification without upgrading legacy artifacts.

    An embedded specification is validated in its declared version.  A
    manifest with no embedded object can only be reconstructed as legacy v1
    from its recorded fields/windows; it is deliberately **not** promoted to
    v2, because its TCP unit and temporal policy were not bound when it was
    prepared.  Callers that need an operational contract must use
    :func:`feature_specification_operational_status`.
    """
    fields = manifest.get("endpoint_feature_fields")
    windows = manifest.get("feature_windows_seconds")
    if (
        not isinstance(fields, list)
        or not all(isinstance(field, str) for field in fields)
        or not isinstance(windows, list)
        or any(
            isinstance(window, bool) or not isinstance(window, (int, float)) for window in windows
        )
    ):
        raise ProcessingError(
            "manifest lacks endpoint_feature_fields/feature_windows_seconds needed to resolve its feature specification"
        )
    field_tuple = tuple(fields)
    window_tuple = tuple(float(window) for window in windows)
    embedded = manifest.get("feature_specification")
    embedded_digest = manifest.get("feature_specification_sha256")
    if embedded is not None:
        specification = validate_feature_specification_object(embedded)
        if (
            tuple(specification["endpoint_fields"]) != field_tuple
            or tuple(specification["feature_windows_seconds"]) != window_tuple
        ):
            raise ProcessingError(
                "embedded feature_specification does not match the manifest endpoint fields/windows"
            )
    else:
        if embedded_digest is not None:
            raise ProcessingError(
                "feature_specification_sha256 is present without the embedded feature "
                "specification; the exact TCP unit and temporal policy cannot be proven "
                "from a digest alone"
            )
        try:
            specification = _build_legacy_feature_specification(field_tuple, window_tuple)
        except (TypeError, ValueError) as exc:
            raise ProcessingError(f"invalid feature specification contract: {exc}") from exc
    if embedded_digest is not None and embedded_digest != feature_specification_sha256(
        specification
    ):
        raise ProcessingError(
            "feature_specification_sha256 does not match the resolved specification"
        )
    declared_columns = manifest.get("feature_columns")
    if declared_columns is not None:
        if (
            not isinstance(declared_columns, list)
            or not declared_columns
            or not all(isinstance(column, str) for column in declared_columns)
            or len(set(declared_columns)) != len(declared_columns)
            or not set(declared_columns).issubset(specification["feature_columns"])
        ):
            raise ProcessingError(
                "manifest feature_columns are not a unique subset of the resolved feature specification"
            )
        # A processing manifest declares the complete prepared feature set.  A
        # model manifest may legitimately declare the train-fitted subset.
        if (
            "processing_schema_version" in manifest
            and declared_columns != specification["feature_columns"]
        ):
            raise ProcessingError(
                "processing manifest feature_columns do not match the resolved feature specification"
            )
    _validate_run_record_tcp_consistency(manifest, specification)
    return specification


def feature_specification_operational_status(
    manifest: dict[str, Any],
    *,
    require_run_record_unit_proof: bool = False,
) -> dict[str, Any]:
    """Classify whether a manifest proves the current operational ML contract.

    This is a contract/provenance classification, not scientific validation.
    Legacy v1 artifacts and v2 artifacts with an unresolved temporal policy
    remain descriptive only.  Local hashes also do not authenticate a human
    approver against someone able to rewrite every local artifact.
    """
    try:
        specification = resolve_feature_specification(manifest)
    except ProcessingError as exc:
        return {
            "status": "invalid",
            "operationally_validated": False,
            "specification": None,
            "specification_version": None,
            "feature_specification_sha256": None,
            "tcp_retransmission_unit": None,
            "tcp_unit_provenance": None,
            "feature_temporal_policy": normalize_feature_temporal_policy(None),
            "reasons": [str(exc)],
        }
    reasons: list[str] = []
    version = specification["spec_version"]
    digest = feature_specification_sha256(specification)
    if manifest.get("feature_specification") is None:
        reasons.append("feature specification is not embedded in the manifest")
    if manifest.get("feature_specification_sha256") != digest:
        reasons.append("feature specification SHA-256 is missing or stale")
    if version != FEATURE_SPEC_VERSION:
        reasons.append(
            "legacy v1 feature specification does not bind the exact TCP unit or "
            "a resolved feature temporal policy"
        )
    temporal_policy = feature_temporal_policy_from_specification(specification)
    if temporal_policy["status"] != "resolved":
        reasons.append("feature temporal policy is unresolved")
    fields = tuple(specification["endpoint_fields"])
    tcp_provenance = declared_tcp_retransmission_units_from_run_records(
        manifest.get("run_records"), fields
    )
    tcp_unit = specification.get("tcp_retransmission_unit")
    if "tcp_retransmission_delta" in fields and tcp_unit not in TCP_RETRANSMISSION_UNITS:
        reasons.append("exact tcp_retransmission_delta unit is missing")
        tcp_unit = None
    if require_run_record_unit_proof and "tcp_retransmission_delta" in fields:
        if tcp_provenance["status"] != "declared":
            reasons.append("run records do not prove one exact tcp_retransmission_delta unit")
        elif tcp_provenance["unit"] != tcp_unit:
            reasons.append("run-record TCP unit differs from the feature specification")
    if version != FEATURE_SPEC_VERSION:
        status = "legacy_unvalidated"
    elif any("TCP" in reason or "tcp_" in reason for reason in reasons):
        status = "missing_tcp_unit_provenance"
    elif temporal_policy["status"] != "resolved":
        status = "unresolved_temporal_policy"
    elif reasons:
        status = "incomplete_current_contract"
    else:
        status = "current_contract_complete"
    return {
        "status": status,
        "operationally_validated": not reasons,
        "specification": specification,
        "specification_version": version,
        "feature_specification_sha256": digest,
        "tcp_retransmission_unit": tcp_unit,
        "tcp_unit_provenance": tcp_provenance,
        "feature_temporal_policy": temporal_policy,
        "reasons": reasons,
    }


def require_operational_feature_specification(
    manifest: dict[str, Any],
    *,
    require_run_record_unit_proof: bool = False,
) -> dict[str, Any]:
    """Return the current specification or raise with its explicit blockers."""
    status = feature_specification_operational_status(
        manifest, require_run_record_unit_proof=require_run_record_unit_proof
    )
    if not status["operationally_validated"]:
        raise ProcessingError(
            "feature specification is not a complete current operational contract: "
            + "; ".join(status["reasons"])
        )
    specification = status["specification"]
    if not isinstance(specification, dict):  # Defensive; status logic prevents this.
        raise ProcessingError("feature specification could not be resolved")
    return specification


def _parse_time(value: str, numeric_unit: str) -> float:
    try:
        return normalize_timestamps([value], numeric_unit=numeric_unit)[0]
    except ValueError as exc:
        raise ProcessingError(str(exc)) from exc


def _feature_numeric_value(value: Any, *, description: str) -> float:
    if isinstance(value, bool):
        raise ProcessingError(f"{description} must be numeric or missing")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProcessingError(f"{description} must be numeric or missing: {value!r}") from exc
    if not math.isfinite(number):
        raise ProcessingError(f"{description} must be finite or missing")
    return number


def _finite_feature_value(value: Any, *, description: str) -> float | None:
    if value is None or value == "":
        return None
    return _feature_numeric_value(value, description=description)


def _classify_feature_temporal_eligibility_validated(
    row: dict[str, Any],
    feature_columns: tuple[str, ...],
    specification: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    temporal_policy = feature_temporal_policy_from_specification(specification)
    fields = tuple(specification["endpoint_fields"])
    windows = tuple(float(window) for window in specification["feature_windows_seconds"])
    selected = set(feature_columns)
    measurement_columns: list[str] = []
    pairs: list[tuple[str, float, list[str], str]] = []
    for field in fields:
        statistics = specification["statistics_by_endpoint_field"][field]
        for window in windows:
            suffix = f"{window:g}s"
            pair_columns = [
                f"{field}_{suffix}_{statistic}"
                for statistic in statistics
                if f"{field}_{suffix}_{statistic}" in selected
            ]
            pair_measurement_columns = [
                column for column in pair_columns if not column.endswith("_valid_fraction")
            ]
            if pair_columns:
                pairs.append((field, window, pair_measurement_columns, suffix))
                measurement_columns.extend(pair_measurement_columns)

    def observed_columns(columns: list[str]) -> list[str]:
        observed: list[str] = []
        for column in columns:
            value = _finite_feature_value(row.get(column), description=f"feature {column!r}")
            if value is not None:
                observed.append(column)
        return observed

    if temporal_policy["status"] != "resolved":
        reason = (
            "legacy_feature_specification"
            if specification["spec_version"] == LEGACY_FEATURE_SPEC_VERSION
            else "feature_temporal_policy_unresolved"
        )
        eligible = observed_columns(measurement_columns)
        return row, {
            "status": "descriptive_only",
            "reason": reason,
            "policy_status": temporal_policy["status"],
            "eligible_measurement_feature_columns": eligible,
            "masked_feature_columns": [],
            "field_window_reasons": {
                f"{field}:{suffix}": reason for field, _window, _columns, suffix in pairs
            },
        }

    window_state: dict[str, dict[str, Any]] = {}
    for window in windows:
        suffix = f"{window:g}s"
        names = {
            "sample_count": f"window_{suffix}_sample_count",
            "observed_span_seconds": f"window_{suffix}_observed_span_seconds",
            "max_gap_seconds": f"window_{suffix}_max_gap_seconds",
        }
        values: dict[str, float | None] = {}
        invalid = False
        for key, name in names.items():
            if name not in row:
                invalid = True
                values[key] = None
                continue
            try:
                values[key] = _finite_feature_value(
                    row.get(name), description=f"temporal diagnostic {name!r}"
                )
            except ProcessingError:
                invalid = True
                values[key] = None
        sample_count = values["sample_count"]
        observed_span = values["observed_span_seconds"]
        max_gap = values["max_gap_seconds"]
        if (
            sample_count is None
            or observed_span is None
            or max_gap is None
            or sample_count < 0
            or not float(sample_count).is_integer()
            or observed_span < 0
            or max_gap < 0
            or observed_span > window + max(1e-9, abs(window) * 1e-12)
        ):
            invalid = True
        temporal_coverage = None if observed_span is None else min(observed_span / window, 1.0)
        reason = None
        if invalid:
            reason = "missing_or_invalid_temporal_diagnostics"
        elif max_gap is not None and max_gap > float(temporal_policy["max_gap_seconds"]):
            reason = "excessive_sampling_gap"
        elif temporal_coverage is not None and temporal_coverage < float(
            temporal_policy["minimum_temporal_coverage_fraction"]
        ):
            reason = "insufficient_observation_history"
        window_state[suffix] = {
            "sample_count": sample_count,
            "observed_span_seconds": observed_span,
            "max_gap_seconds": max_gap,
            "temporal_coverage_fraction": temporal_coverage,
            "reason": reason,
        }

    masked_columns: list[str] = []
    eligible_columns: list[str] = []
    pair_reasons: dict[str, str] = {}
    observed_pair_blockers: list[str] = []
    all_pair_reasons: list[str] = []
    any_samples = False
    for field, _window, pair_measurement_columns, suffix in pairs:
        state = window_state[suffix]
        sample_count = state["sample_count"]
        if sample_count is not None and sample_count > 0:
            any_samples = True
        observed = observed_columns(pair_measurement_columns)
        valid_fraction_name = f"{field}_{suffix}_valid_fraction"
        valid_fraction: float | None = None
        valid_fraction_invalid = False
        if valid_fraction_name in row:
            try:
                valid_fraction = _finite_feature_value(
                    row.get(valid_fraction_name),
                    description=f"feature {valid_fraction_name!r}",
                )
            except ProcessingError:
                valid_fraction_invalid = True
            if valid_fraction is not None and not -1e-12 <= valid_fraction <= 1 + 1e-12:
                valid_fraction_invalid = True
        minimum_valid_fraction = float(temporal_policy["minimum_valid_fraction"])
        if state["reason"] == "missing_or_invalid_temporal_diagnostics" or (
            valid_fraction_invalid or (valid_fraction is None and minimum_valid_fraction > 0)
        ):
            reason = "missing_or_invalid_temporal_diagnostics"
        elif state["reason"] is not None:
            reason = str(state["reason"])
        elif sample_count == 0:
            reason = "no_observed_measurement"
        elif not observed or valid_fraction == 0:
            reason = "missing_measurements"
        elif valid_fraction is not None and valid_fraction < minimum_valid_fraction:
            reason = "inadequate_observed_coverage"
        else:
            reason = "eligible"
        pair_key = f"{field}:{suffix}"
        pair_reasons[pair_key] = reason
        all_pair_reasons.append(reason)
        if reason == "eligible":
            eligible_columns.extend(observed)
        else:
            masked_columns.extend(
                column
                for column in pair_measurement_columns
                if _finite_feature_value(row.get(column), description=f"feature {column!r}")
                is not None
            )
            if observed or (valid_fraction is not None and valid_fraction > 0):
                observed_pair_blockers.append(reason)

    if eligible_columns:
        status = "eligible"
        reason = "eligible"
    else:
        status = "not_eligible"
        priority = (
            "excessive_sampling_gap",
            "insufficient_observation_history",
            "inadequate_observed_coverage",
            "missing_or_invalid_temporal_diagnostics",
            "missing_measurements",
            "no_observed_measurement",
        )
        reason = next(
            (item for item in priority if item in observed_pair_blockers),
            None,
        )
        if reason is None:
            if not any_samples and "no_observed_measurement" in all_pair_reasons:
                reason = "no_observed_measurement"
            elif "missing_measurements" in all_pair_reasons:
                reason = "missing_measurements"
            elif "missing_or_invalid_temporal_diagnostics" in all_pair_reasons:
                reason = "missing_or_invalid_temporal_diagnostics"
            elif all_pair_reasons:
                reason = next(item for item in priority if item in all_pair_reasons)
            else:
                reason = "no_observed_measurement"

    masked_row = row
    if masked_columns:
        masked_row = dict(row)
        for _field, _window, pair_measurement_columns, suffix in pairs:
            if pair_reasons[f"{_field}:{suffix}"] != "eligible":
                for column in pair_measurement_columns:
                    masked_row[column] = None
    return masked_row, {
        "status": status,
        "reason": reason,
        "policy_status": temporal_policy["status"],
        "eligible_measurement_feature_columns": eligible_columns,
        "masked_feature_columns": masked_columns,
        "field_window_reasons": pair_reasons,
        "window_diagnostics": window_state,
    }


def apply_feature_temporal_policy_to_rows(
    rows: list[dict[str, Any]],
    feature_columns: list[str] | tuple[str, ...],
    specification: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Mask model-facing summaries that fail the resolved temporal policy.

    Descriptive feature values in prepared artifacts are not rewritten.  This
    function returns the exact row view that training, selection, registry
    refits, and inference must use, plus one transparent eligibility record
    per anchor.  With no current resolved policy it returns the rows
    unchanged and marks them descriptive-only.
    """
    columns = tuple(feature_columns)
    if not columns or len(set(columns)) != len(columns):
        raise ProcessingError("feature_columns must be non-empty and unique")
    if specification is None:
        classifications = [
            {
                "status": "descriptive_only",
                "reason": "feature_specification_not_supplied",
                "policy_status": "unresolved",
                "eligible_measurement_feature_columns": [
                    column
                    for column in columns
                    if not column.endswith("_valid_fraction")
                    and _finite_feature_value(row.get(column), description=f"feature {column!r}")
                    is not None
                ],
                "masked_feature_columns": [],
                "field_window_reasons": {},
            }
            for row in rows
        ]
        return rows, classifications
    validated = validate_feature_specification_object(specification)
    if not set(columns).issubset(validated["feature_columns"]):
        raise ProcessingError("feature_columns are outside the supplied feature specification")
    output: list[dict[str, Any]] = []
    classifications: list[dict[str, Any]] = []
    for row in rows:
        masked, classification = _classify_feature_temporal_eligibility_validated(
            row, columns, validated
        )
        output.append(masked)
        classifications.append(classification)
    return output, classifications


def apply_feature_temporal_policy(
    row: dict[str, Any],
    feature_columns: list[str] | tuple[str, ...],
    specification: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows, classifications = apply_feature_temporal_policy_to_rows(
        [row], feature_columns, specification
    )
    return rows[0], classifications[0]


def classify_feature_temporal_eligibility(
    row: dict[str, Any],
    feature_columns: list[str] | tuple[str, ...],
    specification: dict[str, Any] | None,
) -> dict[str, Any]:
    return apply_feature_temporal_policy(row, feature_columns, specification)[1]


def feature_temporal_eligibility_counts(
    classifications: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons = Counter(str(item.get("reason")) for item in classifications)
    statuses = Counter(str(item.get("status")) for item in classifications)
    return {
        "anchor_count": len(classifications),
        "eligible_anchor_count": statuses.get("eligible", 0),
        "not_eligible_anchor_count": statuses.get("not_eligible", 0),
        "descriptive_only_anchor_count": statuses.get("descriptive_only", 0),
        "reason_counts": dict(sorted(reasons.items())),
        "status_counts": dict(sorted(statuses.items())),
    }


def feature_contract_metadata(
    specification: dict[str, Any] | None,
    classifications: list[dict[str, Any]] | None = None,
    *,
    scope: str = "all_rows",
) -> dict[str, Any]:
    """Return the one canonical feature-contract record used by ML artifacts."""
    if specification is None:
        return {
            "feature_specification": None,
            "feature_specification_sha256": None,
            "feature_specification_status": "not_supplied_descriptive_only",
            "tcp_retransmission_unit": None,
            "feature_temporal_policy": feature_temporal_policy_from_specification(None),
            "feature_temporal_eligibility": {
                "status": "not_evaluated_without_feature_specification",
                "scope": scope,
            },
        }
    validated = validate_feature_specification_object(specification)
    policy = feature_temporal_policy_from_specification(validated)
    if validated["spec_version"] == LEGACY_FEATURE_SPEC_VERSION:
        status = "supplied_legacy_descriptive_only"
    elif policy["status"] == "resolved":
        status = "supplied_current_resolved"
    else:
        status = "supplied_current_unresolved_descriptive_only"
    eligibility = (
        feature_temporal_eligibility_counts(classifications)
        if classifications is not None
        else {"status": "not_classified"}
    )
    return {
        "feature_specification": validated,
        "feature_specification_sha256": feature_specification_sha256(validated),
        "feature_specification_status": status,
        "tcp_retransmission_unit": validated.get("tcp_retransmission_unit"),
        "feature_temporal_policy": policy,
        "feature_temporal_eligibility": {**eligibility, "scope": scope},
    }


def _normalize_window_rows(
    rows: list[dict[str, Any]], endpoint_fields: tuple[str, ...]
) -> list[dict[str, float | None]]:
    """Convert feature inputs once to finite floats, preserving blanks as missing."""
    normalized: list[dict[str, float | None]] = []
    for index, row in enumerate(rows):
        try:
            raw_timestamp = row["timestamp_seconds"]
        except (KeyError, TypeError) as exc:
            raise ProcessingError(
                f"feature construction row {index} lacks a finite timestamp_seconds"
            ) from exc
        item: dict[str, float | None] = {
            "timestamp_seconds": _feature_numeric_value(
                raw_timestamp, description=f"row {index} timestamp_seconds"
            )
        }
        for field in endpoint_fields:
            try:
                raw_value = row[field]
            except (KeyError, TypeError) as exc:
                raise ProcessingError(
                    f"feature construction row {index} lacks endpoint field {field!r}"
                ) from exc
            item[field] = (
                None
                if raw_value is None or raw_value == ""
                else _feature_numeric_value(
                    raw_value, description=f"row {index} endpoint field {field!r}"
                )
            )
        normalized.append(item)
    return normalized


def _window_statistics(
    times: list[float], values: list[float]
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return mean, population std, OLS slope/second, and sum for observed pairs."""
    if not values:
        return None, None, None, None
    total = math.fsum(values)
    mean = total / len(values)
    if len(values) == 1:
        return mean, 0.0, None, total
    time_mean = math.fsum(times) / len(times)
    time_variation = 0.0
    value_variation = 0.0
    covariance = 0.0
    for timestamp, value in zip(times, values, strict=True):
        time_delta = timestamp - time_mean
        value_delta = value - mean
        time_variation += time_delta * time_delta
        value_variation += value_delta * value_delta
        covariance += time_delta * value_delta
    return (
        mean,
        math.sqrt(max(value_variation, 0.0) / len(values)),
        None if time_variation == 0 else covariance / time_variation,
        total,
    )


def _window_slice_features_from_normalized(
    rows: list[dict[str, float | None]],
    window_seconds: float,
    endpoint_fields: tuple[str, ...],
) -> dict[str, float | None]:
    features: dict[str, float | None] = {}
    suffix = f"{window_seconds:g}s"
    sample_times: list[float] = []
    previous_timestamp: float | None = None
    max_gap = 0.0
    for row in rows:
        timestamp = row["timestamp_seconds"]
        if timestamp is None:  # Defensive: normalized rows always carry a timestamp.
            raise ProcessingError("normalized feature row lacks timestamp_seconds")
        if previous_timestamp is not None:
            max_gap = max(max_gap, timestamp - previous_timestamp)
        previous_timestamp = timestamp
        sample_times.append(timestamp)
    features[f"window_{suffix}_sample_count"] = len(rows)
    features[f"window_{suffix}_observed_span_seconds"] = (
        sample_times[-1] - sample_times[0]
        if len(sample_times) > 1
        else 0.0
        if sample_times
        else None
    )
    features[f"window_{suffix}_max_gap_seconds"] = max_gap if sample_times else None
    for field in endpoint_fields:
        observed_times: list[float] = []
        values: list[float] = []
        for timestamp, row in zip(sample_times, rows, strict=True):
            value = row[field]
            if value is not None:
                observed_times.append(timestamp)
                values.append(value)
        stem = f"{field}_{suffix}"
        mean, std, slope, total = _window_statistics(observed_times, values)
        features[f"{stem}_latest"] = values[-1] if values else None
        features[f"{stem}_mean"] = mean
        features[f"{stem}_std"] = std
        features[f"{stem}_slope_per_s"] = slope
        features[f"{stem}_valid_fraction"] = len(values) / len(rows) if rows else None
        if field == "tcp_retransmission_delta":
            features[f"{stem}_sum"] = total
    return features


def _window_slice_features(
    rows: list[dict[str, Any]],
    window_seconds: float,
    endpoint_fields: tuple[str, ...],
) -> dict[str, float | None]:
    return _window_slice_features_from_normalized(
        _normalize_window_rows(rows, endpoint_fields), window_seconds, endpoint_fields
    )


def _window_features(
    run_rows: list[dict[str, Any]],
    anchor: int,
    window_seconds: float,
    endpoint_fields: tuple[str, ...],
) -> dict[str, float | None]:
    """Compatibility helper for one anchor; batch callers should prefer the builder below."""
    end = _feature_numeric_value(
        run_rows[anchor]["timestamp_seconds"], description="timestamp_seconds"
    )
    start = end - window_seconds
    rows = [
        row
        for row in run_rows[: anchor + 1]
        if start
        <= _feature_numeric_value(row["timestamp_seconds"], description="timestamp_seconds")
        <= end
    ]
    return _window_slice_features(rows, window_seconds, endpoint_fields)


def build_trailing_window_features(
    run_rows: list[dict[str, Any]],
    feature_windows_seconds: tuple[float, ...],
    endpoint_fields: tuple[str, ...] = ENDPOINT_FIELDS,
) -> list[dict[str, float | None]]:
    """Build descriptive trailing features for one already time-ordered run/sequence.

    A monotone left pointer is maintained for each window.  This preserves
    the exact inclusive ``[anchor-window, anchor]`` semantics of
    :func:`_window_features` without rescanning the whole run prefix for
    every anchor.  Statistics are still computed from the actual observed
    values in each window, avoiding numerically fragile prefix-sum shortcuts.
    """
    windows = tuple(float(window) for window in feature_windows_seconds)
    _validate_feature_contract(endpoint_fields, windows)
    if not run_rows:
        return []
    normalized_rows = _normalize_window_rows(run_rows, endpoint_fields)
    times: list[float] = []
    for row in normalized_rows:
        timestamp = row["timestamp_seconds"]
        if timestamp is None:  # Defensive: normalization requires a finite timestamp.
            raise ProcessingError("normalized feature row lacks timestamp_seconds")
        if times and timestamp <= times[-1]:
            raise ProcessingError("feature construction timestamps must be strictly increasing")
        times.append(timestamp)
    left_by_window = {window: 0 for window in windows}
    output: list[dict[str, float | None]] = []
    for anchor, anchor_time in enumerate(times):
        features: dict[str, float | None] = {}
        for window in windows:
            left = left_by_window[window]
            start = anchor_time - window
            while times[left] < start:
                left += 1
            left_by_window[window] = left
            features.update(
                _window_slice_features_from_normalized(
                    normalized_rows[left : anchor + 1], window, endpoint_fields
                )
            )
        output.append(features)
    return output


def _load_run(
    root: Path, entry: dict[str, Any], expected_feature_fields: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    required = (
        "run_id",
        "configuration_id",
        "file",
        "bottleneck_rate_bps",
        "timestamp_unit",
        "column_map",
    )
    missing = [field for field in required if field not in entry]
    if missing:
        raise ProcessingError(f"run manifest entry missing keys: {', '.join(missing)}")
    if not isinstance(entry["run_id"], str) or not entry["run_id"]:
        raise ProcessingError("run_id must be a non-empty string")
    if not isinstance(entry["configuration_id"], str) or not entry["configuration_id"]:
        raise ProcessingError(f"{entry['run_id']}: configuration_id must be a non-empty string")
    if not isinstance(entry["file"], str) or not entry["file"]:
        raise ProcessingError(f"{entry['run_id']}: file must be a relative path string")
    rate = parse_finite_number(
        entry["bottleneck_rate_bps"], name="bottleneck_rate_bps", minimum=1e-9
    )
    if rate is None:
        raise ProcessingError(f"{entry['run_id']}: bottleneck_rate_bps is required")
    if not isinstance(entry["timestamp_unit"], str) or entry["timestamp_unit"] not in {
        "seconds",
        "milliseconds",
        "microseconds",
        "nanoseconds",
    }:
        raise ProcessingError(
            f"{entry['run_id']}: timestamp_unit must explicitly be seconds, milliseconds, microseconds, or nanoseconds; ISO-8601 timestamps use seconds"
        )
    mapping = entry["column_map"]
    if not isinstance(mapping, dict) or not REQUIRED_RUN_FIELDS <= set(mapping):
        raise ProcessingError(
            f"{entry['run_id']}: column_map must map timestamp and queue_backlog_bytes"
        )
    mapped_endpoint = set(mapping) & set(ENDPOINT_FIELDS)
    units = entry.get("column_units")
    mapped_qdisc_counts = set(mapping) & set(OPTIONAL_QDISC_COUNT_FIELDS)
    expected_unit_fields = mapped_endpoint | mapped_qdisc_counts | {"queue_backlog_bytes"}
    if not isinstance(units, dict) or set(units) != expected_unit_fields:
        raise ProcessingError(
            f"{entry['run_id']}: column_units must explicitly declare units for {sorted(expected_unit_fields)}"
        )
    for field in mapped_endpoint:
        if not isinstance(units[field], str) or units[field] not in ENDPOINT_UNIT_FACTORS[field]:
            raise ProcessingError(
                f"{entry['run_id']}: unsupported unit {units[field]!r} for {field}; "
                f"choose from {sorted(ENDPOINT_UNIT_FACTORS[field])}"
            )
    if units["queue_backlog_bytes"] != "bytes":
        raise ProcessingError(
            f"{entry['run_id']}: queue_backlog_bytes numeric unit must be declared 'bytes'"
        )
    for field in mapped_qdisc_counts:
        if units[field] != "count":
            raise ProcessingError(f"{entry['run_id']}: {field} unit must be declared 'count'")
    if not mapped_endpoint.intersection(expected_feature_fields):
        raise ProcessingError(f"{entry['run_id']}: map at least one configured endpoint feature")
    disallowed_endpoint = mapped_endpoint - expected_feature_fields
    if disallowed_endpoint:
        raise ProcessingError(
            f"{entry['run_id']}: endpoint fields are not in configured allowlist: {sorted(disallowed_endpoint)}"
        )
    unknown = set(mapping) - _ALLOWED_MAP_FIELDS
    if unknown:
        raise ProcessingError(f"{entry['run_id']}: unsupported canonical fields: {sorted(unknown)}")
    if not all(isinstance(value, str) and value for value in mapping.values()):
        raise ProcessingError(f"{entry['run_id']}: source column names must be non-empty strings")
    if len(set(mapping.values())) != len(mapping.values()):
        raise ProcessingError(
            f"{entry['run_id']}: one source column cannot be reused for multiple canonical measurements"
        )
    source = _safe_child(root, entry["file"])
    source_sha_before = file_sha256(source)
    headers, raw_rows = _read_rows(source)
    source_sha_after = file_sha256(source)
    if source_sha_before != source_sha_after:
        raise ProcessingError(f"{entry['run_id']}: raw source changed while being read")
    absent = sorted(set(mapping.values()) - set(headers))
    if absent:
        raise ProcessingError(f"{entry['run_id']}: mapped source columns are absent: {absent}")
    suffix_unit = _timestamp_suffix_unit(mapping["timestamp"])
    if suffix_unit is not None and suffix_unit != entry["timestamp_unit"]:
        raise ProcessingError(
            f"{entry['run_id']}: timestamp_unit {entry['timestamp_unit']!r} conflicts with "
            f"timestamp source header suffix ({suffix_unit})"
        )
    feature_source_mapping_warnings = []
    for field, source_column in mapping.items():
        if field not in ENDPOINT_FIELDS:
            continue
        high_risk, ambiguous = _source_header_risk(source_column)
        if high_risk:
            raise ProcessingError(
                f"{entry['run_id']}: endpoint feature {field!r} maps to leakage-prone source header "
                f"{source_column!r} (tokens: {high_risk}); inspect the schema and choose a verified endpoint measurement"
            )
        if ambiguous:
            feature_source_mapping_warnings.append(
                {
                    "canonical_feature": field,
                    "source_column": source_column,
                    "basis": f"ambiguous source token(s) {ambiguous}; inspect actual measurement semantics",
                }
            )

    parsed_times: list[float | None] = []
    exclusions: list[dict[str, Any]] = []
    for i, raw in enumerate(raw_rows):
        try:
            parsed_times.append(
                _parse_time(raw[mapping["timestamp"]] or "", entry["timestamp_unit"])
            )
        except ProcessingError as exc:
            parsed_times.append(None)
            exclusions.append(
                {
                    "run_id": entry["run_id"],
                    "source_row": i + 2,
                    "reason": "invalid_timestamp",
                    "detail": str(exc),
                }
            )
    valid_values = [
        raw_rows[i][mapping["timestamp"]]
        for i, value in enumerate(parsed_times)
        if value is not None
    ]
    try:
        normalized_valid_times = normalize_timestamps(
            valid_values, numeric_unit=entry["timestamp_unit"]
        )
    except ValueError as exc:
        raise ProcessingError(f"{entry['run_id']}: timestamp sequence invalid: {exc}") from exc
    iterator = iter(normalized_valid_times)
    parsed_times = [next(iterator) if value is not None else None for value in parsed_times]

    out: list[dict[str, Any]] = []
    parse_error_counts: Counter[str] = Counter()
    missing_measurement_counts: Counter[str] = Counter()
    for i, (raw, timestamp) in enumerate(zip(raw_rows, parsed_times)):
        if timestamp is None:
            continue
        item: dict[str, Any] = {
            "run_id": entry["run_id"],
            "configuration_id": entry["configuration_id"],
            "source_file": entry["file"],
            "source_row": i + 2,
            "timestamp_original": raw[mapping["timestamp"]],
            "timestamp_seconds": timestamp,
            "phase_observed": raw.get(mapping.get("phase_observed", ""))
            if mapping.get("phase_observed")
            else None,
            "phase_commanded": raw.get(mapping.get("phase_commanded", ""))
            if mapping.get("phase_commanded")
            else None,
        }
        for field in OPTIONAL_QDISC_COUNT_FIELDS:
            item[field] = None
            if field in mapping:
                try:
                    count_value = parse_finite_number(raw[mapping[field]], name=field, minimum=0)
                    if count_value is not None and not count_value.is_integer():
                        raise ValueError(f"{field} must be a non-negative integer count")
                    item[field] = int(count_value) if count_value is not None else None
                    if item[field] is None:
                        missing_measurement_counts[field] += 1
                except ValueError as exc:
                    parse_error_counts[field] += 1
                    exclusions.append(
                        {
                            "run_id": entry["run_id"],
                            "source_row": i + 2,
                            "reason": "invalid_qdisc_measurement",
                            "detail": str(exc),
                        }
                    )
        for field in ENDPOINT_FIELDS:
            item[field] = None
            if field in mapping:
                try:
                    item[field] = parse_finite_number(
                        raw[mapping[field]],
                        name=field,
                        minimum=0,
                        maximum=100
                        if field == "loss_fraction" and units[field] == "percent"
                        else 1
                        if field == "loss_fraction"
                        else None,
                    )
                    if (
                        item[field] is not None
                        and field == "tcp_retransmission_delta"
                        and not item[field].is_integer()
                    ):
                        raise ValueError(
                            "tcp_retransmission_delta must be a non-negative integer increment"
                        )
                    if item[field] is not None:
                        item[field] *= ENDPOINT_UNIT_FACTORS[field][units[field]]
                        if not math.isfinite(item[field]):
                            raise ValueError(f"converted {field} is non-finite")
                    if item[field] is None:
                        missing_measurement_counts[field] += 1
                except ValueError as exc:
                    parse_error_counts[field] += 1
                    exclusions.append(
                        {
                            "run_id": entry["run_id"],
                            "source_row": i + 2,
                            "reason": "invalid_endpoint_measurement",
                            "detail": str(exc),
                        }
                    )
        try:
            item["queue_backlog_bytes"] = parse_backlog_bytes(raw[mapping["queue_backlog_bytes"]])
            if item["queue_backlog_bytes"] is None:
                missing_measurement_counts["queue_backlog_bytes"] += 1
        except ValueError as exc:
            item["queue_backlog_bytes"] = None
            parse_error_counts["queue_backlog_bytes"] += 1
            exclusions.append(
                {
                    "run_id": entry["run_id"],
                    "source_row": i + 2,
                    "reason": "invalid_qdisc_measurement",
                    "detail": str(exc),
                }
            )
        item["qdisc_queue_delay_proxy_ms"] = (
            item["queue_backlog_bytes"] * 8 / rate * 1000
            if item["queue_backlog_bytes"] is not None
            else None
        )
        out.append(item)
    metadata = entry.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ProcessingError(f"{entry['run_id']}: metadata must be an object")
    return (
        out,
        exclusions,
        {
            "run_id": entry["run_id"],
            "configuration_id": entry["configuration_id"],
            "source_file": entry["file"],
            "source_sha256": source_sha_after,
            "run_metadata": metadata,
            "feature_source_mapping_warnings": feature_source_mapping_warnings,
            "source_row_count": len(raw_rows),
            "retained_timestamp_rows": len(out),
            "measurement_parse_error_counts": dict(parse_error_counts),
            "missing_measurement_counts": dict(missing_measurement_counts),
            "bottleneck_rate_bps": rate,
            "timestamp_unit": entry["timestamp_unit"],
            "timestamp_source_column": mapping["timestamp"],
            "column_map": mapping,
            "column_units": units,
        },
    )


def _tcp_unit_from_run_records_for_preparation(
    run_records: list[dict[str, Any]], selected_features: tuple[str, ...]
) -> str | None:
    declared = declared_tcp_retransmission_units_from_run_records(run_records, selected_features)
    if "tcp_retransmission_delta" not in selected_features:
        return None
    if declared["status"] == "conflicting":
        raise ProcessingError(
            _tcp_unit_conflict_message(declared["units_by_run"])
            + "; preparation stopped before writing a combined output"
        )
    if declared["status"] == "invalid":
        raise ProcessingError(
            "tcp_retransmission_delta has an unsupported unit declaration for run(s): "
            + ", ".join(declared["invalid_run_ids"])
            + "; preparation stopped before writing a combined output"
        )
    if declared["status"] != "declared":
        if declared["missing_run_ids"]:
            detail = "run(s) map tcp_retransmission_delta without declaring its unit: " + ", ".join(
                declared["missing_run_ids"]
            )
        else:
            detail = "no run maps the selected tcp_retransmission_delta feature"
        raise ProcessingError(
            "cannot bind the exact tcp_retransmission_delta unit: "
            + detail
            + "; preparation stopped before writing a combined output"
        )
    return str(declared["unit"])


def _preflight_tcp_unit_consistency(entries: list[Any], selected_features: tuple[str, ...]) -> None:
    """Reject an explicit cross-run TCP-unit conflict before reading run CSVs.

    Malformed entries are left for the detailed per-run loader errors; this
    early check only handles structurally readable declarations so a clear
    packets-versus-bytes conflict cannot produce a combined artifact.
    """
    if "tcp_retransmission_delta" not in selected_features:
        return
    records: list[dict[str, Any]] = []
    structurally_readable = True
    any_mapped = False
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("column_map"), dict):
            structurally_readable = False
            continue
        records.append(
            {
                "run_id": entry.get("run_id"),
                "column_map": entry.get("column_map"),
                "column_units": entry.get("column_units"),
            }
        )
        any_mapped = any_mapped or "tcp_retransmission_delta" in entry["column_map"]
    if not structurally_readable or not records:
        return
    declared = declared_tcp_retransmission_units_from_run_records(records, selected_features)
    if declared["status"] == "conflicting":
        raise ProcessingError(
            _tcp_unit_conflict_message(declared["units_by_run"])
            + "; preparation stopped before writing a combined output"
        )
    if any_mapped and declared["status"] == "invalid":
        raise ProcessingError(
            "tcp_retransmission_delta has an unsupported unit declaration for run(s): "
            + ", ".join(declared["invalid_run_ids"])
            + "; preparation stopped before writing a combined output"
        )
    if not any_mapped:
        raise ProcessingError(
            "cannot bind the exact tcp_retransmission_delta unit: no run maps the "
            "selected feature; preparation stopped before writing a combined output"
        )


def process_dataset(
    *,
    raw_root: Path,
    manifest_path: Path,
    output_dir: Path,
    queue_delay_threshold_ms: float,
    consecutive_samples_required: int,
    prediction_horizon_ms: float,
    observation_window_ms: float,
    max_ground_truth_gap_ms: float,
    feature_windows_seconds: list[float],
    endpoint_feature_fields: list[str] | None = None,
    feature_temporal_policy: dict[str, Any] | None = None,
    experiment_config_sha256: str | None = None,
    protocol_status: str | None = None,
    audit_report_path: Path | None = None,
) -> dict[str, Any]:
    """Create an aligned analysis table from an explicitly mapped run manifest.

    This supports only a single aligned CSV per run. Separate endpoint and qdisc
    logs require an inspected, explicit adapter; they are never guessed/merged.
    """
    if not raw_root.is_dir():
        raise ProcessingError(f"raw root does not exist: {raw_root}")
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise ProcessingError(f"run manifest does not exist: {manifest_path}")
    manifest_sha_before = file_sha256(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProcessingError(f"cannot parse run manifest as JSON: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 2
        or not isinstance(manifest.get("runs"), list)
    ):
        raise ProcessingError(
            "run manifest must be JSON object with schema_version=2 and a runs list"
        )
    if not manifest["runs"]:
        raise ProcessingError("run manifest contains no runs")
    alignment_review = verify_alignment_review(raw_root.resolve(), manifest, manifest_path)
    run_ids = [entry.get("run_id") for entry in manifest["runs"] if isinstance(entry, dict)]
    if (
        len(run_ids) != len(manifest["runs"])
        or any(not isinstance(run_id, str) or not run_id for run_id in run_ids)
        or len(set(run_ids)) != len(run_ids)
    ):
        raise ProcessingError(
            "each manifest run must be an object with a unique non-empty string run_id"
        )
    audit_record: dict[str, Any] = {"status": "not_provided"}
    audit_files: dict[str, dict[str, Any]] = {}
    audit_sha_before: str | None = None
    if audit_report_path is not None:
        audit_path = Path(audit_report_path).resolve()
        audit_sha_before = file_sha256(audit_path)
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProcessingError(f"cannot read raw audit report: {exc}") from exc
        if not isinstance(audit, dict) or audit.get("report_schema_version") != 2:
            raise ProcessingError("raw audit report must use supported report_schema_version=2")
        if not audit.get("audit_completed"):
            raise ProcessingError("raw audit report is incomplete or contains structural errors")
        try:
            audited_root = Path(audit["input_directory"]).resolve()
            audit_items = audit["files"]
        except (KeyError, TypeError) as exc:
            raise ProcessingError("raw audit report lacks input_directory or files") from exc
        if audited_root != raw_root.resolve() or not isinstance(audit_items, list):
            raise ProcessingError(
                "raw audit report input directory does not match configured raw_root"
            )
        for item in audit_items:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise ProcessingError("raw audit report contains an invalid file record")
            if item["path"] in audit_files:
                raise ProcessingError(f"raw audit report repeats file path {item['path']!r}")
            audit_files[item["path"]] = item
        if audit.get("csv_file_count") != len(audit_files):
            raise ProcessingError("raw audit CSV file count disagrees with its detailed inventory")
        current_csv_paths = {
            path.relative_to(raw_root.resolve()).as_posix()
            for path in raw_root.resolve().rglob("*")
            if path.is_file() and path.suffix.lower() == ".csv"
        }
        if current_csv_paths != set(audit_files):
            raise ProcessingError(
                "raw CSV inventory changed since audit; rerun audit before preparation "
                f"(added={sorted(current_csv_paths - set(audit_files))}, removed={sorted(set(audit_files) - current_csv_paths)})"
            )
        for relative, audit_item in audit_files.items():
            audited_source = _safe_child(raw_root.resolve(), relative)
            if audit_item.get("status") == "error":
                raise ProcessingError(f"raw audit reports structural errors for {relative!r}")
            if (
                not isinstance(audit_item.get("sha256"), str)
                or file_sha256(audited_source) != audit_item["sha256"]
            ):
                raise ProcessingError(f"raw audit hash is stale for file {relative!r}")
        audit_record = {
            "status": "verified",
            "report_sha256": file_sha256(audit_path),
            "report_path": audit_path.name,
            "generated_at_utc": audit.get("generated_at_utc"),
            "csv_file_count": audit.get("csv_file_count"),
            "files_with_warnings": audit.get("files_with_warnings"),
            "files": [
                {
                    "path": item.get("path"),
                    "sha256": item.get("sha256"),
                    "status": item.get("status"),
                    "row_count": item.get("row_count"),
                    "issue_codes": [
                        issue.get("code")
                        for issue in item.get("issues", [])
                        if isinstance(issue, dict)
                    ],
                    "processing_sufficiency": item.get("processing_sufficiency"),
                }
                for item in audit_items
            ],
        }
    settings = [
        queue_delay_threshold_ms,
        prediction_horizon_ms,
        observation_window_ms,
        max_ground_truth_gap_ms,
    ]
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in settings):
        raise ProcessingError(
            "threshold, horizon, observation window, and max gap must be finite and positive"
        )
    if (
        isinstance(consecutive_samples_required, bool)
        or not isinstance(consecutive_samples_required, int)
        or consecutive_samples_required < 1
    ):
        raise ProcessingError("consecutive_samples_required must be an integer >= 1")
    selected_features = tuple(
        ENDPOINT_FIELDS if endpoint_feature_fields is None else endpoint_feature_fields
    )
    try:
        feature_windows = tuple(float(window) for window in feature_windows_seconds)
    except (TypeError, ValueError) as exc:
        raise ProcessingError(
            "feature_windows_seconds must contain finite positive values"
        ) from exc
    _validate_feature_contract(selected_features, feature_windows)
    temporal_policy = normalize_feature_temporal_policy(
        feature_temporal_policy, require_resolved=protocol_status == "protocol_frozen"
    )
    _preflight_tcp_unit_consistency(manifest["runs"], selected_features)
    if max(feature_windows) > observation_window_ms / 1000:
        raise ProcessingError("feature windows cannot exceed labeling.observation_window_ms")
    output_resolved = output_dir.resolve()
    try:
        output_resolved.relative_to(raw_root.resolve())
    except ValueError:
        pass
    else:
        raise ProcessingError("derived output must be outside raw_root")
    if output_resolved.exists():
        raise ProcessingError(f"refusing to overwrite existing output directory: {output_resolved}")

    expected_endpoint = set(selected_features)
    all_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    run_manifests: list[dict[str, Any]] = []
    seen_source_paths: set[str] = set()
    seen_source_hashes: dict[str, str] = {}
    for entry in manifest["runs"]:
        rows, excluded, run_record = _load_run(raw_root.resolve(), entry, expected_endpoint)
        source_path = run_record["source_file"]
        source_hash = run_record["source_sha256"]
        if source_path in seen_source_paths:
            raise ProcessingError(f"manifest reuses source file {source_path!r} for multiple runs")
        if source_hash in seen_source_hashes:
            raise ProcessingError(
                f"manifest files {source_path!r} and {seen_source_hashes[source_hash]!r} have identical content; "
                "review possible duplicated runs before analysis"
            )
        seen_source_paths.add(source_path)
        seen_source_hashes[source_hash] = source_path
        if audit_report_path is not None:
            audit_item = audit_files.get(entry["file"])
            if audit_item is None:
                raise ProcessingError(
                    f"raw audit report does not cover manifest source {entry['file']!r}"
                )
            if audit_item.get("sha256") != run_record["source_sha256"]:
                raise ProcessingError(f"raw audit hash is stale for source {entry['file']!r}")
            if audit_item.get("status") == "error":
                raise ProcessingError(
                    f"raw audit reports structural errors for source {entry['file']!r}"
                )
            _audit_timestamp_unit_compatible(
                audit,
                audit_item,
                run_record["timestamp_source_column"],
                entry["timestamp_unit"],
            )
        if not rows:
            raise ProcessingError(f"{entry['run_id']}: no rows have usable timestamps")
        event_states = derive_persistent_events(
            [row["qdisc_queue_delay_proxy_ms"] for row in rows],
            threshold_ms=queue_delay_threshold_ms,
            consecutive_samples=consecutive_samples_required,
            timestamps_seconds=[row["timestamp_seconds"] for row in rows],
            max_gap_seconds=max_ground_truth_gap_ms / 1000,
        )
        detailed = label_future_onsets_detailed(
            [row["timestamp_seconds"] for row in rows],
            event_states,
            horizon_seconds=prediction_horizon_ms / 1000,
            max_gap_seconds=max_ground_truth_gap_ms / 1000,
        )
        feature_rows = build_trailing_window_features(rows, feature_windows, selected_features)
        for i, row in enumerate(rows):
            row["event_active"] = event_states[i]
            row["future_event_onset"] = detailed[i].label
            row["label_reason"] = detailed[i].reason
            row["event_onset_source_row"] = (
                rows[detailed[i].event_onset_index]["source_row"]
                if detailed[i].event_onset_index is not None
                else None
            )
            row["label_window_ms"] = prediction_horizon_ms
            features = feature_rows[i]
            bad = [
                name
                for name in features
                if any(token in name.lower() for token in _FORBIDDEN_FEATURE_TOKENS)
            ]
            if bad:
                raise AssertionError(f"forbidden leakage feature name(s): {bad}")
            row.update(features)
            if detailed[i].label is None:
                excluded.append(
                    {
                        "run_id": row["run_id"],
                        "source_row": row["source_row"],
                        "reason": detailed[i].reason,
                        "detail": "target window censored; row retained in analysis.csv",
                    }
                )
        all_rows.extend(rows)
        exclusions.extend(excluded)
        run_manifests.append(run_record)

    tcp_retransmission_unit = _tcp_unit_from_run_records_for_preparation(
        run_manifests, selected_features
    )
    feature_specification = build_feature_specification(
        selected_features,
        feature_windows,
        tcp_retransmission_unit=tcp_retransmission_unit,
        feature_temporal_policy=temporal_policy,
    )
    feature_specification_digest = feature_specification_sha256(feature_specification)
    feature_names = list(feature_specification["feature_columns"])
    _masked_rows, temporal_classifications = apply_feature_temporal_policy_to_rows(
        all_rows, feature_names, feature_specification
    )
    # The prepared table retains descriptive trailing summaries plus the
    # diagnostics needed to reproduce the model-facing temporal mask.  The
    # row-level columns make the full-feature preparation outcome explicit;
    # downstream callers still recompute eligibility for their fitted subset.
    for row, classification in zip(all_rows, temporal_classifications, strict=True):
        row["feature_temporal_eligibility_status"] = classification["status"]
        row["feature_temporal_eligibility_reason"] = classification["reason"]
    temporal_eligibility_counts = feature_temporal_eligibility_counts(temporal_classifications)
    feature_specification_status = (
        "current_contract_complete"
        if temporal_policy["status"] == "resolved"
        else "unresolved_temporal_policy_descriptive_only"
    )
    if file_sha256(manifest_path) != manifest_sha_before:
        raise ProcessingError(
            "run manifest changed during preparation; rerun after inputs are stable"
        )
    if verify_alignment_review(raw_root.resolve(), manifest, manifest_path) != alignment_review:
        raise ProcessingError("alignment approval evidence changed during preparation")
    if audit_report_path is not None:
        if file_sha256(Path(audit_report_path).resolve()) != audit_sha_before:
            raise ProcessingError("raw audit report changed during preparation; rerun audit")
        for relative, audit_item in audit_files.items():
            if file_sha256(_safe_child(raw_root.resolve(), relative)) != audit_item["sha256"]:
                raise ProcessingError(f"raw source changed during preparation: {relative!r}")

    output_resolved.parent.mkdir(parents=True, exist_ok=True)
    output_resolved.mkdir(exist_ok=False)
    incomplete_marker = output_resolved / "PREPARATION_INCOMPLETE"
    with incomplete_marker.open("x", encoding="utf-8") as marker:
        marker.write("Preparation did not reach its final manifest step.\n")
    analysis_path = output_resolved / "analysis.csv"
    target_diagnostic_fields = [
        "queue_backlog_bytes",
        "qdisc_queue_delay_proxy_ms",
        "event_active",
        "future_event_onset",
        "label_reason",
        "event_onset_source_row",
        "label_window_ms",
    ]
    feature_names = list(feature_specification["feature_columns"])
    metadata_names = [
        "run_id",
        "configuration_id",
        "source_file",
        "source_row",
        "timestamp_original",
        "timestamp_seconds",
        "phase_observed",
        "phase_commanded",
    ]
    window_diagnostic_names = sorted(
        name
        for name in all_rows[0]
        if name.startswith("window_")
        and name.endswith(("_sample_count", "_observed_span_seconds", "_max_gap_seconds"))
    )
    endpoint_raw_names = list(ENDPOINT_FIELDS)
    qdisc_count_names = list(OPTIONAL_QDISC_COUNT_FIELDS)
    temporal_eligibility_names = [
        "feature_temporal_eligibility_status",
        "feature_temporal_eligibility_reason",
    ]
    fieldnames = (
        metadata_names
        + window_diagnostic_names
        + temporal_eligibility_names
        + endpoint_raw_names
        + qdisc_count_names
        + target_diagnostic_fields
        + feature_names
    )
    with analysis_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(
                {key: "" if row.get(key) is None else row.get(key) for key in fieldnames}
            )
    exclusion_path = output_resolved / "excluded_rows.csv"
    with exclusion_path.open("x", encoding="utf-8", newline="") as stream:
        fields = ["run_id", "source_row", "reason", "detail"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(exclusions)

    label_counts = Counter(
        str(row["future_event_onset"]) if row["future_event_onset"] is not None else "censored"
        for row in all_rows
    )
    exclusion_counts = Counter(item["reason"] for item in exclusions)
    affected_source_rows = {
        (item.get("run_id"), item.get("source_row"))
        for item in exclusions
        if item.get("source_row") is not None
    }
    rows_excluded_invalid_timestamp = len(
        {
            (item.get("run_id"), item.get("source_row"))
            for item in exclusions
            if item.get("reason") == "invalid_timestamp"
        }
    )
    retained_measurement_issue_entries = sum(
        item.get("reason") in {"invalid_endpoint_measurement", "invalid_qdisc_measurement"}
        for item in exclusions
    )
    manifest_record = {
        "processing_schema_version": 2,
        "status": "complete",
        "experiment_config_sha256": experiment_config_sha256,
        "protocol_status": protocol_status or "not_declared",
        "protocol_sha256": experiment_config_sha256
        if protocol_status == "protocol_frozen"
        else None,
        "source_manifest_sha256": file_sha256(manifest_path),
        "source_manifest_path": str(manifest_path),
        "alignment_review": alignment_review,
        "raw_audit": audit_record,
        "run_records": run_manifests,
        "label_parameters": {
            "queue_delay_threshold_ms": queue_delay_threshold_ms,
            "consecutive_samples_required": consecutive_samples_required,
            "prediction_horizon_ms": prediction_horizon_ms,
            "observation_window_ms": observation_window_ms,
            "max_ground_truth_gap_ms": max_ground_truth_gap_ms,
        },
        "feature_windows_seconds": list(feature_windows),
        "endpoint_feature_fields": list(selected_features),
        "feature_specification": feature_specification,
        "feature_specification_sha256": feature_specification_digest,
        "feature_specification_status": feature_specification_status,
        "tcp_retransmission_unit": tcp_retransmission_unit,
        "feature_temporal_policy": temporal_policy,
        "feature_temporal_eligibility_counts": temporal_eligibility_counts,
        "analysis_row_count": len(all_rows),
        "excluded_record_count": len(exclusions),
        "data_quality_log_entry_count": len(exclusions),
        "distinct_source_rows_with_log_entries": len(affected_source_rows),
        "rows_excluded_for_invalid_timestamp": rows_excluded_invalid_timestamp,
        "retained_rows_with_measurement_issue_entries": retained_measurement_issue_entries,
        "censored_label_anchor_count": label_counts.get("censored", 0),
        "exclusion_log_semantics": "timestamp-invalid rows are omitted; invalid-measurement entries identify retained rows with missing values; censored label entries identify retained anchors, not deleted records",
        "exclusion_reason_counts": dict(exclusion_counts),
        "label_counts": dict(label_counts),
        "analysis_sha256": file_sha256(analysis_path),
        "exclusions_sha256": file_sha256(exclusion_path),
        "feature_columns": feature_names,
        "forbidden_predictors": sorted(
            {
                "run_id",
                "configuration_id",
                "phase_observed",
                "phase_commanded",
                *target_diagnostic_fields,
                *endpoint_raw_names,
                *window_diagnostic_names,
                *temporal_eligibility_names,
            }
        ),
        "evidence_status": "source data supplied but experimental provenance not independently verified",
    }
    summary = {
        "run_count": len(run_manifests),
        "configuration_count": len({r["configuration_id"] for r in run_manifests}),
        "analysis_row_count": len(all_rows),
        "excluded_or_censored_count": len(exclusions),
        "data_quality_log_entry_count": len(exclusions),
        "distinct_source_rows_with_log_entries": len(affected_source_rows),
        "rows_excluded_for_invalid_timestamp": rows_excluded_invalid_timestamp,
        "retained_rows_with_measurement_issue_entries": retained_measurement_issue_entries,
        "censored_label_anchor_count": label_counts.get("censored", 0),
        "label_counts": dict(label_counts),
        "feature_columns": feature_names,
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "feature_specification_sha256": feature_specification_digest,
        "feature_specification_status": feature_specification_status,
        "tcp_retransmission_unit": tcp_retransmission_unit,
        "feature_temporal_policy": temporal_policy,
        "feature_temporal_eligibility_counts": temporal_eligibility_counts,
        "evidence_status": "source data supplied but experimental provenance not independently verified",
        "warning": "Processing does not certify that source files are measurements or scientifically valid.",
    }
    summary_path = output_resolved / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_record["summary_sha256"] = file_sha256(summary_path)
    data_dict = [
        "# Processed table dictionary",
        "",
        "`analysis.csv` is derived, never raw. `processing_manifest.json` records source hashes and parameters.",
        "",
        "| Column group | Meaning | Permitted as predictor? |",
        "|---|---|---|",
        "| run/config/source/timestamp/phase | Provenance and grouping metadata | No |",
        "| `window_*_sample_count`, `window_*_observed_span_seconds`, `window_*_max_gap_seconds` | Available trailing-window coverage diagnostics | No |",
        "| `feature_temporal_eligibility_status`, `feature_temporal_eligibility_reason` | Full-prepared-feature temporal-policy outcome for the anchor; fitted subsets are reclassified downstream | No |",
        "| endpoint raw fields | Raw endpoint measurements retained for inspection | Use derived trailing features only |",
        "| `qdisc_drops_delta`, `qdisc_overlimits_delta` | Optional per-sample count increments, only when explicitly mapped with these semantics | No |",
        "| queue backlog/delay/event/label fields | Ground truth and diagnostics | No |",
        f"| `*_latest`, `*_mean`, `*_std`, `*_slope_per_s`, `*_valid_fraction`, `*_sum` | Trailing endpoint-only summaries under `{FEATURE_SPEC_VERSION}` (SHA-256 `{feature_specification_digest}`); window encoded in suffix | Candidate features |",
        "",
        "Censored anchors are retained in `analysis.csv` and listed in `excluded_rows.csv`; they are not converted to negative labels.",
        "",
        f"TCP retransmission unit bound by the feature specification: `{tcp_retransmission_unit or 'not_applicable'}`. Packets and bytes are distinct and are never converted into one another.",
        "",
        f"Feature temporal policy status: `{temporal_policy['status']}`. Descriptive summaries remain in the table; a resolved policy is reapplied as a model-input mask during training, evaluation, registry refitting, robustness analysis, and inference. An unresolved policy is exploratory/descriptive only.",
        "",
    ]
    data_dictionary_path = output_resolved / "DATA_DICTIONARY.md"
    data_dictionary_path.write_text("\n".join(data_dict), encoding="utf-8")
    manifest_record["data_dictionary_sha256"] = file_sha256(data_dictionary_path)
    (output_resolved / "processing_manifest.json").write_text(
        json.dumps(manifest_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    incomplete_marker.unlink()
    return summary
