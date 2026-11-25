"""Explicit, review-gated alignment for separate endpoint and router CSV streams.

This module never estimates clock offset/drift. A checked alignment specification
must declare timestamp semantics, clock identities, a reviewed evidence file,
all matching policies, and (when needed) a calibrated constant offset.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import median
from typing import Any

from .processing import ENDPOINT_FIELDS, ENDPOINT_UNIT_FACTORS, OPTIONAL_QDISC_COUNT_FIELDS


class AlignmentError(ValueError):
    """An alignment specification or source stream cannot be used safely."""


_TIME_SCALE = {
    "seconds": Decimal(1),
    "milliseconds": Decimal("0.001"),
    "microseconds": Decimal("0.000001"),
    "nanoseconds": Decimal("0.000000001"),
}
_TIMESTAMP_SUFFIX = {
    "_ns": "nanoseconds",
    "_us": "microseconds",
    "_ms": "milliseconds",
    "_seconds": "seconds",
    "_second": "seconds",
    "_s": "seconds",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _require_keys(value: Any, required: set[str], allowed: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AlignmentError(f"{label} must be a JSON object")
    missing, extra = required - set(value), set(value) - allowed
    if missing or extra:
        raise AlignmentError(
            f"{label} keys invalid (missing={sorted(missing)}, unexpected={sorted(extra)})"
        )
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AlignmentError(f"{label} must be a non-empty string")
    return value.strip()


def _decimal(value: Any, label: str, *, minimum: Decimal | None = None) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise AlignmentError(f"{label} must be a finite number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise AlignmentError(f"{label} must be a finite number") from exc
    if not result.is_finite() or (minimum is not None and result < minimum):
        raise AlignmentError(
            f"{label} must be finite" + (f" and >= {minimum}" if minimum is not None else "")
        )
    return result


def _relative_file(root: Path, value: Any, label: str) -> Path:
    relative = _text(value, label)
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise AlignmentError(f"{label} must be a relative path within raw-root")
    probe = root
    for part in candidate.parts:
        probe = probe / part
        if probe.is_symlink():
            raise AlignmentError(f"{label} must not traverse symlinks")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise AlignmentError(f"{label} escapes raw-root") from exc
    if not resolved.is_file():
        raise AlignmentError(f"{label} does not exist: {relative}")
    if resolved.is_symlink():
        raise AlignmentError(f"{label} must not be a symlink")
    return resolved


def _unit_suffix(column: str) -> str | None:
    name = column.strip().lower()
    for suffix, unit in _TIMESTAMP_SUFFIX.items():
        if name.endswith(suffix):
            return unit
    return None


def _iso_epoch_seconds(value: str, label: str) -> Decimal:
    text = value.strip()
    match = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|z|[+-]\d{2}:?\d{2})",
        text,
    )
    if match is None:
        raise AlignmentError(f"{label} must be ISO-8601 with an explicit timezone offset")
    base, fraction, offset = match.groups()
    if offset.lower() == "z":
        offset = "+00:00"
    elif len(offset) == 5 and ":" not in offset:
        offset = offset[:3] + ":" + offset[3:]
    try:
        parsed = datetime.fromisoformat(base + offset)
    except ValueError as exc:
        raise AlignmentError(f"{label} is not a valid timezone-bearing ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AlignmentError(
            f"{label} is timezone-naive; explicit timezone evidence/conversion is required"
        )
    utc = parsed.astimezone(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = utc - epoch
    whole_seconds = Decimal(delta.days * 86400 + delta.seconds)
    fractional_seconds = Decimal("0." + fraction) if fraction else Decimal(0)
    return whole_seconds + fractional_seconds


def _parse_time(value: str | None, stream: dict[str, Any], row_number: int) -> Decimal:
    label = f"{stream['name']} timestamp at CSV record {row_number}"
    if value is None or not value.strip():
        raise AlignmentError(f"{label} is missing")
    if stream["timestamp_format"] == "iso8601_offset":
        return _iso_epoch_seconds(value, label)
    try:
        numeric = Decimal(value.strip())
    except InvalidOperation as exc:
        raise AlignmentError(f"{label} is not numeric") from exc
    if not numeric.is_finite():
        raise AlignmentError(f"{label} is non-finite")
    return numeric * _TIME_SCALE[stream["timestamp_unit"]]


def _read_stream(
    root: Path,
    spec: dict[str, Any],
    name: str,
    *,
    collect_rows: bool = True,
    allow_empty: bool = False,
) -> dict[str, Any]:
    required = {"file", "timestamp_column", "timestamp_format", "timestamp_unit", "fields"}
    stream = _require_keys(spec, required, required, f"streams.{name}")
    stream = dict(stream)
    stream["name"] = name
    stream["file"] = _text(stream["file"], f"streams.{name}.file")
    stream["path"] = _relative_file(root, stream["file"], f"streams.{name}.file")
    stream["timestamp_column"] = _text(
        stream["timestamp_column"], f"streams.{name}.timestamp_column"
    )
    fmt = stream["timestamp_format"]
    if not isinstance(fmt, str) or fmt not in {"numeric", "iso8601_offset"}:
        raise AlignmentError(f"streams.{name}.timestamp_format must be numeric or iso8601_offset")
    unit = stream["timestamp_unit"]
    if fmt == "numeric" and (not isinstance(unit, str) or unit not in _TIME_SCALE):
        raise AlignmentError(
            f"streams.{name}.timestamp_unit must explicitly name seconds/milliseconds/microseconds/nanoseconds"
        )
    if fmt == "iso8601_offset" and unit != "seconds":
        raise AlignmentError(
            f"streams.{name}: offset-bearing ISO timestamps must declare timestamp_unit='seconds'"
        )
    suffix_unit = _unit_suffix(stream["timestamp_column"])
    if suffix_unit and suffix_unit != unit:
        raise AlignmentError(
            f"streams.{name}: timestamp unit conflicts with header suffix {suffix_unit!r}"
        )
    fields = stream["fields"]
    if not isinstance(fields, dict) or not fields:
        raise AlignmentError(f"streams.{name}.fields must be a non-empty mapping")
    used_columns = [stream["timestamp_column"]]
    normalized_fields: dict[str, dict[str, Any]] = {}
    if name == "endpoint":
        allowed_canonical = set(ENDPOINT_FIELDS) | {"phase_observed", "phase_commanded"}
    else:
        allowed_canonical = set(OPTIONAL_QDISC_COUNT_FIELDS) | {
            "queue_backlog_bytes",
            "phase_observed",
            "phase_commanded",
        }
    unknown = set(fields) - allowed_canonical
    if unknown:
        raise AlignmentError(
            f"streams.{name}.fields has unsupported canonical fields: {sorted(unknown)}"
        )
    for canonical, mapping in fields.items():
        mapping = _require_keys(
            mapping, {"column", "unit"}, {"column", "unit"}, f"streams.{name}.fields.{canonical}"
        )
        column = _text(mapping["column"], f"streams.{name}.fields.{canonical}.column")
        if column in used_columns:
            raise AlignmentError(
                f"streams.{name}: source column {column!r} is mapped more than once"
            )
        used_columns.append(column)
        unit_value = mapping["unit"]
        if canonical in ENDPOINT_FIELDS:
            if (
                not isinstance(unit_value, str)
                or unit_value not in ENDPOINT_UNIT_FACTORS[canonical]
            ):
                raise AlignmentError(f"unsupported source unit {unit_value!r} for {canonical}")
        elif canonical == "queue_backlog_bytes" and unit_value != "bytes":
            raise AlignmentError(
                "queue_backlog_bytes source unit must be explicitly declared 'bytes'"
            )
        elif canonical in OPTIONAL_QDISC_COUNT_FIELDS and unit_value != "count":
            raise AlignmentError(f"{canonical} source unit must be explicitly declared 'count'")
        elif canonical.startswith("phase_") and unit_value is not None:
            raise AlignmentError(f"{canonical} is metadata and must use unit: null")
        normalized_fields[canonical] = {"column": column, "unit": unit_value}
    if name == "endpoint" and not (set(normalized_fields) & set(ENDPOINT_FIELDS)):
        raise AlignmentError("endpoint stream must map at least one canonical endpoint feature")
    if name == "router" and "queue_backlog_bytes" not in normalized_fields:
        raise AlignmentError("router stream must map queue_backlog_bytes")
    stream["fields"] = normalized_fields
    try:
        before = _sha256(stream["path"])
        with stream["path"].open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader, None)
            if not header:
                raise AlignmentError(f"{name} CSV has no header: {stream['file']}")
            if len(header) != len(set(header)):
                raise AlignmentError(f"{name} CSV has duplicate header names")
            absent = sorted(set(used_columns) - set(header))
            if absent:
                raise AlignmentError(f"{name} CSV missing mapped columns: {absent}")
            rows = []
            row_count = 0
            for record_number, values in enumerate(reader, start=2):
                if len(values) != len(header):
                    raise AlignmentError(
                        f"{name} CSV record {record_number} has {len(values)} fields; expected {len(header)}"
                    )
                row_count += 1
                if collect_rows:
                    rows.append({"source_row_number": record_number, **dict(zip(header, values))})
        after = _sha256(stream["path"])
    except (OSError, UnicodeError, csv.Error, StopIteration) as exc:
        raise AlignmentError(f"cannot read {name} stream {stream['file']}: {exc}") from exc
    if before != after:
        raise AlignmentError(f"{name} source changed while being read")
    if row_count == 0 and not allow_empty:
        raise AlignmentError(f"{name} stream contains no data records")
    stream["source_sha256"] = after
    stream["source_row_count"] = row_count
    stream["headers"] = header
    stream["rows"] = rows
    return stream


def _review_clock(
    root: Path,
    review_spec: Any,
    endpoint: dict[str, Any],
    router: dict[str, Any],
    tolerance: Decimal,
) -> dict[str, Any]:
    required = {
        "relation",
        "endpoint_clock_id",
        "router_clock_id",
        "timestamp_semantics",
        "router_offset_to_endpoint_seconds",
        "residual_uncertainty_seconds",
        "max_unmodeled_drift_seconds",
        "reviewed_by",
        "reviewed_at_utc",
        "evidence_file",
        "evidence_sha256",
    }
    review = _require_keys(review_spec, required, required, "clock_review")
    relation = review["relation"]
    if not isinstance(relation, str) or relation not in {"same_clock", "calibrated_fixed_offset"}:
        raise AlignmentError("clock_review.relation must be same_clock or calibrated_fixed_offset")
    endpoint_id = _text(review["endpoint_clock_id"], "clock_review.endpoint_clock_id")
    router_id = _text(review["router_clock_id"], "clock_review.router_clock_id")
    semantics = review["timestamp_semantics"]
    if not isinstance(semantics, str) or semantics not in {"unix_epoch", "shared_monotonic"}:
        raise AlignmentError(
            "clock_review.timestamp_semantics must be unix_epoch or shared_monotonic"
        )
    if semantics == "shared_monotonic" and (
        endpoint["timestamp_format"] != "numeric" or router["timestamp_format"] != "numeric"
    ):
        raise AlignmentError(
            "shared_monotonic semantics require numeric timestamps in both streams"
        )
    offset = _decimal(
        review["router_offset_to_endpoint_seconds"],
        "clock_review.router_offset_to_endpoint_seconds",
    )
    uncertainty = _decimal(
        review["residual_uncertainty_seconds"],
        "clock_review.residual_uncertainty_seconds",
        minimum=Decimal(0),
    )
    drift = _decimal(
        review["max_unmodeled_drift_seconds"],
        "clock_review.max_unmodeled_drift_seconds",
        minimum=Decimal(0),
    )
    if uncertainty + drift > tolerance:
        raise AlignmentError(
            "residual clock uncertainty plus unmodeled drift exceeds max_tolerance_seconds"
        )
    if relation == "same_clock" and (endpoint_id != router_id or offset != 0):
        raise AlignmentError("same_clock requires identical clock IDs and a zero router offset")
    if relation == "calibrated_fixed_offset" and endpoint_id == router_id and offset == 0:
        raise AlignmentError(
            "calibrated_fixed_offset requires distinct clock IDs or a non-zero calibrated offset"
        )
    if not isinstance(review["reviewed_at_utc"], str):
        raise AlignmentError(
            "clock_review.reviewed_at_utc must be a timezone-bearing ISO timestamp"
        )
    try:
        reviewed_at = datetime.fromisoformat(review["reviewed_at_utc"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise AlignmentError("clock_review.reviewed_at_utc is invalid") from exc
    if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
        raise AlignmentError("clock_review.reviewed_at_utc must include a timezone")
    reviewer = _text(review["reviewed_by"], "clock_review.reviewed_by")
    evidence_path = _relative_file(root, review["evidence_file"], "clock_review.evidence_file")
    expected_hash = _text(review["evidence_sha256"], "clock_review.evidence_sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise AlignmentError("clock_review.evidence_sha256 must be a lowercase SHA-256 digest")
    actual_hash = _sha256(evidence_path)
    if actual_hash != expected_hash:
        raise AlignmentError("clock review evidence hash does not match the cited file")
    return {
        "relation": relation,
        "endpoint_clock_id": endpoint_id,
        "router_clock_id": router_id,
        "timestamp_semantics": semantics,
        "router_offset_to_endpoint_seconds": str(offset),
        "residual_uncertainty_seconds": str(uncertainty),
        "max_unmodeled_drift_seconds": str(drift),
        "reviewed_by": reviewer,
        "reviewed_at_utc": reviewed_at.isoformat(),
        "evidence_file": str(review["evidence_file"]),
        "evidence_sha256": actual_hash,
        "scientific_approval_status": "not_asserted_by_software",
    }


def _timestamp_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [records[i]["time"] - records[i - 1]["time"] for i in range(1, len(records))]
    positive = sorted(delta for delta in deltas if delta > 0)
    duplicates = sum(delta == 0 for delta in deltas)
    duplicate_timestamp_values = len(records) - len({item["time"] for item in records})
    out_of_order = sum(delta < 0 for delta in deltas)
    if positive:
        middle = len(positive) // 2
        cadence = (
            positive[middle] if len(positive) % 2 else (positive[middle - 1] + positive[middle]) / 2
        )
        gaps = sum(delta > cadence * Decimal("1.5") for delta in positive)
    else:
        cadence, gaps = None, None
    return {
        "valid_timestamp_count": len(records),
        "range_seconds": {
            "minimum": str(min(item["time"] for item in records)),
            "maximum": str(max(item["time"] for item in records)),
        },
        "adjacent_duplicate_count": duplicates,
        "duplicate_timestamp_value_count": duplicate_timestamp_values,
        "out_of_order_adjacent_count": out_of_order,
        "positive_interval_count": len(positive),
        "median_positive_cadence_seconds": str(cadence) if cadence is not None else None,
        "positive_intervals_over_1_5x_median": gaps,
        "gap_rule": "descriptive 1.5x-median heuristic only; not an accepted protocol threshold",
    }


def _prepare_records(
    stream: dict[str, Any],
    offset: Decimal,
    *,
    invalid_policy: str,
    ordering_policy: str,
    duplicate_policy: str,
    exclusions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parsed = []
    invalid_rows = []
    for row in stream["rows"]:
        try:
            raw_time = row[stream["timestamp_column"]]
            corrected = _parse_time(raw_time, stream, row["source_row_number"]) + offset
            parsed.append(
                {
                    "raw": row,
                    "raw_timestamp": raw_time,
                    "time": corrected,
                    "source_row_number": row["source_row_number"],
                }
            )
        except AlignmentError as exc:
            invalid_rows.append(row["source_row_number"])
            if invalid_policy == "error":
                raise
            exclusions.append(
                {
                    "stream": stream["name"],
                    "source_row_number": row["source_row_number"],
                    "reason": "invalid_timestamp",
                    "detail": str(exc),
                }
            )
    diagnostics = (
        _timestamp_diagnostics(parsed)
        if parsed
        else {
            "valid_timestamp_count": 0,
            "range_seconds": None,
            "adjacent_duplicate_count": 0,
            "duplicate_timestamp_value_count": 0,
            "out_of_order_adjacent_count": 0,
            "positive_interval_count": 0,
            "median_positive_cadence_seconds": None,
            "positive_intervals_over_1_5x_median": None,
        }
    )
    if diagnostics["out_of_order_adjacent_count"] and ordering_policy == "error":
        raise AlignmentError(
            f"{stream['name']} timestamp sequence is out of order; choose sort_and_report only after review"
        )
    if duplicate_policy == "error" and diagnostics["adjacent_duplicate_count"]:
        raise AlignmentError(f"{stream['name']} contains duplicate adjacent timestamps")
    # Group all equal corrected timestamp values, not only adjacent source rows.
    grouped: dict[Decimal, list[dict[str, Any]]] = defaultdict(list)
    for item in parsed:
        grouped[item["time"]].append(item)
    deduplicated = []
    duplicate_dropped_rows = []
    for timestamp, group in grouped.items():
        group.sort(key=lambda item: item["source_row_number"])
        if len(group) > 1:
            if duplicate_policy == "error":
                raise AlignmentError(
                    f"{stream['name']} has {len(group)} records at corrected timestamp {timestamp}"
                )
            chosen = group[0] if duplicate_policy == "keep_first" else group[-1]
            for removed in group:
                if removed is not chosen:
                    duplicate_dropped_rows.append(
                        {
                            "stream": stream["name"],
                            "source_row_number": removed["source_row_number"],
                            "reason": "duplicate_timestamp_dropped",
                            "kept_source_row_number": chosen["source_row_number"],
                        }
                    )
                    exclusions.append(duplicate_dropped_rows[-1])
            deduplicated.append(chosen)
        else:
            deduplicated.append(group[0])
    deduplicated.sort(key=lambda item: (item["time"], item["source_row_number"]))
    diagnostics["invalid_timestamp_row_numbers"] = invalid_rows
    diagnostics["duplicate_dropped_rows"] = duplicate_dropped_rows
    diagnostics["records_after_duplicate_policy"] = len(deduplicated)
    diagnostics["source_ordering_policy"] = ordering_policy
    diagnostics["duplicate_policy"] = duplicate_policy
    return deduplicated, diagnostics


def _nearest_matches(
    endpoint: list[dict[str, Any]], router: list[dict[str, Any]], tolerance: Decimal, strategy: str
) -> tuple[list[int | None], set[int]]:
    router_times = [item["time"] for item in router]
    matches: list[int | None] = [None] * len(endpoint)
    used: set[int] = set()

    if strategy == "backward_asof_reuse":
        # Causal as-of matching: only use the latest router observation
        # whose corrected timestamp is <= the endpoint timestamp.
        # A router observation may be reused by multiple endpoint rows.
        for i, record in enumerate(endpoint):
            pos = bisect.bisect_right(router_times, record["time"]) - 1
            if pos < 0:
                continue

            age = record["time"] - router[pos]["time"]
            if age <= tolerance:
                matches[i] = pos
                used.add(pos)

        return matches, used

    if strategy == "nearest_reuse":
        for i, record in enumerate(endpoint):
            pos = bisect.bisect_left(router_times, record["time"])
            candidates = [j for j in (pos - 1, pos) if 0 <= j < len(router)]
            if not candidates:
                continue
            best = min(
                candidates,
                key=lambda j: (
                    abs(router[j]["time"] - record["time"]),
                    router[j]["time"],
                    router[j]["source_row_number"],
                ),
            )
            if abs(router[best]["time"] - record["time"]) <= tolerance:
                matches[i] = best
                used.add(best)
        return matches, used
    # Union-find successor/predecessor structures make nearest-unused lookup O(alpha(n)).
    n = len(router)
    next_parent = list(range(n + 1))
    prev_parent = list(range(n + 1))

    def next_free(index: int) -> int:
        if index >= n:
            return n
        while next_parent[index] != index:
            next_parent[index] = next_parent[next_parent[index]]
            index = next_parent[index]
        return index

    def prev_free(index: int) -> int:
        if index < 0:
            return -1
        while prev_parent[index] != index:
            prev_parent[index] = prev_parent[prev_parent[index]]
            index = prev_parent[index]
        return index

    for i, record in enumerate(endpoint):
        pos = bisect.bisect_left(router_times, record["time"])
        right, left = next_free(pos), prev_free(pos - 1)
        candidates = [index for index in (left, right) if 0 <= index < n]
        if not candidates:
            continue
        best = min(
            candidates,
            key=lambda j: (
                abs(router[j]["time"] - record["time"]),
                router[j]["time"],
                router[j]["source_row_number"],
            ),
        )
        if abs(router[best]["time"] - record["time"]) <= tolerance:
            matches[i] = best
            used.add(best)
            next_parent[best] = next_free(best + 1)
            prev_parent[best] = prev_free(best - 1)
    return matches, used


def _format_decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _safe_stem(index: int) -> str:
    return f"run-{index:04d}"


def _validate_spec(
    root: Path, spec_path: Path, spec: Any
) -> tuple[dict[str, Any], list[dict[str, Any]], Decimal]:
    required = {"schema_version", "alignment_policy", "runs"}
    top = _require_keys(spec, required, required, "alignment spec")
    if (
        isinstance(top["schema_version"], bool)
        or not isinstance(top["schema_version"], int)
        or top["schema_version"] != 1
    ):
        raise AlignmentError("alignment spec schema_version must be integer 1")
    policy = _require_keys(
        top["alignment_policy"],
        {
            "max_tolerance_seconds",
            "matching_strategy",
            "endpoint_duplicate_policy",
            "router_duplicate_policy",
            "invalid_timestamp_policy",
            "out_of_order_policy",
            "unmatched_endpoint_policy",
            "unmatched_router_policy",
            "missing_measurements_policy",
        },
        {
            "max_tolerance_seconds",
            "matching_strategy",
            "endpoint_duplicate_policy",
            "router_duplicate_policy",
            "invalid_timestamp_policy",
            "out_of_order_policy",
            "unmatched_endpoint_policy",
            "unmatched_router_policy",
            "missing_measurements_policy",
        },
        "alignment_policy",
    )
    tolerance = _decimal(
        policy["max_tolerance_seconds"], "max_tolerance_seconds", minimum=Decimal(0)
    )
    if tolerance == 0:
        raise AlignmentError("max_tolerance_seconds must be positive")
    enumerations = {
        "matching_strategy": {
            "nearest_one_to_one",
            "nearest_reuse",
            "backward_asof_reuse",
        },
        "endpoint_duplicate_policy": {"error", "keep_first", "keep_last"},
        "router_duplicate_policy": {"error", "keep_first", "keep_last"},
        "invalid_timestamp_policy": {"error", "exclude_and_report"},
        "out_of_order_policy": {"error", "sort_and_report"},
        "unmatched_endpoint_policy": {"error", "keep_nulls", "drop"},
        "unmatched_router_policy": {"error", "report_only"},
        "missing_measurements_policy": {
            "preserve_empty",
            "drop_endpoint_if_any_mapped_field_blank",
        },
    }
    for name, allowed in enumerations.items():
        if not isinstance(policy[name], str) or policy[name] not in allowed:
            raise AlignmentError(f"alignment_policy.{name} must be one of {sorted(allowed)}")
    runs = top["runs"]
    if not isinstance(runs, list) or not runs:
        raise AlignmentError("runs must be a non-empty list")
    normalized_runs = []
    ids = set()
    cross_run_units: dict[str, set[str]] = defaultdict(set)
    for index, raw_run in enumerate(runs, start=1):
        label = f"runs[{index - 1}]"
        required_run = {
            "run_id",
            "configuration_id",
            "bottleneck_rate_bps",
            "streams",
            "clock_review",
        }
        run = _require_keys(raw_run, required_run, required_run, label)
        run_id, configuration_id = (
            _text(run["run_id"], f"{label}.run_id"),
            _text(run["configuration_id"], f"{label}.configuration_id"),
        )
        if run_id in ids:
            raise AlignmentError(f"duplicate run_id {run_id!r}")
        ids.add(run_id)
        rate = _decimal(
            run["bottleneck_rate_bps"], f"{label}.bottleneck_rate_bps", minimum=Decimal(0)
        )
        try:
            rate_float = float(rate)
        except OverflowError as exc:
            raise AlignmentError(
                f"{label}.bottleneck_rate_bps is outside the preparation pipeline numeric range"
            ) from exc
        if rate == 0 or not math.isfinite(rate_float):
            raise AlignmentError(
                f"{label}.bottleneck_rate_bps must be a finite positive number representable by the preparation pipeline"
            )
        streams_spec = _require_keys(
            run["streams"], {"endpoint", "router"}, {"endpoint", "router"}, f"{label}.streams"
        )
        endpoint = _read_stream(root, streams_spec["endpoint"], "endpoint", collect_rows=False)
        router = _read_stream(
            root,
            streams_spec["router"],
            "router",
            collect_rows=False,
            allow_empty=policy["matching_strategy"] == "backward_asof_reuse",
        )
        if endpoint["path"] == router["path"]:
            raise AlignmentError(f"{label}: endpoint and router streams must be distinct files")
        offset = Decimal(0)
        review = _review_clock(root, run["clock_review"], endpoint, router, tolerance)
        if (
            endpoint["timestamp_format"] != router["timestamp_format"]
            and review["timestamp_semantics"] != "unix_epoch"
        ):
            raise AlignmentError(
                "mixed timestamp formats are supported only with reviewed unix_epoch semantics"
            )
        if review["timestamp_semantics"] == "unix_epoch" and (
            endpoint["timestamp_format"] == "numeric"
            and endpoint["timestamp_unit"] not in _TIME_SCALE
            or router["timestamp_format"] == "numeric"
            and router["timestamp_unit"] not in _TIME_SCALE
        ):
            raise AlignmentError(
                "epoch numeric timestamps require an explicit unit for each numeric stream"
            )
        if review["relation"] == "calibrated_fixed_offset":
            offset = Decimal(review["router_offset_to_endpoint_seconds"])
        for stream in (endpoint, router):
            for canonical, mapping in stream["fields"].items():
                if mapping["unit"] is not None:
                    cross_run_units[canonical].add(mapping["unit"])
        normalized_runs.append(
            {
                "run_id": run_id,
                "configuration_id": configuration_id,
                "bottleneck_rate_bps": rate,
                "endpoint": endpoint,
                "router": router,
                "streams_spec": streams_spec,
                "clock_review": review,
                "router_offset_seconds": offset,
            }
        )
    if len(cross_run_units.get("tcp_retransmission_delta", set())) > 1:
        raise AlignmentError(
            "tcp_retransmission_delta units must agree across runs (packets and bytes are not interchangeable)"
        )
    return policy, normalized_runs, tolerance


def align_streams(*, raw_root: Path, spec_path: Path, output_dir: Path) -> dict[str, Any]:
    """Align explicitly reviewed endpoint/router CSV streams into v2 aligned-run inputs."""
    root = raw_root.resolve()
    if not root.is_dir():
        raise AlignmentError(f"raw-root does not exist or is not a directory: {root}")
    if raw_root.is_symlink():
        raise AlignmentError("raw-root itself must not be a symlink")
    if spec_path.is_symlink():
        raise AlignmentError("alignment specification must not be a symlink")
    spec_file = spec_path.resolve()
    if not spec_file.is_file():
        raise AlignmentError(f"alignment specification does not exist: {spec_file}")
    output = output_dir.resolve()
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise AlignmentError("alignment output directory must be outside raw-root")
    if output.exists():
        raise AlignmentError(f"refusing to overwrite existing output directory: {output}")
    spec_hash_before = _sha256(spec_file)
    try:
        spec = json.loads(spec_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AlignmentError(f"cannot read alignment specification JSON: {exc}") from exc
    policy, runs, tolerance = _validate_spec(root, spec_file, spec)
    if _sha256(spec_file) != spec_hash_before:
        raise AlignmentError("alignment specification changed while it was read")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    marker = output / "ALIGNMENT_INCOMPLETE"
    marker.write_text("Alignment did not reach final manifest creation.\n", encoding="utf-8")
    provenance_dir = output / "provenance"
    exclusions_dir = output / "exclusions"
    aligned_dir = output / "aligned"
    for directory in (provenance_dir, exclusions_dir, aligned_dir):
        directory.mkdir()
    manifest_runs = []
    run_reports = []
    output_hashes = {}
    for run_index, run in enumerate(runs, start=1):
        stem = _safe_stem(run_index)
        endpoint_data = _read_stream(
            root, run["streams_spec"]["endpoint"], "endpoint", collect_rows=True
        )
        router_data = _read_stream(
            root,
            run["streams_spec"]["router"],
            "router",
            collect_rows=True,
            allow_empty=policy["matching_strategy"] == "backward_asof_reuse",
        )
        for stream_name, planned, current in (
            ("endpoint", run["endpoint"], endpoint_data),
            ("router", run["router"], router_data),
        ):
            if current["source_sha256"] != planned["source_sha256"]:
                raise AlignmentError(
                    f"raw {stream_name} source changed after alignment preflight: {planned['file']}"
                )
        endpoint_records, endpoint_diag = _prepare_records(
            endpoint_data,
            Decimal(0),
            invalid_policy=policy["invalid_timestamp_policy"],
            ordering_policy=policy["out_of_order_policy"],
            duplicate_policy=policy["endpoint_duplicate_policy"],
            exclusions=[],
        )
        router_records, router_diag = _prepare_records(
            router_data,
            run["router_offset_seconds"],
            invalid_policy=policy["invalid_timestamp_policy"],
            ordering_policy=policy["out_of_order_policy"],
            duplicate_policy=policy["router_duplicate_policy"],
            exclusions=[],
        )
        if not endpoint_records:
            raise AlignmentError(
                f"run {run['run_id']}: endpoint stream has no valid timestamps after the declared policy"
            )
        matches, used_router = _nearest_matches(
            endpoint_records, router_records, tolerance, policy["matching_strategy"]
        )
        if policy["unmatched_endpoint_policy"] == "error" and any(
            value is None for value in matches
        ):
            raise AlignmentError(
                f"run {run['run_id']}: endpoint samples are unmatched within the declared tolerance"
            )
        timestamp_origin = endpoint_records[0]["time"]
        output_fields = ["timestamp_seconds"]
        canonical_fields = []
        units: dict[str, str] = {}
        column_map = {"timestamp": "timestamp_seconds"}
        for stream_name in ("endpoint", "router"):
            stream = run[stream_name]
            for canonical, mapping in stream["fields"].items():
                if canonical in canonical_fields:
                    raise AlignmentError(
                        f"run {run['run_id']}: canonical field mapped twice: {canonical}"
                    )
                canonical_fields.append(canonical)
                output_fields.append(canonical)
                column_map[canonical] = canonical
                if mapping["unit"] is not None:
                    units[canonical] = mapping["unit"]
        aligned_path = aligned_dir / f"{stem}.csv"
        provenance_path = provenance_dir / f"{stem}.jsonl"
        exclusions_path = exclusions_dir / f"{stem}.jsonl"
        previous_output_float: float | None = None
        exclusions: list[dict[str, Any]] = []
        for stream_name, diagnostics in (("endpoint", endpoint_diag), ("router", router_diag)):
            for source_row_number in diagnostics.get("invalid_timestamp_row_numbers", []):
                exclusions.append(
                    {
                        "stream": stream_name,
                        "source_row_number": source_row_number,
                        "reason": "invalid_timestamp",
                    }
                )
            exclusions.extend(diagnostics.get("duplicate_dropped_rows", []))
        used_router_indices = set(used_router)
        unmatched_endpoint_count = 0
        missing_measurement_count = 0
        matched_deltas: list[Decimal] = [
            abs(router_records[router_index]["time"] - endpoint_record["time"])
            for endpoint_record, router_index in zip(endpoint_records, matches)
            if router_index is not None
        ]
        matched_endpoint_count = sum(router_index is not None for router_index in matches)
        unmatched_router = [
            record
            for index, record in enumerate(router_records)
            if index not in used_router_indices
        ]
        if unmatched_router and policy["unmatched_router_policy"] == "error":
            raise AlignmentError(
                f"run {run['run_id']}: router samples remain unmatched after alignment"
            )
        for record in unmatched_router:
            exclusions.append(
                {
                    "stream": "router",
                    "source_row_number": record["source_row_number"],
                    "reason": "unmatched_router_sample",
                }
            )
        output_row_count = 0
        matched_output_count = 0
        with (
            aligned_path.open("x", encoding="utf-8", newline="") as aligned_handle,
            provenance_path.open("x", encoding="utf-8") as provenance_handle,
        ):
            writer = csv.DictWriter(aligned_handle, fieldnames=output_fields, extrasaction="raise")
            writer.writeheader()
            for endpoint_record, router_index in zip(endpoint_records, matches):
                router_record = router_records[router_index] if router_index is not None else None
                if router_record is None:
                    unmatched_endpoint_count += 1
                    if policy["unmatched_endpoint_policy"] == "drop":
                        exclusions.append(
                            {
                                "stream": "endpoint",
                                "source_row_number": endpoint_record["source_row_number"],
                                "reason": "unmatched_endpoint_sample",
                            }
                        )
                        continue
                fields = {}
                for canonical, mapping in run["endpoint"]["fields"].items():
                    fields[canonical] = endpoint_record["raw"][mapping["column"]]
                if router_record is not None:
                    for canonical, mapping in run["router"]["fields"].items():
                        fields[canonical] = router_record["raw"][mapping["column"]]
                else:
                    for canonical in run["router"]["fields"]:
                        fields[canonical] = ""
                missing = [
                    field
                    for field, value in fields.items()
                    if not field.startswith("phase_") and (value is None or not value.strip())
                ]
                if missing:
                    missing_measurement_count += 1
                    if (
                        policy["missing_measurements_policy"]
                        == "drop_endpoint_if_any_mapped_field_blank"
                    ):
                        exclusions.append(
                            {
                                "stream": "aligned",
                                "endpoint_source_row_number": endpoint_record["source_row_number"],
                                "router_source_row_number": router_record["source_row_number"]
                                if router_record
                                else None,
                                "reason": "mapped_measurement_blank",
                                "fields": missing,
                            }
                        )
                        continue
                delta = (
                    None
                    if router_record is None
                    else router_record["time"] - endpoint_record["time"]
                )
                relative_time = endpoint_record["time"] - timestamp_origin
                relative_float = float(relative_time)
                if not math.isfinite(relative_float) or (
                    previous_output_float is not None and relative_float <= previous_output_float
                ):
                    raise AlignmentError(
                        f"run {run['run_id']}: timestamp precision cannot be preserved as seconds in the downstream processor"
                    )
                previous_output_float = relative_float
                aligned_row = {"timestamp_seconds": _format_decimal(relative_time), **fields}
                writer.writerow(aligned_row)
                entry = {
                    "aligned_data_row_index": output_row_count,
                    "aligned_csv_record_number": output_row_count + 2,
                    "endpoint_source_file": run["endpoint"]["file"],
                    "endpoint_source_sha256": run["endpoint"]["source_sha256"],
                    "endpoint_source_row_number": endpoint_record["source_row_number"],
                    "endpoint_timestamp_raw": endpoint_record["raw_timestamp"],
                    "endpoint_timestamp_seconds_after_correction": _format_decimal(
                        endpoint_record["time"]
                    ),
                    "aligned_timestamp_seconds_since_run_origin": aligned_row["timestamp_seconds"],
                    "router_source_file": run["router"]["file"] if router_record else None,
                    "router_source_sha256": run["router"]["source_sha256"]
                    if router_record
                    else None,
                    "router_source_row_number": router_record["source_row_number"]
                    if router_record
                    else None,
                    "router_timestamp_raw": router_record["raw_timestamp"]
                    if router_record
                    else None,
                    "router_timestamp_seconds_after_correction": _format_decimal(
                        router_record["time"]
                    )
                    if router_record
                    else None,
                    "router_minus_endpoint_delta_seconds": _format_decimal(delta)
                    if delta is not None
                    else None,
                    "router_sample_age_seconds": (
                        _format_decimal(endpoint_record["time"] - router_record["time"])
                        if (
                            router_record is not None
                            and policy["matching_strategy"] == "backward_asof_reuse"
                        )
                        else None
                    ),
                    "missing_mapped_fields": missing,
                    "matching_strategy": policy["matching_strategy"],
                    "max_tolerance_seconds": str(tolerance),
                }
                provenance_handle.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
                output_row_count += 1
                matched_output_count += int(router_record is not None)
        if output_row_count == 0:
            raise AlignmentError(
                f"run {run['run_id']}: declared policies exclude every endpoint record"
            )
        with exclusions_path.open("x", encoding="utf-8") as exclusions_handle:
            for entry in exclusions:
                exclusions_handle.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
        aligned_hash = _sha256(aligned_path)
        provenance_hash = _sha256(provenance_path)
        exclusions_hash = _sha256(exclusions_path)
        output_hashes[aligned_path.relative_to(output).as_posix()] = aligned_hash
        output_hashes[provenance_path.relative_to(output).as_posix()] = provenance_hash
        output_hashes[exclusions_path.relative_to(output).as_posix()] = exclusions_hash
        metadata = {
            "alignment_spec_sha256": spec_hash_before,
            "aligned_data_sha256": aligned_hash,
            "alignment_provenance_file": provenance_path.relative_to(output).as_posix(),
            "alignment_provenance_sha256": provenance_hash,
            "alignment_exclusions_file": exclusions_path.relative_to(output).as_posix(),
            "alignment_exclusions_sha256": exclusions_hash,
            "source_streams": {
                "endpoint": {
                    "file": run["endpoint"]["file"],
                    "sha256": run["endpoint"]["source_sha256"],
                },
                "router": {"file": run["router"]["file"], "sha256": run["router"]["source_sha256"]},
            },
            "clock_review": run["clock_review"],
            "timestamp_transform": {
                "output_unit": "seconds",
                "output_origin_seconds_in_endpoint_reference_clock": str(timestamp_origin),
                "output_value": "corrected endpoint timestamp minus earliest valid corrected endpoint timestamp for this run",
            },
            "alignment_status": "mechanically_aligned_under_declared_policy; scientific validity not established",
        }
        manifest_runs.append(
            {
                "run_id": run["run_id"],
                "configuration_id": run["configuration_id"],
                "file": aligned_path.relative_to(output).as_posix(),
                "bottleneck_rate_bps": float(run["bottleneck_rate_bps"]),
                "timestamp_unit": "seconds",
                "column_map": column_map,
                "column_units": units,
                "metadata": metadata,
            }
        )
        run_reports.append(
            {
                "run_id": run["run_id"],
                "configuration_id": run["configuration_id"],
                "endpoint_source_file": run["endpoint"]["file"],
                "endpoint_source_sha256": run["endpoint"]["source_sha256"],
                "router_source_file": run["router"]["file"],
                "router_source_sha256": run["router"]["source_sha256"],
                "endpoint_timestamp_diagnostics": endpoint_diag,
                "router_timestamp_diagnostics": router_diag,
                "endpoint_samples_after_timestamp_policy": len(endpoint_records),
                "router_samples_after_timestamp_policy": len(router_records),
                "aligned_row_count": output_row_count,
                "matched_row_count": matched_output_count,
                "matched_endpoint_sample_count": matched_endpoint_count,
                "matched_samples_dropped_by_missing_measurements_policy": matched_endpoint_count
                - matched_output_count,
                "unmatched_endpoint_row_count": unmatched_endpoint_count,
                "unmatched_router_row_count": len(unmatched_router),
                "matched_absolute_offset_seconds": {
                    "count": len(matched_deltas),
                    "maximum": str(max(matched_deltas)) if matched_deltas else None,
                    "median": str(median(matched_deltas)) if matched_deltas else None,
                },
                "rows_with_blank_mapped_measurements": missing_measurement_count,
                "exclusion_reason_counts": dict(Counter(item["reason"] for item in exclusions)),
                "clock_review": run["clock_review"],
                "output_file": aligned_path.relative_to(output).as_posix(),
                "output_sha256": aligned_hash,
                "provenance_file": provenance_path.relative_to(output).as_posix(),
                "provenance_sha256": provenance_hash,
                "exclusions_file": exclusions_path.relative_to(output).as_posix(),
                "exclusions_sha256": exclusions_hash,
            }
        )
    # Detect mutation of all evidence after alignment calculations, before publishing manifests.
    if _sha256(spec_file) != spec_hash_before:
        raise AlignmentError("alignment specification changed during alignment")
    for run in runs:
        for stream_name in ("endpoint", "router"):
            stream = run[stream_name]
            if _sha256(stream["path"]) != stream["source_sha256"]:
                raise AlignmentError(
                    f"raw {stream_name} source changed during alignment: {stream['file']}"
                )
        evidence_path = _relative_file(
            root, run["clock_review"]["evidence_file"], "clock_review.evidence_file"
        )
        if _sha256(evidence_path) != run["clock_review"]["evidence_sha256"]:
            raise AlignmentError("clock review evidence changed during alignment")
    normalized_policy = {
        key: (str(value) if isinstance(value, Decimal) else value) for key, value in policy.items()
    }
    binding_basis = {
        "alignment_source_root": str(root.resolve()),
        "alignment_spec_file": str(spec_file.resolve()),
        "alignment_spec_sha256": spec_hash_before,
        "alignment_policy": normalized_policy,
        "runs": manifest_runs,
    }
    binding_hash = _canonical_sha256(binding_basis)
    manifest = {
        "schema_version": 2,
        **binding_basis,
        "alignment_binding_sha256": binding_hash,
        "alignment_review": {
            "status": "review_required",
            "reason": "review actual collector semantics and this alignment binding, then add a hash-linked decision record before preparation",
        },
        "alignment_status": "review_required",
    }
    manifest_path = output / "runs-v2.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    output_hashes[manifest_path.name] = _sha256(manifest_path)
    report = {
        "alignment_report_schema_version": 1,
        "status": "complete_mechanical_alignment_review_required",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(root),
        "spec_path": str(spec_file),
        "spec_sha256": spec_hash_before,
        "output_directory": str(output),
        "runs_manifest": manifest_path.name,
        "runs_manifest_sha256": output_hashes[manifest_path.name],
        "alignment_binding_sha256": binding_hash,
        "alignment_policy": normalized_policy,
        "run_count": len(run_reports),
        "runs": run_reports,
        "output_file_hashes": output_hashes,
        "interpretation_limits": [
            "Clock identity, timestamp semantics, offset and uncertainty come from the checked review specification and cited file; software does not infer them.",
            "A constant offset is applied only when declared as calibrated_fixed_offset. Clock drift is not estimated or corrected; its reviewed upper bound must fit within tolerance.",
            "Nearest-time matching is not proof of measurement equivalence. Reuse/one-to-one behavior and all exclusions are explicit in the selected policy and provenance.",
            "Timestamp values are converted to reference-clock seconds for the aligned CSV. Mapped measurement strings and declared units are copied without unit conversion; existing preparation performs its explicit feature conversions.",
            "Successful alignment and processing do not establish qdisc semantics, ground-truth validity, protocol approval, or predictive performance.",
        ],
    }
    report_path = output / "alignment_report.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    output_hashes[report_path.name] = _sha256(report_path)
    marker.unlink()
    return report


def verify_alignment_artifacts(
    report_dir: Path,
    *,
    raw_root: Path | None = None,
    spec_path: Path | None = None,
) -> dict[str, Any]:
    """Verify aligned outputs and, when supplied, the original source/spec hashes."""
    if report_dir.is_symlink():
        return {"status": "incomplete", "errors": ["alignment report directory is a symlink"]}
    directory = report_dir.resolve()
    report_path = directory / "alignment_report.json"
    if not report_path.is_file() or report_path.is_symlink():
        return {
            "status": "incomplete",
            "errors": ["alignment_report.json is missing or is a symlink"],
        }
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {
            "status": "incomplete",
            "errors": [f"cannot parse alignment report: {type(exc).__name__}"],
        }
    errors: list[str] = []
    if (
        not isinstance(report, dict)
        or report.get("alignment_report_schema_version") != 1
        or report.get("status") != "complete_mechanical_alignment_review_required"
    ):
        errors.append("unsupported or incomplete alignment report")
        report = report if isinstance(report, dict) else {}
    expected = report.get("output_file_hashes")
    if not isinstance(expected, dict):
        expected = {}
        errors.append("alignment report lacks output_file_hashes")
    expected_files = set(expected) | {"alignment_report.json"}
    actual_files: set[str] = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            errors.append(
                f"unexpected symlink in alignment output: {path.relative_to(directory).as_posix()}"
            )
        elif path.is_file():
            actual_files.add(path.relative_to(directory).as_posix())
    if actual_files != expected_files:
        errors.append(
            f"alignment output inventory differs (added={sorted(actual_files - expected_files)}, missing={sorted(expected_files - actual_files)})"
        )
    verified_outputs = []
    for relative, expected_hash in expected.items():
        candidate = (directory / relative).resolve()
        try:
            candidate.relative_to(directory)
        except ValueError:
            errors.append(f"alignment output path escapes report directory: {relative!r}")
            continue
        if not candidate.is_file() or candidate.is_symlink():
            errors.append(f"alignment output is missing or unsafe: {relative!r}")
        elif not isinstance(expected_hash, str) or _sha256(candidate) != expected_hash:
            errors.append(f"alignment output hash mismatch: {relative!r}")
        else:
            verified_outputs.append(relative)
    manifest_name = report.get("runs_manifest")
    if isinstance(manifest_name, str) and isinstance(report.get("runs_manifest_sha256"), str):
        if expected.get(manifest_name) != report["runs_manifest_sha256"]:
            errors.append("runs-v2 manifest hash is inconsistent within alignment report")
    else:
        errors.append("alignment report lacks a runs manifest reference")
    if isinstance(manifest_name, str) and Path(manifest_name).name == manifest_name:
        manifest_path = directory / manifest_name
        try:
            generated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            basis = {
                "alignment_source_root": generated_manifest["alignment_source_root"],
                "alignment_spec_file": generated_manifest["alignment_spec_file"],
                "alignment_spec_sha256": generated_manifest["alignment_spec_sha256"],
                "alignment_policy": generated_manifest["alignment_policy"],
                "runs": generated_manifest["runs"],
            }
            binding_hash = _canonical_sha256(basis)
            if (
                generated_manifest.get("alignment_binding_sha256") != binding_hash
                or report.get("alignment_binding_sha256") != binding_hash
            ):
                errors.append("alignment binding fingerprint is inconsistent")
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            errors.append("generated runs manifest has a missing or malformed alignment binding")
    else:
        errors.append("runs manifest path must be a local filename")
    raw_verified: list[str] = []
    if raw_root is not None:
        if raw_root.is_symlink():
            errors.append("source raw-root is a symlink")
        root = raw_root.resolve()
        if not root.is_dir():
            errors.append("source raw-root is missing or is not a directory")
        else:
            for run in report.get("runs", []) if isinstance(report.get("runs"), list) else []:
                for stream_name in ("endpoint", "router"):
                    record = (
                        run.get(f"{stream_name}_source_file") if isinstance(run, dict) else None
                    )
                    expected_hash = (
                        run.get(f"{stream_name}_source_sha256") if isinstance(run, dict) else None
                    )
                    try:
                        source = _relative_file(root, record, f"{stream_name} source")
                        actual_hash = _sha256(source)
                    except (AlignmentError, OSError) as exc:
                        errors.append(str(exc))
                        continue
                    if actual_hash != expected_hash:
                        errors.append(f"original {stream_name} source hash mismatch: {record!r}")
                    else:
                        raw_verified.append(str(record))
                clock = run.get("clock_review", {}) if isinstance(run, dict) else {}
                try:
                    evidence = _relative_file(
                        root, clock.get("evidence_file"), "clock review evidence"
                    )
                    if _sha256(evidence) != clock.get("evidence_sha256"):
                        errors.append("clock review evidence hash mismatch")
                except (AlignmentError, OSError) as exc:
                    errors.append(str(exc))
    spec_verified = False
    if spec_path is not None:
        source_spec = spec_path.resolve()
        if (
            spec_path.is_symlink()
            or not source_spec.is_file()
            or _sha256(source_spec) != report.get("spec_sha256")
        ):
            errors.append("alignment specification is missing or its hash differs from the report")
        else:
            spec_verified = True
    return {
        "status": "complete" if not errors else "incomplete",
        "errors": errors,
        "verified_output_count": len(verified_outputs),
        "verified_original_source_count": len(raw_verified),
        "original_sources_checked": raw_root is not None,
        "alignment_spec_checked": spec_path is not None,
        "alignment_spec_verified": spec_verified,
        "scientific_validity_established": False,
    }
