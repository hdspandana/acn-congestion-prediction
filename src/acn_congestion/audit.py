"""Read-only CSV audits for experimental data. Source files are never modified."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

TIMESTAMP_NAMES = {
    "timestamp",
    "timestamp_utc",
    "timestamp_ns",
    "timestamp_ms",
    "time",
    "time_utc",
    "ts",
    "sample_mono",
    "sample_mono_ns",
    "collected_at",
}
PHASE_NAMES = {"phase", "phase_label", "load_phase", "phase_observed", "phase_commanded"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamp_seconds(
    value: str,
    column: str,
    numeric_unit: str | None = None,
) -> tuple[float | None, str | None]:
    """Parse numeric clocks or ISO timestamps; report whether units were explicit."""
    text = value.strip()
    if not text:
        return None, None
    try:
        number = float(text)
        if not math.isfinite(number):
            return None, None
        lowered = column.lower()
        suffix_unit = (
            "nanoseconds"
            if lowered.endswith("_ns")
            else "microseconds"
            if lowered.endswith("_us")
            else "milliseconds"
            if lowered.endswith("_ms")
            else None
        )
        unit = numeric_unit or suffix_unit or "seconds"
        factors = {"seconds": 1.0, "milliseconds": 1e-3, "microseconds": 1e-6, "nanoseconds": 1e-9}
        kind = (
            "numeric_unit_conflict"
            if numeric_unit and suffix_unit and numeric_unit != suffix_unit
            else (
                "numeric_unit_explicit"
                if numeric_unit
                else f"numeric_unit_inferred_{suffix_unit}"
                if suffix_unit
                else "numeric_unit_assumed_seconds"
            )
        )
        return number * factors[unit], kind
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None, None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc).timestamp(), "naive_assumed_utc"
    offset = parsed.utcoffset()
    offset_minutes = int(offset.total_seconds() // 60) if offset is not None else 0
    return parsed.timestamp(), f"aware_offset_{offset_minutes:+d}min"


def _number(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _value_kind(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"true", "false", "yes", "no"}:
        return "boolean"
    number = _number(value)
    if number is not None:
        if not math.isfinite(number):
            return "nonfinite_numeric"
        return "integer" if value.strip().lstrip("+-").isdigit() else "number"
    if _timestamp_seconds(value, "timestamp")[0] is not None:
        return "timestamp_like"
    return "string"


def _infer_type(kinds: Counter[str], timestamp: bool = False) -> str:
    if timestamp:
        return "timestamp"
    if not kinds:
        return "empty_or_all_missing"
    observed = set(kinds)
    numeric = {"integer", "number", "nonfinite_numeric"}
    if observed <= {"boolean"}:
        return "boolean"
    if observed <= {"integer"}:
        return "integer"
    if observed <= {"integer", "number"}:
        return "number"
    if observed <= numeric:
        return "number_with_nonfinite_values" if "nonfinite_numeric" in observed else "number"
    if observed <= {"timestamp_like"}:
        return "timestamp_like"
    if "timestamp_like" in observed:
        return "timestamp_like_or_mixed"
    return "mixed_or_string" if len(observed) > 1 else "string"


def _issue(severity: str, code: str, message: str, column: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"severity": severity, "code": code, "message": message}
    if column is not None:
        item["column"] = column
    return item


def _interval_summary(
    timestamps: list[float], threshold_factor: float = 1.5
) -> dict[str, Any] | None:
    deltas = [right - left for left, right in pairwise(timestamps) if right > left]
    if not deltas:
        return None
    center = median(deltas)
    deviations = [abs(value - center) for value in deltas]
    ordered = sorted(deltas)
    p90 = ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)]
    return {
        "count_positive_intervals": len(deltas),
        "median_seconds": center,
        "mean_seconds": mean(deltas),
        "stddev_seconds_population": pstdev(deltas) if len(deltas) > 1 else 0.0,
        "mad_seconds": median(deviations),
        "minimum_seconds": min(deltas),
        "p90_seconds": p90,
        "maximum_seconds": max(deltas),
        "gap_threshold_factor": threshold_factor,
        "large_gap_count": sum(value > threshold_factor * center for value in deltas),
    }


def audit_csv(
    path: Path,
    root: Path,
    *,
    required_columns: Iterable[str] = (),
    expected_phases: Iterable[str] = (),
    timestamp_column: str | None = None,
    numeric_timestamp_unit: str | None = None,
) -> dict[str, Any]:
    """Summarize a CSV without repairing it; all row-level anomalies remain visible."""
    valid_units = {"seconds", "milliseconds", "microseconds", "nanoseconds"}
    if numeric_timestamp_unit is not None and numeric_timestamp_unit not in valid_units:
        raise ValueError(f"numeric_timestamp_unit must be one of {sorted(valid_units)}")
    result: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": None,
        "sha256": None,
        "row_count": 0,
        "columns": [],
        "inferred_types": {},
        "missing_by_column": {},
        "missing_percent_by_column": {},
        "nonfinite_by_column": {},
        "distinct_count_by_column": {},
        "distinct_count_is_lower_bound_by_column": {},
        "constant_columns": [],
        "near_constant_candidates": [],
        "duplicate_rows": 0,
        "timestamp_column": None,
        "timestamp_candidate_columns": [],
        "timestamp_parse_failures": 0,
        "duplicate_timestamps": 0,
        "timestamp_order_violations": 0,
        "timestamp_timezone_kinds": {},
        "sampling_intervals": None,
        "phase_values": {},
        "unexpected_phases": [],
        "malformed_row_count": 0,
        "issues": [],
        "errors": [],
        "cadence_seconds": None,
        "gap_count_over_1_5x_median": None,
        "status": "error",
        "processing_sufficiency": {},
    }
    try:
        result["size_bytes"] = path.stat().st_size
        result["sha256"] = _sha256(path)
    except OSError as exc:
        result["issues"].append(_issue("error", "file_read_error", str(exc)))
        result["errors"] = ["file_read_error"]
        result["processing_sufficiency"] = {"sufficient": False, "blockers": ["file_unreadable"]}
        return result
    if result["size_bytes"] == 0:
        result["issues"].append(_issue("error", "empty_file", "file is zero bytes"))
        result["errors"] = ["empty_file"]
        result["processing_sufficiency"] = {"sufficient": False, "blockers": ["empty_file"]}
        return result

    timestamps: list[float] = []
    timezone_kinds: Counter[str] = Counter()
    type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    distinct: dict[str, set[str]] = defaultdict(set)
    value_counts: dict[str, Counter[str]] = defaultdict(Counter)
    row_digests: set[str] = set()
    phase_counts: Counter[str] = Counter()
    nonfinite: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    phase_key: str | None = None
    raw_timestamp_key: str | None = None
    resolved_required = set(required_columns)
    resolved_expected_phases = set(expected_phases)
    missing_required = sorted(resolved_required)

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                result["issues"].append(_issue("error", "missing_header", "CSV has no header row"))
                result["errors"] = ["missing_header"]
                result["processing_sufficiency"] = {
                    "sufficient": False,
                    "blockers": ["missing_header"],
                }
                return result
            source_fields = list(reader.fieldnames)
            fields = [field.strip() if field is not None else "" for field in source_fields]
            result["columns"] = fields
            if len(fields) != len(set(fields)):
                result["issues"].append(
                    _issue("error", "duplicate_column_names", "header contains duplicate names")
                )
                result["errors"] = ["duplicate_column_names"]
                result["processing_sufficiency"] = {
                    "sufficient": False,
                    "blockers": ["duplicate_column_names"],
                }
                return result
            missing_required = sorted(resolved_required - set(fields))
            for name in missing_required:
                result["issues"].append(
                    _issue(
                        "error",
                        "required_column_absent",
                        f"required column {name!r} is absent",
                        name,
                    )
                )
            normalized = {field.strip().lower(): field for field in source_fields if field}
            if timestamp_column is not None:
                timestamp_candidates = (
                    [normalized[timestamp_column.strip().lower()]]
                    if timestamp_column.strip().lower() in normalized
                    else []
                )
            else:
                timestamp_candidates = [
                    normalized[name] for name in sorted(TIMESTAMP_NAMES) if name in normalized
                ]
            result["timestamp_candidate_columns"] = [name.strip() for name in timestamp_candidates]
            raw_timestamp_key = timestamp_candidates[0] if len(timestamp_candidates) == 1 else None
            result["timestamp_column"] = raw_timestamp_key.strip() if raw_timestamp_key else None
            phase_key = next((normalized[name] for name in PHASE_NAMES if name in normalized), None)
            field_map = {raw: raw.strip() for raw in source_fields if raw is not None}
            for field in fields:
                result["missing_by_column"][field] = 0
                result["nonfinite_by_column"][field] = 0

            for row in reader:
                result["row_count"] += 1
                if None in row or any(
                    value is None for key, value in row.items() if key is not None
                ):
                    result["malformed_row_count"] += 1
                normalized_row = {
                    field_map[key]: value for key, value in row.items() if key in field_map
                }
                row_bytes = json.dumps(normalized_row, sort_keys=True, ensure_ascii=False).encode(
                    "utf-8"
                )
                digest = hashlib.sha256(row_bytes).hexdigest()
                if digest in row_digests:
                    result["duplicate_rows"] += 1
                else:
                    row_digests.add(digest)

                for key, raw_value in normalized_row.items():
                    if raw_value is None or not raw_value.strip():
                        missing[key] += 1
                        continue
                    value = raw_value.strip()
                    type_counts[key][_value_kind(value)] += 1
                    if len(distinct[key]) < 100_000:
                        distinct[key].add(value)
                    if value in value_counts[key] or len(value_counts[key]) < 10_000:
                        value_counts[key][value] += 1
                    number = _number(value)
                    if number is not None and not math.isfinite(number):
                        nonfinite[key] += 1
                if raw_timestamp_key is not None:
                    raw_time = row.get(raw_timestamp_key)
                    parsed, timezone_kind = _timestamp_seconds(
                        raw_time or "", raw_timestamp_key, numeric_timestamp_unit
                    )
                    if parsed is None:
                        result["timestamp_parse_failures"] += 1
                    else:
                        timestamps.append(parsed)
                        timezone_kinds[timezone_kind or "unknown"] += 1
                if phase_key is not None:
                    phase_value = row.get(phase_key)
                    if phase_value and phase_value.strip():
                        phase_counts[phase_value.strip()] += 1

        result["missing_by_column"] = {field: missing[field] for field in fields}
        result["missing_percent_by_column"] = {
            field: (100.0 * missing[field] / result["row_count"] if result["row_count"] else None)
            for field in fields
        }
        result["nonfinite_by_column"] = {field: nonfinite[field] for field in fields}
        result["distinct_count_by_column"] = {field: len(distinct[field]) for field in fields}
        result["distinct_count_is_lower_bound_by_column"] = {
            field: len(distinct[field]) >= 100_000 for field in fields
        }
        timestamp_name = result["timestamp_column"]
        result["inferred_types"] = {
            field: _infer_type(type_counts[field], timestamp=(field == timestamp_name))
            for field in fields
        }
        result["constant_columns"] = [
            field
            for field in fields
            if result["row_count"] > 0
            and missing[field] < result["row_count"]
            and len(distinct[field]) == 1
            and len(distinct[field]) < 100_000
        ]
        for field in fields:
            nonmissing = result["row_count"] - missing[field]
            if nonmissing >= 20 and value_counts[field]:
                top_fraction = max(value_counts[field].values()) / nonmissing
                if top_fraction >= 0.995 and len(distinct[field]) > 1:
                    result["near_constant_candidates"].append(
                        {
                            "column": field,
                            "dominant_value_fraction": top_fraction,
                            "rule": "at least 99.5% of non-missing values equal",
                        }
                    )
        result["phase_values"] = dict(sorted(phase_counts.items()))
        if resolved_expected_phases and phase_key is not None:
            result["unexpected_phases"] = sorted(set(phase_counts) - resolved_expected_phases)
        result["timestamp_timezone_kinds"] = dict(timezone_kinds)
        if timestamps:
            result["duplicate_timestamps"] = len(timestamps) - len(set(timestamps))
            result["timestamp_order_violations"] = sum(
                later < earlier for earlier, later in pairwise(timestamps)
            )
            result["sampling_intervals"] = _interval_summary(timestamps)

    except (OSError, UnicodeError, csv.Error) as exc:
        result["issues"].append(_issue("error", "csv_read_error", f"{type(exc).__name__}: {exc}"))

    if result["row_count"] == 0:
        result["issues"].append(
            _issue("error", "no_data_rows", "CSV has a header but no data rows")
        )
    if result["malformed_row_count"]:
        result["issues"].append(
            _issue(
                "error",
                "wrong_field_count",
                f"{result['malformed_row_count']} rows have the wrong number of fields",
            )
        )
    if result["duplicate_rows"]:
        result["issues"].append(
            _issue(
                "warning",
                "duplicate_rows",
                f"{result['duplicate_rows']} exact duplicate rows detected",
            )
        )
    if resolved_expected_phases and phase_key is None:
        result["issues"].append(
            _issue(
                "error",
                "phase_column_not_found",
                "expected phases were configured but no recognized phase column was found",
            )
        )
    if result["timestamp_column"] is None and len(result["timestamp_candidate_columns"]) > 1:
        result["issues"].append(
            _issue(
                "error",
                "ambiguous_timestamp_columns",
                f"multiple timestamp candidates {result['timestamp_candidate_columns']}; pass --timestamp-column explicitly",
            )
        )
    elif result["timestamp_column"] is None:
        result["issues"].append(
            _issue(
                "warning",
                "timestamp_column_not_found",
                "no recognized timestamp column; cadence cannot be audited",
            )
        )
    elif result["timestamp_parse_failures"]:
        result["issues"].append(
            _issue(
                "warning",
                "invalid_timestamps",
                f"{result['timestamp_parse_failures']} timestamp values are blank or invalid",
                result["timestamp_column"],
            )
        )
    if result["duplicate_timestamps"]:
        result["issues"].append(
            _issue(
                "warning",
                "duplicate_timestamps",
                f"{result['duplicate_timestamps']} duplicate timestamp values",
                result["timestamp_column"],
            )
        )
    if result["timestamp_order_violations"]:
        result["issues"].append(
            _issue(
                "warning",
                "timestamp_order",
                f"{result['timestamp_order_violations']} adjacent timestamp order violations",
                result["timestamp_column"],
            )
        )
    timezone_labels = set(timezone_kinds)
    if "numeric_unit_assumed_seconds" in timezone_labels:
        result["issues"].append(
            _issue(
                "warning",
                "numeric_timestamp_unit_assumed",
                "numeric timestamps have no unit suffix or explicit --timestamp-unit; cadence is provisionally treated as seconds",
                result["timestamp_column"],
            )
        )
    if "numeric_unit_conflict" in timezone_labels:
        result["issues"].append(
            _issue(
                "error",
                "numeric_timestamp_unit_conflict",
                "explicit timestamp unit conflicts with the column name's unit suffix",
                result["timestamp_column"],
            )
        )
    if len(timezone_labels) > 1:
        result["issues"].append(
            _issue(
                "warning",
                "mixed_timezone_conventions",
                f"timestamp clock/timezone kinds differ: {sorted(timezone_labels)}",
                result["timestamp_column"],
            )
        )
    if result["sampling_intervals"] and result["sampling_intervals"]["large_gap_count"]:
        result["issues"].append(
            _issue(
                "warning",
                "large_sampling_gaps",
                f"{result['sampling_intervals']['large_gap_count']} intervals exceed 1.5 times the median",
            )
        )
    for field, count in result["missing_by_column"].items():
        if count:
            result["issues"].append(
                _issue(
                    "warning",
                    "missing_values",
                    f"{count}/{result['row_count']} cells missing ({result['missing_percent_by_column'][field]:.2f}%)",
                    field,
                )
            )
    for field, count in result["nonfinite_by_column"].items():
        if count:
            result["issues"].append(
                _issue(
                    "warning",
                    "nonfinite_numeric_values",
                    f"{count} non-finite numeric values",
                    field,
                )
            )
    for field in result["constant_columns"]:
        result["issues"].append(
            _issue("info", "constant_column", "column is constant among non-missing rows", field)
        )
    for field in result["near_constant_candidates"]:
        result["issues"].append(
            _issue(
                "info",
                "near_constant_candidate",
                f"dominant fraction {field['dominant_value_fraction']:.4f}",
                field["column"],
            )
        )
    for phase in result["unexpected_phases"]:
        result["issues"].append(
            _issue(
                "warning",
                "unexpected_phase",
                f"phase value {phase!r} is not in configured expected phases",
                phase_key.strip() if phase_key else None,
            )
        )

    blockers: list[str] = []
    if result["row_count"] == 0:
        blockers.append("no_data_rows")
    if result["duplicate_rows"]:
        blockers.append("duplicate_rows")
    if result["duplicate_timestamps"]:
        blockers.append("duplicate_timestamps")
    if result["timestamp_order_violations"]:
        blockers.append("timestamp_order_violations")
    if len(result["timestamp_candidate_columns"]) > 1 and timestamp_column is None:
        blockers.append("ambiguous_timestamp_columns")
    if resolved_expected_phases and phase_key is None:
        blockers.append("phase_column_missing")
    if result["unexpected_phases"]:
        blockers.append("unexpected_phases")
    if not resolved_required:
        blockers.append("required_columns_not_configured")
    if missing_required:
        blockers.append("required_columns_absent")
    if result["timestamp_column"] is None:
        blockers.append("timestamp_missing")
    if result["timestamp_parse_failures"]:
        blockers.append("invalid_or_missing_timestamps")
    if "numeric_unit_assumed_seconds" in timezone_labels:
        blockers.append("timestamp_unit_unverified")
    if "numeric_unit_conflict" in timezone_labels:
        blockers.append("timestamp_unit_conflict")
    if any(missing.get(name, 0) for name in resolved_required):
        blockers.append("required_values_missing")
    if result["malformed_row_count"]:
        blockers.append("malformed_rows")
    if any(nonfinite.get(name, 0) for name in resolved_required):
        blockers.append("nonfinite_required_values")
    result["processing_sufficiency"] = {"sufficient": not blockers, "blockers": blockers}
    severities = [issue["severity"] for issue in result["issues"]]
    result["status"] = (
        "error" if "error" in severities else "warning" if "warning" in severities else "ok"
    )
    result["errors"] = [issue["code"] for issue in result["issues"] if issue["severity"] == "error"]
    if result["sampling_intervals"]:
        intervals = result["sampling_intervals"]
        result["cadence_seconds"] = {
            "median": intervals["median_seconds"],
            "minimum_positive": intervals["minimum_seconds"],
            "maximum_positive": intervals["maximum_seconds"],
        }
        result["gap_count_over_1_5x_median"] = intervals["large_gap_count"]
    return result


def audit_directory(
    directory: Path,
    *,
    required_columns: Iterable[str] = (),
    expected_phases: Iterable[str] = (),
    timestamp_column: str | None = None,
    numeric_timestamp_unit: str | None = None,
) -> dict[str, Any]:
    """Audit CSV files recursively; report empty directories as incomplete, not successful."""
    root = directory.resolve()
    required = tuple(required_columns)
    phases = tuple(expected_phases)
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".csv"
    )
    audited = [
        audit_csv(
            path,
            root,
            required_columns=required,
            expected_phases=phases,
            timestamp_column=timestamp_column,
            numeric_timestamp_unit=numeric_timestamp_unit,
        )
        for path in files
    ]
    by_hash: dict[str, list[str]] = defaultdict(list)
    for item in audited:
        if item["sha256"]:
            by_hash[item["sha256"]].append(item["path"])
    possible_duplicates = [paths for paths in by_hash.values() if len(paths) > 1]
    for paths in possible_duplicates:
        for item in audited:
            if item["path"] in paths:
                item["issues"].append(
                    _issue(
                        "warning",
                        "identical_file_hash",
                        f"identical content also found at: {', '.join(p for p in paths if p != item['path'])}",
                    )
                )
                if item["status"] == "ok":
                    item["status"] = "warning"
    return {
        "report_schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(root),
        "timestamp_column_override": timestamp_column,
        "numeric_timestamp_unit_override": numeric_timestamp_unit,
        "csv_file_count": len(audited),
        "files_with_errors": sum(item["status"] == "error" for item in audited),
        "files_with_warnings": sum(item["status"] == "warning" for item in audited),
        "possible_duplicate_files": possible_duplicates,
        "audit_completed": bool(audited) and all(item["status"] != "error" for item in audited),
        "data_present": bool(audited),
        "interpretation": "Audit completion is not a scientific-validity verdict. An empty directory is incomplete; anomalies are reported, not repaired or silently excluded.",
        "files": audited,
    }


def render_summary(report: dict[str, Any]) -> str:
    """Produce a concise, human-readable summary from a machine report."""
    lines = [
        "# Raw CSV audit summary",
        "",
        f"- Input: `{report['input_directory']}`",
        f"- CSV files: {report['csv_file_count']}",
        f"- Audit completed: {report['audit_completed']}",
        f"- Files with errors/warnings: {report['files_with_errors']}/{report['files_with_warnings']}",
        "- This is a data-quality report, not evidence of model performance.",
        "",
    ]
    if not report["files"]:
        lines.append(
            "No CSV files found. No data were modified; this is not a successful empty experiment."
        )
    for item in report["files"]:
        cadence = item.get("sampling_intervals")
        cadence_text = (
            f"median cadence {cadence['median_seconds']:.6g}s; gaps {cadence['large_gap_count']}"
            if cadence
            else "cadence unavailable"
        )
        lines.extend(
            [
                f"## {item['path']}",
                "",
                f"- Status: {item['status']}; rows: {item['row_count']}; bytes: {item['size_bytes']}",
                f"- SHA-256: `{item['sha256']}`",
                f"- Columns: {', '.join(item['columns']) or '(none)'}",
                f"- Timestamp: {item['timestamp_column'] or '(not recognized)'}; {cadence_text}",
                f"- Processing sufficient under configured required fields: {item['processing_sufficiency'].get('sufficient', False)}",
            ]
        )
        if item["issues"]:
            lines.append("- Findings:")
            lines.extend(
                f"  - {issue['severity'].upper()} `{issue['code']}`: {issue['message']}"
                for issue in item["issues"]
            )
        else:
            lines.append("- Findings: none")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
