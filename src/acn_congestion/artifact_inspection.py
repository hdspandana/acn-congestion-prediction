"""Read-only, value-sparing inventory of collector and experiment artifacts.

This is a forensic orientation aid, not a schema adapter or scientific validator.
Column/path labels are reported as candidates only; raw measurement values are
never copied into the report.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


class ArtifactInspectionError(ValueError):
    """Artifact root or report path is invalid."""


_METADATA_KEYS = {
    "run_id",
    "configuration_id",
    "config_id",
    "experiment_id",
    "protocol_id",
    "schema_version",
    "timestamp_unit",
    "time_unit",
    "clock_source",
    "timezone",
    "start_time",
    "end_time",
    "duration_seconds",
    "bottleneck_rate_bps",
    "rate_bps",
    "qdisc_type",
    "qdisc_kind",
    "workload_type",
    "workload_name",
    "queue_setting",
    "source_file",
    "file",
    "column_map",
    "column_units",
    "timestamp_column",
    "feature_columns",
    "phase_schedule",
    "offered_load_bps",
    "burst_bytes",
    "queue_limit",
    "collector_version",
    "protocol_version",
}
_MAX_STRUCTURED_METADATA_BYTES = 8 * 1024 * 1024
_MAX_REPORTED_METADATA_FIELDS = 200
_MAX_DISTINCT_IDENTIFIERS = 100

_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_NUMBER_WITH_UNIT_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-zµ%/]+)\s*$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unit_hint(name: str) -> str | None:
    value = name.strip().lower()
    patterns = (
        (r"(?:^|[_-])(?:ns|nanoseconds?)$", "nanoseconds"),
        (r"(?:^|[_-])(?:us|microseconds?)$", "microseconds"),
        (r"(?:^|[_-])(?:ms|milliseconds?)$", "milliseconds"),
        (r"(?:^|[_-])(?:s|sec|seconds?)$", "seconds"),
        (r"(?:^|[_-])(?:bytes?|octets?)$", "bytes"),
        (r"(?:^|[_-])(?:bps|bit_per_second|bits_per_second)$", "bits_per_second"),
        (r"(?:^|[_-])(?:packets?|pkts?)$", "packets"),
        (r"(?:^|[_-])(?:count|total|delta)$", "count_or_delta_unspecified"),
        (r"(?:^|[_-])(?:percent|pct)$", "percent"),
    )
    for pattern, hint in patterns:
        if re.search(pattern, value):
            return hint
    return None


def _name_roles(name: str) -> list[str]:
    """Name-only candidates. These strings are never asserted to be semantics."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name).lower()
    tokens = set(re.findall(r"[a-z0-9]+", text))
    roles: list[str] = []
    if tokens & {
        "rtt",
        "latency",
        "ping",
        "loss",
        "goodput",
        "throughput",
        "retrans",
        "retransmission",
        "cwnd",
        "tcp",
        "endpoint",
        "client",
    }:
        roles.append("endpoint_telemetry_name_candidate")
    if tokens & {
        "queue",
        "qdisc",
        "backlog",
        "overlimit",
        "drop",
        "buffer",
        "qlen",
        "fifo",
        "router",
    }:
        roles.append("router_or_queue_name_candidate")
    if tokens & {"run", "runid", "config", "configuration", "experiment", "trial"}:
        roles.append("run_or_configuration_metadata_name_candidate")
    if tokens & {"phase", "stage", "load"}:
        roles.append("workload_or_phase_metadata_name_candidate")
    return roles


def _classify_cell(value: str | None) -> tuple[str, str | None]:
    if value is None or not value.strip():
        return "missing", None
    text = value.strip()
    lowered = text.lower()
    if lowered in {
        "nan",
        "+nan",
        "-nan",
        "inf",
        "+inf",
        "-inf",
        "infinity",
        "+infinity",
        "-infinity",
    }:
        return "nonfinite_numeric_token", None
    if lowered in {"true", "false", "yes", "no"}:
        return "boolean_like", None
    if _NUMERIC_RE.fullmatch(text):
        try:
            number = float(text)
        except ValueError:
            return "string", None
        if not math.isfinite(number):
            return "nonfinite_numeric", None
        return (
            "integer"
            if number.is_integer() and not any(c in text.lower() for c in (".", "e"))
            else "number"
        ), None
    with_unit = _NUMBER_WITH_UNIT_RE.fullmatch(text)
    if with_unit:
        try:
            number = float(with_unit.group(1))
        except ValueError:
            number = math.nan
        if not math.isfinite(number):
            return "nonfinite_numeric_with_unit", with_unit.group(2)
        return "numeric_text_with_unit", with_unit.group(2)
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
        return "iso_datetime_like", None
    except ValueError:
        return "string", None


def _parse_timestamp(value: str) -> tuple[str, float | datetime] | None:
    text = value.strip()
    if not text:
        return None
    if _NUMERIC_RE.fullmatch(text):
        number = float(text)
        if math.isfinite(number):
            return "numeric_raw", number
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return ("iso_aware", parsed) if parsed.tzinfo is not None else ("iso_naive", parsed)


def _candidate_timestamp_header(name: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", name.lower()))
    return (
        bool(
            tokens
            & {"time", "timestamp", "date", "clock", "ts", "epoch", "mono", "monotonic", "datetime"}
        )
        or name.lower().startswith(("time_", "timestamp_", "capture_time", "sample_time", "mono_"))
        or bool(re.search(r"(?:^|[_-])t(?:_?(?:ns|us|ms|s))?$", name.lower()))
    )


def _type_summary(counts: Counter[str]) -> str:
    present = set(counts) - {"missing"}
    if not present:
        return "all_missing"
    if len(present) == 1:
        return next(iter(present))
    return "mixed:" + ",".join(sorted(present))


class _TimeProfile:
    def __init__(self, column: str):
        self.column = column
        self.valid_count = 0
        self.invalid_count = 0
        self.kind_counts: Counter[str] = Counter()
        self.offsets: set[str] = set()
        self.numeric_min: float | None = None
        self.numeric_max: float | None = None
        self.iso_ranges: dict[str, tuple[datetime, datetime]] = {}
        self.previous: tuple[str, float | datetime] | None = None
        self.positive_interval_counts: Counter[str] = Counter()
        self.positive_interval_samples: dict[str, list[float]] = {}
        self.interval_ranges: dict[str, tuple[float, float]] = {}
        seed = int(hashlib.sha256(column.encode("utf-8")).hexdigest()[:16], 16)
        self._sample_rng = random.Random(seed)
        self._interval_sample_limit = 50_000
        self.duplicate_adjacent = 0
        self.out_of_order_adjacent = 0
        self.invalid_edges = 0
        self._distinct: set[tuple[str, str]] = set()
        self._distinct_saturated = False

    def _record_positive_interval(self, kind: str, delta: float) -> None:
        count = self.positive_interval_counts[kind] + 1
        self.positive_interval_counts[kind] = count
        if kind in self.interval_ranges:
            low, high = self.interval_ranges[kind]
            self.interval_ranges[kind] = (min(low, delta), max(high, delta))
        else:
            self.interval_ranges[kind] = (delta, delta)
        sample = self.positive_interval_samples.setdefault(kind, [])
        if len(sample) < self._interval_sample_limit:
            sample.append(delta)
        else:
            replacement = self._sample_rng.randrange(count)
            if replacement < self._interval_sample_limit:
                sample[replacement] = delta

    def add(self, value: str | None) -> None:
        parsed = _parse_timestamp(value or "")
        if parsed is None:
            self.invalid_count += 1
            self.invalid_edges += self.previous is not None
            self.previous = None
            return
        kind, current = parsed
        self.valid_count += 1
        self.kind_counts[kind] += 1
        if kind == "numeric_raw":
            number = float(current)
            self.numeric_min = number if self.numeric_min is None else min(self.numeric_min, number)
            self.numeric_max = number if self.numeric_max is None else max(self.numeric_max, number)
            canonical = format(number, ".17g")
        else:
            dt = current
            assert isinstance(dt, datetime)
            if kind == "iso_aware":
                offset = dt.utcoffset()
                self.offsets.add(str(offset))
            if kind in self.iso_ranges:
                previous_min, previous_max = self.iso_ranges[kind]
                self.iso_ranges[kind] = (min(previous_min, dt), max(previous_max, dt))
            else:
                self.iso_ranges[kind] = (dt, dt)
            canonical = dt.isoformat()
        distinct_key = (kind, canonical)
        if distinct_key in self._distinct:
            pass
        elif len(self._distinct) < 250_000:
            self._distinct.add(distinct_key)
        else:
            self._distinct_saturated = True
        if self.previous is not None:
            previous_kind, previous_value = self.previous
            if previous_kind != kind:
                self.invalid_edges += 1
            else:
                try:
                    delta = (
                        float(current) - float(previous_value)
                        if kind == "numeric_raw"
                        else (current - previous_value).total_seconds()
                    )
                except (TypeError, ValueError, OverflowError):
                    self.invalid_edges += 1
                else:
                    if not math.isfinite(delta):
                        self.invalid_edges += 1
                    elif delta == 0:
                        self.duplicate_adjacent += 1
                    elif delta < 0:
                        self.out_of_order_adjacent += 1
                    else:
                        self._record_positive_interval(kind, delta)
        self.previous = parsed

    def report(self, *, header_unit_hint: str | None) -> dict[str, Any]:
        # Never combine intervals expressed in different numeric/ISO formats.
        one_kind = len(self.kind_counts) == 1
        only_kind = next(iter(self.kind_counts), None)
        intervals = sorted(self.positive_interval_samples.get(only_kind, [])) if one_kind else []
        if intervals:
            center = median(intervals)
            p90 = intervals[min(len(intervals) - 1, math.ceil(0.9 * len(intervals)) - 1)]
            low, high = self.interval_ranges[only_kind]
            total_intervals = self.positive_interval_counts[only_kind]
            exact_sample = len(intervals) == total_intervals
            sampled_gap_count = sum(value > 1.5 * center for value in intervals)
            cadence = {
                "unit": "raw_timestamp_units"
                if only_kind == "numeric_raw"
                else "seconds_between_ISO_values",
                "median_of_deterministic_interval_sample": center,
                "minimum_positive_exact": low,
                "p90_of_deterministic_interval_sample": p90,
                "maximum_positive_exact": high,
                "positive_interval_count_exact": total_intervals,
                "interval_sample_count": len(intervals),
                "sampling_method": "all_intervals"
                if exact_sample
                else "deterministic_reservoir_sample",
                "gaps_over_1_5x_sample_median_exact": sampled_gap_count if exact_sample else None,
                "sampled_fraction_over_1_5x_sample_median": (
                    None if exact_sample else sampled_gap_count / len(intervals)
                ),
                "gap_rule": "descriptive 1.5x sample-median heuristic only; not an accepted protocol gap threshold",
            }
        else:
            cadence = None
        kinds = sorted(self.kind_counts)
        formats = "mixed" if len(kinds) > 1 else (kinds[0] if kinds else "unavailable")
        result: dict[str, Any] = {
            "column": self.column,
            "detection_basis": ["header_name_candidate"]
            if _candidate_timestamp_header(self.column)
            else ["timestamp_like_values"],
            "value_formats": dict(self.kind_counts),
            "dominant_format": formats,
            "valid_timestamp_like_count": self.valid_count,
            "invalid_or_missing_count": self.invalid_count,
            "duplicates_adjacent": self.duplicate_adjacent,
            "duplicate_value_count_lower_bound": max(0, self.valid_count - len(self._distinct)),
            "distinct_tracking_saturated": self._distinct_saturated,
            "out_of_order_adjacent_count": self.out_of_order_adjacent,
            "invalid_or_mixed_adjacent_edges": self.invalid_edges,
            "cadence": cadence,
            "header_unit_hint": header_unit_hint,
            "unit_status": "unknown_unverified"
            if header_unit_hint is None
            else "header_hint_only_unverified",
            "timezone_offsets_observed": sorted(self.offsets),
        }
        if self.numeric_min is not None:
            result["numeric_raw_range"] = {
                "minimum": self.numeric_min,
                "maximum": self.numeric_max,
                "units": "not normalized; see unverified header hint",
            }
        if self.iso_ranges:
            result["ISO_raw_ranges_by_timezone_kind"] = {
                kind: {
                    "minimum": bounds[0].isoformat(),
                    "maximum": bounds[1].isoformat(),
                    "timezone_status": "offset_or_Z_present"
                    if kind == "iso_aware"
                    else "timezone_naive_unknown",
                }
                for kind, bounds in sorted(self.iso_ranges.items())
            }
        if len(self.kind_counts) > 1:
            result["cadence_note"] = (
                "not summarized because timestamp-like values use mixed formats/units"
            )
        return result


def _timestamp_name_hint(name: str) -> str | None:
    hint = _unit_hint(name)
    return hint if hint in {"nanoseconds", "microseconds", "milliseconds", "seconds"} else None


def _sniff_format(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    extension_map = {
        ".csv": "CSV",
        ".tsv": "tab-separated text",
        ".json": "JSON",
        ".jsonl": "JSON Lines",
        ".ndjson": "newline-delimited JSON",
        ".yaml": "YAML text (not parsed)",
        ".yml": "YAML text (not parsed)",
        ".toml": "TOML text (not parsed)",
        ".log": "log text (not parsed)",
        ".txt": "plain text (not parsed)",
        ".sh": "shell script (not executed)",
        ".py": "Python source (not executed)",
        ".pcap": "packet capture (not decoded)",
        ".pcapng": "packet capture (not decoded)",
        ".gz": "gzip-compressed data (not decompressed)",
        ".zip": "ZIP archive (not extracted)",
        ".sqlite": "SQLite database (not opened)",
        ".db": "database or unknown binary (not opened)",
        ".parquet": "Parquet data (not decoded)",
    }
    try:
        with path.open("rb") as stream:
            magic = stream.read(16)
    except OSError:
        magic = b""
    if (
        magic[:4] == b"\xd4\xc3\xb2\xa1"
        or magic[:4] == b"\xa1\xb2\xc3\xd4"
        or magic[:4] in {b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d"}
    ):
        detected = "pcap binary signature"
    elif magic[:4] == b"\x0a\x0d\x0d\x0a":
        detected = "pcapng binary signature"
    elif magic[:2] == b"\x1f\x8b":
        detected = "gzip binary signature"
    elif magic[:4] == b"PK\x03\x04":
        detected = "ZIP binary signature"
    elif magic[:16] == b"SQLite format 3\x00":
        detected = "SQLite binary signature"
    elif magic[:4] == b"PAR1":
        detected = "Parquet binary signature"
    else:
        detected = "text-or-unrecognized"
    return {
        "extension": suffix or "(none)",
        "extension_claim": extension_map.get(suffix, "unrecognized extension"),
        "magic_signature": detected,
    }


def _safe_metadata_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return {"type": "nonfinite_number"}
    if isinstance(value, str) and len(value) > 240:
        return {"type": "string", "preview": value[:240], "truncated": True}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return {"type": "array", "length": len(value)}
    if isinstance(value, dict):
        return {"type": "object", "keys": sorted(str(key) for key in value)[:100]}
    return {"type": type(value).__name__}


def _extract_structured_metadata(document: Any) -> dict[str, Any]:
    safe_fields: list[dict[str, Any]] = []
    structural_keys: Counter[str] = Counter()
    object_count = 0
    list_counts: list[dict[str, Any]] = []
    stack: list[tuple[str, Any, int]] = [("$", document, 0)]
    while stack and len(safe_fields) < _MAX_REPORTED_METADATA_FIELDS:
        path, value, depth = stack.pop()
        if depth > 10:
            continue
        if isinstance(value, dict):
            object_count += 1
            for key, child in value.items():
                key_text = str(key)
                structural_keys[key_text] += 1
                child_path = f"{path}.{key_text}"
                if key_text.lower() in _METADATA_KEYS:
                    if key_text.lower() in {"column_map", "column_units"} and isinstance(
                        child, dict
                    ):
                        safe_value = {
                            str(map_key): _safe_metadata_value(map_value)
                            for map_key, map_value in list(child.items())[:100]
                        }
                    else:
                        safe_value = _safe_metadata_value(child)
                    safe_fields.append(
                        {"path": child_path, "field": key_text, "value_or_shape": safe_value}
                    )
                if isinstance(child, dict):
                    stack.append((child_path, child, depth + 1))
                elif isinstance(child, list):
                    list_counts.append({"path": child_path, "length": len(child)})
                    # Inspect bounded metadata structure, never serialize the list's values.
                    for index, element in enumerate(child[:500]):
                        if isinstance(element, (dict, list)):
                            stack.append((f"{child_path}[{index}]", element, depth + 1))
        elif isinstance(value, list):
            list_counts.append({"path": path, "length": len(value)})
    return {
        "root_type": type(document).__name__,
        "object_count_scanned": object_count,
        "key_occurrence_counts": dict(sorted(structural_keys.items())[:300]),
        "array_lengths": list_counts[:300],
        "allowlisted_metadata_fields": safe_fields[:_MAX_REPORTED_METADATA_FIELDS],
        "metadata_extraction_note": "Only allowlisted metadata keys and structural keys/counts are shown; arbitrary values and measurement arrays are not emitted. These fields remain unverified claims from source files.",
        "metadata_field_limit_reached": len(safe_fields) >= _MAX_REPORTED_METADATA_FIELDS,
    }


def _inspect_json(path: Path, *, jsonl: bool) -> dict[str, Any]:
    size = path.stat().st_size
    if jsonl:
        keys: Counter[str] = Counter()
        safe_values: list[dict[str, Any]] = []
        records = malformed = 0
        with path.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                records += 1
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if isinstance(value, dict):
                    for key, item in value.items():
                        keys[str(key)] += 1
                        if (
                            str(key).lower() in _METADATA_KEYS
                            and len(safe_values) < _MAX_REPORTED_METADATA_FIELDS
                        ):
                            safe_values.append(
                                {
                                    "line": line_number,
                                    "field": str(key),
                                    "value_or_shape": _safe_metadata_value(item),
                                }
                            )
        return {
            "parse_status": "line_scan_complete",
            "record_count": records,
            "malformed_record_count": malformed,
            "top_level_key_occurrence_counts": dict(sorted(keys.items())[:300]),
            "allowlisted_metadata_fields": safe_values,
            "content_disclosure": "record values omitted",
        }
    if size > _MAX_STRUCTURED_METADATA_BYTES:
        return {
            "parse_status": "not_parsed_size_limit",
            "size_limit_bytes": _MAX_STRUCTURED_METADATA_BYTES,
            "content_disclosure": "content omitted",
        }
    try:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {
            "parse_status": "invalid_or_unreadable",
            "error_type": type(exc).__name__,
            "content_disclosure": "content omitted",
        }
    return {"parse_status": "valid_json", **_extract_structured_metadata(document)}


def _inspect_text(path: Path, *, yaml_keys_only: bool = False) -> dict[str, Any]:
    line_count = 0
    key_counts: Counter[str] = Counter()
    timestamp_token_count = 0
    decode_error = False
    try:
        with path.open("r", encoding="utf-8-sig", errors="strict") as stream:
            for line in stream:
                line_count += 1
                if yaml_keys_only:
                    match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*:", line)
                    if match:
                        key_counts[match.group(1)] += 1
                if re.search(r"\d{4}-\d\d-\d\d[T ]\d\d:\d\d", line):
                    timestamp_token_count += 1
    except (OSError, UnicodeError):
        decode_error = True
    result: dict[str, Any] = {
        "line_count": line_count,
        "timestamp_like_line_count": timestamp_token_count,
        "content_disclosure": "line contents omitted",
        "text_scan_status": "decode_error" if decode_error else "complete",
    }
    if yaml_keys_only:
        result["detected_key_names_only"] = dict(sorted(key_counts.items())[:300])
        result["parse_status"] = "key_scan_only_yaml_not_parsed"
    return result


def _csv_delimiter(path: Path, suffix: str) -> str:
    if suffix == ".tsv":
        return "\t"
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _inspect_csv(path: Path, relative_path: str) -> dict[str, Any]:
    delimiter = _csv_delimiter(path, path.suffix.lower())
    column_counts: list[Counter[str]] = []
    missing_counts: list[int] = []
    numeric_min: list[float | None] = []
    numeric_max: list[float | None] = []
    numeric_unit_tokens: list[Counter[str]] = []
    missing_token_candidates: list[Counter[str]] = []
    distinct_ids: list[set[str]] = []
    header: list[str] = []
    row_count = malformed_long = malformed_short = 0
    timestamp_profiles: dict[int, _TimeProfile] = {}
    timestamp_names: set[int] = set()
    time_like_values: set[int] = set()
    file_role_terms = _name_roles(relative_path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter=delimiter, strict=True)
        try:
            raw_header = next(reader)
        except StopIteration:
            return {
                "parse_status": "empty_file",
                "row_count": 0,
                "delimiter": repr(delimiter),
                "columns": [],
                "content_disclosure": "no measurement values emitted",
            }
        header = [item.strip() for item in raw_header]
        for index, name in enumerate(header):
            column_counts.append(Counter())
            missing_counts.append(0)
            numeric_min.append(None)
            numeric_max.append(None)
            numeric_unit_tokens.append(Counter())
            missing_token_candidates.append(Counter())
            distinct_ids.append(set())
            if _candidate_timestamp_header(name):
                timestamp_names.add(index)
                timestamp_profiles[index] = _TimeProfile(name)
        for row in reader:
            row_count += 1
            if len(row) > len(header):
                malformed_long += 1
            elif len(row) < len(header):
                malformed_short += 1
            for index, name in enumerate(header):
                value = row[index] if index < len(row) else None
                kind, unit_token = _classify_cell(value)
                column_counts[index][kind] += 1
                if kind == "missing":
                    missing_counts[index] += 1
                if value is not None and value.strip().lower() in {
                    "na",
                    "n/a",
                    "null",
                    "none",
                    "nan",
                    "-",
                }:
                    missing_token_candidates[index][value.strip()] += 1
                if unit_token:
                    numeric_unit_tokens[index][unit_token] += 1
                if kind in {"integer", "number"} and value is not None:
                    number = float(value.strip())
                    numeric_min[index] = (
                        number if numeric_min[index] is None else min(numeric_min[index], number)
                    )
                    numeric_max[index] = (
                        number if numeric_max[index] is None else max(numeric_max[index], number)
                    )
                if (
                    name.strip().lower()
                    in {"run_id", "configuration_id", "config_id", "experiment_id"}
                    and value
                    and len(distinct_ids[index]) < _MAX_DISTINCT_IDENTIFIERS
                ):
                    distinct_ids[index].add(value.strip())
                if index in timestamp_names:
                    timestamp_profiles[index].add(value)
                elif kind == "iso_datetime_like":
                    if index not in time_like_values:
                        profile = timestamp_profiles.setdefault(index, _TimeProfile(name))
                        profile.invalid_count = row_count - 1
                        time_like_values.add(index)
                    timestamp_profiles[index].add(value)
                elif index in time_like_values:
                    timestamp_profiles[index].add(value)
    candidates = sorted(timestamp_names | time_like_values)
    columns = []
    endpoint_file_candidate = "endpoint_telemetry_name_candidate" in file_role_terms
    router_file_candidate = "router_or_queue_name_candidate" in file_role_terms
    for index, name in enumerate(header):
        roles = _name_roles(name)
        if endpoint_file_candidate:
            roles = sorted(set(roles + ["endpoint_file_path_name_candidate"]))
        if router_file_candidate:
            roles = sorted(set(roles + ["router_or_queue_file_path_name_candidate"]))
        item: dict[str, Any] = {
            "index": index,
            "name": name,
            "detected_type_counts": dict(column_counts[index]),
            "detected_type_summary": _type_summary(column_counts[index]),
            "missing_count": missing_counts[index],
            "missing_fraction": missing_counts[index] / row_count if row_count else None,
            "possible_missing_token_counts_not_reclassified": dict(
                sorted(missing_token_candidates[index].items())
            ),
            "numeric_raw_range": (
                {
                    "minimum": numeric_min[index],
                    "maximum": numeric_max[index],
                    "unit": "not normalized; semantics unknown",
                }
                if numeric_min[index] is not None
                else None
            ),
            "observed_numeric_text_unit_tokens": dict(sorted(numeric_unit_tokens[index].items())),
            "header_unit_hint": _unit_hint(name),
            "unit_status": "header_or_text_token_hint_only_unverified"
            if (_unit_hint(name) or numeric_unit_tokens[index])
            else "unknown_unverified",
            "name_based_role_candidates": roles,
            "semantic_status": "unknown_human_confirmation_required",
        }
        if name.strip().lower() in {"run_id", "configuration_id", "config_id", "experiment_id"}:
            item["distinct_value_count_lower_bound"] = len(distinct_ids[index])
            item["distinct_tracking_saturated"] = (
                len(distinct_ids[index]) >= _MAX_DISTINCT_IDENTIFIERS
            )
            item["identifier_values_sample"] = sorted(distinct_ids[index])[
                :_MAX_DISTINCT_IDENTIFIERS
            ]
            item["identifier_sample_note"] = (
                "values are included only for columns with identifier-like header names; header semantics remain unverified"
            )
        if index in candidates:
            item["timestamp_profile"] = timestamp_profiles[index].report(
                header_unit_hint=_timestamp_name_hint(name)
            )
            item["timestamp_profile"]["column_index"] = index
            item["timestamp_profile"]["detection_basis"] = (
                ["header_name_candidate", "timestamp_like_values"]
                if index in timestamp_names and index in time_like_values
                else ["header_name_candidate"]
                if index in timestamp_names
                else ["timestamp_like_values"]
            )
        columns.append(item)
    endpoint_cols = [
        col["name"]
        for col in columns
        if "endpoint_telemetry_name_candidate" in col["name_based_role_candidates"]
    ]
    router_cols = [
        col["name"]
        for col in columns
        if "router_or_queue_name_candidate" in col["name_based_role_candidates"]
    ]
    return {
        "parse_status": "complete",
        "delimiter": "TAB" if delimiter == "\t" else delimiter,
        "header": header,
        "duplicate_header_names": sorted({name for name in header if header.count(name) > 1}),
        "column_count": len(header),
        "row_count": row_count,
        "malformed_row_count": malformed_long + malformed_short,
        "rows_with_extra_fields": malformed_long,
        "rows_with_short_fields": malformed_short,
        "columns": columns,
        "timestamp_candidates": [
            {
                **timestamp_profiles[index].report(
                    header_unit_hint=_timestamp_name_hint(header[index])
                ),
                "column_index": index,
            }
            for index in candidates
        ],
        "file_role_candidates": file_role_terms,
        "endpoint_column_name_candidates": endpoint_cols,
        "router_or_qdisc_column_name_candidates": router_cols,
        "stream_interpretation": "name-only candidates; same-file columns do not prove temporal alignment or measurement semantics",
        "content_disclosure": "raw measurement values are not emitted; only type counts, missingness, ranges, selected identifier-like values, hashes, and timestamp diagnostics are reported",
    }


def _walk_files(root: Path) -> tuple[list[Path], list[dict[str, Any]], list[dict[str, Any]]]:
    files: list[Path] = []
    symlinks: list[dict[str, Any]] = []
    walk_errors: list[dict[str, Any]] = []

    def record_walk_error(error: OSError) -> None:
        walk_errors.append(
            {"path": str(error.filename or root), "error_type": type(error).__name__}
        )

    for directory, subdirs, filenames in os.walk(
        root, followlinks=False, onerror=record_walk_error
    ):
        current = Path(directory)
        kept_dirs = []
        for name in subdirs:
            item = current / name
            if item.is_symlink():
                symlinks.append(
                    {
                        "path": item.relative_to(root).as_posix(),
                        "kind": "directory_symlink",
                        "followed": False,
                    }
                )
            else:
                kept_dirs.append(name)
        subdirs[:] = kept_dirs
        for name in filenames:
            item = current / name
            if item.is_symlink():
                symlinks.append(
                    {
                        "path": item.relative_to(root).as_posix(),
                        "kind": "file_symlink",
                        "followed": False,
                    }
                )
            elif item.is_file():
                files.append(item)
    return sorted(files), symlinks, walk_errors


def _inspect_file(root: Path, path: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    try:
        before = path.stat()
        digest_before = _sha256(path)
    except OSError as exc:
        return {
            "path": relative,
            "status": "unreadable_or_disappeared",
            "error_type": type(exc).__name__,
        }
    item: dict[str, Any] = {
        "path": relative,
        "size_bytes": before.st_size,
        "format": _sniff_format(path),
        "sha256": digest_before,
        "status": "inspected",
        "parse_status": "metadata_only",
    }
    suffix = path.suffix.lower()
    try:
        if suffix in {".csv", ".tsv"}:
            item["csv"] = _inspect_csv(path, relative)
            item["parse_status"] = item["csv"]["parse_status"]
        elif suffix == ".json":
            item["structured_metadata"] = _inspect_json(path, jsonl=False)
            item["parse_status"] = item["structured_metadata"]["parse_status"]
        elif suffix in {".jsonl", ".ndjson"}:
            item["structured_metadata"] = _inspect_json(path, jsonl=True)
            item["parse_status"] = item["structured_metadata"]["parse_status"]
        elif suffix in {".yaml", ".yml"}:
            item["text_metadata"] = _inspect_text(path, yaml_keys_only=True)
            item["parse_status"] = item["text_metadata"]["parse_status"]
        elif suffix in {".txt", ".log", ".sh", ".py", ".toml", ".md"}:
            item["text_metadata"] = _inspect_text(path)
            item["parse_status"] = "text_metadata_only"
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        item["status"] = "inspection_error"
        item["parse_status"] = "inspection_error"
        item["error_type"] = type(exc).__name__
    try:
        after = path.stat()
        digest_after = _sha256(path)
        changed = (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or digest_before != digest_after
        )
    except OSError:
        changed = True
    item["changed_during_inspection"] = changed
    if changed:
        item["status"] = "changed_during_inspection"
    return item


def _primary_time_profile(file_report: dict[str, Any]) -> dict[str, Any] | None:
    candidates = file_report.get("csv", {}).get("timestamp_candidates", [])
    return next(
        (candidate for candidate in candidates if candidate.get("valid_timestamp_like_count", 0)),
        None,
    )


def _coverage_comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    a, b = _primary_time_profile(left), _primary_time_profile(right)
    result: dict[str, Any] = {
        "left_timestamp_candidate": a and a.get("column"),
        "right_timestamp_candidate": b and b.get("column"),
        "alignment_status": "unknown; no join or merge performed",
    }
    if not a or not b:
        result["coverage_status"] = "unavailable_timestamp_candidate"
        return result
    af, bf = a.get("dominant_format"), b.get("dominant_format")
    au, bu = a.get("header_unit_hint"), b.get("header_unit_hint")
    if af == bf == "numeric_raw":
        if not au or not bu or au != bu:
            result["coverage_status"] = (
                "incomparable_numeric_ranges_unverified_or_different_unit_hints"
            )
            result["unit_hints"] = [au, bu]
            return result
        ar, br = a.get("numeric_raw_range", {}), b.get("numeric_raw_range", {})
        overlaps = max(ar["minimum"], br["minimum"]) <= min(ar["maximum"], br["maximum"])
        result.update(
            {
                "coverage_status": "same_unverified_header_unit_raw_range_overlap"
                if overlaps
                else "same_unverified_header_unit_raw_ranges_disjoint",
                "raw_range_overlap": overlaps,
                "unit_hint": au,
                "note": "Range overlap is descriptive only; clock identity/offset/drift and sample correspondence remain unknown",
            }
        )
        return result
    if af == bf == "iso_aware":
        ar = a.get("ISO_raw_ranges_by_timezone_kind", {}).get("iso_aware", {})
        br = b.get("ISO_raw_ranges_by_timezone_kind", {}).get("iso_aware", {})
        try:
            a0, a1 = datetime.fromisoformat(ar["minimum"]), datetime.fromisoformat(ar["maximum"])
            b0, b1 = datetime.fromisoformat(br["minimum"]), datetime.fromisoformat(br["maximum"])
            overlaps = max(a0, b0) <= min(a1, b1)
        except (KeyError, TypeError, ValueError):
            result["coverage_status"] = "unavailable_aware_ISO_ranges"
            return result
        result.update(
            {
                "coverage_status": "offset_ISO_range_overlap_only"
                if overlaps
                else "offset_ISO_ranges_disjoint",
                "range_overlap": overlaps,
                "note": "Offset-bearing ISO ranges compared by absolute time only; overlap does not establish synchronized clocks or paired samples",
            }
        )
        return result
    if af == bf == "iso_naive":
        result["coverage_status"] = "timezone_naive_incomparable_without_timezone_evidence"
        return result
    result["coverage_status"] = "incompatible_timestamp_formats_no_conversion"
    return result


def _stream_assessment(files: list[dict[str, Any]]) -> dict[str, Any]:
    csv_files = [item for item in files if isinstance(item.get("csv"), dict)]
    endpoint_files = [
        item
        for item in csv_files
        if item["csv"].get("endpoint_column_name_candidates")
        or "endpoint_telemetry_name_candidate" in item["csv"].get("file_role_candidates", [])
    ]
    router_files = [
        item
        for item in csv_files
        if item["csv"].get("router_or_qdisc_column_name_candidates")
        or "router_or_queue_name_candidate" in item["csv"].get("file_role_candidates", [])
    ]
    co_located = sorted(
        {item["path"] for item in endpoint_files} & {item["path"] for item in router_files}
    )
    separate = []
    for left in endpoint_files:
        for right in router_files:
            if left["path"] != right["path"] and len(separate) < 500:
                separate.append(
                    {
                        "endpoint_file_candidate": left["path"],
                        "router_file_candidate": right["path"],
                        "coverage_comparison": _coverage_comparison(left, right),
                    }
                )
    if co_located and separate:
        pattern = "mixed_co_located_and_separate_stream_candidates"
    elif co_located:
        pattern = "same_file_endpoint_and_router_field_name_candidates"
    elif separate:
        pattern = "separate_endpoint_and_router_file_name_candidates"
    else:
        pattern = "stream_roles_unknown_from_names"
    return {
        "pattern": pattern,
        "endpoint_file_name_candidates": sorted({item["path"] for item in endpoint_files}),
        "router_or_qdisc_file_name_candidates": sorted({item["path"] for item in router_files}),
        "same_file_candidates": co_located,
        "separate_file_pairs": separate[:500],
        "alignment_decision": "not established; no timestamp normalization, join, interpolation, or sample alignment was performed",
        "candidate_basis": "file/column names only; semantics require collector-code and trace review",
        "human_evidence_needed": [
            "clock source and timezone",
            "timestamp unit",
            "clock offset/drift",
            "sampling cadence",
            "start/stop and phase relationship",
            "measurement semantics",
            "join key and tolerance",
        ],
    }


def inspect_artifact_tree(root: Path) -> dict[str, Any]:
    """Recursively inspect an artifact tree without modifying it or emitting measurements."""
    if root.is_symlink():
        raise ArtifactInspectionError("artifact root itself must not be a symlink")
    resolved = root.resolve()
    if not resolved.is_dir():
        raise ArtifactInspectionError(
            f"artifact root does not exist or is not a directory: {resolved}"
        )
    paths, symlinks, walk_errors = _walk_files(resolved)
    reports = [_inspect_file(resolved, path) for path in paths]
    return {
        "artifact_inspection_schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_root": str(resolved),
        "file_count": len(reports),
        "symlink_count_not_followed": len(symlinks),
        "walk_error_count": len(walk_errors),
        "walk_errors": walk_errors,
        "files": reports,
        "symlinks_not_followed": symlinks,
        "stream_alignment_assessment": _stream_assessment(reports),
        "interpretation_limits": [
            "This inspection is read-only and does not copy complete measurement rows into the report.",
            "Header/path names and observed suffixes are hints only; field semantics and units remain unknown until verified against source code, documentation, and acquisition behavior.",
            "Timestamp unit hints are not applied as conversions. Numeric timestamp ranges/cadence stay in raw numeric units unless their meaning is independently reviewed.",
            "Time-range overlap does not prove clock synchronization or aligned samples. No merge, resampling, tolerance join, or label construction is performed.",
            "Hashes establish file identity for later comparison, not authenticity or scientific validity.",
        ],
    }


def write_inspection_report(report: dict[str, Any], output_path: Path) -> Path:
    output_path = output_path.resolve()
    root = Path(report["artifact_root"]).resolve()
    try:
        output_path.relative_to(root)
    except ValueError:
        pass
    else:
        raise ArtifactInspectionError(
            "inspection report must be written outside the inspected artifact tree"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise ArtifactInspectionError(
            f"refusing to overwrite existing inspection report: {output_path}"
        ) from exc
    return output_path
