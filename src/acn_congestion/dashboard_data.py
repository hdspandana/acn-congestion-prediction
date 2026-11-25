"""Safe, bounded, read-only data access for the optional ACN Streamlit console.

The dashboard deliberately has no arbitrary path browser and does not duplicate
labeling, preparation, alignment, or model logic. Expensive work is explicit and
source fingerprints are rechecked before cached chart previews are returned.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import uuid
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

from . import __version__
from .artifact_inspection import ArtifactInspectionError, inspect_artifact_tree
from .audit import audit_directory, render_summary
from .config import ConfigError, load_config, verify_protocol_freeze_evidence
from .ground_truth import GroundTruthReportError, verify_ground_truth_report
from .processing import (
    ProcessingError,
    feature_specification_operational_status,
    feature_specification_sha256,
    feature_temporal_policy_from_specification,
    file_sha256,
    normalize_feature_temporal_policy,
    verify_alignment_review,
)
from .reporting import ReportError, verify_evaluation_artifacts
from .robustness import RobustnessError, verify_robustness_artifacts

MAX_JSON_BYTES = 24 * 1024 * 1024
MAX_DOWNLOAD_BYTES = 16 * 1024 * 1024
MAX_CHART_POINTS = 1800
MAX_INVENTORY_FILES = 20_000
MAX_DASHBOARD_AUDIT_BYTES = 64 * 1024 * 1024


class DashboardDataError(ValueError):
    """A configured research artifact is missing, unsafe, stale, or malformed."""


@dataclass(frozen=True)
class Workspace:
    project_root: Path
    raw_root: Path
    processed_root: Path
    reports_root: Path
    config_path: Path
    manifest_path: Path | None
    audit_report_path: Path | None
    alignment_spec_path: Path | None
    configuration_errors: tuple[str, ...] = ()


def _configured_path(
    value: str | None, default: Path, project_root: Path, *, allowed_roots: tuple[Path, ...]
) -> Path:
    candidate = Path(value).expanduser() if value else default
    was_absolute = candidate.is_absolute()
    if not was_absolute:
        candidate = project_root / candidate
    if candidate.is_symlink():
        raise DashboardDataError("configured path itself must not be a symlink")
    resolved = candidate.resolve()
    if not was_absolute and not any(
        _is_relative_to(resolved, root.resolve()) for root in allowed_roots
    ):
        raise DashboardDataError("relative configured paths must remain within the project root")
    return resolved


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_workspace(
    environ: Mapping[str, str] | None = None, project_root: Path | None = None
) -> Workspace:
    """Resolve a small, explicit set of source roots; no path is taken from a form."""
    env = os.environ if environ is None else environ
    root_candidate = (
        Path(env.get("ACN_PROJECT_ROOT", "")).expanduser()
        if env.get("ACN_PROJECT_ROOT")
        else (project_root if project_root is not None else Path(__file__).resolve().parents[2])
    )
    if not root_candidate.is_absolute():
        root_candidate = root_candidate.resolve()
    if root_candidate.is_symlink():
        raise DashboardDataError("ACN_PROJECT_ROOT must not be a symlink")
    root = root_candidate.resolve()
    if not root.is_dir():
        raise DashboardDataError("ACN_PROJECT_ROOT is missing or is not a directory")

    errors: list[str] = []
    try:
        raw = _configured_path(
            env.get("ACN_RAW_ROOT"), root / "data" / "raw", root, allowed_roots=(root,)
        )
        processed = _configured_path(
            env.get("ACN_PROCESSED_ROOT"), root / "data" / "processed", root, allowed_roots=(root,)
        )
        reports = _configured_path(
            env.get("ACN_REPORTS_ROOT"), root / "reports", root, allowed_roots=(root,)
        )
        config = _configured_path(
            env.get("ACN_CONFIG_PATH"),
            root / "configs" / "experiment.yaml",
            root,
            allowed_roots=(root,),
        )
    except DashboardDataError as exc:
        raise DashboardDataError(str(exc)) from exc

    def configured_file(key: str, default: Path | None, allowed: tuple[Path, ...]) -> Path | None:
        value = env.get(key)
        if value:
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = root / candidate
            if candidate.is_symlink():
                errors.append(f"{key}: configured files must not be symlinks")
                return None
            resolved = candidate.resolve()
            if not any(
                _is_relative_to(resolved, allowed_root.resolve()) for allowed_root in allowed
            ):
                errors.append(f"{key}: file must be within its declared source/report root")
                return None
            return resolved
        if default is not None and default.is_file() and not default.is_symlink():
            return default.resolve()
        return None

    manifest_default = next(
        (
            path
            for path in (raw / "runs-v2.json", raw / "runs.json")
            if path.is_file() and not path.is_symlink()
        ),
        None,
    )
    manifest = configured_file("ACN_RUN_MANIFEST", manifest_default, (root, raw))
    audit_default = next(
        (
            path
            for path in (
                reports / "audit.json",
                reports / "audits" / "latest" / "audit.json",
            )
            if path.is_file() and not path.is_symlink()
        ),
        None,
    )
    if audit_default is None:
        audit_candidates = (
            sorted(
                (
                    path
                    for path in reports.glob("audits/*/audit.json")
                    if path.is_file() and not path.is_symlink()
                ),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
            if reports.is_dir()
            else []
        )
        audit_default = audit_candidates[0] if audit_candidates else None
    audit_path = configured_file("ACN_AUDIT_REPORT", audit_default, (root, reports))
    spec = configured_file("ACN_ALIGNMENT_SPEC", None, (root, raw))
    return Workspace(
        root, raw, processed, reports, config, manifest, audit_path, spec, tuple(errors)
    )


def display_path(path: Path | None, workspace: Workspace) -> str:
    if path is None:
        return "Not configured"
    try:
        return path.resolve().relative_to(workspace.project_root.resolve()).as_posix()
    except ValueError:
        return f"External source: {path.name}"


def safe_child(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    """Resolve a relative artifact path while refusing traversal and symlinks."""
    if not isinstance(relative, str) or not relative:
        raise DashboardDataError("artifact path must be a non-empty relative string")
    item = PurePosixPath(relative)
    if item.is_absolute() or ".." in item.parts or "" in item.parts:
        raise DashboardDataError("artifact path may not be absolute or traverse parent directories")
    root_path = root.resolve()
    probe = root_path
    for part in item.parts:
        probe = probe / part
        if probe.is_symlink():
            raise DashboardDataError("artifact path must not traverse symlinks")
    candidate = (root_path / Path(*item.parts)).resolve()
    if not _is_relative_to(candidate, root_path):
        raise DashboardDataError("artifact path escapes its allowed root")
    if must_exist and not candidate.is_file():
        raise DashboardDataError("artifact file is missing")
    return candidate


def read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DashboardDataError("JSON report is missing or unsafe")
    if path.stat().st_size > MAX_JSON_BYTES:
        raise DashboardDataError("JSON report exceeds the dashboard's safe metadata size limit")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DashboardDataError(f"cannot parse JSON report: {exc}") from exc
    if not isinstance(payload, dict):
        raise DashboardDataError("JSON report root must be an object")
    return payload


def scan_csv_inventory(root: Path, *, limit: int = MAX_INVENTORY_FILES) -> dict[str, Any]:
    """Count files without following symlinks or reading measurement content."""
    if root.is_symlink():
        return {
            "status": "unsafe",
            "csv_count": 0,
            "file_count": 0,
            "message": "Configured raw root is a symlink.",
        }
    if not root.is_dir():
        return {
            "status": "missing",
            "csv_count": 0,
            "file_count": 0,
            "message": "No raw artifact directory is configured or present.",
        }
    csv_paths: list[str] = []
    file_count = 0
    symlinks = 0
    truncated = False
    for current, dirs, files in os.walk(root, followlinks=False):
        symlink_dirs = [name for name in dirs if (Path(current) / name).is_symlink()]
        symlinks += len(symlink_dirs)
        dirs[:] = sorted(name for name in dirs if name not in symlink_dirs)
        for name in sorted(files):
            candidate = Path(current) / name
            if candidate.is_symlink():
                symlinks += 1
                continue
            if not candidate.is_file():
                continue
            file_count += 1
            if candidate.suffix.lower() == ".csv":
                csv_paths.append(candidate.relative_to(root).as_posix())
            if file_count > limit:
                truncated = True
                break
        if truncated:
            break
    return {
        "status": "discovered" if csv_paths else "no_data",
        "csv_count": len(csv_paths),
        "file_count": file_count,
        "symlink_count_not_followed": symlinks,
        "truncated": truncated,
        "csv_paths": csv_paths[:limit],
        "message": "File-name scan only; contents and hashes are not verified.",
    }


def verify_audit_inventory(raw_root: Path, audit_path: Path | None) -> dict[str, Any]:
    """Recheck report identity, complete CSV inventory and all current source hashes."""
    if audit_path is None or not audit_path.is_file():
        return {
            "status": "not_audited",
            "errors": ["No audit report is configured or present."],
            "report": None,
        }
    try:
        report = read_json(audit_path)
        root = raw_root.resolve()
        if report.get("report_schema_version") != 2 or report.get("input_directory") != str(root):
            raise DashboardDataError("audit report schema or raw-root linkage does not match")
        if report.get("audit_completed") is not True:
            raise DashboardDataError("audit report records incomplete/failed inspection")
        records = report.get("files")
        if not isinstance(records, list) or report.get("csv_file_count") != len(records):
            raise DashboardDataError("audit file list/count is malformed")
        record_map: dict[str, dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                raise DashboardDataError("audit contains a malformed file record")
            if record["path"] in record_map:
                raise DashboardDataError("audit contains duplicate file paths")
            record_map[record["path"]] = record
        current_inventory = scan_csv_inventory(root)
        if current_inventory.get("truncated"):
            raise DashboardDataError(
                f"raw inventory exceeds the dashboard limit of {MAX_INVENTORY_FILES:,} regular files"
            )
        if current_inventory.get("symlink_count_not_followed", 0):
            raise DashboardDataError("raw tree contains symlinks; audit cannot be trusted")
        actual = set(current_inventory.get("csv_paths", []))
        if actual != set(record_map):
            raise DashboardDataError(
                f"raw CSV inventory changed (added={sorted(actual - set(record_map))[:8]}, "
                f"missing={sorted(set(record_map) - actual)[:8]})"
            )
        for relative, record in record_map.items():
            source = safe_child(root, relative)
            digest = record.get("sha256")
            if (
                record.get("status") == "error"
                or not isinstance(digest, str)
                or file_sha256(source) != digest
            ):
                raise DashboardDataError(f"audit is failed or stale for {relative!r}")
        return {
            "status": "passed",
            "errors": [],
            "report": report,
            "report_sha256": file_sha256(audit_path),
            "verified_csv_count": len(actual),
        }
    except (DashboardDataError, OSError, ValueError) as exc:
        return {"status": "failed_or_stale", "errors": [str(exc)], "report": None}


def run_audit_safely(
    raw_root: Path,
    reports_root: Path,
    *,
    timestamp_column: str | None = None,
    numeric_timestamp_unit: str | None = None,
) -> dict[str, Any]:
    """Run the existing read-only audit and write only to a new, isolated report folder."""
    raw = raw_root.resolve()
    reports = reports_root.resolve()
    if raw_root.is_symlink() or not raw.is_dir():
        raise DashboardDataError("raw root is missing or unsafe")
    if reports_root.is_symlink():
        raise DashboardDataError("reports root must not be a symlink")
    if _is_relative_to(reports, raw):
        raise DashboardDataError("audit outputs must remain outside the raw source tree")
    audits_root = reports / "audits"
    if audits_root.is_symlink():
        raise DashboardDataError("audit output directory must not be a symlink")
    if audits_root.exists() and not audits_root.is_dir():
        raise DashboardDataError("audit output path exists and is not a directory")
    if not _is_relative_to(audits_root.resolve(), reports):
        raise DashboardDataError("audit output directory escapes the configured reports root")
    inventory = scan_csv_inventory(raw)
    if inventory.get("status") == "unsafe" or inventory.get("symlink_count_not_followed", 0):
        raise DashboardDataError("refusing to audit a raw tree containing symlinks")
    if inventory.get("truncated"):
        raise DashboardDataError(
            "refusing to audit a raw tree exceeding the bounded dashboard inventory"
        )
    total_bytes = 0
    for relative in inventory.get("csv_paths", []):
        total_bytes += safe_child(raw, relative).stat().st_size
        if total_bytes > MAX_DASHBOARD_AUDIT_BYTES:
            raise DashboardDataError(
                f"dashboard audit is limited to {MAX_DASHBOARD_AUDIT_BYTES:,} total CSV bytes; use the CLI with appropriate resource planning"
            )
    report = audit_directory(
        raw, timestamp_column=timestamp_column, numeric_timestamp_unit=numeric_timestamp_unit
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = reports / "audits" / f"audit-{stamp}-{uuid.uuid4().hex[:8]}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise DashboardDataError("fresh audit output directory already exists; retry") from exc
    marker = destination / "INCOMPLETE"
    marker.write_text("Audit report writing did not complete.\n", encoding="utf-8")
    audit_path = destination / "audit.json"
    summary_path = destination / "summary.md"
    # Remove INCOMPLETE only after both outputs are written; a failure before
    # unlinking leaves the marker and any partial output visible for inspection.
    with audit_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with summary_path.open("x", encoding="utf-8") as handle:
        handle.write(render_summary(report))
    marker.unlink()
    return {
        "status": "complete" if report["audit_completed"] else "failed",
        "report": report,
        "audit_path": audit_path,
        "summary_path": summary_path,
        "output_dir": destination,
    }


def inspect_raw_artifacts(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise DashboardDataError("raw artifact root is missing or unsafe")
    file_count = scan_csv_inventory(root).get("file_count", 0)
    if file_count > MAX_INVENTORY_FILES:
        raise DashboardDataError(f"inventory is bounded to {MAX_INVENTORY_FILES:,} files")
    try:
        return inspect_artifact_tree(root)
    except (ArtifactInspectionError, OSError) as exc:
        raise DashboardDataError(str(exc)) from exc


def verify_processed_artifacts(processed_root: Path) -> dict[str, Any]:
    """Verify checksums from the existing processing manifest without loading rows."""
    if processed_root.is_symlink() or not processed_root.is_dir():
        return {
            "status": "missing",
            "errors": ["Processed-data directory is unavailable."],
            "manifest": None,
        }
    if (processed_root / "PREPARATION_INCOMPLETE").exists():
        return {
            "status": "incomplete",
            "errors": ["Processing left PREPARATION_INCOMPLETE."],
            "manifest": None,
        }
    manifest_path = processed_root / "processing_manifest.json"
    if manifest_path.is_symlink():
        return {
            "status": "invalid",
            "errors": ["Processing manifest must not be a symlink."],
            "manifest": None,
        }
    if not manifest_path.is_file():
        return {
            "status": "missing",
            "errors": ["No processing manifest is available."],
            "manifest": None,
        }
    try:
        manifest = read_json(manifest_path)
        if manifest.get("processing_schema_version") != 2 or manifest.get("status") != "complete":
            raise DashboardDataError("processing manifest is not schema-v2 complete")
        names = {
            "analysis.csv": "analysis_sha256",
            "excluded_rows.csv": "exclusions_sha256",
            "summary.json": "summary_sha256",
            "DATA_DICTIONARY.md": "data_dictionary_sha256",
        }
        verified = []
        for relative, hash_key in names.items():
            artifact = safe_child(processed_root, relative)
            expected = manifest.get(hash_key)
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise DashboardDataError(
                    f"processing manifest has an invalid checksum for {relative}"
                )
            if file_sha256(artifact) != expected:
                raise DashboardDataError(f"processed artifact checksum mismatch: {relative}")
            verified.append(relative)
        with (processed_root / "analysis.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            reader = csv.reader(handle)
            headers = next(reader, None)
        if not headers or len(set(headers)) != len(headers):
            raise DashboardDataError("analysis.csv header is empty or duplicated")
        if not isinstance(manifest.get("run_records"), list) or not manifest["run_records"]:
            raise DashboardDataError("processing manifest has no run provenance records")
        feature_contract = feature_specification_operational_status(
            manifest,
            require_run_record_unit_proof=True,
        )
        feature_specification = feature_contract.get("specification")
        if isinstance(feature_specification, dict):
            feature_policy = feature_temporal_policy_from_specification(feature_specification)
            if (
                "tcp_retransmission_unit" in manifest
                and manifest.get("tcp_retransmission_unit")
                != feature_specification.get("tcp_retransmission_unit")
            ) or (
                "feature_temporal_policy" in manifest
                and manifest.get("feature_temporal_policy") != feature_policy
            ):
                raise DashboardDataError(
                    "processing manifest TCP unit or feature temporal policy differs from its feature specification"
                )
            summary = read_json(safe_child(processed_root, "summary.json"))
            if (
                (
                    "feature_specification" in summary
                    and summary.get("feature_specification") != feature_specification
                )
                or (
                    "feature_specification_sha256" in summary
                    and summary.get("feature_specification_sha256")
                    != feature_specification_sha256(feature_specification)
                )
                or (
                    "tcp_retransmission_unit" in summary
                    and summary.get("tcp_retransmission_unit")
                    != feature_specification.get("tcp_retransmission_unit")
                )
                or (
                    "feature_temporal_policy" in summary
                    and summary.get("feature_temporal_policy") != feature_policy
                )
            ):
                raise DashboardDataError(
                    "processed summary feature specification, TCP unit, or temporal policy differs from the processing manifest"
                )
        return {
            "status": "verified",
            "errors": [],
            "manifest": manifest,
            "verified_artifacts": verified,
            "columns": headers,
            "analysis_sha256": manifest["analysis_sha256"],
            "manifest_sha256": file_sha256(processed_root / "processing_manifest.json"),
            "feature_specification": feature_specification,
            "feature_specification_sha256": feature_contract.get("feature_specification_sha256"),
            "feature_specification_status": feature_contract.get("status"),
            "feature_specification_operationally_validated": feature_contract.get(
                "operationally_validated"
            ),
            "tcp_retransmission_unit": feature_contract.get("tcp_retransmission_unit"),
            "tcp_retransmission_unit_provenance": feature_contract.get("tcp_unit_provenance"),
            "feature_temporal_policy": feature_contract.get("feature_temporal_policy"),
            "feature_contract_reasons": feature_contract.get("reasons", []),
        }
    except (DashboardDataError, OSError, UnicodeError, StopIteration) as exc:
        return {"status": "invalid", "errors": [str(exc)], "manifest": None}


def _parse_finite(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _state(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


@lru_cache(maxsize=12)
def _read_preview_cached(
    path_text: str,
    digest: str,
    run_id: str,
    fields: tuple[str, ...],
    time_start: float | None,
    time_end: float | None,
    max_points: int,
    expected_rows: int,
    max_gap_seconds: float,
) -> dict[str, Any]:
    path = Path(path_text)
    stride = max(1, math.ceil(expected_rows / max_points))
    labels = Counter()
    feature_missing = Counter()
    selected_rows = 0
    selected_time_min: float | None = None
    selected_time_max: float | None = None
    all_time_min: float | None = None
    all_time_max: float | None = None
    previous_time: float | None = None
    buckets: dict[int, dict[str, Any]] = {}
    event_count = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            missing_columns = sorted((set(fields) | {"run_id", "timestamp_seconds"}) - set(header))
            if missing_columns:
                raise DashboardDataError(f"analysis.csv lacks chart columns: {missing_columns}")
            for raw in reader:
                if raw.get("run_id") != run_id:
                    continue
                timestamp = _parse_finite(raw.get("timestamp_seconds"))
                if timestamp is None:
                    raise DashboardDataError("selected run contains a missing/non-finite timestamp")
                if previous_time is not None and timestamp <= previous_time:
                    raise DashboardDataError("selected run timestamps are not strictly increasing")
                time_gap_before = (
                    previous_time is not None and timestamp - previous_time > max_gap_seconds
                )
                previous_time = timestamp
                all_time_min = timestamp if all_time_min is None else min(all_time_min, timestamp)
                all_time_max = timestamp if all_time_max is None else max(all_time_max, timestamp)
                selected_rows += 1
                label = raw.get("future_event_onset", "")
                if label not in ("", "0", "1"):
                    raise DashboardDataError("selected run has an invalid future_event_onset value")
                if label == "1":
                    labels["positive"] += 1
                elif label == "0":
                    labels["negative"] += 1
                else:
                    labels["censored_or_missing"] += 1
                raw_event_state = raw.get("event_active")
                event_state = _state(raw_event_state)
                if raw_event_state not in (None, "") and str(
                    raw_event_state
                ).strip().lower() not in {"true", "false"}:
                    raise DashboardDataError("selected run has an invalid event_active value")
                if event_state is True:
                    event_count += 1
                if (time_start is not None and timestamp < time_start) or (
                    time_end is not None and timestamp > time_end
                ):
                    continue
                selected_time_min = (
                    timestamp if selected_time_min is None else min(selected_time_min, timestamp)
                )
                selected_time_max = (
                    timestamp if selected_time_max is None else max(selected_time_max, timestamp)
                )
                row_index = selected_rows - 1
                bucket_index = min(max_points - 1, row_index // stride)
                bucket = buckets.setdefault(
                    bucket_index,
                    {
                        "timestamp_seconds": timestamp,
                        "values": {field: None for field in fields},
                        "missing": {field: False for field in fields},
                        "event_states": set(),
                        "labels": set(),
                        "time_gap": False,
                        "sample_count": 0,
                    },
                )
                bucket["timestamp_seconds"] = timestamp
                bucket["sample_count"] += 1
                if time_gap_before:
                    bucket["time_gap"] = True
                if event_state is None:
                    bucket["event_states"].add("unknown")
                else:
                    bucket["event_states"].add(event_state)
                if label == "1":
                    bucket["labels"].add(1)
                elif label == "0":
                    bucket["labels"].add(0)
                else:
                    bucket["labels"].add(None)
                for field in fields:
                    raw_number = raw.get(field)
                    number = _parse_finite(raw_number)
                    if raw_number not in (None, "") and number is None:
                        raise DashboardDataError(
                            f"selected run has a non-finite or non-numeric value in {field!r}"
                        )
                    if number is None:
                        bucket["missing"][field] = True
                        feature_missing[field] += 1
                    else:
                        bucket["values"][field] = number
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DashboardDataError(f"cannot read processed telemetry: {exc}") from exc
    if selected_rows == 0:
        raise DashboardDataError(f"run {run_id!r} is absent from analysis.csv")
    if selected_rows != expected_rows:
        raise DashboardDataError(
            f"manifest says {expected_rows} rows for this run, analysis.csv contains {selected_rows}"
        )
    points: list[dict[str, Any]] = []
    for _, bucket in sorted(buckets.items()):
        gap = bucket["time_gap"]
        point: dict[str, Any] = {
            "timestamp_seconds": bucket["timestamp_seconds"],
            "bucket_sample_count": bucket["sample_count"],
        }
        for field in fields:
            point[field] = None if gap or bucket["missing"][field] else bucket["values"][field]
        states = bucket["event_states"]
        point["event_active"] = (
            states.pop() if len(states) == 1 and "unknown" not in states else None
        )
        bucket_labels = bucket["labels"]
        point["future_event_onset"] = (
            1 if bucket_labels == {1} else 0 if bucket_labels == {0} else None
        )
        point["bucket_contains_gap_or_missing"] = gap or any(bucket["missing"].values())
        points.append(point)
    return {
        "run_id": run_id,
        "points": points,
        "fields": list(fields),
        "source_rows": selected_rows,
        "display_points": len(points),
        "time_min": all_time_min,
        "time_max": all_time_max,
        "selected_time_min": selected_time_min,
        "selected_time_max": selected_time_max,
        "label_counts": dict(labels),
        "observed_active_rows": event_count,
        "missing_counts_in_selected_range": dict(feature_missing),
        "max_gap_seconds": max_gap_seconds,
        "decimation": f"row-bucket last observation; stride {stride}; any missing value or declared max-gap crossing breaks the plotted segment",
        "display_limit": max_points,
    }


def load_bounded_run_preview(
    processed_root: Path,
    manifest: dict[str, Any],
    run_id: str,
    fields: tuple[str, ...],
    *,
    time_start: float | None = None,
    time_end: float | None = None,
    max_points: int = MAX_CHART_POINTS,
) -> dict[str, Any]:
    """Read one run in a bounded-memory pass; cache key contains content + settings hashes."""
    if not 1 <= max_points <= MAX_CHART_POINTS:
        raise DashboardDataError(f"display point limit must be in [1,{MAX_CHART_POINTS}]")
    if time_start is not None and time_end is not None and time_start > time_end:
        raise DashboardDataError("time-range start must not exceed its end")
    if not fields or len(set(fields)) != len(fields):
        raise DashboardDataError(
            "select one or more distinct columns from the verified analysis schema"
        )
    path = safe_child(processed_root, "analysis.csv")
    digest = file_sha256(path)
    expected_digest = manifest.get("analysis_sha256")
    if digest != expected_digest:
        raise DashboardDataError(
            "analysis.csv has changed since its processing manifest was verified"
        )
    run_records = manifest.get("run_records")
    record = (
        next(
            (
                item
                for item in run_records
                if isinstance(item, dict) and item.get("run_id") == run_id
            ),
            None,
        )
        if isinstance(run_records, list)
        else None
    )
    if record is None:
        raise DashboardDataError("selected run is not declared in the processing manifest")
    expected_rows = record.get("retained_timestamp_rows")
    if isinstance(expected_rows, bool) or not isinstance(expected_rows, int) or expected_rows < 1:
        raise DashboardDataError(
            "processing manifest has no valid retained row count for the selected run"
        )
    max_gap = manifest.get("label_parameters", {}).get("max_ground_truth_gap_ms")
    if (
        isinstance(max_gap, bool)
        or not isinstance(max_gap, (int, float))
        or not math.isfinite(max_gap)
        or max_gap <= 0
    ):
        raise DashboardDataError("processing manifest lacks a valid declared gap limit")
    preview = _read_preview_cached(
        str(path),
        digest,
        run_id,
        fields,
        time_start,
        time_end,
        max_points,
        expected_rows,
        float(max_gap) / 1000,
    )
    if file_sha256(path) != digest:
        raise DashboardDataError("processed source changed while serving the cached preview")
    return preview


def clear_preview_cache() -> None:
    _read_preview_cached.cache_clear()


def safe_download(
    path: Path, allowed_root: Path, *, limit_bytes: int = MAX_DOWNLOAD_BYTES
) -> bytes:
    """Read a small verified artifact under an allowlisted root; never follows symlinks."""
    try:
        root = allowed_root.resolve()
        lexical = path if path.is_absolute() else root / path
        if lexical.is_symlink():
            raise DashboardDataError(
                "download path is outside the allowed artifact directory or unsafe"
            )
        try:
            relative = lexical.absolute().relative_to(root)
        except ValueError as exc:
            raise DashboardDataError(
                "download path is outside the allowed artifact directory or unsafe"
            ) from exc
        if ".." in relative.parts:
            raise DashboardDataError("download path may not traverse parent directories")
        probe = root
        for part in relative.parts:
            probe = probe / part
            if probe.is_symlink():
                raise DashboardDataError("download path traverses a symlink")
        resolved = lexical.resolve()
        if not _is_relative_to(resolved, root) or not resolved.is_file():
            raise DashboardDataError(
                "download artifact is missing or outside the allowed directory"
            )
        if not resolved.is_file():
            raise DashboardDataError("download artifact is missing")
        if resolved.stat().st_size > limit_bytes:
            raise DashboardDataError(
                f"artifact exceeds the in-app download limit of {limit_bytes:,} bytes"
            )
        data = resolved.read_bytes()
        if len(data) > limit_bytes:
            raise DashboardDataError("artifact grew beyond the download limit while being read")
        return data
    except OSError as exc:
        raise DashboardDataError(f"cannot read download artifact: {exc}") from exc


def verify_workspace_lineage_sources(
    workspace: Workspace, processed: dict[str, Any], audit_status: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Verify expensive current raw/audit/manifest sources once for a refresh snapshot."""
    if processed.get("status") != "verified":
        raise DashboardDataError("processed artifacts could not be verified")
    processing = processed["manifest"]
    audit_status = audit_status or verify_audit_inventory(
        workspace.raw_root, workspace.audit_report_path
    )
    if audit_status.get("status") != "passed":
        raise DashboardDataError("current raw audit inventory/hashes could not be verified")
    if audit_status.get("report_sha256") != processing.get("raw_audit", {}).get("report_sha256"):
        raise DashboardDataError("current audit report is not the one linked by processed data")
    if workspace.manifest_path is None or not workspace.manifest_path.is_file():
        raise DashboardDataError("current run manifest is unavailable")
    manifest_hash = file_sha256(workspace.manifest_path)
    if manifest_hash != processing.get("source_manifest_sha256"):
        raise DashboardDataError("current run manifest hash differs from processed provenance")
    source_manifest_path = processing.get("source_manifest_path")
    if (
        source_manifest_path
        and Path(source_manifest_path).resolve() != workspace.manifest_path.resolve()
    ):
        raise DashboardDataError("processed data points to a different run-manifest path")
    run_records = processing.get("run_records")
    if not isinstance(run_records, list) or not run_records:
        raise DashboardDataError("processed provenance contains no source run records")
    for source in run_records:
        relative = source.get("source_file") if isinstance(source, dict) else None
        expected = source.get("source_sha256") if isinstance(source, dict) else None
        original = safe_child(workspace.raw_root, relative)
        if file_sha256(original) != expected:
            raise DashboardDataError(f"source hash is stale for {relative!r}")
    if processing.get("alignment_review", {}).get("status") == "approved":
        alignment_manifest = read_json(workspace.manifest_path)
        verify_alignment_review(
            workspace.raw_root.resolve(), alignment_manifest, workspace.manifest_path
        )
    return {
        "status": "verified",
        "processing_manifest_sha256": processed.get("manifest_sha256"),
        "audit_report_sha256": audit_status.get("report_sha256"),
        "source_manifest_sha256": manifest_hash,
    }


def verify_current_lineage(
    workspace: Workspace,
    processed: dict[str, Any],
    report: dict[str, Any],
    *,
    require_frozen: bool = False,
    shared_sources: dict[str, Any] | None = None,
    report_run_records: list[dict[str, Any]] | None = None,
) -> None:
    """Check report links against current artifacts and a shared refresh-time source proof."""
    if processed.get("status") != "verified":
        raise DashboardDataError("processed artifacts could not be verified")
    processing = processed["manifest"]
    feature_contract = feature_specification_operational_status(
        processing,
        require_run_record_unit_proof=True,
    )
    prepared_specification = feature_contract.get("specification")
    report_feature_keys = {
        "feature_specification",
        "feature_specification_sha256",
        "tcp_retransmission_unit",
        "feature_temporal_policy",
        "processing_compatibility",
    }
    report_has_feature_contract = bool(report_feature_keys & set(report))
    if require_frozen and not feature_contract.get("operationally_validated"):
        reasons = "; ".join(str(reason) for reason in feature_contract.get("reasons", []))
        raise DashboardDataError(
            "confirmatory report's prepared feature contract is not operationally "
            f"validated: {reasons or feature_contract.get('status')}"
        )
    if isinstance(prepared_specification, dict):
        prepared_policy = feature_temporal_policy_from_specification(prepared_specification)
        prepared_digest = feature_specification_sha256(prepared_specification)
        if (
            "tcp_retransmission_unit" in processing
            and processing.get("tcp_retransmission_unit")
            != prepared_specification.get("tcp_retransmission_unit")
        ) or (
            "feature_temporal_policy" in processing
            and processing.get("feature_temporal_policy") != prepared_policy
        ):
            raise DashboardDataError(
                "processing manifest TCP unit or temporal policy differs from its feature specification"
            )
        if require_frozen or report_has_feature_contract:
            compatibility = report.get("processing_compatibility")
            compatibility = compatibility if isinstance(compatibility, dict) else {}
            if (
                report.get("feature_specification") != prepared_specification
                or report.get("feature_specification_sha256") != prepared_digest
                or report.get("tcp_retransmission_unit")
                != prepared_specification.get("tcp_retransmission_unit")
                or report.get("feature_temporal_policy") != prepared_policy
                or (
                    (require_frozen or "processing_compatibility" in report)
                    and (
                        compatibility.get("feature_specification") != prepared_specification
                        or compatibility.get("feature_specification_sha256") != prepared_digest
                        or compatibility.get("tcp_retransmission_unit")
                        != prepared_specification.get("tcp_retransmission_unit")
                        or compatibility.get("feature_temporal_policy") != prepared_policy
                    )
                )
            ):
                raise DashboardDataError(
                    "report feature specification, TCP retransmission unit, or temporal policy differs from prepared provenance"
                )
    elif require_frozen or report_has_feature_contract:
        raise DashboardDataError(
            "report feature contract cannot be checked because the prepared feature specification is invalid"
        )
    if report_run_records is None:
        report_run_records = report.get("source_runs")
    if report_run_records is not None:

        def run_links(records: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
            links: dict[str, tuple[str, str]] = {}
            for item in records:
                if not isinstance(item, dict):
                    raise DashboardDataError("report contains a malformed source-run record")
                run_id, source_file, source_hash = (
                    item.get("run_id"),
                    item.get("source_file"),
                    item.get("source_sha256"),
                )
                if not all(
                    isinstance(value, str) and value for value in (run_id, source_file, source_hash)
                ):
                    raise DashboardDataError("report contains an incomplete run-source link")
                if run_id in links:
                    raise DashboardDataError("report contains duplicate source run IDs")
                links[run_id] = (source_file, source_hash)
            return links

        expected_records = processing.get("run_records")
        if not isinstance(expected_records, list):
            raise DashboardDataError("processing manifest lacks source run records")
        if run_links(report_run_records) != run_links(expected_records):
            raise DashboardDataError("report source-run links differ from the processing manifest")
    elif require_frozen:
        raise DashboardDataError("confirmatory report lacks source-run provenance records")
    if report.get("analysis_sha256") != processed.get("analysis_sha256") or report.get(
        "processing_manifest_sha256"
    ) != processed.get("manifest_sha256"):
        raise DashboardDataError("report does not match current processed artifacts")
    if report.get("source_manifest_sha256") and report.get(
        "source_manifest_sha256"
    ) != processing.get("source_manifest_sha256"):
        raise DashboardDataError(
            "report source-manifest fingerprint differs from processed provenance"
        )
    report_audit = report.get("raw_audit")
    if isinstance(report_audit, dict):
        current_audit_ref = processing.get("raw_audit", {})
        if report_audit.get("report_sha256") != current_audit_ref.get("report_sha256"):
            raise DashboardDataError("report audit fingerprint differs from processed provenance")
    config_hash = processing.get("experiment_config_sha256")
    if config_hash:
        if not workspace.config_path.is_file() or file_sha256(workspace.config_path) != config_hash:
            raise DashboardDataError(
                "current configuration hash differs from the processed artifact"
            )
        config = load_config(workspace.config_path)
        if require_frozen or config.get("status") == "protocol_frozen":
            if config.get("status") != "protocol_frozen":
                raise DashboardDataError(
                    "confirmatory report requires a frozen current configuration"
                )
            verify_protocol_freeze_evidence(config, workspace.config_path)
        if isinstance(prepared_specification, dict):
            try:
                configured_policy = normalize_feature_temporal_policy(
                    (config.get("processing") or {}).get("feature_temporal_policy"),
                    require_resolved=require_frozen,
                )
            except ProcessingError as exc:
                raise DashboardDataError(str(exc)) from exc
            if configured_policy != feature_temporal_policy_from_specification(
                prepared_specification
            ):
                raise DashboardDataError(
                    "prepared feature temporal policy differs from the current configuration"
                )
        if report.get("config_sha256") and report.get("config_sha256") != config_hash:
            raise DashboardDataError("report configuration fingerprint differs from processed data")
    elif require_frozen:
        raise DashboardDataError("confirmatory report lacks a processed configuration fingerprint")
    shared = shared_sources or verify_workspace_lineage_sources(workspace, processed)
    if shared.get("status") != "verified":
        raise DashboardDataError(
            shared.get("error", "current raw source linkage could not be verified")
        )
    if shared.get("processing_manifest_sha256") != processed.get("manifest_sha256"):
        raise DashboardDataError(
            "shared source verification does not match current processed artifacts"
        )
    if shared.get("audit_report_sha256") != processing.get("raw_audit", {}).get("report_sha256"):
        raise DashboardDataError("current audit report is not the one linked by processed data")
    if shared.get("source_manifest_sha256") != processing.get("source_manifest_sha256"):
        raise DashboardDataError("current run manifest is not the one linked by processed data")


def verify_evaluation_bundle(
    report_dir: Path,
    workspace: Workspace,
    *,
    processed_artifacts: dict[str, Any] | None = None,
    shared_sources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check report-internal integrity plus source, protocol, processed and audit linkage."""
    try:
        internal = verify_evaluation_artifacts(report_dir, processed_dir=workspace.processed_root)
        report = read_json(report_dir / "evaluation_manifest.json")
        status = report.get("status")
        if status != "complete":
            return {
                "status": "failed_or_incomplete",
                "internal_status": internal["status"],
                "report": report,
                "errors": [f"evaluation outcome is {status!r}, not complete"],
            }
        if report.get("protocol_status") != "protocol_frozen" or report.get(
            "protocol_sha256"
        ) != report.get("config_sha256"):
            raise DashboardDataError("evaluation manifest does not identify its frozen protocol")
        processed = processed_artifacts or verify_processed_artifacts(workspace.processed_root)
        verify_current_lineage(
            workspace, processed, report, require_frozen=True, shared_sources=shared_sources
        )
        if report.get("source_manifest_sha256") != processed["manifest"].get(
            "source_manifest_sha256"
        ):
            raise DashboardDataError(
                "evaluation source-manifest fingerprint differs from processed provenance"
            )
        return {
            "status": "complete_validated_provenance",
            "internal_status": internal["status"],
            "report": report,
            "errors": [],
            "verified_artifacts": internal.get("verified_artifacts", []),
            "processed_verified": True,
            "raw_sources_verified": True,
            "audit_verified": True,
            "config_verified": True,
            "protocol_freeze_verified": True,
            "feature_specification": report.get("feature_specification"),
            "feature_specification_sha256": report.get("feature_specification_sha256"),
            "feature_specification_status": (
                report.get("processing_compatibility", {}).get("feature_specification_status")
            ),
            "tcp_retransmission_unit": report.get("tcp_retransmission_unit"),
            "feature_temporal_policy": report.get("feature_temporal_policy"),
        }
    except (
        DashboardDataError,
        ReportError,
        ConfigError,
        ProcessingError,
        OSError,
        ValueError,
    ) as exc:
        return {
            "status": "provenance_unverified",
            "errors": [str(exc)],
            "report": None,
            "internal_status": "unknown",
        }


def verify_ground_truth_bundle(
    report_dir: Path,
    workspace: Workspace,
    *,
    processed_artifacts: dict[str, Any] | None = None,
    shared_sources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        internal = verify_ground_truth_report(report_dir)
        report = read_json(report_dir / "ground_truth_report.json")
        processed = processed_artifacts or verify_processed_artifacts(workspace.processed_root)
        verify_current_lineage(
            workspace,
            processed,
            report,
            shared_sources=shared_sources,
            report_run_records=report.get("per_run"),
        )
        return {"status": "verified", "report": report, "internal": internal, "errors": []}
    except (GroundTruthReportError, DashboardDataError, OSError, ValueError) as exc:
        return {"status": "unverified", "report": None, "errors": [str(exc)]}


def verify_robustness_bundle(
    report_dir: Path,
    workspace: Workspace,
    *,
    processed_artifacts: dict[str, Any] | None = None,
    shared_sources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        internal = verify_robustness_artifacts(
            report_dir,
            processed_dir=workspace.processed_root,
            raw_root=workspace.raw_root,
        )
        report = read_json(report_dir / "robustness_manifest.json")
        outcome_status = internal.get("status")
        if (
            outcome_status not in {"complete", "partial_failure"}
            or report.get("status") != outcome_status
        ):
            raise DashboardDataError(
                "robustness manifest outcome disagrees with its verified artifact record"
            )
        processed = processed_artifacts or verify_processed_artifacts(workspace.processed_root)
        verify_current_lineage(workspace, processed, report, shared_sources=shared_sources)
        return {
            "status": "verified" if outcome_status == "complete" else "partial_failure",
            "outcome_status": outcome_status,
            "provenance_status": "current_lineage_verified",
            "report": report,
            "internal": internal,
            "errors": (
                []
                if outcome_status == "complete"
                else ["robustness outcome is partial_failure, not complete"]
            ),
        }
    except (RobustnessError, DashboardDataError, OSError, ValueError) as exc:
        return {"status": "unverified", "report": None, "errors": [str(exc)]}


def discover_report_directories(
    reports_root: Path, manifest_name: str, *, max_reports: int = 250
) -> list[Path]:
    if reports_root.is_symlink() or not reports_root.is_dir():
        return []
    matches: list[Path] = []
    inspected = 0
    for current, dirs, files in os.walk(reports_root, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(current) / name).is_symlink()]
        inspected += len(dirs) + len(files)
        if inspected > MAX_INVENTORY_FILES:
            break
        if manifest_name in files:
            candidate = Path(current) / manifest_name
            if not candidate.is_symlink():
                matches.append(candidate.parent)
    matches.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    return matches[:max_reports]


def classify_workspace_state(
    *,
    inventory: dict[str, Any],
    audit: dict[str, Any],
    protocol_status: str | None,
    processed_status: str | None,
    alignment_status: str | None,
    evaluation_status: str | None = None,
    evaluation_validated: bool = False,
    evaluation_in_progress: bool = False,
) -> tuple[str, str]:
    """Return a visible state label and concise cause; no color-based semantics."""
    if inventory.get("status") in {"missing", "no_data"} or inventory.get("csv_count", 0) == 0:
        return (
            "No data available",
            "No source CSVs are configured/discovered; no analysis result is available.",
        )
    if evaluation_in_progress:
        return (
            "Evaluation in progress",
            "An INCOMPLETE marker exists; do not interpret partial output as a result.",
        )
    if evaluation_status in {"failed", "partial_failure", "failed_or_incomplete", "incomplete"}:
        return (
            "Evaluation failed or incomplete",
            "The latest report is failed/partial or contains unresolved method failures.",
        )
    if evaluation_validated:
        return (
            "Evaluation completed with validated provenance",
            "Report artifacts and linked input/config/source hashes were rechecked.",
        )
    if audit.get("status") == "failed_or_stale":
        return (
            "Audit failed or stale",
            "The audit failed, its inventory differs, or source hashes changed.",
        )
    if audit.get("status") != "passed":
        return (
            "Data discovered but not audited",
            "An explicit audit is required before readiness can be established.",
        )
    if alignment_status in {"review_required", "invalid", "approved_pending"}:
        return (
            "Awaiting human alignment approval",
            "Separate-stream alignment remains blocked until its exact fingerprint is reviewed.",
        )
    if protocol_status == "exploratory_unfrozen" or processed_status == "exploratory_unfrozen":
        return (
            "Descriptive / exploratory analysis",
            "These labels or results are not confirmatory; re-prepare after a separate protocol freeze.",
        )
    if protocol_status != "protocol_frozen":
        return (
            "Protocol draft",
            "Research-defining choices remain unresolved; confirmatory evaluation is blocked.",
        )
    if processed_status == "protocol_frozen":
        return (
            "Frozen preparation",
            "Prepared inputs match the frozen protocol, but no current validated evaluation is shown.",
        )
    return (
        "Audit passed; schema/clock semantics remain unverified",
        "A completed audit is not semantic, timing, or scientific approval.",
    )


def available_feature_units(field: str, run_record: dict[str, Any] | None = None) -> str:
    mapping = (run_record or {}).get("column_units", {})
    if field == "tcp_retransmission_delta":
        unit = mapping.get(field) if isinstance(mapping, dict) else None
        if unit in {"packets", "bytes"}:
            return f"{unit} increment/sample (source-declared)"
        return "exact packets-or-bytes unit not proven"
    fixed = {
        "timestamp_seconds": "seconds since run origin",
        "timestamp_original": "source clock representation",
        "rtt_ms": "ms",
        "loss_fraction": "fraction",
        "goodput_bps": "bits/s",
        "queue_backlog_bytes": "bytes",
        "qdisc_queue_delay_proxy_ms": "ms (backlog-derived serialization-delay proxy)",
        "qdisc_drops_delta": "count/sample (only if explicitly mapped)",
        "qdisc_overlimits_delta": "count/sample (only if explicitly mapped)",
    }
    if field in fixed:
        return fixed[field]
    for canonical, unit in mapping.items():
        if field.startswith(canonical + "_"):
            return str(unit)
    return "derived feature; see processed data dictionary"


def app_version() -> str:
    return __version__
