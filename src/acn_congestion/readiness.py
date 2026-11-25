"""Machine-readable data and protocol readiness checks; never a scientific approval."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import ConfigError, load_config, validate_config, verify_protocol_freeze_evidence
from .processing import (
    ENDPOINT_FIELDS,
    ProcessingError,
    _audit_timestamp_unit_compatible,
    _load_run,
    declared_tcp_retransmission_units_from_run_records,
    feature_specification_operational_status,
    file_sha256,
    normalize_feature_temporal_policy,
    verify_alignment_review,
)
from .reporting import ReportError, load_analysis_dataset, validate_processing_compatibility
from .splits import SplitError, make_group_split


class ReadinessError(ValueError):
    """Readiness report inputs could not be inspected safely."""


def _check(
    checks: list[dict[str, Any]], check_id: str, status: str, message: str, evidence: Any = None
) -> None:
    item: dict[str, Any] = {"id": check_id, "status": status, "message": message}
    if evidence is not None:
        item["evidence"] = evidence
    checks.append(item)


def _safe_source(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ProcessingError(f"source path escapes raw root: {relative!r}") from exc
    if not path.is_file():
        raise ProcessingError(f"required source file is missing: {relative!r}")
    return path


def _event_counts(rows: list[dict[str, Any]], max_gap_seconds: float | None) -> dict[str, int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("run_id"))].append(row)
    identified = 0
    observed_active_episodes = 0
    unidentified = 0
    for run in grouped.values():
        previous_state: bool | None = None
        previous_time: float | None = None
        for row in run:
            state = row.get("event_active")
            time = float(row["timestamp_seconds"])
            gap = (
                previous_time is not None
                and max_gap_seconds is not None
                and time - previous_time > max_gap_seconds
            )
            if state is True and (previous_state is not True or gap):
                observed_active_episodes += 1
                if previous_state is False and not gap:
                    identified += 1
                else:
                    unidentified += 1
            previous_state = state
            previous_time = time
    return {
        "identified_onset_events": identified,
        "observed_active_episodes_including_left_or_gap_censored": observed_active_episodes,
        "episodes_without_identified_onset": unidentified,
    }


def build_readiness_report(
    *,
    raw_root: Path,
    manifest_path: Path,
    audit_report_path: Path,
    config_path: Path | None = None,
    processed_dir: Path | None = None,
    minimum_independent_configurations: int | None = None,
    minimum_runs_per_configuration: int | None = None,
    minimum_identified_events: int | None = None,
    minimum_events_per_partition: int | None = None,
) -> dict[str, Any]:
    """Inspect the explicit v2 aligned-CSV contract and optional prepared artifacts.

    Numeric minima are optional user-supplied gates. They are compared mechanically
    and are never described as evidence of power or sample sufficiency.
    """
    checks: list[dict[str, Any]] = []
    root = raw_root.resolve()
    manifest_file = manifest_path.resolve()
    audit_file = audit_report_path.resolve()
    config_file = config_path.resolve() if config_path is not None else None
    for name, value in (
        ("minimum_independent_configurations", minimum_independent_configurations),
        ("minimum_runs_per_configuration", minimum_runs_per_configuration),
        ("minimum_identified_events", minimum_identified_events),
        ("minimum_events_per_partition", minimum_events_per_partition),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ReadinessError(f"{name} must be a positive integer when supplied")

    raw_ok = root.is_dir()
    _check(
        checks,
        "raw_root",
        "pass" if raw_ok else "fail",
        "raw root exists" if raw_ok else "raw root is missing or is not a directory",
        str(root),
    )
    manifest_ok = manifest_file.is_file()
    _check(
        checks,
        "run_manifest_file",
        "pass" if manifest_ok else "fail",
        "run manifest exists" if manifest_ok else "run manifest file is missing",
        str(manifest_file),
    )
    audit_ok = audit_file.is_file()
    _check(
        checks,
        "audit_report_file",
        "pass" if audit_ok else "fail",
        "audit report exists" if audit_ok else "audit report file is missing",
        str(audit_file),
    )

    configuration: dict[str, Any] | None = None
    freeze_verified = False
    if config_file is not None:
        try:
            configuration = load_config(config_file)
            if configuration.get("status") == "protocol_frozen":
                validate_config(configuration, require_analysis_ready=True)
                verify_protocol_freeze_evidence(configuration, config_file)
                freeze_verified = True
                _check(
                    checks,
                    "protocol_freeze",
                    "pass",
                    "frozen config and decision-record file/hash verify; scientific justification remains human-reviewed",
                    {
                        "config_sha256": file_sha256(config_file),
                        "protocol_id": configuration.get("protocol_freeze", {}).get("protocol_id"),
                    },
                )
            else:
                _check(
                    checks,
                    "protocol_freeze",
                    "warning",
                    "config is not frozen; descriptive inspection may proceed, confirmatory evaluation is blocked",
                    {"status": configuration.get("status")},
                )
        except (ConfigError, OSError, ValueError) as exc:
            _check(checks, "protocol_freeze", "fail", f"config/freeze evidence is invalid: {exc}")
    else:
        _check(
            checks,
            "protocol_freeze",
            "unknown",
            "no config file supplied; protocol freeze cannot be checked",
        )

    if configuration is not None:
        try:
            feature_temporal_policy = normalize_feature_temporal_policy(
                (configuration.get("processing") or {}).get("feature_temporal_policy")
            )
        except ProcessingError as exc:
            _check(
                checks,
                "feature_temporal_policy",
                "fail",
                f"configured feature temporal policy is invalid: {exc}",
            )
        else:
            resolved = feature_temporal_policy["status"] == "resolved"
            _check(
                checks,
                "feature_temporal_policy",
                "pass"
                if resolved
                else "fail"
                if configuration.get("status") == "protocol_frozen"
                else "warning",
                (
                    "configured feature-gap/coverage policy is explicitly resolved; "
                    "software does not validate its scientific appropriateness"
                    if resolved
                    else "feature-gap/coverage policy is explicitly unresolved; descriptive "
                    "analysis remains non-confirmatory and confirmatory evaluation/model "
                    "eligibility are blocked"
                ),
                feature_temporal_policy,
            )
    elif config_file is not None:
        _check(
            checks,
            "feature_temporal_policy",
            "unknown",
            "feature temporal policy cannot be checked because the config did not load",
        )
    else:
        _check(
            checks,
            "feature_temporal_policy",
            "unknown",
            "no config file supplied; feature temporal policy cannot be checked",
        )

    manifest: dict[str, Any] | None = None
    if manifest_ok:
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            valid_manifest = (
                isinstance(manifest, dict)
                and manifest.get("schema_version") == 2
                and isinstance(manifest.get("runs"), list)
                and bool(manifest["runs"])
            )
            if not valid_manifest:
                raise ReadinessError(
                    "expected JSON object, schema_version=2, and a non-empty runs list"
                )
            run_ids = [entry.get("run_id") for entry in manifest["runs"] if isinstance(entry, dict)]
            if (
                len(run_ids) != len(manifest["runs"])
                or any(not isinstance(item, str) or not item for item in run_ids)
                or len(set(run_ids)) != len(run_ids)
            ):
                raise ReadinessError("runs must have unique, non-empty string run_id values")
            _check(
                checks,
                "manifest_schema",
                "pass",
                "manifest uses schema version 2 with unique run IDs",
                {"run_count": len(run_ids), "manifest_sha256": file_sha256(manifest_file)},
            )
        except (OSError, json.JSONDecodeError, ReadinessError) as exc:
            manifest = None
            _check(checks, "manifest_schema", "fail", f"invalid run manifest: {exc}")
    else:
        _check(checks, "manifest_schema", "fail", "run manifest is unavailable")

    if manifest is not None:
        configured_fields = tuple(
            (configuration.get("evaluation", {}) if configuration is not None else {}).get(
                "endpoint_feature_allowlist", ENDPOINT_FIELDS
            )
        )
        tcp_provenance = declared_tcp_retransmission_units_from_run_records(
            manifest.get("runs"), configured_fields
        )
        evidence = {
            **tcp_provenance,
            "configured_endpoint_fields": list(configured_fields),
        }
        if tcp_provenance["status"] == "not_applicable":
            _check(
                checks,
                "tcp_retransmission_unit_consistency",
                "not_applicable",
                "configured endpoint features do not use tcp_retransmission_delta",
                evidence,
            )
        elif tcp_provenance["status"] == "declared":
            _check(
                checks,
                "tcp_retransmission_unit_consistency",
                "pass",
                "every run mapping tcp_retransmission_delta declares the same exact unit",
                evidence,
            )
        elif tcp_provenance["status"] == "missing" and not tcp_provenance["runs_using_feature"]:
            _check(
                checks,
                "tcp_retransmission_unit_consistency",
                "warning",
                "configured features include tcp_retransmission_delta but no run maps it; "
                "preparation under that exact feature list cannot bind a TCP unit",
                evidence,
            )
        else:
            _check(
                checks,
                "tcp_retransmission_unit_consistency",
                "fail",
                "runs mapping tcp_retransmission_delta have missing, invalid, or conflicting "
                "packets/bytes declarations; packets and bytes are not interchangeable",
                evidence,
            )
    else:
        _check(
            checks,
            "tcp_retransmission_unit_consistency",
            "unknown",
            "cross-run TCP retransmission units await a valid run manifest",
        )

    audit: dict[str, Any] | None = None
    audit_items: dict[str, dict[str, Any]] = {}
    if audit_ok:
        try:
            audit = json.loads(audit_file.read_text(encoding="utf-8"))
            if not isinstance(audit, dict) or audit.get("report_schema_version") != 2:
                raise ReadinessError("expected audit report_schema_version=2")
            audited_directory = audit.get("input_directory")
            if not isinstance(audited_directory, str) or not audited_directory:
                raise ReadinessError("audit input_directory is missing or malformed")
            if Path(audited_directory).resolve() != root:
                raise ReadinessError("audit input_directory does not match raw_root")
            items = audit.get("files")
            if not isinstance(items, list) or any(
                not isinstance(item, dict) or not isinstance(item.get("path"), str)
                for item in items
            ):
                raise ReadinessError("audit report has malformed file inventory")
            audit_items = {item["path"]: item for item in items}
            if len(audit_items) != len(items) or audit.get("csv_file_count") != len(items):
                raise ReadinessError("audit file count/inventory is duplicated or inconsistent")
            if audit.get("audit_completed") is not True:
                raise ReadinessError("raw audit is not complete")
            actual_paths = {
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() == ".csv"
            }
            if actual_paths != set(audit_items):
                raise ReadinessError(
                    f"raw CSV inventory differs from audit (added={sorted(actual_paths - set(audit_items))}, "
                    f"missing={sorted(set(audit_items) - actual_paths)})"
                )
            stale = [
                relative
                for relative, item in audit_items.items()
                if not _safe_source(root, relative).is_file()
                or file_sha256(_safe_source(root, relative)) != item.get("sha256")
            ]
            if stale:
                raise ReadinessError(f"audit hashes are missing or stale for: {stale}")
            _check(
                checks,
                "audit_inventory_and_hashes",
                "pass",
                "completed audit inventory and source hashes match the current raw CSV tree",
                {"csv_file_count": len(audit_items), "audit_sha256": file_sha256(audit_file)},
            )
        except (OSError, json.JSONDecodeError, ReadinessError, ProcessingError) as exc:
            audit = None
            audit_items = {}
            _check(
                checks,
                "audit_inventory_and_hashes",
                "fail",
                f"audit cannot be trusted for preparation: {exc}",
            )
    else:
        _check(checks, "audit_inventory_and_hashes", "fail", "audit report is unavailable")

    run_records: list[dict[str, Any]] = []
    source_failures: list[dict[str, str]] = []
    target_counts: Counter[str] = Counter()
    timestamp_statuses: list[str] = []
    if manifest is not None and raw_ok:
        allowed_fields = set(ENDPOINT_FIELDS)
        if configuration is not None:
            allowed_fields = set(
                configuration.get("evaluation", {}).get("endpoint_feature_allowlist", [])
            )
        try:
            alignment_review = verify_alignment_review(root, manifest, manifest_file)
            review_status = alignment_review["status"]
            _check(
                checks,
                "alignment_review",
                "fail" if review_status == "review_required" else "pass",
                "alignment provenance has a hash-linked human decision"
                if review_status == "approved"
                else "no stream-alignment review gate applies to this manifest"
                if review_status == "not_applicable"
                else "stream-alignment approval is incomplete",
                alignment_review,
            )
        except ProcessingError as exc:
            alignment_review = {"status": "invalid", "message": str(exc)}
            _check(checks, "alignment_review", "fail", str(exc), alignment_review)
        seen_hashes: dict[str, str] = {}
        for entry in manifest["runs"]:
            run_id = entry.get("run_id", "<invalid>") if isinstance(entry, dict) else "<invalid>"
            try:
                if not isinstance(entry, dict):
                    raise ProcessingError("run entry must be an object")
                rows, _, record = _load_run(root, entry, allowed_fields)
                relative = entry["file"]
                source_hash = record["source_sha256"]
                if source_hash in seen_hashes:
                    raise ProcessingError(
                        f"source content duplicates run {seen_hashes[source_hash]!r}; not an independent run file"
                    )
                seen_hashes[source_hash] = run_id
                audit_item = audit_items.get(relative)
                if audit is None or audit_item is None:
                    raise ProcessingError(
                        f"source {relative!r} is not covered by a current valid audit"
                    )
                if audit_item.get("status") == "error" or audit_item.get("sha256") != source_hash:
                    raise ProcessingError(f"audit error or hash mismatch for {relative!r}")
                _audit_timestamp_unit_compatible(
                    audit,
                    audit_item,
                    record["timestamp_source_column"],
                    entry["timestamp_unit"],
                )
                kinds = audit_item.get("timestamp_timezone_kinds", {})
                has_numeric = any(
                    isinstance(kind, str) and kind.startswith("numeric_") for kind in kinds
                )
                corroborated = bool(kinds) and (
                    not has_numeric
                    or bool(
                        audit.get("numeric_timestamp_unit_override")
                        or record.get("timestamp_source_column", "")
                        .lower()
                        .endswith(("_ns", "_us", "_ms"))
                    )
                )
                timestamp_statuses.append("corroborated" if corroborated else "unverified")
                target_counts["rows"] += len(rows)
                target_counts["valid_queue_backlog_rows"] += sum(
                    row.get("queue_backlog_bytes") is not None for row in rows
                )
                run_records.append(record)
            except (KeyError, TypeError, OSError, ProcessingError, ValueError) as exc:
                source_failures.append({"run_id": str(run_id), "message": str(exc)})
        if source_failures:
            _check(
                checks,
                "source_schemas_and_units",
                "fail",
                "one or more required run files, mapped columns, units, or measurements failed ingestion validation",
                source_failures,
            )
        else:
            _check(
                checks,
                "source_schemas_and_units",
                "pass",
                "every manifest source passes the same read-only schema/unit parser used by preparation",
                {
                    "validated_runs": len(run_records),
                    "required_per_run": [
                        "timestamp",
                        "queue_backlog_bytes",
                        "at least one configured endpoint feature",
                    ],
                    "unique_source_hashes": len(seen_hashes),
                },
            )
        all_timestamps_corroborated = len(timestamp_statuses) == len(manifest["runs"]) and all(
            value == "corroborated" for value in timestamp_statuses
        )
        _check(
            checks,
            "timestamp_semantics",
            "pass" if all_timestamps_corroborated else "fail",
            "timestamp source/unit matches audit evidence; timezone-naive ISO timestamps are rejected. Clock truth and cross-stream alignment still need human evidence."
            if all_timestamps_corroborated
            else "timestamp unit/format is uncorroborated or at least one run could not be checked",
            {
                "per_run_status": timestamp_statuses,
                "alignment_review": "human review required: clock source, timezone/offset, drift, and endpoint/qdisc alignment",
            },
        )
        valid_target = target_counts.get("valid_queue_backlog_rows", 0)
        total_target = target_counts.get("rows", 0)
        if total_target and valid_target:
            _check(
                checks,
                "ground_truth_measurements",
                "warning",
                "mapped queue-backlog values are parseable, but qdisc source, units, interface, and configured-rate semantics require human verification",
                {
                    "valid_backlog_rows": valid_target,
                    "timestamped_rows": total_target,
                    "missing_or_invalid_backlog_rows": total_target - valid_target,
                    "valid_fraction": valid_target / total_target,
                },
            )
        else:
            _check(
                checks,
                "ground_truth_measurements",
                "fail",
                "no parseable queue-backlog ground-truth measurements were found",
            )
    elif manifest is None:
        _check(
            checks,
            "source_schemas_and_units",
            "unknown",
            "source schema validation awaits a valid run manifest",
        )
        _check(checks, "timestamp_semantics", "unknown", "timestamp semantics cannot be assessed")
        _check(checks, "ground_truth_measurements", "unknown", "ground truth cannot be assessed")

    group_counts = Counter(record["configuration_id"] for record in run_records)
    distinct_run_count = len(run_records)
    distinct_config_count = len(group_counts)
    independent_gate = {
        "status": "not_assessed",
        "minimum_supplied": minimum_independent_configurations,
    }
    if minimum_independent_configurations is not None:
        independent_gate = {
            "status": "pass"
            if distinct_config_count >= minimum_independent_configurations
            else "below_user_supplied_minimum",
            "observed_configurations": distinct_config_count,
            "minimum_supplied": minimum_independent_configurations,
            "note": "A user-supplied count gate is not a power or scientific sufficiency justification.",
        }
    _check(
        checks,
        "independent_configuration_count",
        independent_gate["status"],
        "independent configuration count compared with explicitly supplied gate"
        if minimum_independent_configurations
        else "configuration count reported; scientific minimum not supplied",
        {
            "configuration_count": distinct_config_count,
            "run_count": distinct_run_count,
            "runs_per_configuration": dict(sorted(group_counts.items())),
            "mechanical_three_partition_minimum_met": distinct_config_count >= 3,
            "user_gate": independent_gate,
        },
    )
    replication_gate = {
        "status": "not_assessed",
        "minimum_supplied": minimum_runs_per_configuration,
    }
    if minimum_runs_per_configuration is not None:
        under = {
            key: count
            for key, count in group_counts.items()
            if count < minimum_runs_per_configuration
        }
        replication_gate = {
            "status": "pass" if not under else "below_user_supplied_minimum",
            "minimum_supplied": minimum_runs_per_configuration,
            "under_minimum": under,
            "note": "Repeated runs are nested within configurations and are not independent configuration units.",
        }
    _check(
        checks,
        "run_replication",
        replication_gate["status"],
        "repeats are counted per configuration; they do not increase the configuration-level independent N",
        replication_gate,
    )
    _check(
        checks,
        "endpoint_qdisc_alignment",
        "warning",
        "the current adapter accepts one aligned CSV per run; software cannot prove common clocks or endpoint/router alignment",
        {"adapter_contract": "one aligned CSV per run", "human_review_required": True},
    )

    prepared_rows: list[dict[str, Any]] | None = None
    processing_manifest: dict[str, Any] | None = None
    processed_status = "not_supplied"
    event_results: dict[str, Any] = {"status": "not_assessed"}
    if processed_dir is not None:
        processed_root = processed_dir.resolve()
        try:
            prepared_rows, processing_manifest = load_analysis_dataset(processed_root)
            processed_status = "valid"
            source_manifest_match = processing_manifest.get(
                "source_manifest_sha256"
            ) == file_sha256(manifest_file)
            if not source_manifest_match:
                raise ReadinessError(
                    "processing manifest is linked to a different/stale run manifest"
                )
            raw_audit = processing_manifest.get("raw_audit", {})
            if not isinstance(raw_audit, dict) or raw_audit.get("report_sha256") != file_sha256(
                audit_file
            ):
                raise ReadinessError(
                    "processing manifest is linked to a different/stale raw audit report"
                )
            raw_records = processing_manifest.get("run_records", [])
            stale_raw = []
            for record in raw_records:
                relative = record.get("source_file")
                try:
                    source = _safe_source(root, relative)
                    if file_sha256(source) != record.get("source_sha256"):
                        stale_raw.append(relative)
                except (ProcessingError, TypeError):
                    stale_raw.append(str(relative))
            if stale_raw:
                raise ReadinessError(
                    f"processed data refer to missing/changed raw sources: {stale_raw}"
                )
            _check(
                checks,
                "processed_artifacts",
                "pass",
                "typed loader checks analysis, exclusions, summary, manifest, and data-quality reconciliation",
                {
                    "analysis_rows": len(prepared_rows),
                    "analysis_sha256": processing_manifest.get("analysis_sha256"),
                    "protocol_status": processing_manifest.get("protocol_status"),
                },
            )
            feature_contract = feature_specification_operational_status(
                processing_manifest, require_run_record_unit_proof=True
            )
            feature_contract_status = {
                "current_contract_complete": "pass",
                "legacy_unvalidated": "warning",
                "unresolved_temporal_policy": "warning",
            }.get(str(feature_contract.get("status")), "fail")
            _check(
                checks,
                "processed_feature_contract",
                feature_contract_status,
                (
                    "prepared artifacts bind the current feature specification, exact TCP "
                    "unit provenance, and resolved temporal policy"
                    if feature_contract_status == "pass"
                    else "prepared feature contract is descriptive-only or incomplete; "
                    "confirmatory evaluation and model eligibility remain blocked"
                ),
                {
                    "status": feature_contract.get("status"),
                    "operationally_validated": feature_contract.get("operationally_validated"),
                    "specification_version": feature_contract.get("specification_version"),
                    "feature_specification_sha256": feature_contract.get(
                        "feature_specification_sha256"
                    ),
                    "tcp_retransmission_unit": feature_contract.get("tcp_retransmission_unit"),
                    "tcp_unit_provenance": feature_contract.get("tcp_unit_provenance"),
                    "feature_temporal_policy": feature_contract.get("feature_temporal_policy"),
                    "feature_temporal_eligibility_counts": processing_manifest.get(
                        "feature_temporal_eligibility_counts"
                    ),
                    "reasons": feature_contract.get("reasons"),
                },
            )
            max_gap_s = (
                float(
                    processing_manifest.get("label_parameters", {}).get(
                        "max_ground_truth_gap_ms", 0
                    )
                )
                / 1000
            )
            event_counts = _event_counts(prepared_rows, max_gap_s if max_gap_s > 0 else None)
            event_counts["positive_onset_anchors"] = sum(
                row.get("future_event_onset") == 1 for row in prepared_rows
            )
            event_counts["negative_anchors"] = sum(
                row.get("future_event_onset") == 0 for row in prepared_rows
            )
            event_counts["censored_or_missing_anchors"] = sum(
                row.get("future_event_onset") is None for row in prepared_rows
            )
            event_results = {"status": "counted", **event_counts}
            if minimum_identified_events is None:
                event_results["minimum_gate"] = "not_assessed"
            else:
                event_results["minimum_gate"] = (
                    "pass"
                    if event_counts["identified_onset_events"] >= minimum_identified_events
                    else "below_user_supplied_minimum"
                )
                event_results["minimum_identified_events_supplied"] = minimum_identified_events
                event_results["minimum_gate_note"] = (
                    "A count threshold is not a power or sufficiency justification."
                )
            _check(
                checks,
                "event_counts",
                event_results["minimum_gate"],
                "prepared event episodes and label denominators counted; no event-count threshold is assumed",
                event_results,
            )
            if configuration is not None and freeze_verified:
                compatibility = validate_processing_compatibility(
                    processing_manifest,
                    configuration,
                    current_config_sha256=file_sha256(config_file),
                    config_path=config_file,
                )
                _check(
                    checks,
                    "frozen_processing_compatibility",
                    "pass",
                    "prepared artifacts match exact frozen config bytes, labels, features, and decision evidence",
                    compatibility,
                )
            else:
                _check(
                    checks,
                    "frozen_processing_compatibility",
                    "warning",
                    "prepared data are not confirmed as analysis-ready under a verified frozen config",
                )
            if configuration is not None and prepared_rows:
                grouping = configuration.get("evaluation", {}).get(
                    "primary_grouping", "configuration_id"
                )
                evaluation = configuration.get("evaluation", {})
                try:
                    split = make_group_split(
                        prepared_rows,
                        grouping=grouping,
                        seed=int(evaluation.get("random_seed", 2026)),
                        train_fraction=float(evaluation.get("train_fraction", 0.6)),
                        validation_fraction=float(evaluation.get("validation_fraction", 0.2)),
                    )
                    class_counts = {
                        part: {
                            "positive": split["counts"][part]["positive"],
                            "negative": split["counts"][part]["negative"],
                        }
                        for part in ("train", "validation", "test")
                    }
                    class_coverage = all(
                        value["positive"] > 0 and value["negative"] > 0
                        for value in class_counts.values()
                    )
                    _check(
                        checks,
                        "grouped_split_preview",
                        "warning",
                        "deterministic no-label-stratification preview only; evaluation will re-create it from frozen parameters",
                        {
                            "grouping": split["grouping"],
                            "groups": split["groups"],
                            "counts": split["counts"],
                            "classes_present": class_counts,
                        },
                    )
                    _check(
                        checks,
                        "split_class_coverage",
                        "pass" if class_coverage else "fail",
                        "all preview partitions contain both label classes"
                        if class_coverage
                        else "at least one preview partition lacks positive or negative labels; configured model evaluation may fail",
                        class_counts,
                    )
                    max_gap_for_events = (
                        float(
                            processing_manifest.get("label_parameters", {}).get(
                                "max_ground_truth_gap_ms", 0
                            )
                        )
                        / 1000
                    )
                    partition_event_counts = {
                        part: _event_counts(
                            [prepared_rows[index] for index in split["row_indices"][part]],
                            max_gap_for_events if max_gap_for_events > 0 else None,
                        )
                        for part in ("train", "validation", "test")
                    }
                    partition_event_status = "not_assessed"
                    if minimum_events_per_partition is not None:
                        under = {
                            part: counts["identified_onset_events"]
                            for part, counts in partition_event_counts.items()
                            if counts["identified_onset_events"] < minimum_events_per_partition
                        }
                        partition_event_status = (
                            "pass" if not under else "below_user_supplied_minimum"
                        )
                    _check(
                        checks,
                        "partition_event_counts",
                        partition_event_status,
                        "identified onset episodes counted separately by partition; no minimum is assumed",
                        {
                            "counts": partition_event_counts,
                            "minimum_per_partition_supplied": minimum_events_per_partition,
                            "note": "Any supplied threshold is a mechanical gate, not evidence of power or adequate precision.",
                        },
                    )
                except (SplitError, ValueError) as exc:
                    _check(
                        checks,
                        "grouped_split_preview",
                        "fail",
                        f"grouped split cannot be formed: {exc}",
                    )
        except (OSError, ReportError, ReadinessError, ConfigError, ValueError) as exc:
            processed_status = "invalid"
            prepared_rows = None
            processing_manifest = None
            _check(
                checks,
                "processed_artifacts",
                "fail",
                f"processed data are missing, stale, or invalid: {exc}",
            )
            if not any(item["id"] == "processed_feature_contract" for item in checks):
                _check(
                    checks,
                    "processed_feature_contract",
                    "unknown",
                    "prepared feature contract awaits valid processed artifacts",
                )
            _check(checks, "event_counts", "unknown", "event counts await valid prepared artifacts")
    else:
        _check(
            checks,
            "processed_artifacts",
            "not_applicable",
            "no prepared directory supplied; raw data may still support descriptive preparation",
        )
        _check(
            checks,
            "processed_feature_contract",
            "not_applicable",
            "no prepared directory supplied; no prepared feature contract can be classified",
        )
        _check(
            checks,
            "event_counts",
            "unknown",
            "event counts require derived labels/ground-truth report",
        )

    failures = [item["id"] for item in checks if item["status"] == "fail"]
    count_gates_below = [
        item["id"] for item in checks if item["status"] == "below_user_supplied_minimum"
    ]
    descriptive_ids = {
        "raw_root",
        "run_manifest_file",
        "audit_report_file",
        "manifest_schema",
        "audit_inventory_and_hashes",
        "source_schemas_and_units",
        "tcp_retransmission_unit_consistency",
        "timestamp_semantics",
        "ground_truth_measurements",
    }
    descriptive_check_status = {
        item["id"]: item["status"] for item in checks if item["id"] in descriptive_ids
    }
    descriptive_usable = descriptive_ids <= set(descriptive_check_status) and all(
        value in {"pass", "warning"}
        or (check_id == "tcp_retransmission_unit_consistency" and value == "not_applicable")
        for check_id, value in descriptive_check_status.items()
    )
    if not raw_ok or not manifest_ok or not audit_ok:
        overall = "input_blocked"
    elif failures or count_gates_below:
        overall = "blocked_or_below_declared_gate"
    elif processed_status == "valid" and freeze_verified:
        overall = "mechanical_checks_passed_human_scientific_review_still_required"
    elif raw_ok and manifest is not None and audit is not None:
        overall = "descriptive_inputs_available_protocol_or_prepared_evaluation_still_blocked"
    else:
        overall = "input_blocked"
    return {
        "readiness_report_schema_version": 1,
        "status": overall,
        "descriptive_data_status": (
            "available_for_manual_descriptive_review"
            if descriptive_usable
            else "blocked_by_missing_or_invalid_data_requirements"
        ),
        "confirmatory_status": (
            "blocked_protocol_not_frozen"
            if not freeze_verified
            else "blocked_no_valid_frozen_preparation"
            if processed_status != "valid"
            else "blocked_by_mechanical_checks"
            if failures or count_gates_below
            else "mechanical_prerequisites_passed_human_scientific_review_required"
        ),
        "evidence_status": "software readiness check only; never experimental or scientific validation",
        "confirmatory_claim_authorized": False,
        "paths": {
            "raw_root": str(root),
            "manifest": str(manifest_file),
            "audit_report": str(audit_file),
            "config": str(config_file) if config_file else None,
            "processed_dir": str(processed_dir.resolve()) if processed_dir else None,
        },
        "counts": {
            "manifest_run_count_validated": distinct_run_count,
            "configuration_count": distinct_config_count,
            "runs_per_configuration": dict(sorted(group_counts.items())),
            "valid_backlog_rows": target_counts.get("valid_queue_backlog_rows", 0),
            "timestamped_source_rows": target_counts.get("rows", 0),
        },
        "source_mapping_inventory": [
            {
                "run_id": record["run_id"],
                "configuration_id": record["configuration_id"],
                "source_file": record["source_file"],
                "source_sha256": record["source_sha256"],
                "source_row_count": record["source_row_count"],
                "retained_timestamp_rows": record["retained_timestamp_rows"],
                "timestamp_source_column": record["timestamp_source_column"],
                "timestamp_unit": record["timestamp_unit"],
                "column_map": record["column_map"],
                "column_units": record["column_units"],
                "bottleneck_rate_bps": record["bottleneck_rate_bps"],
                "measurement_parse_error_counts": record["measurement_parse_error_counts"],
                "missing_measurement_counts": record["missing_measurement_counts"],
                "feature_source_mapping_warnings": record["feature_source_mapping_warnings"],
            }
            for record in run_records
        ],
        "event_counts": event_results,
        "count_gate_caveat": "User-supplied numeric gates are mechanical comparisons only; they do not establish sample sufficiency, power, validity, or generalization.",
        "human_review_required": [
            "actual collector semantics and endpoint feature availability",
            "qdisc/interface/rate/backlog source and queue-delay proxy assumptions",
            "clock source/timezone/drift and endpoint-router synchronization",
            "target sampling cadence, missingness, censoring, and scientific event definitions",
            "run-validity/exclusion justification and protocol decision evidence",
            "independent configuration count, event counts, power/precision and claim scope",
        ],
        "blocking_check_ids": failures + count_gates_below,
        "checks": checks,
    }


def write_readiness_report(report: dict[str, Any], output_path: Path) -> Path:
    """Write a new machine-readable JSON report outside raw input directories."""
    output_path = output_path.resolve()
    raw_root = Path(report["paths"]["raw_root"]).resolve()
    try:
        output_path.relative_to(raw_root)
    except ValueError:
        pass
    else:
        raise ReadinessError("readiness report output must be outside raw_root")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise ReadinessError(
            f"refusing to overwrite existing readiness report: {output_path}"
        ) from exc
    return output_path
