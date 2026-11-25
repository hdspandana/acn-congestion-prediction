"""Command-line interface for audit, configuration validation, processing and evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .alignment import AlignmentError, align_streams, verify_alignment_artifacts
from .artifact_inspection import (
    ArtifactInspectionError,
    inspect_artifact_tree,
    write_inspection_report,
)
from .audit import audit_directory, render_summary
from .config import ConfigError, load_config, validate_config, verify_protocol_freeze_evidence
from .evaluation import EvaluationError, evaluate_baseline_events, evaluate_model_events
from .ground_truth import (
    GroundTruthReportError,
    build_ground_truth_report,
    verify_ground_truth_report,
)
from .models import ModelError
from .preflight import preflight
from .processing import ProcessingError, feature_contract_metadata, process_dataset
from .readiness import ReadinessError, build_readiness_report, write_readiness_report
from .reporting import (
    ReportError,
    load_analysis_dataset,
    sha256_file,
    validate_processing_compatibility,
    verify_evaluation_artifacts,
    write_evaluation_artifacts,
)
from .robustness import (
    RobustnessError,
    evaluate_circular_shift_negative_control,
    evaluate_feature_ablation,
    evaluate_model_seed_stability,
    evaluate_target_sensitivity,
    verify_robustness_artifacts,
    write_robustness_artifacts,
)
from .splits import SplitError, make_factor_holdout_split, make_group_split

LOG = logging.getLogger("acn")


def _outside_input(output: Path, input_root: Path) -> Path:
    resolved = output.resolve()
    try:
        resolved.relative_to(input_root.resolve())
    except ValueError:
        return resolved
    raise ValueError(f"refusing to write output inside source input tree: {resolved}")


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
    except FileExistsError as exc:
        raise ValueError(f"refusing to overwrite existing output: {path}") from exc


def _readiness(args: argparse.Namespace) -> int:
    report = build_readiness_report(
        raw_root=args.raw_root,
        manifest_path=args.manifest,
        audit_report_path=args.audit_report,
        config_path=args.config,
        processed_dir=args.processed_dir,
        minimum_independent_configurations=args.minimum_independent_configurations,
        minimum_runs_per_configuration=args.minimum_runs_per_configuration,
        minimum_identified_events=args.minimum_identified_events,
        minimum_events_per_partition=args.minimum_events_per_partition,
    )
    if args.output:
        write_readiness_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 2 if report["status"] in {"input_blocked", "blocked_or_below_declared_gate"} else 0


def _audit(args: argparse.Namespace) -> int:
    source = args.input.resolve()
    if not source.is_dir():
        raise ValueError(f"input directory does not exist or is not a directory: {source}")
    required = list(args.required_column or [])
    phases = list(args.expected_phase or [])
    if args.config:
        config = load_config(args.config)
        audit_config = config.get("audit", {})
        if not required:
            required = audit_config.get("required_columns", [])
        if not phases:
            phases = audit_config.get("expected_phases", [])
    json_path = _outside_input(args.output, source)
    summary_path = _outside_input(args.summary_output or args.output.with_suffix(".md"), source)
    if json_path == summary_path:
        raise ValueError("JSON report and human summary must use different paths")
    if json_path.exists() or summary_path.exists():
        raise ValueError("refusing to overwrite an existing JSON or summary report")
    report = audit_directory(
        source,
        required_columns=required,
        expected_phases=phases,
        timestamp_column=args.timestamp_column,
        numeric_timestamp_unit=args.timestamp_unit,
    )
    _write_new(json_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    try:
        _write_new(summary_path, render_summary(report))
    except Exception:
        LOG.exception("JSON audit report exists but the human summary could not be written")
        raise
    print(f"Inspected {report['csv_file_count']} CSV files: {json_path} and {summary_path}")
    print(
        f"Errors={report['files_with_errors']} warnings={report['files_with_warnings']} audit_completed={report['audit_completed']}"
    )
    return 0 if report["audit_completed"] else 2


def _inspect_artifacts(args: argparse.Namespace) -> int:
    if args.input.is_symlink():
        raise ArtifactInspectionError("artifact root itself must not be a symlink")
    source = args.input.resolve()
    if not source.is_dir():
        raise ArtifactInspectionError(
            f"artifact root does not exist or is not a directory: {source}"
        )
    output = _outside_input(args.output, source)
    report = inspect_artifact_tree(source)
    write_inspection_report(report, output)
    issues = sum(
        item.get("status") != "inspected"
        or item.get("parse_status") in {"invalid_or_unreadable", "inspection_error", "empty_file"}
        for item in report["files"]
    )
    print(
        json.dumps(
            {
                "inspection_report": str(output),
                "artifact_root": str(source),
                "file_count": report["file_count"],
                "symlink_count_not_followed": report["symlink_count_not_followed"],
                "walk_error_count": report["walk_error_count"],
                "files_with_inspection_issues": issues,
                "stream_pattern": report["stream_alignment_assessment"]["pattern"],
                "alignment_performed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _align_streams(args: argparse.Namespace) -> int:
    report = align_streams(raw_root=args.raw_root, spec_path=args.spec, output_dir=args.output_dir)
    print(
        json.dumps(
            {
                "status": report["status"],
                "alignment_report": str(args.output_dir.resolve() / "alignment_report.json"),
                "runs_manifest": str(args.output_dir.resolve() / report["runs_manifest"]),
                "run_count": report["run_count"],
                "alignment_review_required": True,
                "scientific_validity_established": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _verify_factor_definition(specification: dict, specification_path: Path) -> dict:
    factor = specification.get("factor")
    if not isinstance(factor, str) or not factor.strip():
        raise RobustnessError("factor must be a non-empty metadata path")
    definition = specification.get("factor_definition")
    required = {
        "factor",
        "semantic_definition",
        "unit",
        "source",
        "reviewed_by",
        "reviewed_at_utc",
        "evidence_file",
        "evidence_sha256",
    }
    if not isinstance(definition, dict) or set(definition) != required:
        raise RobustnessError(f"factor_definition must contain exactly {sorted(required)}")
    if definition["factor"] != specification.get("factor"):
        raise RobustnessError("factor_definition.factor must match the selected factor path")
    for key in ("semantic_definition", "unit", "source", "reviewed_by", "evidence_file"):
        if not isinstance(definition[key], str) or not definition[key].strip():
            raise RobustnessError(f"factor_definition.{key} must be a non-empty string")
    reviewed_value = definition["reviewed_at_utc"]
    if not isinstance(reviewed_value, str):
        raise RobustnessError(
            "factor_definition.reviewed_at_utc must be an ISO timestamp with timezone"
        )
    try:
        reviewed_at = datetime.fromisoformat(reviewed_value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise RobustnessError(
            "factor_definition.reviewed_at_utc must be an ISO timestamp with timezone"
        ) from exc
    if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
        raise RobustnessError("factor_definition.reviewed_at_utc must include a timezone")
    digest = definition["evidence_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(ch not in "0123456789abcdef" for ch in digest)
    ):
        raise RobustnessError("factor_definition.evidence_sha256 must be a lowercase SHA-256")
    relative = Path(definition["evidence_file"])
    if relative.is_absolute() or ".." in relative.parts:
        raise RobustnessError(
            "factor-definition evidence path must be relative to the specification directory"
        )
    base = specification_path.resolve().parent
    candidate = base / relative
    probe = base
    for part in relative.parts:
        probe = probe / part
        if probe.is_symlink():
            raise RobustnessError("factor-definition evidence path must not traverse symlinks")
    candidate = candidate.resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise RobustnessError(
            "factor-definition evidence path escapes the specification directory"
        ) from exc
    if not candidate.is_file() or sha256_file(candidate) != digest:
        raise RobustnessError(
            "factor-definition evidence file is missing or its SHA-256 does not match"
        )
    return {**definition, "reviewed_at_utc": reviewed_at.isoformat(), "evidence_sha256": digest}


def _inspect_alignment_report(args: argparse.Namespace) -> int:
    result = verify_alignment_artifacts(
        args.report_dir, raw_root=args.raw_root, spec_path=args.spec
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "complete" else 2


def _validate_config(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    validate_config(config, require_analysis_ready=args.analysis_ready)
    if args.analysis_ready:
        verify_protocol_freeze_evidence(config, args.config)
    print(f"Configuration valid (JSON syntax): {args.config}")
    if not args.analysis_ready and config.get("status") == "protocol_draft_not_frozen":
        print("Draft accepted; unresolved scientific fields will block analysis-ready commands.")
    return 0


def _preflight(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    validate_config(config, require_analysis_ready=False)
    result = preflight(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ready_for_manual_review" else 2


def _prepare(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    config = load_config(config_path)
    project_root = config_path.parent.parent
    settings = config["processing"]
    if args.exploratory:
        if config.get("status") != "protocol_draft_not_frozen":
            raise ConfigError(
                "--exploratory is only for a draft config; frozen preparation must use its frozen values"
            )
        required_args = {
            "--raw-root": args.raw_root,
            "--manifest": args.manifest,
            "--output": args.output,
            "--queue-delay-threshold-ms": args.queue_delay_threshold_ms,
            "--consecutive-samples-required": args.consecutive_samples_required,
            "--prediction-horizon-ms": args.prediction_horizon_ms,
            "--observation-window-ms": args.observation_window_ms,
            "--max-ground-truth-gap-ms": args.max_ground_truth_gap_ms,
            "--feature-window-seconds": args.feature_window_seconds,
        }
        missing = [flag for flag, value in required_args.items() if value is None]
        if missing:
            raise ConfigError(
                "exploratory preparation requires explicit provisional values for: "
                + ", ".join(missing)
            )
        labels = {
            "queue_delay_threshold_ms": args.queue_delay_threshold_ms,
            "consecutive_samples_required": args.consecutive_samples_required,
            "prediction_horizon_ms": args.prediction_horizon_ms,
            "observation_window_ms": args.observation_window_ms,
            "max_ground_truth_gap_ms": args.max_ground_truth_gap_ms,
        }
        feature_windows = args.feature_window_seconds
        protocol_status = "exploratory_unfrozen"
        raw_root, manifest, output = args.raw_root, args.manifest, args.output
    else:
        validate_config(config, require_analysis_ready=True)
        verify_protocol_freeze_evidence(config, config_path)
        labels = config["labeling"]
        feature_windows = settings["feature_windows_seconds"]
        raw_root = args.raw_root or project_root / settings["raw_root"]
        manifest = args.manifest or project_root / settings["manifest"]
        output = args.output or project_root / settings["output_root"]
        protocol_status = "protocol_frozen"
    if manifest is None:
        raise ConfigError(
            "a reviewed aligned-run manifest must be specified (config processing.manifest or --manifest)"
        )
    result = process_dataset(
        raw_root=Path(raw_root),
        manifest_path=Path(manifest),
        output_dir=Path(output),
        queue_delay_threshold_ms=float(labels["queue_delay_threshold_ms"]),
        consecutive_samples_required=int(labels["consecutive_samples_required"]),
        prediction_horizon_ms=float(labels["prediction_horizon_ms"]),
        observation_window_ms=float(labels["observation_window_ms"]),
        max_ground_truth_gap_ms=float(labels["max_ground_truth_gap_ms"]),
        feature_windows_seconds=list(feature_windows),
        endpoint_feature_fields=list(config["evaluation"]["endpoint_feature_allowlist"]),
        feature_temporal_policy=settings.get("feature_temporal_policy"),
        experiment_config_sha256=sha256_file(config_path),
        protocol_status=protocol_status,
        audit_report_path=args.audit_report,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    config = load_config(config_path)
    validate_config(config, require_analysis_ready=True)
    verify_protocol_freeze_evidence(config, config_path)
    project_root = config_path.parent.parent
    processed_dir = args.processed_dir or project_root / config["processing"]["output_root"]
    evaluation_root = project_root / config["evaluation"]["output_root"]
    output_dir = (
        args.output
        or evaluation_root
        / f"evaluation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}"
    )
    rows, processing_manifest = load_analysis_dataset(Path(processed_dir))
    config_hash = sha256_file(config_path)
    compatibility = validate_processing_compatibility(
        processing_manifest,
        config,
        current_config_sha256=config_hash,
        config_path=config_path,
    )
    evaluation = config["evaluation"]
    if args.grouping is not None and args.grouping != evaluation["primary_grouping"]:
        raise ConfigError(
            "evaluate --grouping cannot override the frozen primary grouping; use an explicitly exploratory robustness analysis"
        )
    if args.seed is not None and args.seed != evaluation["random_seed"]:
        raise ConfigError(
            "evaluate --seed cannot override the frozen primary seed; use an explicitly exploratory robustness analysis"
        )
    grouping = evaluation["primary_grouping"]
    random_seed = evaluation["random_seed"]
    split = make_group_split(
        rows,
        grouping=grouping,
        seed=random_seed,
        train_fraction=evaluation["train_fraction"],
        validation_fraction=evaluation["validation_fraction"],
    )
    labels = config["labeling"]
    common = {
        "max_false_alarms_per_minute": float(evaluation["max_false_alarms_per_minute"]),
        "horizon_seconds": float(labels["prediction_horizon_ms"]) / 1000,
        "max_gap_seconds": float(labels["max_ground_truth_gap_ms"]) / 1000,
        "alarm_merge_gap_seconds": float(evaluation["alarm_merge_gap_ms"]) / 1000,
        "bootstrap_resamples": int(evaluation["bootstrap_resamples"]),
        "bootstrap_confidence": float(evaluation["bootstrap_confidence"]),
        "bootstrap_seed": int(random_seed),
        "feature_specification": compatibility["feature_specification"],
    }
    results: dict[str, dict] = {}
    failures: dict[str, str] = {}
    feature_columns = processing_manifest["feature_columns"]
    try:
        baseline_results = evaluate_baseline_events(
            rows,
            split,
            feature_column=evaluation["baseline_feature"],
            eligible_feature_columns=feature_columns,
            ewma_alpha=float(evaluation["ewma_alpha"]),
            direction=evaluation["baseline_feature_direction"],
            **common,
        )
        results.update(baseline_results)
    except (EvaluationError, ModelError, ValueError) as exc:
        failures["static_threshold"] = str(exc)
        failures["ewma"] = str(exc)
    for model_name in evaluation["models_to_evaluate"]:
        try:
            results[model_name] = evaluate_model_events(
                rows,
                split,
                model_name=model_name,
                feature_columns=feature_columns,
                random_seed=random_seed,
                class_weight=evaluation.get("class_weight"),
                model_selection=evaluation.get("model_selection"),
                **common,
            )
        except (EvaluationError, ModelError, ValueError, ImportError) as exc:
            failures[model_name] = f"{type(exc).__name__}: {exc}"
    artifact_result = write_evaluation_artifacts(
        Path(output_dir),
        config_hash=config_hash,
        data_hash=processing_manifest["analysis_sha256"],
        split=split,
        results=results,
        configuration=config,
        failures=failures,
        processing_manifest=processing_manifest,
        processing_manifest_sha256=sha256_file(Path(processed_dir) / "processing_manifest.json"),
        compatibility=compatibility,
        config_path=config_path,
    )
    print(
        json.dumps(
            {
                **artifact_result,
                "failures": failures,
                "data_origin": processing_manifest.get("evidence_status", "unknown"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 2 if failures else 0


def _inspect_report(args: argparse.Namespace) -> int:
    result = verify_evaluation_artifacts(
        args.report_dir,
        processed_dir=args.processed_dir,
        raw_root=args.raw_root,
        audit_report_path=args.audit_report,
        run_manifest_path=args.run_manifest,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "complete" else 2


def _validate_target(args: argparse.Namespace) -> int:
    output = _outside_input(args.output, args.processed_dir)
    result = build_ground_truth_report(args.processed_dir, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _inspect_target_report(args: argparse.Namespace) -> int:
    result = verify_ground_truth_report(
        args.report_dir,
        processed_dir=args.processed_dir,
        raw_root=args.raw_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _robustness(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    config = load_config(config_path)
    validate_config(config, require_analysis_ready=True)
    verify_protocol_freeze_evidence(config, config_path)
    processed_dir = args.processed_dir.resolve()
    rows, processing_manifest = load_analysis_dataset(processed_dir)
    config_hash = sha256_file(config_path)
    compatibility = validate_processing_compatibility(
        processing_manifest,
        config,
        current_config_sha256=config_hash,
        config_path=config_path,
    )
    if args.spec.is_symlink():
        raise RobustnessError("robustness specification must not be a symlink")
    specification_sha256 = sha256_file(args.spec)
    try:
        specification = json.loads(args.spec.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RobustnessError(f"cannot read robustness specification JSON: {exc}") from exc
    if sha256_file(args.spec) != specification_sha256:
        raise RobustnessError("robustness specification changed while it was read")
    if not isinstance(specification, dict) or not isinstance(specification.get("analysis"), str):
        raise RobustnessError("specification must be an object with an 'analysis' value")
    allowed_keys = {
        "target_sensitivity": {"analysis", "thresholds_ms", "horizons_seconds"},
        "feature_ablation": {"analysis", "model_name"},
        "circular_shift_negative_control": {"analysis", "model_name", "minimum_shift_samples"},
        "model_seed_stability": {"analysis", "model_name", "seeds"},
        "factor_level_holdout": {
            "analysis",
            "factor",
            "held_out_values",
            "train_fraction",
            "factor_definition",
        },
    }
    analysis = specification["analysis"]
    if analysis not in allowed_keys or set(specification) != allowed_keys[analysis]:
        raise RobustnessError(
            f"unknown analysis or specification keys; expected exact keys {allowed_keys.get(analysis)}"
        )
    factor_definition = (
        _verify_factor_definition(specification, args.spec)
        if analysis == "factor_level_holdout"
        else None
    )
    evaluation = config["evaluation"]
    labels = config["labeling"]
    features = processing_manifest["feature_columns"]
    seed = args.seed if args.seed is not None else int(evaluation["random_seed"])
    grouping = args.grouping or evaluation["primary_grouping"]
    if analysis == "factor_level_holdout":
        if args.grouping is not None:
            raise RobustnessError(
                "factor_level_holdout defines configuration grouping and cannot use --grouping"
            )
        run_records = {
            record["run_id"]: record for record in processing_manifest.get("run_records", [])
        }
        for row in rows:
            record = run_records.get(row.get("run_id"))
            if record is None:
                raise RobustnessError(
                    f"missing processing-manifest record for run {row.get('run_id')!r}"
                )
            row["bottleneck_rate_bps"] = record.get("bottleneck_rate_bps")
            row["run_metadata"] = record.get("run_metadata", {})
        split = make_factor_holdout_split(
            rows,
            factor=specification["factor"],
            held_out_values=specification["held_out_values"],
            seed=seed,
            train_fraction=specification["train_fraction"],
        )
    else:
        split = make_group_split(
            rows,
            grouping=grouping,
            seed=seed,
            train_fraction=evaluation["train_fraction"],
            validation_fraction=evaluation["validation_fraction"],
        )
    max_gap = float(labels["max_ground_truth_gap_ms"]) / 1000
    common = {
        "baseline_feature": evaluation["baseline_feature"],
        "baseline_direction": evaluation["baseline_feature_direction"],
        "eligible_feature_columns": features,
        "max_false_alarms_per_minute": float(evaluation["max_false_alarms_per_minute"]),
        "alarm_merge_gap_seconds": float(evaluation["alarm_merge_gap_ms"]) / 1000,
        "random_seed": seed,
        "class_weight": evaluation.get("class_weight"),
        "bootstrap_resamples": int(evaluation["bootstrap_resamples"]),
        "bootstrap_confidence": float(evaluation["bootstrap_confidence"]),
        "feature_specification": compatibility["feature_specification"],
    }
    if analysis == "target_sensitivity":
        result = evaluate_target_sensitivity(
            rows,
            split,
            thresholds_ms=specification["thresholds_ms"],
            horizons_seconds=specification["horizons_seconds"],
            persistence_samples=int(labels["consecutive_samples_required"]),
            max_gap_seconds=max_gap,
            model_names=evaluation["models_to_evaluate"],
            **common,
        )
    elif analysis == "feature_ablation":
        result = {
            "analysis_type": "exploratory feature ablation",
            "grouping": split["grouping"],
            "split_seed": split["seed"],
            "methods": evaluate_feature_ablation(
                rows,
                split,
                model_name=specification["model_name"],
                feature_columns=features,
                max_gap_seconds=max_gap,
                horizon_seconds=float(labels["prediction_horizon_ms"]) / 1000,
                bootstrap_seed=seed,
                **{
                    key: value
                    for key, value in common.items()
                    if key
                    not in {"baseline_feature", "baseline_direction", "eligible_feature_columns"}
                },
            ),
        }
    elif analysis == "circular_shift_negative_control":
        result = evaluate_circular_shift_negative_control(
            rows,
            split,
            model_name=specification["model_name"],
            feature_columns=features,
            minimum_shift_samples=specification["minimum_shift_samples"],
            max_gap_seconds=max_gap,
            horizon_seconds=float(labels["prediction_horizon_ms"]) / 1000,
            bootstrap_seed=seed,
            **{
                key: value
                for key, value in common.items()
                if key not in {"baseline_feature", "baseline_direction", "eligible_feature_columns"}
            },
        )
        result["analysis_type"] = "exploratory circular-shift negative control"
    elif analysis == "factor_level_holdout":
        factor_methods: dict[str, Any] = {}
        try:
            factor_methods.update(
                evaluate_baseline_events(
                    rows,
                    split,
                    feature_column=evaluation["baseline_feature"],
                    eligible_feature_columns=features,
                    ewma_alpha=float(evaluation["ewma_alpha"]),
                    direction=evaluation["baseline_feature_direction"],
                    max_false_alarms_per_minute=float(evaluation["max_false_alarms_per_minute"]),
                    horizon_seconds=float(labels["prediction_horizon_ms"]) / 1000,
                    max_gap_seconds=max_gap,
                    alarm_merge_gap_seconds=float(evaluation["alarm_merge_gap_ms"]) / 1000,
                    bootstrap_resamples=int(evaluation["bootstrap_resamples"]),
                    bootstrap_confidence=float(evaluation["bootstrap_confidence"]),
                    bootstrap_seed=seed,
                    feature_specification=compatibility["feature_specification"],
                )
            )
        except (EvaluationError, ModelError, ValueError) as exc:
            factor_methods["baselines"] = {
                "status": "failed",
                "failure": f"{type(exc).__name__}: {exc}",
                **feature_contract_metadata(
                    compatibility["feature_specification"], scope="failed_method"
                ),
            }
        for model_name in evaluation["models_to_evaluate"]:
            try:
                factor_methods[model_name] = evaluate_model_events(
                    rows,
                    split,
                    model_name=model_name,
                    feature_columns=features,
                    random_seed=seed,
                    class_weight=evaluation.get("class_weight"),
                    max_false_alarms_per_minute=float(evaluation["max_false_alarms_per_minute"]),
                    horizon_seconds=float(labels["prediction_horizon_ms"]) / 1000,
                    max_gap_seconds=max_gap,
                    alarm_merge_gap_seconds=float(evaluation["alarm_merge_gap_ms"]) / 1000,
                    bootstrap_resamples=int(evaluation["bootstrap_resamples"]),
                    bootstrap_confidence=float(evaluation["bootstrap_confidence"]),
                    bootstrap_seed=seed,
                    feature_specification=compatibility["feature_specification"],
                )
            except (EvaluationError, ModelError, ValueError, ImportError) as exc:
                factor_methods[model_name] = {
                    "status": "failed",
                    "failure": f"{type(exc).__name__}: {exc}",
                    **feature_contract_metadata(
                        compatibility["feature_specification"], scope="failed_method"
                    ),
                }
        failures_present = any(value.get("status") == "failed" for value in factor_methods.values())
        result = {
            "analysis_type": "exploratory factor-level holdout; not the frozen primary analysis",
            "factor": specification["factor"],
            "factor_definition": factor_definition,
            "held_out_values": specification["held_out_values"],
            "method_failures_present": failures_present,
            "methods": factor_methods,
            "interpretation": "Test configurations are restricted to the declared factor level(s); this does not establish broader unseen-factor transfer, and all model/threshold choices remain validation-only within non-held-out levels.",
        }
    else:
        result = evaluate_model_seed_stability(
            rows,
            split,
            model_name=specification["model_name"],
            feature_columns=features,
            seeds=specification["seeds"],
            max_false_alarms_per_minute=float(evaluation["max_false_alarms_per_minute"]),
            horizon_seconds=float(labels["prediction_horizon_ms"]) / 1000,
            max_gap_seconds=max_gap,
            alarm_merge_gap_seconds=float(evaluation["alarm_merge_gap_ms"]) / 1000,
            class_weight=evaluation.get("class_weight"),
            bootstrap_resamples=int(evaluation["bootstrap_resamples"]),
            bootstrap_confidence=float(evaluation["bootstrap_confidence"]),
            bootstrap_seed=seed,
            feature_specification=compatibility["feature_specification"],
        )
    result["split_manifest"] = split
    result["specification"] = specification
    if sha256_file(args.spec) != specification_sha256:
        raise RobustnessError("robustness specification changed during analysis")
    if (
        analysis == "factor_level_holdout"
        and _verify_factor_definition(specification, args.spec) != factor_definition
    ):
        raise RobustnessError("factor-definition evidence changed during analysis")
    output = _outside_input(args.output, processed_dir)
    artifact = write_robustness_artifacts(
        output,
        result=result,
        config_sha256=config_hash,
        analysis_sha256=processing_manifest["analysis_sha256"],
        processing_manifest_sha256=sha256_file(processed_dir / "processing_manifest.json"),
        processing_manifest=processing_manifest,
        specification_sha256=specification_sha256,
        configuration=config,
        config_path=config_path,
    )
    print(
        json.dumps(
            {**artifact, "processing_compatibility": compatibility}, indent=2, sort_keys=True
        )
    )
    return 2 if artifact["status"] == "partial_failure" else 0


def _inspect_robustness(args: argparse.Namespace) -> int:
    result = verify_robustness_artifacts(
        args.report_dir,
        processed_dir=args.processed_dir,
        raw_root=args.raw_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "complete" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acn", description=__doc__)
    parser.add_argument("--version", action="version", version=f"acn {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    readiness = sub.add_parser(
        "readiness", help="machine-readable raw-data and evaluation readiness report"
    )
    readiness.add_argument("--raw-root", type=Path, required=True)
    readiness.add_argument("--manifest", type=Path, required=True)
    readiness.add_argument("--audit-report", type=Path, required=True)
    readiness.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    readiness.add_argument("--processed-dir", type=Path)
    readiness.add_argument(
        "--minimum-independent-configurations",
        type=int,
        help="optional user-declared mechanical count gate; not a sufficiency claim",
    )
    readiness.add_argument(
        "--minimum-runs-per-configuration",
        type=int,
        help="optional user-declared replication gate; repeats remain nested",
    )
    readiness.add_argument(
        "--minimum-identified-events",
        type=int,
        help="optional user-declared total event-count gate; not a power claim",
    )
    readiness.add_argument(
        "--minimum-events-per-partition",
        type=int,
        help="optional user-declared event-count gate for each deterministic split",
    )
    readiness.add_argument("--output", type=Path, help="write a new JSON report outside raw-root")
    readiness.set_defaults(handler=_readiness)

    artifact_inspect = sub.add_parser(
        "inspect-artifacts",
        help="read-only local inventory of experiment artifacts without emitting measurement rows",
    )
    artifact_inspect.add_argument(
        "input", type=Path, help="artifact directory; symlinks are listed but not followed"
    )
    artifact_inspect.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new JSON inventory path outside the artifact directory",
    )
    artifact_inspect.set_defaults(handler=_inspect_artifacts)

    align = sub.add_parser(
        "align-streams",
        help="align separate streams only under a reviewed, explicit clock/matching specification",
    )
    align.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="directory containing original endpoint/router CSVs and cited clock evidence",
    )
    align.add_argument(
        "--spec", type=Path, required=True, help="version-1 JSON alignment specification"
    )
    align.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new directory outside raw-root; contains aligned CSVs, provenance, exclusions, report, and v2 manifest",
    )
    align.set_defaults(handler=_align_streams)

    inspect_alignment = sub.add_parser(
        "inspect-alignment-report",
        help="verify aligned outputs and optional original-source/spec hashes",
    )
    inspect_alignment.add_argument("--report-dir", type=Path, required=True)
    inspect_alignment.add_argument(
        "--raw-root",
        type=Path,
        help="also verify original endpoint/router sources and clock-evidence hashes",
    )
    inspect_alignment.add_argument(
        "--spec", type=Path, help="also verify the alignment specification hash"
    )
    inspect_alignment.set_defaults(handler=_inspect_alignment_report)

    audit = sub.add_parser("audit", help="read-only recursive CSV quality audit")
    audit.add_argument("input", type=Path)
    audit.add_argument("--output", type=Path, required=True, help="new JSON report path")
    audit.add_argument(
        "--summary-output", type=Path, help="new Markdown summary (default: JSON sibling .md)"
    )
    audit.add_argument("--config", type=Path, help="optional JSON-syntax experiment config")
    audit.add_argument(
        "--timestamp-column",
        help="timestamp column (otherwise auto-detected; ambiguous candidates require this)",
    )
    audit.add_argument(
        "--timestamp-unit",
        choices=("seconds", "milliseconds", "microseconds", "nanoseconds"),
        help="explicit unit for numeric timestamps; suffixes such as _ns may be inferred",
    )
    audit.add_argument("--required-column", action="append", help="required column; repeatable")
    audit.add_argument("--expected-phase", action="append", help="allowed phase; repeatable")
    audit.set_defaults(handler=_audit)

    validate = sub.add_parser("validate-config", help="validate experiment config")
    validate.add_argument("config", type=Path)
    validate.add_argument(
        "--analysis-ready", action="store_true", help="require all frozen analysis parameters"
    )
    validate.set_defaults(handler=_validate_config)

    preflight_cmd = sub.add_parser(
        "preflight", help="read-only testbed inspection; never configures namespaces or traffic"
    )
    preflight_cmd.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    preflight_cmd.set_defaults(handler=_preflight)

    prepare = sub.add_parser(
        "prepare", help="prepare derived data from the explicit aligned-CSV manifest contract"
    )
    prepare.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    prepare.add_argument("--raw-root", type=Path)
    prepare.add_argument("--manifest", type=Path)
    prepare.add_argument(
        "--audit-report",
        type=Path,
        required=True,
        help="completed raw audit JSON whose file hashes match the manifest inputs",
    )
    prepare.add_argument("--output", type=Path)
    prepare.add_argument(
        "--exploratory",
        action="store_true",
        help="prepare draft-data labels for descriptive review only; cannot be evaluated as confirmatory data",
    )
    prepare.add_argument("--queue-delay-threshold-ms", type=float)
    prepare.add_argument("--consecutive-samples-required", type=int)
    prepare.add_argument("--prediction-horizon-ms", type=float)
    prepare.add_argument("--observation-window-ms", type=float)
    prepare.add_argument("--max-ground-truth-gap-ms", type=float)
    prepare.add_argument("--feature-window-seconds", type=float, nargs="+")
    prepare.set_defaults(handler=_prepare)

    evaluate = sub.add_parser(
        "evaluate", help="grouped baselines and ML with validation-only event-FAPM calibration"
    )
    evaluate.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    evaluate.add_argument("--processed-dir", type=Path)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--grouping", choices=("configuration_id", "run_id"))
    evaluate.add_argument(
        "--seed", type=int, help="override split, model, and bootstrap random seed"
    )
    evaluate.set_defaults(handler=_evaluate)

    inspect = sub.add_parser(
        "inspect-report", help="verify evaluation completion marker and artifact checksums"
    )
    inspect.add_argument("--report-dir", type=Path, required=True)
    inspect.add_argument(
        "--processed-dir",
        type=Path,
        help="also verify input analysis.csv and processing manifest against report hashes",
    )
    inspect.add_argument(
        "--raw-root",
        type=Path,
        help="also verify original raw source files against recorded SHA-256 hashes",
    )
    inspect.add_argument(
        "--audit-report", type=Path, help="also verify the raw audit JSON and its file inventory"
    )
    inspect.add_argument(
        "--run-manifest", type=Path, help="also verify the exact version-2 run manifest SHA-256"
    )
    inspect.set_defaults(handler=_inspect_report)

    target = sub.add_parser(
        "validate-target",
        help="build descriptive target/ground-truth validation report from prepared data",
    )
    target.add_argument("--processed-dir", type=Path, required=True)
    target.add_argument(
        "--output", type=Path, required=True, help="new report directory outside processed data"
    )
    target.set_defaults(handler=_validate_target)

    inspect_target = sub.add_parser(
        "inspect-target-report", help="verify target report hashes and optional source provenance"
    )
    inspect_target.add_argument("--report-dir", type=Path, required=True)
    inspect_target.add_argument("--processed-dir", type=Path)
    inspect_target.add_argument(
        "--raw-root",
        type=Path,
        help="also verify original raw file hashes from processing manifest",
    )
    inspect_target.set_defaults(handler=_inspect_target_report)

    robustness = sub.add_parser(
        "robustness",
        help="run explicit exploratory sensitivity, ablation, controls, or factor-level holdout",
    )
    robustness.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    robustness.add_argument("--processed-dir", type=Path, required=True)
    robustness.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="JSON specification for target_sensitivity, feature_ablation, circular_shift_negative_control, model_seed_stability, or factor_level_holdout",
    )
    robustness.add_argument(
        "--output", type=Path, required=True, help="new exploratory report directory"
    )
    robustness.add_argument("--grouping", choices=("configuration_id", "run_id"))
    robustness.add_argument("--seed", type=int, help="override split/model/bootstrap random seed")
    robustness.set_defaults(handler=_robustness)

    inspect_robustness = sub.add_parser(
        "inspect-robustness-report",
        help="verify exploratory report checksums and optional input provenance",
    )
    inspect_robustness.add_argument("--report-dir", type=Path, required=True)
    inspect_robustness.add_argument("--processed-dir", type=Path)
    inspect_robustness.add_argument("--raw-root", type=Path)
    inspect_robustness.set_defaults(handler=_inspect_robustness)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (
        ConfigError,
        ProcessingError,
        EvaluationError,
        ModelError,
        ReportError,
        ReadinessError,
        GroundTruthReportError,
        RobustnessError,
        SplitError,
        AlignmentError,
        ArtifactInspectionError,
        OSError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
