"""Descriptive target-validation report; never tunes target thresholds from model outcomes."""

from __future__ import annotations

import csv
import html
import json
import math
import statistics
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

from .processing import file_sha256
from .reporting import ReportError, load_analysis_dataset


class GroundTruthReportError(ValueError):
    """Ground-truth validation report cannot be generated or verified."""


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def _event_intervals(rows: list[dict[str, Any]], max_gap: float) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    start: int | None = None
    for index, row in enumerate(rows):
        active = row.get("event_active")
        previous = rows[index - 1] if index else None
        gap_break = bool(
            previous and row["timestamp_seconds"] - previous["timestamp_seconds"] > max_gap
        )
        if active is True and (start is None or gap_break):
            if start is not None:
                events.append(
                    _finish_event(rows, start, index - 1, None, max_gap, left_censored=None)
                )
            start = index
        elif active is not True and start is not None:
            events.append(_finish_event(rows, start, index - 1, index, max_gap, left_censored=None))
            start = None
    if start is not None:
        events.append(_finish_event(rows, start, len(rows) - 1, None, max_gap, left_censored=None))
    return events


def _finish_event(
    rows: list[dict[str, Any]],
    start: int,
    end: int,
    next_index: int | None,
    max_gap: float,
    left_censored: bool | None,
) -> dict[str, Any]:
    first = rows[start]
    last = rows[end]
    previous = rows[start - 1] if start > 0 else None
    next_row = rows[next_index] if next_index is not None else None
    preceding_gap_ok = bool(
        previous and first["timestamp_seconds"] - previous["timestamp_seconds"] <= max_gap
    )
    following_gap_ok = bool(
        next_row and next_row["timestamp_seconds"] - last["timestamp_seconds"] <= max_gap
    )
    onset_lower = (
        previous["timestamp_seconds"]
        if preceding_gap_ok and previous.get("event_active") is False
        else None
    )
    onset_upper = first["timestamp_seconds"] if onset_lower is not None else None
    end_lower = last["timestamp_seconds"]
    end_upper = (
        next_row["timestamp_seconds"]
        if following_gap_ok and next_row.get("event_active") is False
        else None
    )
    duration_lower = max(
        0.0, end_lower - (onset_upper if onset_upper is not None else first["timestamp_seconds"])
    )
    duration_upper = (
        max(0.0, end_upper - onset_lower)
        if end_upper is not None and onset_lower is not None
        else None
    )
    return {
        "run_id": first["run_id"],
        "configuration_id": first["configuration_id"],
        "observed_start_seconds": first["timestamp_seconds"],
        "observed_end_seconds": last["timestamp_seconds"],
        "onset_interval_lower_exclusive_seconds": onset_lower,
        "onset_interval_upper_inclusive_seconds": onset_upper,
        "onset_interval_identified": onset_lower is not None,
        "end_interval_lower_inclusive_seconds": end_lower,
        "end_interval_upper_exclusive_seconds": end_upper,
        "duration_lower_bound_seconds": duration_lower,
        "duration_upper_bound_seconds": duration_upper,
        "left_censored_or_gap_ambiguous": onset_lower is None,
        "right_censored_or_gap_ambiguous": end_upper is None,
        "sample_count": end - start + 1,
    }


def _run_summary(
    rows: list[dict[str, Any]],
    run_record: dict[str, Any],
    label_parameters: dict[str, Any],
    feature_columns: list[str],
    audit_files: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    run_id = rows[0]["run_id"]
    config_id = rows[0]["configuration_id"]
    times = [float(row["timestamp_seconds"]) for row in rows]
    gaps = [b - a for a, b in pairwise(times)]
    max_gap = float(label_parameters["max_ground_truth_gap_ms"]) / 1000
    proxy_values = [row.get("qdisc_queue_delay_proxy_ms") for row in rows]
    backlog_values = [row.get("queue_backlog_bytes") for row in rows]
    states = [row.get("event_active") for row in rows]
    labels = [row.get("future_event_onset") for row in rows]
    events = _event_intervals(rows, max_gap)
    n = len(rows)
    delay_valid = [value for value in proxy_values if value is not None]
    backlog_valid = [value for value in backlog_values if value is not None]
    phases_compared = [
        (
            str(row.get("phase_commanded") or "").strip(),
            str(row.get("phase_observed") or "").strip(),
        )
        for row in rows
    ]
    phases_compared = [
        (commanded, observed) for commanded, observed in phases_compared if commanded and observed
    ]
    phase_mismatch = sum(commanded != observed for commanded, observed in phases_compared)
    source_file = run_record.get("source_file")
    audit_record = audit_files.get(source_file, {})
    warnings: list[str] = []
    if not delay_valid:
        warnings.append("no_valid_queue_delay_proxy_measurements")
    if len(backlog_valid) < len(backlog_values):
        warnings.append("backlog_measurements_missing_or_invalid")
    if len([gap for gap in gaps if gap > max_gap]):
        warnings.append("sampling_gaps_exceed_ground_truth_gap_limit")
    if labels.count(None) / n >= 0.5:
        warnings.append("at_least_half_of_anchors_have_censored_or_missing_labels")
    if run_record.get("measurement_parse_error_counts", {}).get("queue_backlog_bytes", 0):
        warnings.append("invalid_queue_backlog_rows_recorded_during_preparation")
    audit_sufficiency = audit_record.get("processing_sufficiency")
    if isinstance(audit_sufficiency, dict) and audit_sufficiency.get("sufficient") is not True:
        warnings.append("raw_audit_did_not_certify_processing_sufficiency")
    if phase_mismatch:
        warnings.append("commanded_and_observed_phase_values_differ")
    if run_record.get("feature_source_mapping_warnings"):
        warnings.append("endpoint_source_headers_need_semantic_leakage_review")
    constant_features = [
        column
        for column in feature_columns
        if len({row.get(column) for row in rows if row.get(column) is not None}) == 1
        and any(row.get(column) is not None for row in rows)
    ]
    if constant_features:
        warnings.append("constant_features_within_run:" + ",".join(constant_features))
    absent_features = [
        column for column in feature_columns if not any(row.get(column) is not None for row in rows)
    ]
    if absent_features:
        warnings.append("entirely_missing_features_within_run:" + ",".join(absent_features))
    unit_map = run_record.get("column_units", {})
    window_names = sorted(
        {
            key.removesuffix("_sample_count")
            for key in rows[0]
            if key.startswith("window_") and key.endswith("_sample_count")
        }
    )
    window_coverage = {}
    for stem in window_names:
        window_seconds = float(stem.removeprefix("window_").removesuffix("s"))
        sample_counts = [row.get(stem + "_sample_count") for row in rows]
        spans = [row.get(stem + "_observed_span_seconds") for row in rows]
        gaps_in_window = [row.get(stem + "_max_gap_seconds") for row in rows]
        window_coverage[stem] = {
            "anchors_with_no_samples": sum(value == 0 for value in sample_counts),
            "anchors_with_partial_window_span": sum(
                value is not None and value < window_seconds for value in spans
            ),
            "anchors_within_full_window_span": sum(
                value is not None and value >= window_seconds for value in spans
            ),
            "anchors_with_gap_over_ground_truth_limit": sum(
                value is not None and value > max_gap for value in gaps_in_window
            ),
        }
    return {
        "run_id": run_id,
        "configuration_id": config_id,
        "source_file": source_file,
        "source_sha256": run_record.get("source_sha256"),
        "source_rows": run_record.get("source_row_count"),
        "retained_timestamp_rows": n,
        "sampling": {
            "observed_span_seconds": times[-1] - times[0] if len(times) > 1 else 0.0,
            "median_interval_seconds": statistics.median(gaps) if gaps else None,
            "p95_interval_seconds": _percentile(gaps, 0.95),
            "maximum_interval_seconds": max(gaps) if gaps else None,
            "interval_count": len(gaps),
            "gaps_over_ground_truth_limit": sum(gap > max_gap for gap in gaps),
            "expected_max_gap_seconds_from_protocol": max_gap,
            "trailing_feature_window_coverage": window_coverage,
        },
        "target_measurements": {
            "backlog_valid_count": len(backlog_valid),
            "backlog_missing_count": n - len(backlog_valid),
            "backlog_min_bytes": min(backlog_valid) if backlog_valid else None,
            "backlog_max_bytes": max(backlog_valid) if backlog_valid else None,
            "queue_delay_proxy_valid_count": len(delay_valid),
            "queue_delay_proxy_missing_count": n - len(delay_valid),
            "queue_delay_proxy_min_ms": min(delay_valid) if delay_valid else None,
            "queue_delay_proxy_max_ms": max(delay_valid) if delay_valid else None,
            "configured_threshold_ms": label_parameters["queue_delay_threshold_ms"],
            "threshold_is_model_independent": True,
            "invalid_measurement_counts": run_record.get("measurement_parse_error_counts", {}),
            "missing_measurement_counts": run_record.get("missing_measurement_counts", {}),
        },
        "labels": {
            "active_sample_count": sum(state is True for state in states),
            "inactive_sample_count": sum(state is False for state in states),
            "unknown_state_count": sum(state is None for state in states),
            "positive_onset_anchor_count": labels.count(1),
            "negative_anchor_count": labels.count(0),
            "censored_or_missing_anchor_count": labels.count(None),
            "event_count_observed_in_trace": len(events),
            "event_frequency_per_observed_minute": (
                len(events) / ((times[-1] - times[0]) / 60)
                if len(times) > 1 and times[-1] > times[0]
                else None
            ),
            "events_with_identified_onset_interval": sum(
                event["onset_interval_identified"] for event in events
            ),
            "events_left_or_gap_censored": sum(
                event["left_censored_or_gap_ambiguous"] for event in events
            ),
            "events_right_or_gap_censored": sum(
                event["right_censored_or_gap_ambiguous"] for event in events
            ),
            "events": events,
        },
        "qdisc_counts": {
            field: {
                "observed_count_sum": sum(
                    int(row[field]) for row in rows if row.get(field) is not None
                ),
                "missing_count": sum(row.get(field) is None for row in rows),
                "semantics": "explicit per-sample count increments only; cumulative counters are not inferred",
            }
            for field in ("qdisc_drops_delta", "qdisc_overlimits_delta")
        },
        "phase_alignment": {
            "rows_with_both_commanded_and_observed": len(phases_compared),
            "mismatch_row_count": phase_mismatch,
        },
        "declared_source_units": unit_map,
        "feature_source_mapping_warnings": run_record.get("feature_source_mapping_warnings", []),
        "feature_quality": {
            column: {
                "observed_count": sum(row.get(column) is not None for row in rows),
                "missing_count": sum(row.get(column) is None for row in rows),
                "distinct_value_count": len(
                    {row.get(column) for row in rows if row.get(column) is not None}
                ),
            }
            for column in feature_columns
        },
        "raw_audit_processing_sufficiency": audit_sufficiency,
        "quality_warnings": warnings,
    }


def _svg_series(
    rows: list[dict[str, Any]],
    field: str,
    title: str,
    color: str,
    *,
    threshold: float | None = None,
) -> str:
    width, height, left, right, top, bottom = 800, 210, 55, 18, 28, 34
    values = [row.get(field) for row in rows if row.get(field) is not None]
    if not values:
        return f'<svg viewBox="0 0 {width} {height}" role="img"><text x="20" y="35">{html.escape(title)} — no valid observations</text></svg>'
    times = [row["timestamp_seconds"] for row in rows]
    t_min, t_max = times[0], times[-1]
    y_min, y_max = min(values), max(values)
    if y_min == y_max:
        pad = max(abs(y_min) * 0.05, 1.0)
        y_min, y_max = y_min - pad, y_max + pad

    def x_map(t: float) -> float:
        return (
            left + (t - t_min) / (t_max - t_min) * (width - left - right)
            if t_max > t_min
            else (width + left - right) / 2
        )

    def y_map(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * (height - top - bottom)

    groups: list[list[str]] = []
    current: list[str] = []
    # Uniform decimation limits static-report size; all event intervals remain listed in the table.
    stride = max(1, math.ceil(len(rows) / 1800))
    chosen = set(range(0, len(rows), stride)) | {len(rows) - 1}
    for i, row in enumerate(rows):
        value = row.get(field)
        if value is None:
            if current:
                groups.append(current)
                current = []
            continue
        if i not in chosen:
            continue
        current.append(f"{x_map(float(row['timestamp_seconds'])):.1f},{y_map(float(value)):.1f}")
    if current:
        groups.append(current)
    polylines = "".join(
        f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{" ".join(group)}"/>'
        for group in groups
        if group
    )
    threshold_line = ""
    if threshold is not None and y_min <= threshold <= y_max:
        yy = y_map(threshold)
        threshold_line = f'<line x1="{left}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}" stroke="#b91c1c" stroke-dasharray="6 4"/><text x="{left + 6}" y="{yy - 4:.1f}" fill="#991b1b">predeclared threshold {threshold:g} ms</text>'
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
        f'<text x="{left}" y="17" font-size="14">{html.escape(title)}</text>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#64748b"/>'
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#64748b"/>'
        f'<text x="4" y="{top + 8}" font-size="10">{y_max:.4g}</text>'
        f'<text x="4" y="{height - bottom}" font-size="10">{y_min:.4g}</text>'
        f"{threshold_line}{polylines}"
        f'<text x="{left}" y="{height - 7}" font-size="10">t={t_min:.4g}s</text>'
        f'<text x="{width - right - 44}" y="{height - 7}" font-size="10">t={t_max:.4g}s</text></svg>'
    )


def _event_strip(rows: list[dict[str, Any]]) -> str:
    width, height, left, right = 800, 76, 55, 18
    times = [row["timestamp_seconds"] for row in rows]
    t0, t1 = times[0], times[-1]

    def x_map(t: float) -> float:
        return (
            left + (t - t0) / (t1 - t0) * (width - left - right)
            if t1 > t0
            else (width + left - right) / 2
        )

    rects = []
    active = False
    start = None
    for i, row in enumerate(rows):
        state = row.get("event_active")
        if state is True and not active:
            start, active = row["timestamp_seconds"], True
        if active and (state is not True or i == len(rows) - 1):
            end = row["timestamp_seconds"]
            rects.append(
                f'<rect x="{x_map(start):.1f}" y="28" width="{max(1, x_map(end) - x_map(start)):.1f}" height="18" fill="#dc2626"/>'
            )
            active = False
    missing = []
    for row in rows:
        if row.get("event_active") is None:
            x = x_map(row["timestamp_seconds"])
            missing.append(
                f'<line x1="{x:.1f}" y1="24" x2="{x:.1f}" y2="50" stroke="#64748b" stroke-dasharray="2 2"/>'
            )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="event active intervals">'
        f'<text x="{left}" y="17" font-size="14">Derived event-active samples (red); unknown target samples (dashed)</text>'
        f"{''.join(rects)}{''.join(missing)}</svg>"
    )


def build_ground_truth_report(processed_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Create a checksum-traceable descriptive report for the already-prepared target."""
    try:
        rows, processing_manifest = load_analysis_dataset(processed_dir)
    except ReportError as exc:
        raise GroundTruthReportError(str(exc)) from exc
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise GroundTruthReportError(
            f"refusing to overwrite existing report directory: {output_dir}"
        )
    run_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        run_rows.setdefault(row["run_id"], []).append(row)
    records = processing_manifest.get("run_records", [])
    run_records = {record["run_id"]: record for record in records}
    raw_audit = processing_manifest.get("raw_audit", {})
    audit_files = {
        record.get("path"): record
        for record in raw_audit.get("files", [])
        if isinstance(record, dict)
    }
    params = processing_manifest["label_parameters"]
    feature_columns = processing_manifest["feature_columns"]
    per_run = []
    for run_id, run in run_rows.items():
        per_run.append(_run_summary(run, run_records[run_id], params, feature_columns, audit_files))
    total_rows = len(rows)
    all_events = [event for item in per_run for event in item["labels"]["events"]]
    feature_quality: dict[str, dict[str, Any]] = {}
    for column in feature_columns:
        values = [row.get(column) for row in rows if row.get(column) is not None]
        feature_quality[column] = {
            "observed_count": len(values),
            "missing_count": total_rows - len(values),
            "distinct_value_count": len(set(values)),
            "constant_when_observed": bool(values) and len(set(values)) == 1,
            "entirely_missing": not values,
        }
    inconsistent_units: dict[str, list[str]] = {}
    for field in ("rtt_ms", "loss_fraction", "goodput_bps", "tcp_retransmission_delta"):
        units = sorted(
            {
                record.get("column_units", {}).get(field)
                for record in records
                if record.get("column_units", {}).get(field) is not None
            }
        )
        if len(units) > 1:
            inconsistent_units[field] = units
    dataset = {
        "ground_truth_report_schema_version": 1,
        "evidence_status": "descriptive validation of supplied prepared data; source measurement provenance is not independently established",
        "protocol_status": processing_manifest.get("protocol_status", "not_declared"),
        "protocol_sha256": processing_manifest.get("protocol_sha256"),
        "analysis_sha256": processing_manifest["analysis_sha256"],
        "processing_manifest_sha256": file_sha256(processed_dir / "processing_manifest.json"),
        "source_manifest_sha256": processing_manifest.get("source_manifest_sha256"),
        "raw_audit": raw_audit,
        "label_parameters": params,
        "feature_windows_seconds": processing_manifest["feature_windows_seconds"],
        "feature_columns": feature_columns,
        "row_count": total_rows,
        "run_count": len(per_run),
        "configuration_count": len({item["configuration_id"] for item in per_run}),
        "event_count_observed_in_trace": len(all_events),
        "events_with_identified_onset_interval": sum(
            event["onset_interval_identified"] for event in all_events
        ),
        "inconsistent_declared_source_units_normalized_by_adapter": inconsistent_units,
        "feature_quality": feature_quality,
        "per_run": per_run,
        "limitations": [
            "qdisc_queue_delay_proxy_ms = backlog_bytes * 8 / declared_bottleneck_rate_bps; it is a serialization-delay proxy, not direct per-packet queueing-delay measurement.",
            "Observed event onset intervals reflect discrete sampling and the configured persistence rule; an interval crossing a long gap is marked ambiguous/censored.",
            "No target threshold or horizon is selected or optimized from model results by this report.",
            "Drops/overlimits are summarized only when the explicit manifest maps per-sample qdisc_drops_delta/qdisc_overlimits_delta counts; cumulative counter semantics are not inferred.",
            "Synthetic software tests do not validate the Kali collector, qdisc behavior, clocks, or experimental protocol.",
            (
                "This report uses exploratory labels from an unfrozen protocol; do not treat its outputs as confirmatory, and re-prepare after any separately approved protocol freeze."
                if processing_manifest.get("protocol_status") != "protocol_frozen"
                else "The data were prepared under a frozen protocol; this descriptive report still does not establish scientific validity."
            ),
        ],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    marker = output_dir / "INCOMPLETE"
    marker.write_text(
        "Ground-truth report generation did not reach its manifest commit step.\n", encoding="utf-8"
    )
    json_path = output_dir / "ground_truth_report.json"
    json_path.write_text(
        json.dumps(dataset, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    events_path = output_dir / "events.csv"
    event_fields = [
        "run_id",
        "configuration_id",
        "observed_start_seconds",
        "observed_end_seconds",
        "onset_interval_lower_exclusive_seconds",
        "onset_interval_upper_inclusive_seconds",
        "onset_interval_identified",
        "end_interval_lower_inclusive_seconds",
        "end_interval_upper_exclusive_seconds",
        "duration_lower_bound_seconds",
        "duration_upper_bound_seconds",
        "left_censored_or_gap_ambiguous",
        "right_censored_or_gap_ambiguous",
        "sample_count",
    ]
    with events_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=event_fields)
        writer.writeheader()
        writer.writerows(all_events)
    sections = []
    threshold = float(params["queue_delay_threshold_ms"])
    for run_id, run in run_rows.items():
        summary = next(item for item in per_run if item["run_id"] == run_id)
        event_rows_html = []
        for event in summary["labels"]["events"]:
            onset = (
                f"({event['onset_interval_lower_exclusive_seconds']:.6g}, "
                f"{event['onset_interval_upper_inclusive_seconds']:.6g}] s"
                if event["onset_interval_identified"]
                else "unidentified / gap-censored"
            )
            duration_upper = event["duration_upper_bound_seconds"]
            duration = (
                f">={event['duration_lower_bound_seconds']:.6g} s; upper {duration_upper:.6g} s"
                if duration_upper is not None
                else f">={event['duration_lower_bound_seconds']:.6g} s; right-censored"
            )
            event_rows_html.append(
                f"<tr><td>{event['observed_start_seconds']:.6g}</td>"
                f"<td>{event['observed_end_seconds']:.6g}</td><td>{html.escape(onset)}</td>"
                f"<td>{html.escape(duration)}</td><td>{event['sample_count']}</td></tr>"
            )
        event_table = (
            "<table><thead><tr><th>Observed start (s)</th><th>Observed end (s)</th>"
            "<th>Onset interval</th><th>Duration bounds</th><th>Active samples</th></tr></thead><tbody>"
            + (
                "".join(event_rows_html)
                if event_rows_html
                else '<tr><td colspan="5">No fully observed event intervals</td></tr>'
            )
            + "</tbody></table>"
        )
        drops = summary["qdisc_counts"]["qdisc_drops_delta"]
        overlimits = summary["qdisc_counts"]["qdisc_overlimits_delta"]
        sections.append(
            f"<section><h2>Run {html.escape(run_id)} · configuration {html.escape(summary['configuration_id'])}</h2>"
            f"<p>Rows: {len(run)}; events: {summary['labels']['event_count_observed_in_trace']}; "
            f"events/minute of observed trace: {summary['labels']['event_frequency_per_observed_minute']}; "
            f"censored labels: {summary['labels']['censored_or_missing_anchor_count']}; "
            f"backlog missing/invalid: {summary['target_measurements']['backlog_missing_count']}; "
            f"proxy missing/invalid: {summary['target_measurements']['queue_delay_proxy_missing_count']}; "
            f"invalid-measurement counts: {html.escape(str(summary['target_measurements']['invalid_measurement_counts']))}; "
            f"drops increment sum: {drops['observed_count_sum']} ({drops['missing_count']} missing); "
            f"overlimits increment sum: {overlimits['observed_count_sum']} ({overlimits['missing_count']} missing); "
            f"phase mismatch rows: {summary['phase_alignment']['mismatch_row_count']}; "
            f"quality warnings: {html.escape(', '.join(summary['quality_warnings']) or 'none recorded')}</p>"
            f"{_svg_series(run, 'queue_backlog_bytes', 'Raw qdisc backlog (bytes)', '#2563eb')}"
            f"{_svg_series(run, 'qdisc_queue_delay_proxy_ms', 'Backlog-derived serialization-delay proxy (ms)', '#ea580c', threshold=threshold)}"
            f"{_event_strip(run)}<h3>Derived event intervals</h3>{event_table}</section>"
        )
    unit_warning = " ".join(
        f"Declared source units for {field} differ across runs ({', '.join(units)}); normalized values are converted to canonical units."
        for field, units in inconsistent_units.items()
    )
    warning_banner = '<p class="warning"><strong>Important:</strong> the plotted queue-delay quantity is a backlog-derived serialization-delay proxy, not direct packet-delay ground truth. This report describes the configured target and measurements; it does not validate the collector or optimize thresholds.</p>'
    summary_rows = []
    for item in per_run:
        median_cadence = item["sampling"]["median_interval_seconds"]
        quality_warnings = html.escape(", ".join(item["quality_warnings"]) or "none")
        summary_rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
            "<td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(str(item["run_id"])),
                html.escape(str(item["configuration_id"])),
                item["retained_timestamp_rows"],
                item["labels"]["event_count_observed_in_trace"],
                item["labels"]["positive_onset_anchor_count"],
                item["labels"]["censored_or_missing_anchor_count"],
                median_cadence if median_cadence is not None else "—",
                item["sampling"]["gaps_over_ground_truth_limit"],
                quality_warnings,
            )
        )
    limitations_html = "".join(f"<li>{html.escape(text)}</li>" for text in dataset["limitations"])
    html_content = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ground-truth validation report</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#172033}}section{{border-top:1px solid #cbd5e1;padding:1rem 0}}svg{{width:100%;height:auto;background:#f8fafc;border:1px solid #e2e8f0;margin:.4rem 0}}.warning{{background:#fff7ed;border-left:4px solid #ea580c;padding:.8rem}}table{{border-collapse:collapse}}td,th{{border:1px solid #cbd5e1;padding:.4rem;text-align:left}}code{{overflow-wrap:anywhere}}</style>
<h1>Ground-truth validation report</h1>{warning_banner}<p class="warning">Protocol status: {html.escape(str(dataset["protocol_status"]))}. This report is descriptive only; unfrozen exploratory labels are not confirmatory and must be regenerated after a protocol freeze.</p><p>Evidence status: descriptive report for prepared data; provenance not independently verified. Analysis SHA-256: <code>{dataset["analysis_sha256"]}</code></p>
<p>Threshold: {threshold:g} ms; persistence: {params["consecutive_samples_required"]} samples; horizon: {params["prediction_horizon_ms"]} ms; max target gap: {params["max_ground_truth_gap_ms"]} ms. These values are read from the preparation manifest, not chosen by this report.</p>
<p>Runs: {len(per_run)}; configurations: {dataset["configuration_count"]}; events observed: {len(all_events)}; events with bounded onset interval: {dataset["events_with_identified_onset_interval"]}.</p>
{('<p class="warning">' + html.escape(unit_warning) + "</p>") if unit_warning else ""}
<h2>Per-run target and data-quality summary</h2><table><thead><tr><th>Run</th><th>Config</th><th>Rows</th><th>Events</th><th>Positive anchors</th><th>Censored anchors</th><th>Median cadence (s)</th><th>Long gaps</th><th>Quality warnings</th></tr></thead><tbody>
{"".join(summary_rows)}
</tbody></table><h2>Run timelines</h2>{"".join(sections)}<h2>Target and scope notes</h2><ul>{limitations_html}</ul></html>"""
    html_path = output_dir / "ground_truth_report.html"
    html_path.write_text(html_content, encoding="utf-8")
    artifact_names = [json_path.name, events_path.name, html_path.name]
    manifest = {
        "ground_truth_report_schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_sha256": dataset["analysis_sha256"],
        "protocol_sha256": dataset["protocol_sha256"],
        "processing_manifest_sha256": dataset["processing_manifest_sha256"],
        "source_manifest_sha256": dataset["source_manifest_sha256"],
        "artifact_sha256": {name: file_sha256(output_dir / name) for name in artifact_names},
        "run_count": dataset["run_count"],
        "configuration_count": dataset["configuration_count"],
        "event_count_observed_in_trace": dataset["event_count_observed_in_trace"],
        "evidence_status": dataset["evidence_status"],
    }
    (output_dir / "ground_truth_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    marker.unlink()
    return {
        "status": "complete",
        "output_dir": str(output_dir),
        "run_count": len(per_run),
        "event_count": len(all_events),
        "report": str(html_path),
    }


def verify_ground_truth_report(
    output_dir: Path, *, processed_dir: Path | None = None, raw_root: Path | None = None
) -> dict[str, Any]:
    root = output_dir.resolve()
    if (root / "INCOMPLETE").exists():
        raise GroundTruthReportError("ground-truth report is marked incomplete")
    manifest_path = root / "ground_truth_manifest.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise GroundTruthReportError(
            "ground-truth report directory lacks ground_truth_manifest.json"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GroundTruthReportError(f"invalid ground-truth report manifest: {exc}") from exc
    if (
        manifest.get("ground_truth_report_schema_version") != 1
        or manifest.get("status") != "complete"
    ):
        raise GroundTruthReportError("ground-truth report manifest is unsupported or incomplete")
    artifacts = manifest.get("artifact_sha256")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "ground_truth_report.json",
        "events.csv",
        "ground_truth_report.html",
    }:
        raise GroundTruthReportError("ground-truth report manifest artifact inventory is invalid")
    for relative, expected in artifacts.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise GroundTruthReportError(
                f"artifact path escapes report directory: {relative}"
            ) from exc
        if not path.is_file() or file_sha256(path) != expected:
            raise GroundTruthReportError(f"missing or checksum-mismatched artifact: {relative}")
    allowed = set(artifacts) | {"ground_truth_manifest.json"}
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")}
    if any(path.is_symlink() or path.is_dir() for path in root.rglob("*")) or actual != allowed:
        raise GroundTruthReportError(
            f"unexpected or missing report files: expected {sorted(allowed)}, found {sorted(actual)}"
        )
    report_payload = json.loads((root / "ground_truth_report.json").read_text(encoding="utf-8"))
    if manifest.get("protocol_sha256") != report_payload.get("protocol_sha256"):
        raise GroundTruthReportError(
            "ground-truth protocol fingerprint disagrees with report content"
        )
    if processed_dir is not None:
        try:
            _, processing_manifest = load_analysis_dataset(processed_dir)
        except ReportError as exc:
            raise GroundTruthReportError(str(exc)) from exc
        if processing_manifest.get("analysis_sha256") != manifest.get("analysis_sha256"):
            raise GroundTruthReportError(
                "prepared analysis does not match ground-truth report input"
            )
        if file_sha256(processed_dir / "processing_manifest.json") != manifest.get(
            "processing_manifest_sha256"
        ):
            raise GroundTruthReportError(
                "processing manifest does not match ground-truth report input"
            )
    if raw_root is not None:
        try:
            report = json.loads((root / "ground_truth_report.json").read_text(encoding="utf-8"))
            for run in report["per_run"]:
                source = (raw_root / run["source_file"]).resolve()
                source.relative_to(raw_root.resolve())
                if not source.is_file() or file_sha256(source) != run["source_sha256"]:
                    raise GroundTruthReportError(
                        f"raw source missing or hash mismatch: {run['source_file']}"
                    )
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, GroundTruthReportError):
                raise
            raise GroundTruthReportError(f"could not verify raw source inventory: {exc}") from exc
    return {
        "status": "verified",
        "output_dir": str(root),
        "verified_artifacts": sorted(artifacts),
        "prepared_data_verified": processed_dir is not None,
        "raw_sources_verified": raw_root is not None,
    }
