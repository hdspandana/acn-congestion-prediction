"""Optional Streamlit research console; consequential research operations stay in the CLI."""

from __future__ import annotations

import csv
import html
import io
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Streamlit executes a file directly, which otherwise removes package context.
# Resolve only this source tree (or use the installed package) for that entry point.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "acn_congestion"

from .config import ConfigError, load_config, verify_protocol_freeze_evidence
from .dashboard_data import (
    MAX_CHART_POINTS,
    MAX_DOWNLOAD_BYTES,
    MAX_INVENTORY_FILES,
    DashboardDataError,
    Workspace,
    app_version,
    available_feature_units,
    classify_workspace_state,
    clear_preview_cache,
    discover_report_directories,
    display_path,
    load_bounded_run_preview,
    read_json,
    resolve_workspace,
    run_audit_safely,
    safe_child,
    safe_download,
    scan_csv_inventory,
    verify_audit_inventory,
    verify_evaluation_bundle,
    verify_ground_truth_bundle,
    verify_processed_artifacts,
    verify_robustness_bundle,
    verify_workspace_lineage_sources,
)
from .processing import ProcessingError, file_sha256
from .readiness import ReadinessError, build_readiness_report

_NAVIGATION = (
    "Overview",
    "Data & artifacts",
    "Telemetry",
    "Ground truth & labels",
    "Protocol workspace",
    "Model evaluation",
    "Robustness & provenance",
    "Reports",
)
MAX_ROBUSTNESS_RENDER_BYTES = 1024 * 1024


def _safe_markdown(path: Path, *, maximum: int = 500_000) -> str | None:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _theme_css(mode: str) -> str:
    dark = mode == "Dark"
    bg = "#0B1220" if dark else "#F1F5F9"
    surface = "#111827" if dark else "#FFFFFF"
    fg = "#F8FAFC" if dark else "#111827"
    muted = "#CBD5E1" if dark else "#475569"
    border = "#475569" if dark else "#CBD5E1"
    input_bg = "#1F2937" if dark else "#FFFFFF"
    return f"""
    <style>
      :root {{ color-scheme: {"dark" if dark else "light"}; }}
      html, body, .stApp, [data-testid="stAppViewContainer"] {{
        background: {bg} !important; color: {fg} !important;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        font-size: 16px; line-height: 1.55;
      }}
      [data-testid="stSidebar"] {{ background: {surface} !important; border-right: 1px solid {border}; }}
      [data-testid="stHeader"] {{ background: {bg} !important; }}
      [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] {{ color: {fg}; }}
      [data-testid="stCaptionContainer"] {{ color: {muted}; }}
      h1, h2, h3 {{ letter-spacing: -0.025em; color: {fg} !important; }}
      h1 {{ font-size: clamp(1.85rem, 3vw, 2.5rem); font-weight: 720; }}
      h2 {{ font-size: 1.35rem; font-weight: 680; margin-top: 1.15rem; }}
      a {{ color: {"#67E8F9" if dark else "#0E7490"} !important; text-underline-offset: 3px; }}
      [data-testid="stMetric"] {{ background: {surface}; border: 1px solid {border}; border-radius: 8px; padding: .7rem .9rem; }}
      [data-testid="stMetricLabel"] {{ color: {muted} !important; font-size: .9rem; }}
      [data-testid="stMetricValue"] {{ color: {fg} !important; font-size: 1.5rem; }}
      [data-testid="stDataFrame"] {{ border: 1px solid {border}; border-radius: 8px; }}
      [data-testid="stTextInput"] input, [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
      [data-testid="stNumberInput"] input, [data-testid="stTextArea"] textarea {{
        background: {input_bg} !important; color: {fg} !important; border-color: {border} !important;
      }}
      [data-testid="stCheckbox"] label, [data-testid="stRadio"] label {{ color: {fg} !important; }}
      button, [role="button"], [role="radio"], [role="checkbox"], a {{
        min-height: 2.25rem;
      }}
      button:focus-visible, [role="button"]:focus-visible, [role="radio"]:focus-visible,
      [role="checkbox"]:focus-visible, a:focus-visible, input:focus-visible {{
        outline: 3px solid #22D3EE !important; outline-offset: 3px !important;
      }}
      [data-testid="stButton"] button[kind="primary"] {{
        background: #0891B2 !important; color: #FFFFFF !important; border: 1px solid #0891B2 !important;
      }}
      [data-testid="stButton"] button:hover {{ border-color: #22D3EE !important; }}
      [data-testid="stAlert"] {{ border-radius: 8px; }}
      .acn-kicker {{ color: {"#67E8F9" if dark else "#0E7490"}; font-size: .76rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }}
      .acn-subtitle {{ color: {muted}; margin-top: -.4rem; }}
      .acn-rule {{ border: 0; border-top: 1px solid {border}; margin: 1rem 0 1.25rem; }}
      .acn-state {{ display: inline-block; border: 1px solid {border}; border-left: 4px solid #0891B2;
        background: {surface}; border-radius: 7px; padding: .65rem .9rem; margin: .25rem 0 .6rem; font-weight: 650; }}
      .acn-note {{ color: {muted}; font-size: .94rem; }}
      code {{ color: {"#A5F3FC" if dark else "#155E75"}; }}
    </style>
    """


def _render_header(st: Any, title: str, subtitle: str) -> None:
    st.markdown(
        '<div class="acn-kicker">ACN · Network observability research</div>', unsafe_allow_html=True
    )
    st.title(title)
    st.markdown(f'<p class="acn-subtitle">{html.escape(subtitle)}</p>', unsafe_allow_html=True)


def _config_state(workspace: Workspace) -> tuple[dict[str, Any] | None, str, str | None]:
    try:
        config = load_config(workspace.config_path)
        status = str(config.get("status", "not_declared"))
        if status == "protocol_frozen":
            verify_protocol_freeze_evidence(config, workspace.config_path)
            return config, "protocol_frozen", None
        return config, status, None
    except (ConfigError, OSError, ValueError) as exc:
        return None, "invalid_or_unavailable", str(exc)


def _read_manifest(workspace: Workspace) -> tuple[dict[str, Any] | None, str | None]:
    if workspace.manifest_path is None:
        return None, "No run manifest configured or found in the declared raw root."
    try:
        return read_json(workspace.manifest_path), None
    except DashboardDataError as exc:
        return None, str(exc)


def _has_incomplete_marker(root: Path) -> bool:
    if not root.is_dir() or root.is_symlink():
        return False
    inspected = 0
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(current) / name).is_symlink()]
        if {"INCOMPLETE", "PREPARATION_INCOMPLETE"}.intersection(files):
            return True
        inspected += len(files) + len(dirs)
        if inspected > MAX_INVENTORY_FILES:
            return True  # Conservative: an oversized tree cannot be declared marker-free.
    return False


def _refresh_snapshot(workspace: Workspace) -> dict[str, Any]:
    inventory = scan_csv_inventory(workspace.raw_root)
    audit = verify_audit_inventory(workspace.raw_root, workspace.audit_report_path)
    processed = verify_processed_artifacts(workspace.processed_root)
    if processed.get("status") == "verified":
        try:
            shared_sources = verify_workspace_lineage_sources(workspace, processed, audit)
        except (DashboardDataError, OSError, ValueError) as exc:
            shared_sources = {"status": "unverified", "error": str(exc)}
    else:
        shared_sources = {
            "status": "unverified",
            "error": "processed artifacts could not be verified",
        }
    config, protocol_state, config_error = _config_state(workspace)
    manifest, manifest_error = _read_manifest(workspace)
    alignment_status: str | None = None
    if manifest is not None and "alignment_status" in manifest:
        alignment_status = str(manifest.get("alignment_status"))
        if alignment_status == "approved":
            try:
                from .processing import verify_alignment_review

                verify_alignment_review(
                    workspace.raw_root.resolve(), manifest, workspace.manifest_path
                )
            except Exception:  # noqa: BLE001 - verification failure must fail closed in the UI.
                alignment_status = "invalid"
    eval_dirs = discover_report_directories(workspace.reports_root, "evaluation_manifest.json")
    evaluations: list[dict[str, Any]] = []
    latest_evaluation: dict[str, Any] | None = None
    latest_valid_evaluation: dict[str, Any] | None = None
    for directory in eval_dirs[:3]:
        result = verify_evaluation_bundle(
            directory, workspace, processed_artifacts=processed, shared_sources=shared_sources
        )
        item = {"path": directory, **result}
        evaluations.append(item)
        if latest_evaluation is None:
            latest_evaluation = item
        if (
            result.get("status") == "complete_validated_provenance"
            and latest_valid_evaluation is None
        ):
            latest_valid_evaluation = item
    ground_truth_reports: list[dict[str, Any]] = []
    for directory in discover_report_directories(
        workspace.reports_root, "ground_truth_report.json", max_reports=3
    ):
        result = verify_ground_truth_bundle(
            directory, workspace, processed_artifacts=processed, shared_sources=shared_sources
        )
        ground_truth_reports.append({"path": directory, **result})
    robustness_reports: list[dict[str, Any]] = []
    for directory in discover_report_directories(
        workspace.reports_root, "robustness_manifest.json", max_reports=3
    ):
        result = verify_robustness_bundle(
            directory, workspace, processed_artifacts=processed, shared_sources=shared_sources
        )
        robustness_reports.append({"path": directory, **result})
    processed_protocol = None
    if processed.get("status") == "verified":
        processed_protocol = processed["manifest"].get("protocol_status")
    evaluation_status = None
    if latest_evaluation is not None:
        report = latest_evaluation.get("report") or {}
        evaluation_status = report.get("status")
        if latest_evaluation.get("status") == "provenance_unverified":
            evaluation_status = "incomplete"
    state, state_note = classify_workspace_state(
        inventory=inventory,
        audit=audit,
        protocol_status=protocol_state,
        processed_status=processed_protocol,
        alignment_status=alignment_status,
        evaluation_status=evaluation_status,
        evaluation_validated=latest_valid_evaluation is not None,
        evaluation_in_progress=_has_incomplete_marker(workspace.reports_root)
        or _has_incomplete_marker(workspace.processed_root),
    )
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "inventory": inventory,
        "audit": audit,
        "processed": processed,
        "config": config,
        "protocol_state": protocol_state,
        "config_error": config_error,
        "manifest": manifest,
        "manifest_error": manifest_error,
        "alignment_status": alignment_status,
        "evaluations": evaluations,
        "latest_evaluation": latest_evaluation,
        "latest_valid_evaluation": latest_valid_evaluation,
        "ground_truth_reports": ground_truth_reports,
        "robustness_reports": robustness_reports,
        "state": state,
        "state_note": state_note,
    }


def _format_count(value: Any) -> str:
    return f"{value:,}" if isinstance(value, int) and not isinstance(value, bool) else "—"


def _show_state(st: Any, label: str, note: str) -> None:
    safe_label = html.escape(label)
    st.markdown(f'<div class="acn-state">Status: {safe_label}</div>', unsafe_allow_html=True)
    st.write(note)


def _read_limited_json(root: Path, name: str) -> dict[str, Any] | None:
    try:
        return read_json(safe_child(root, name))
    except (DashboardDataError, OSError):
        return None


def _overview(st: Any, workspace: Workspace, snapshot: dict[str, Any] | None) -> None:
    _render_header(
        st,
        "Research console",
        "A provenance-first workspace for ACN telemetry, labels, protocol and evaluation artifacts.",
    )
    quick = scan_csv_inventory(workspace.raw_root)
    protocol_state = snapshot.get("protocol_state") if snapshot else _config_state(workspace)[1]
    if snapshot:
        _show_state(st, snapshot["state"], snapshot["state_note"])
        st.caption(
            f"Integrity snapshot: {snapshot['checked_at_utc']} · re-run verification after any source change."
        )
    else:
        initial_state, note = classify_workspace_state(
            inventory=quick,
            audit={"status": "not_audited"},
            protocol_status=protocol_state,
            processed_status=None,
            alignment_status=None,
        )
        _show_state(st, initial_state, note)
        st.caption("No hashes or full-file parses have been run in this session yet.")
    if st.button(
        "Refresh integrity and provenance checks",
        type="primary",
        help="Hashes current sources and derived files, verifies configured audit/report links, and checks at most three recent evaluation reports. Read-only; may take time.",
    ):
        with st.spinner("Checking declared source and artifact fingerprints…"):
            st.session_state["acn_snapshot"] = _refresh_snapshot(workspace)
            st.session_state.pop("acn_processed_verified", None)
            st.session_state.pop("acn_telemetry_request", None)
            st.session_state.pop("acn_inventory_report", None)
            st.session_state.pop("acn_inventory_checked_at", None)
            st.session_state.pop("acn_readiness_report", None)
            st.session_state.pop("acn_readiness_checked_at", None)
            clear_preview_cache()
        st.rerun()
    st.markdown("<hr class='acn-rule'>", unsafe_allow_html=True)
    if quick.get("csv_count", 0) == 0:
        st.info(
            "**No data available.** No source CSVs were found in the configured raw root. Set `ACN_RAW_ROOT` to the preserved data directory, then run an audit. No model result is being displayed."
        )
    elif not snapshot or snapshot.get("audit", {}).get("status") != "passed":
        st.warning(
            "Source files were discovered, but current audit hashes/inventory have not passed verification. Do not interpret them as validated data."
        )
    if snapshot:
        cols = st.columns(4)
        cols[0].metric(
            "CSV files discovered", _format_count(snapshot["inventory"].get("csv_count"))
        )
        processed = snapshot["processed"]
        cols[1].metric(
            "Verified runs",
            _format_count(
                processed.get("manifest", {}).get("run_count")
                if processed.get("status") == "verified"
                else None
            ),
        )
        cols[2].metric(
            "Verified configurations",
            _format_count(
                processed.get("manifest", {}).get("configuration_count")
                if processed.get("status") == "verified"
                else None
            ),
        )
        cols[3].metric("Protocol", snapshot.get("protocol_state", "unknown"))
        latest = snapshot.get("latest_evaluation")
        if latest:
            report = latest.get("report") or {}
            st.subheader("Latest evaluation artifact")
            status = report.get("status", latest.get("status"))
            st.write(
                f"Outcome: **{status}** · provenance check: **{latest.get('status')}** · artifact directory: `{display_path(latest['path'], workspace)}`"
            )
            if latest.get("status") != "complete_validated_provenance":
                st.warning(
                    "This report is not being presented as a validated evaluation result. Review the blockers below."
                )
                for message in latest.get("errors", []):
                    st.write(f"- {message}")
        elif snapshot.get("latest_valid_evaluation"):
            st.subheader("Most recent validated evaluation")
            st.write(display_path(snapshot["latest_valid_evaluation"]["path"], workspace))
        else:
            st.info(
                "No complete evaluation with currently rechecked data/configuration provenance was found."
            )
        if snapshot.get("config_error"):
            st.error(f"Configuration verification failed: {snapshot['config_error']}")
        if snapshot.get("manifest_error"):
            st.caption(snapshot["manifest_error"])
        with st.expander("Configured sources (paths are intentionally abbreviated)"):
            st.write(
                {
                    "raw": display_path(workspace.raw_root, workspace),
                    "processed": display_path(workspace.processed_root, workspace),
                    "reports": display_path(workspace.reports_root, workspace),
                    "config": display_path(workspace.config_path, workspace),
                    "run_manifest": display_path(workspace.manifest_path, workspace),
                    "audit_report": display_path(workspace.audit_report_path, workspace),
                }
            )
    st.subheader("Outstanding evidence blockers")
    blockers = [
        "No actual Kali collector artifacts are included by default; configure only preserved, reviewed sources.",
        "Schema, timestamp units/clocks, qdisc semantics and source-to-run mappings require evidence-based review.",
        "Research thresholds/windows and protocol freeze remain human decisions; the UI never approves them.",
        "Synthetic tests and benchmark values are software verification only, not measured network results.",
    ]
    for blocker in blockers:
        st.write(f"- {blocker}")


def _inventory_frame(report: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for item in report.get("files", []):
        result.append(
            {
                "relative_path": item.get("path"),
                "size_bytes": item.get("size_bytes"),
                "sha256": item.get("sha256"),
                "format": item.get("format"),
                "status": item.get("status"),
                "parse_status": item.get("parse_status"),
                "changed_during_inspection": item.get("changed_during_inspection"),
            }
        )
    return result


def _data_page(st: Any, workspace: Workspace, snapshot: dict[str, Any] | None) -> None:
    _render_header(
        st,
        "Data & artifacts",
        "Inventory, source audit, schema mappings and row-level provenance. All scans are explicit and read-only.",
    )
    st.warning(
        "Inventory and audit observations are detected facts about files, not proof that a column means what its name suggests. Units, timestamp clocks and qdisc semantics remain unverified until reviewed."
    )
    tab_inventory, tab_audit, tab_mapping = st.tabs(["Inventory", "Audit", "Mappings & provenance"])
    with tab_inventory:
        st.caption(
            "Hashing/profile inspection reads every selected file. It runs only when requested; very large artifact trees can take time."
        )
        if st.button(
            "Inspect raw artifact tree",
            help="Uses the existing read-only artifact inspector; no file contents are copied to the report.",
        ):
            try:
                from .dashboard_data import inspect_raw_artifacts

                with st.spinner("Hashing and profiling the configured raw tree…"):
                    st.session_state["acn_inventory_report"] = inspect_raw_artifacts(
                        workspace.raw_root
                    )
                st.session_state["acn_inventory_checked_at"] = datetime.now(
                    timezone.utc
                ).isoformat()
            except DashboardDataError as exc:
                st.error(f"Inventory could not be completed: {exc}")
        report = st.session_state.get("acn_inventory_report")
        if report:
            st.caption(
                f"Inventory snapshot: {st.session_state.get('acn_inventory_checked_at')} · {report.get('file_count')} files · symlinks skipped: {report.get('symlink_count_not_followed')}"
            )
            inventory_rows = _inventory_frame(report)
            if inventory_rows:
                st.dataframe(inventory_rows[:500], use_container_width=True, hide_index=True)
                if len(inventory_rows) > 500:
                    st.info(
                        "Showing the first 500 relative paths. The underlying inspection report contains the full bounded inventory."
                    )
            else:
                st.info("No regular files were inventoried.")
            assessment = report.get("stream_alignment_assessment", {})
            st.write(
                f"Name-based stream pattern: **{assessment.get('pattern', 'unknown')}**. Alignment: **not established**."
            )
            with st.expander("Detected candidate names and limits"):
                st.json(
                    {
                        "endpoint_candidates": assessment.get("endpoint_file_name_candidates", []),
                        "router_or_qdisc_candidates": assessment.get(
                            "router_or_qdisc_file_name_candidates", []
                        ),
                        "alignment_decision": assessment.get("alignment_decision"),
                        "candidate_basis": assessment.get("candidate_basis"),
                    }
                )
        else:
            st.info(
                "No inventory snapshot has been created in this session. Use the button to inspect the configured tree."
            )
    with tab_audit:
        st.subheader("Current audit status")
        audit = snapshot.get("audit") if snapshot else {"status": "not_checked", "errors": []}
        if audit.get("status") == "passed":
            report = audit["report"]
            st.success(
                f"Audit hashes and CSV inventory rechecked: {audit['verified_csv_count']} CSV files. This is a structural audit, not semantic/scientific approval."
            )
            rows = []
            for item in report.get("files", []):
                rows.append(
                    {
                        "path": item.get("path"),
                        "rows": item.get("row_count"),
                        "bytes": item.get("size_bytes"),
                        "status": item.get("status"),
                        "malformed_rows": item.get("malformed_row_count"),
                        "duplicate_rows": item.get("duplicate_rows"),
                        "duplicate_timestamps": item.get("duplicate_timestamps"),
                        "timestamp_column": item.get("timestamp_column"),
                        "timestamp_kind": item.get("timestamp_timezone_kinds"),
                        "sha256": item.get("sha256"),
                    }
                )
            st.dataframe(rows, use_container_width=True, hide_index=True)
            issue_rows = []
            issue_count = 0
            for item in report.get("files", []):
                for issue in item.get("issues", []):
                    issue_count += 1
                    if len(issue_rows) < 1000:
                        issue_rows.append(
                            {
                                "file": item.get("path"),
                                "severity": issue.get("severity"),
                                "code": issue.get("code"),
                                "message": issue.get("message"),
                            }
                        )
            if issue_rows:
                st.subheader("Audit warnings and errors")
                st.dataframe(issue_rows, use_container_width=True, hide_index=True)
                if issue_count > len(issue_rows):
                    st.caption(
                        f"Showing {len(issue_rows):,} of {issue_count:,} recorded issue rows."
                    )
            else:
                st.info(
                    "The audit recorded no structural error/warning items; that still does not validate units or meaning."
                )
        elif audit.get("status") == "failed_or_stale":
            st.error("Audit is failed, stale, or linked to a different source tree.")
            for error in audit.get("errors", []):
                st.write(f"- {error}")
        elif audit.get("status") == "not_checked":
            st.info(
                "Audit hashes were not rechecked in this session. Refresh Overview integrity checks to inspect the current configured audit."
            )
        else:
            st.info("No completed audit is configured. Data are not considered audit-passed.")
        st.markdown("#### Run a new read-only audit")
        st.caption(
            "This reads the raw tree and writes a new audit report under the configured reports root; it never edits raw files. Timestamp overrides are technical inputs and require independent evidence."
        )
        with st.form("acn-audit-form"):
            timestamp_override = st.text_input(
                "Optional timestamp header override",
                value="",
                help="Leave blank to use the audit tool's conservative candidate detection. This is a column name, not a path.",
            )
            unit_choice = st.selectbox(
                "Numeric timestamp unit",
                ["Not specified", "seconds", "milliseconds", "microseconds", "nanoseconds"],
                index=0,
            )
            confirmed = st.checkbox(
                "I understand this performs a new read-only audit and creates a new report directory."
            )
            submitted = st.form_submit_button("Run read-only audit", disabled=not confirmed)
        if submitted:
            try:
                with st.spinner(
                    "Auditing and hashing the source tree; no progress estimate is available…"
                ):
                    result = run_audit_safely(
                        workspace.raw_root,
                        workspace.reports_root,
                        timestamp_column=timestamp_override.strip() or None,
                        numeric_timestamp_unit=None
                        if unit_choice == "Not specified"
                        else unit_choice,
                    )
                st.session_state["acn_audit_created"] = result
                st.session_state["acn_snapshot"] = None
                st.session_state.pop("acn_readiness_report", None)
                st.session_state.pop("acn_readiness_checked_at", None)
                st.success(
                    f"New audit report: {display_path(result['audit_path'], workspace)} · status: {result['status']}"
                )
                st.caption(
                    "The app will use this report after refresh; audit completion does not verify collector semantics or scientific validity."
                )
            except (DashboardDataError, OSError, ValueError) as exc:
                st.error(f"Audit failed safely: {exc}")
        created = st.session_state.get("acn_audit_created")
        if created:
            try:
                data = safe_download(created["audit_path"], workspace.reports_root)
                st.download_button(
                    "Download latest audit JSON",
                    data=data,
                    file_name=created["audit_path"].name,
                    mime="application/json",
                    key="acn-audit-download",
                )
            except DashboardDataError as exc:
                st.caption(f"In-app download unavailable: {exc}")
    with tab_mapping:
        processed = snapshot.get("processed") if snapshot else None
        if processed is None:
            st.info(
                "Refresh Overview integrity checks before viewing mappings; no processed artifacts are read automatically on this page."
            )
        elif processed.get("status") != "verified":
            st.info(
                "No checksum-verified processing manifest is available; source-to-canonical mappings are not shown as verified."
            )
            for error in processed.get("errors", []):
                st.write(f"- {error}")
        else:
            manifest = processed["manifest"]
            st.write(
                f"Processing protocol state: **{manifest.get('protocol_status', 'not declared')}**. SHA-256: `{processed['manifest_sha256']}`."
            )
            if manifest.get("alignment_review", {}).get("status") == "approved":
                st.warning(
                    "Alignment review is recorded as approved by the linked decision record; this is not protocol or scientific approval."
                )
                st.code(
                    str(
                        manifest["alignment_review"].get("alignment_binding_sha256", "not recorded")
                    )
                )
            mapping_rows = []
            for run in manifest.get("run_records", []):
                for canonical, source_column in run.get("column_map", {}).items():
                    mapping_rows.append(
                        {
                            "run_id": run.get("run_id"),
                            "configuration_id": run.get("configuration_id"),
                            "source_file": run.get("source_file"),
                            "canonical_field": canonical,
                            "observed_column": source_column,
                            "declared_unit": run.get("column_units", {}).get(canonical),
                            "source_sha256": run.get("source_sha256"),
                        }
                    )
            st.dataframe(mapping_rows[:2000], use_container_width=True, hide_index=True)
            with st.expander("Audit exclusions and processing quality counts"):
                st.json(
                    {
                        "exclusion_reason_counts": manifest.get("exclusion_reason_counts"),
                        "missing/parse errors by run": [
                            {
                                "run_id": item.get("run_id"),
                                "missing": item.get("missing_measurement_counts"),
                                "parse_errors": item.get("measurement_parse_error_counts"),
                            }
                            for item in manifest.get("run_records", [])
                        ],
                        "source_manifest_sha256": manifest.get("source_manifest_sha256"),
                    }
                )


def _metric_label(field: str) -> str:
    return {
        "rtt_ms": "Round-trip time",
        "goodput_bps": "Throughput",
        "loss_fraction": "Packet loss fraction",
        "tcp_retransmission_delta": "TCP retransmission delta",
        "queue_backlog_bytes": "Queue backlog",
        "qdisc_queue_delay_proxy_ms": "Backlog-derived serialization-delay proxy",
        "qdisc_drops_delta": "Mapped qdisc drops delta",
        "qdisc_overlimits_delta": "Mapped qdisc overlimits delta",
    }.get(field, field)


def _line_chart(
    go: Any, st: Any, preview: dict[str, Any], field: str, run_record: dict[str, Any]
) -> None:
    points = preview["points"]
    unit = available_feature_units(field, run_record)
    derived = (
        field in set(run_record.get("feature_columns", []))
        or re.search(r"_\d+(?:\.\d+)?s_", field) is not None
    )
    kind = "Processed derived endpoint feature" if derived else "Processed raw/target field"
    st.markdown(f"**{_metric_label(field)}** · `{unit}` · {kind}")
    if field == "qdisc_queue_delay_proxy_ms":
        st.caption(
            "Backlog × 8 / declared bottleneck rate is a serialization-delay proxy, not a direct per-packet queueing-delay measurement."
        )
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[point["timestamp_seconds"] for point in points],
            y=[point.get(field) for point in points],
            mode="lines",
            name=_metric_label(field),
            connectgaps=False,
            line={"color": "#0891B2", "width": 2},
            customdata=[
                [point.get("bucket_sample_count"), point.get("bucket_contains_gap_or_missing")]
                for point in points
            ],
            hovertemplate="t=%{x:.6g} s<br>value=%{y}<br>bucket rows=%{customdata[0]}<br>contains gap/missing=%{customdata[1]}<extra></extra>",
        )
    )
    figure.update_layout(
        height=310,
        margin={"l": 45, "r": 20, "t": 20, "b": 45},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#CBD5E1" if st.session_state.get("acn_theme") == "Dark" else "#334155"},
        xaxis_title="Seconds since run origin",
        yaxis_title=unit,
        showlegend=False,
        hovermode="x unified",
    )
    figure.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,.22)")
    figure.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,.22)")
    st.plotly_chart(figure, use_container_width=True, key=f"acn-series-{field}")


def _event_chart(go: Any, st: Any, preview: dict[str, Any]) -> None:
    points = preview["points"]
    if not points or not any(point.get("event_active") is not None for point in points):
        st.info("No usable derived event-active labels are available for an event overlay.")
        return
    x = [point["timestamp_seconds"] for point in points]
    active = [1 if point.get("event_active") is True else None for point in points]
    inactive = [0 if point.get("event_active") is False else None for point in points]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x,
            y=inactive,
            mode="lines",
            line_shape="hv",
            line={"color": "#64748B", "width": 2},
            name="Observed inactive",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=active,
            mode="lines",
            line_shape="hv",
            line={"color": "#DC2626", "width": 3},
            name="Observed event-active",
        )
    )
    figure.update_layout(
        height=180,
        margin={"l": 45, "r": 20, "t": 20, "b": 45},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis={"tickvals": [0, 1], "ticktext": ["inactive", "active"]},
        xaxis_title="Seconds since run origin",
        hovermode="x unified",
    )
    st.plotly_chart(figure, use_container_width=True, key="acn-event-strip")
    st.caption(
        "Event states come from the supplied processed labels. Sampled transitions are not exact physical onset times; unknown/censored buckets are left blank."
    )


def _telemetry_page(st: Any, workspace: Workspace, snapshot: dict[str, Any] | None) -> None:
    _render_header(
        st,
        "Telemetry explorer",
        "A bounded, server-side preview of checksum-verified processed rows for one declared run.",
    )
    st.info(
        "The chart is a decimated view, not a replacement dataset. Missing values and declared long-gap crossings break lines; any bucket containing a missing value is shown as a gap rather than interpolated."
    )
    if not st.session_state.get("acn_processed_verified") and st.button(
        "Verify processed artifacts before plotting"
    ):
        with st.spinner("Rechecking processing-manifest checksums…"):
            status = verify_processed_artifacts(workspace.processed_root)
        st.session_state["acn_processed_verified"] = status
    verified = st.session_state.get("acn_processed_verified")
    if not isinstance(verified, dict) or verified.get("status") != "verified":
        st.warning(
            "No verified processed data are available. Prepare data through the existing validated CLI workflow; the dashboard does not create labels or prepared datasets."
        )
        if isinstance(verified, dict):
            for error in verified.get("errors", []):
                st.write(f"- {error}")
        return
    manifest = verified["manifest"]
    headers = verified["columns"]
    run_records = manifest.get("run_records", [])
    run_choices = [record.get("run_id") for record in run_records if isinstance(record, dict)]
    if not run_choices:
        st.error("Verified processing manifest contains no run records.")
        return
    by_run = {
        record["run_id"]: record
        for record in run_records
        if isinstance(record, dict) and record.get("run_id")
    }
    selected_run = st.selectbox("Run", run_choices, key="acn-telemetry-run")
    record = by_run[selected_run]
    fields_in_order = [
        "rtt_ms",
        "goodput_bps",
        "loss_fraction",
        "tcp_retransmission_delta",
        "queue_backlog_bytes",
        "qdisc_queue_delay_proxy_ms",
        "qdisc_drops_delta",
        "qdisc_overlimits_delta",
    ] + list(manifest.get("feature_columns", []))
    available = list(dict.fromkeys(field for field in fields_in_order if field in headers))
    if not available:
        st.info("The verified analysis table contains no supported telemetry columns to plot.")
        return
    defaults = [
        field
        for field in available
        if field
        in {
            "rtt_ms",
            "goodput_bps",
            "loss_fraction",
            "queue_backlog_bytes",
            "qdisc_queue_delay_proxy_ms",
        }
    ][:3] or available[:1]
    selected_fields = st.multiselect(
        "Series (only columns present in analysis.csv)",
        available,
        default=defaults,
        key="acn-telemetry-fields",
    )
    if st.button("Load bounded preview", type="primary", disabled=not selected_fields):
        st.session_state["acn_telemetry_request"] = (selected_run, tuple(selected_fields))
    request = st.session_state.get("acn_telemetry_request")
    if request != (selected_run, tuple(selected_fields)):
        st.info("Choose a run and available fields, then load the bounded preview.")
        return
    try:
        with st.spinner("Streaming the selected run into a bounded preview…"):
            full_preview = load_bounded_run_preview(
                workspace.processed_root,
                manifest,
                selected_run,
                tuple(selected_fields),
                max_points=MAX_CHART_POINTS,
            )
        start, end = full_preview["time_min"], full_preview["time_max"]
        selected_preview = full_preview
        if start is not None and end is not None and end > start:
            selected_range = st.slider(
                "Displayed time range (seconds since run origin)",
                min_value=float(start),
                max_value=float(end),
                value=(float(start), float(end)),
                key=f"acn-time-range-{selected_run}",
            )
            if tuple(map(float, selected_range)) != (float(start), float(end)):
                selected_preview = load_bounded_run_preview(
                    workspace.processed_root,
                    manifest,
                    selected_run,
                    tuple(selected_fields),
                    time_start=float(selected_range[0]),
                    time_end=float(selected_range[1]),
                    max_points=MAX_CHART_POINTS,
                )
        st.caption(
            f"Source rows in run: {_format_count(selected_preview['source_rows'])} · displayed buckets: {_format_count(selected_preview['display_points'])} (limit {MAX_CHART_POINTS:,}) · exact target counts: {selected_preview['label_counts']}"
        )
        st.caption(
            f"Decimation: {selected_preview['decimation']}. Range filtering is performed server-side; source CSV is unchanged."
        )
        import plotly.graph_objects as go

        for field in selected_fields:
            _line_chart(go, st, selected_preview, field, record)
        _event_chart(go, st, selected_preview)
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "timestamp_seconds",
                *selected_fields,
                "event_active",
                "future_event_onset",
                "bucket_sample_count",
                "bucket_contains_gap_or_missing",
            ],
        )
        writer.writeheader()
        writer.writerows(selected_preview["points"])
        st.download_button(
            "Download displayed bounded preview CSV",
            data=output.getvalue().encode("utf-8"),
            file_name=f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', selected_run)}-preview.csv",
            mime="text/csv",
            help="This is only the decimated chart preview, not an export of the underlying source data.",
        )
        with st.expander("Run source provenance and configuration"):
            st.write(
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
                        "missing_measurement_counts",
                        "measurement_parse_error_counts",
                    )
                }
            )
            if isinstance(record.get("run_metadata"), dict):
                st.json(record["run_metadata"])
    except DashboardDataError as exc:
        st.error(f"Telemetry preview was not loaded: {exc}")


def _find_ground_truth_report(
    snapshot: dict[str, Any] | None,
) -> tuple[Path | None, dict[str, Any] | None]:
    if snapshot is None:
        return None, None
    for result in snapshot.get("ground_truth_reports", []):
        if result.get("status") == "verified":
            return result["path"], result
    return None, None


def _ground_truth_page(st: Any, workspace: Workspace, snapshot: dict[str, Any] | None) -> None:
    _render_header(
        st,
        "Ground truth & labels",
        "Review target settings, event intervals, censoring and alignment state without changing the protocol.",
    )
    if snapshot is None:
        st.info(
            "Refresh Overview integrity checks first. This page does not hash or parse processed data automatically."
        )
        return
    processed = snapshot.get("processed")
    if processed.get("status") != "verified":
        st.info(
            "No checksum-verified prepared dataset is available. Labeling has not been run in the dashboard; use the existing preparation workflow after inputs and protocol decisions are reviewed."
        )
        return
    manifest = processed["manifest"]
    protocol_status = manifest.get("protocol_status", "not_declared")
    if protocol_status == "protocol_frozen":
        st.warning(
            "Prepared labels are associated with a frozen configuration fingerprint; that does not independently establish label validity."
        )
    else:
        st.warning(
            "These labels are exploratory/unfrozen. They must not be used for confirmatory evaluation; re-prepare only after an independent protocol freeze."
        )
    st.subheader("Declared target definition")
    st.json(manifest.get("label_parameters", {}))
    st.caption(
        "Queue delay uses qdisc backlog × 8 / declared bottleneck rate: a serialization-delay proxy, not direct per-packet queueing delay."
    )
    cols = st.columns(4)
    cols[0].metric("Runs", _format_count(manifest.get("run_count")))
    cols[1].metric("Configurations", _format_count(manifest.get("configuration_count")))
    cols[2].metric("Label counts", str(manifest.get("label_counts", {})))
    cols[3].metric("Censored anchors", _format_count(manifest.get("censored_label_anchor_count")))
    alignment = manifest.get("alignment_review", {"status": "not_applicable"})
    st.write(
        f"Input alignment approval: **{alignment.get('status', 'unknown')}**. This is separate from target/protocol approval."
    )
    with st.expander("Censoring, missingness and exclusions"):
        st.json(
            {
                "exclusion_reason_counts": manifest.get("exclusion_reason_counts"),
                "rows_excluded_for_invalid_timestamp": manifest.get(
                    "rows_excluded_for_invalid_timestamp"
                ),
                "retained_rows_with_measurement_issue_entries": manifest.get(
                    "retained_rows_with_measurement_issue_entries"
                ),
                "censored_label_anchor_count": manifest.get("censored_label_anchor_count"),
                "evidence_status": manifest.get("evidence_status"),
            }
        )
    directory, report_result = _find_ground_truth_report(snapshot)
    if directory and report_result:
        report = report_result["report"]
        st.subheader("Verified descriptive ground-truth report")
        st.caption(
            f"Source analysis SHA-256: `{report.get('analysis_sha256')}` · protocol state: `{report.get('protocol_status')}`"
        )
        st.write(
            {
                key: report.get(key)
                for key in (
                    "run_count",
                    "configuration_count",
                    "row_count",
                    "event_count_observed_in_trace",
                    "events_with_identified_onset_interval",
                    "feature_quality",
                    "inconsistent_declared_source_units_normalized_by_adapter",
                )
            }
        )
        if report.get("per_run"):
            st.dataframe(
                [
                    {
                        "run_id": run.get("run_id"),
                        "configuration_id": run.get("configuration_id"),
                        "rows": run.get("retained_timestamp_rows"),
                        "events": run.get("labels", {}).get("event_count_observed_in_trace"),
                        "positive_onset_anchors": run.get("labels", {}).get(
                            "positive_onset_anchor_count"
                        ),
                        "censored_anchors": run.get("labels", {}).get(
                            "censored_or_missing_anchor_count"
                        ),
                        "onset_uncertainty": "interval/censored; see events.csv",
                    }
                    for run in report["per_run"]
                ],
                use_container_width=True,
                hide_index=True,
            )
        try:
            events_path = safe_child(directory, "events.csv")
            event_bytes = safe_download(events_path, directory, limit_bytes=2_000_000)
            event_rows = list(csv.DictReader(io.StringIO(event_bytes.decode("utf-8"))))
            st.dataframe(event_rows[:1000], use_container_width=True, hide_index=True)
            if len(event_rows) > 1000:
                st.caption(
                    "Showing 1,000 event intervals; download the full report for remaining records."
                )
        except (DashboardDataError, UnicodeError, csv.Error) as exc:
            st.warning(f"Event interval table unavailable: {exc}")
    else:
        if snapshot is None:
            st.info(
                "Refresh Overview integrity checks to verify ground-truth report links. Existing CLI: `acn validate-target --processed-dir … --output <new-report-dir>`."
            )
        else:
            st.info(
                "No ground-truth report among the three most recent candidates linked to the current verified sources. Existing CLI: `acn validate-target --processed-dir … --output <new-report-dir>`."
            )
    st.caption(
        "Threshold crossings and event boundaries are sample-interval estimates; onset uncertainty and gap censoring must remain visible in any interpretation."
    )


def _protocol_page(st: Any, workspace: Workspace, snapshot: dict[str, Any] | None) -> None:
    _render_header(
        st,
        "Protocol workspace",
        "See which decisions are frozen, which are unresolved, and what evidence each requires.",
    )
    config = snapshot.get("config") if snapshot else _config_state(workspace)[0]
    if config is None:
        st.error(
            "Experiment configuration is missing or invalid. No approval workflow is offered here."
        )
        return
    protocol_status = str(config.get("status", "not_declared"))
    if protocol_status == "protocol_frozen":
        st.success(
            "Protocol status is frozen in the configuration file. The UI does not alter or approve that decision."
        )
        try:
            verify_protocol_freeze_evidence(config, workspace.config_path)
            st.write(
                {
                    "config_sha256": file_sha256(workspace.config_path),
                    "protocol_freeze": config.get("protocol_freeze"),
                }
            )
        except Exception as exc:  # noqa: BLE001 - report any freeze-evidence failure in the UI.
            st.error(f"Freeze evidence did not verify: {exc}")
    else:
        st.warning(
            f"Protocol status: {protocol_status}. Confirmatory evaluation remains blocked until a separately reviewed freeze record is supplied."
        )
        st.code(
            file_sha256(workspace.config_path)
            if workspace.config_path.is_file()
            else "config hash unavailable"
        )
    st.subheader("Research-defining settings")
    for section_name in ("labeling", "evaluation", "features", "data_quality", "testbed"):
        section = config.get(section_name)
        if isinstance(section, dict):
            with st.expander(
                section_name.replace("_", " ").title(), expanded=(section_name == "labeling")
            ):
                st.json(section)
                unresolved = [
                    key
                    for key, value in section.items()
                    if value is None and key not in {"notes", "description"}
                ]
                if unresolved:
                    st.warning("Unresolved values: " + ", ".join(unresolved))
    decision_path = workspace.project_root / "docs" / "DECISION_REGISTER.md"
    decision_text = _safe_markdown(decision_path)
    if decision_text:
        with st.expander("Decision register", expanded=True):
            st.markdown(decision_text)
    matrix_path = workspace.project_root / "docs" / "CONFIGURATION_MATRIX.md"
    matrix_text = _safe_markdown(matrix_path)
    if matrix_text:
        with st.expander("Configuration matrix"):
            st.markdown(matrix_text)
    manifest, error = _read_manifest(workspace)
    if manifest and manifest.get("alignment_status"):
        st.subheader("Alignment review")
        st.write(
            f"Status: **{manifest.get('alignment_status')}** · review: **{manifest.get('alignment_review', {}).get('status', 'missing')}**"
        )
        if manifest.get("alignment_binding_sha256"):
            st.code(manifest["alignment_binding_sha256"])
        st.caption(
            "No interface to approve alignment is provided. The checked decision record is a human responsibility and does not prove clock truth."
        )
    elif error:
        st.caption(error)
    if workspace.alignment_spec_path and workspace.alignment_spec_path.is_file():
        st.subheader("Declared alignment specification")
        try:
            spec = read_json(workspace.alignment_spec_path)
            st.write(
                {
                    "spec_sha256": file_sha256(workspace.alignment_spec_path),
                    "policy": spec.get("alignment_policy"),
                    "run_count": len(spec.get("runs", [])),
                    "run_ids": [
                        item.get("run_id")
                        for item in spec.get("runs", [])
                        if isinstance(item, dict)
                    ],
                }
            )
        except DashboardDataError as exc:
            st.error(str(exc))
    processed = snapshot.get("processed") if snapshot else None
    if processed is None:
        st.info("Refresh Overview integrity checks before inspecting saved evaluation splits.")
    elif processed.get("status") == "verified":
        st.subheader("Evaluation group assignment")
        evaluation = snapshot.get("latest_valid_evaluation") if snapshot else None
        if evaluation:
            try:
                split = read_json(safe_child(evaluation["path"], "splits.json"))
                st.write(
                    {
                        "grouping": split.get("grouping"),
                        "seed": split.get("seed"),
                        "group assignments": split.get("groups"),
                        "counts": split.get("counts"),
                    }
                )
                if split.get("grouping") != "configuration_id":
                    st.warning(
                        "Run grouping allows configuration overlap; it is not an unseen-configuration generalization test."
                    )
            except DashboardDataError as exc:
                st.error(str(exc))
        else:
            st.info("No currently provenance-verified evaluation split is available.")


def _safe_report_download(
    st: Any,
    root: Path,
    relative: str,
    label: str,
    mime: str,
    *,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
) -> None:
    try:
        path = safe_child(root, relative)
        data = safe_download(path, root, limit_bytes=max_bytes)
        st.download_button(
            label, data=data, file_name=path.name, mime=mime, key=f"download-{root.name}-{relative}"
        )
    except DashboardDataError as exc:
        st.caption(f"{label} unavailable: {exc}")


def _evaluation_page(st: Any, workspace: Workspace, snapshot: dict[str, Any] | None) -> None:
    _render_header(
        st,
        "Model evaluation",
        "Only fully linked reports are shown as validated. Baselines, models and event metrics remain contextual—not a leaderboard.",
    )
    if not snapshot:
        st.info("Refresh Overview integrity checks to inspect evaluation reports.")
        return
    latest = snapshot.get("latest_evaluation")
    if not latest:
        st.info(
            "No evaluation report exists. The dashboard will not synthesize metrics or present test fixtures as results."
        )
        return
    report = latest.get("report") or {}
    status = report.get("status", latest.get("status"))
    if latest.get("status") != "complete_validated_provenance":
        st.error(
            f"Latest report is not validated: outcome `{status}`; linkage status `{latest.get('status')}`."
        )
        for message in latest.get("errors", []):
            st.write(f"- {message}")
        if report.get("method_failures"):
            st.subheader("Recorded method failures")
            st.json(report["method_failures"])
        st.info(
            "Partial, failed or provenance-unverified reports are not presented as successful metrics."
        )
        return
    st.success(
        "Complete evaluation report, processed artifacts, configuration/freeze, audit, run manifest and source hashes were rechecked in this session."
    )
    st.caption(
        f"Report directory: `{display_path(latest['path'], workspace)}` · created: {report.get('created_at_utc')} · protocol: `{report.get('protocol_status')}`"
    )
    st.write(
        {
            "dataset SHA-256": report.get("analysis_sha256"),
            "config SHA-256": report.get("config_sha256"),
            "run-manifest SHA-256": report.get("source_manifest_sha256"),
            "split grouping": report.get("split_grouping"),
            "test groups": (report.get("split_counts") or {}).get("test"),
            "label counts": report.get("label_counts"),
            "software": report.get("software"),
        }
    )
    try:
        metrics = read_json(safe_child(latest["path"], "metrics.json"))
        splits = read_json(safe_child(latest["path"], "splits.json"))
    except DashboardDataError as exc:
        st.error(f"Evaluation artifacts could not be loaded: {exc}")
        return
    rows = []
    per_configuration = []
    per_run = []
    for method, result in metrics.items():
        if method == "method_failures" or not isinstance(result, dict):
            continue
        event = result.get("test_event_metrics", {})
        window = result.get("test_window_metrics_secondary", result.get("test_metrics", {}))
        lead = event.get("warning_lead_time_seconds", {})
        rows.append(
            {
                "method": method,
                "event_count": event.get("event_count"),
                "event_recall": event.get("event_recall"),
                "event_precision": event.get("event_precision"),
                "false_alarms_per_minute": event.get("false_alarms_per_minute"),
                "median_warning_lead_seconds": lead.get("median")
                if isinstance(lead, dict)
                else None,
                "PR average precision (secondary)": window.get("average_precision"),
                "scored anchors": window.get("n_scored"),
                "test threshold": result.get("threshold_calibration", {}).get("threshold"),
                "method type": "baseline" if method in {"static_threshold", "ewma"} else "ML model",
            }
        )
        uncertainty = result.get("test_uncertainty", {})
        if uncertainty:
            for metric, interval in uncertainty.items():
                if isinstance(interval, dict):
                    per_configuration.append(
                        {
                            "method": method,
                            "metric": metric,
                            "cluster_count": interval.get("cluster_count"),
                            "estimable": interval.get("estimable"),
                            "interval": interval.get("interval"),
                            "limitation": interval.get("warning") or interval.get("reason"),
                        }
                    )
        for item in (
            event.get("per_configuration", {}).values()
            if isinstance(event.get("per_configuration"), dict)
            else []
        ):
            per_run.append({"method": method, "scope": "configuration", **item})
        for item in event.get("per_run", []) if isinstance(event.get("per_run"), list) else []:
            per_run.append({"method": method, "scope": "run", **item})
    if rows:
        st.subheader("Held-out event and secondary anchor metrics")
        st.caption(
            "Event metrics are primary in this report. PR average precision is a secondary row/anchor metric; it is undefined when supported classes are absent. Null means undefined/not estimated—not zero. No single-metric model ranking is applied."
        )
        st.dataframe(rows, use_container_width=True, hide_index=True)
    if per_configuration:
        with st.expander("Configuration-cluster uncertainty"):
            st.dataframe(per_configuration, use_container_width=True, hide_index=True)
    if per_run:
        with st.expander("Per-configuration and per-run test outcomes"):
            st.dataframe(per_run[:2000], use_container_width=True, hide_index=True)
            if len(per_run) > 2000:
                st.caption(
                    "Showing the first 2,000 rows; full metrics JSON is available for download."
                )
    if report.get("failed_methods") or metrics.get("method_failures"):
        st.warning("Method failures are recorded; review them before interpretation.")
        st.json(metrics.get("method_failures", {}))
    st.subheader("Saved evaluation split")
    st.write(
        {
            "grouping": splits.get("grouping"),
            "seed": splits.get("seed"),
            "groups": splits.get("groups"),
            "counts": splits.get("counts"),
        }
    )
    if splits.get("grouping") != "configuration_id":
        st.warning(
            "This split groups by run, which permits configuration overlap; it is not an unseen-configuration test."
        )
    st.caption(
        "Thresholds were selected on validation groups only. Event matching/lead-time definitions and limitations are in the saved report."
    )
    with st.expander("Metric definitions"):
        st.markdown(
            "- **Event recall/precision:** future events matched to warning episodes under the declared merge and lead-time rules.\n- **False alarms/minute:** unmatched warning episodes divided by eligible event-free exposure; censored/gapped intervals are excluded and reported.\n- **Warning lead time:** interval between a credited warning and observed event onset sample; discrete sampling means onset is interval-uncertain.\n- **PR average precision:** secondary anchor-level ranking metric, not an independent-event score.\n- **Configuration-cluster interval:** bootstrap over held-out configuration groups; null/unestimable intervals must remain null."
        )


def _robustness_page(st: Any, workspace: Workspace, snapshot: dict[str, Any] | None) -> None:
    _render_header(
        st,
        "Robustness & provenance",
        "Inspect exploratory sensitivity, ablation, controls, uncertainty and exact data lineage.",
    )
    reports = snapshot.get("robustness_reports", []) if snapshot else []
    if not reports:
        st.info(
            "Refresh Overview integrity checks to verify up to three recent robustness reports. Analyses are not launched from the dashboard; use an explicit, predeclared CLI specification after protocol freeze."
        )
        return
    report_options: list[tuple[str, Path, dict[str, Any]]] = []
    for item in reports:
        directory = item["path"]
        report_options.append((f"{directory.name} · {item.get('status')}", directory, item))
    label = st.selectbox(
        "Robustness artifact", [item[0] for item in report_options], key="acn-robustness-report"
    )
    _, directory, result = next(item for item in report_options if item[0] == label)
    if result.get("status") not in {"verified", "partial_failure"}:
        st.warning("This report is not linked to the currently verified processed dataset.")
        for error in result.get("errors", []):
            st.write(f"- {error}")
        return
    report = result["report"]
    st.warning(
        "Robustness outputs are exploratory unless the exact specification and analysis plan were frozen before outcomes were inspected."
    )
    if report.get("status") != "complete":
        st.error(
            f"Robustness outcome is {report.get('status')!r}; integrity verification does not convert a partial failure into a complete result."
        )
    st.write(
        {
            key: report.get(key)
            for key in (
                "status",
                "protocol_status",
                "analysis_sha256",
                "config_sha256",
                "processing_manifest_sha256",
                "source_manifest_sha256",
                "specification_sha256",
                "source_runs",
            )
        }
    )
    try:
        result_path = safe_child(directory, "robustness.json")
        if result_path.stat().st_size > MAX_ROBUSTNESS_RENDER_BYTES:
            st.info(
                "Robustness JSON exceeds the 1 MiB browser-render limit; the Reports page offers downloads up to 16 MiB, otherwise use the local artifact path."
            )
        else:
            result_json = read_json(result_path)
            st.json(result_json)
    except (DashboardDataError, OSError) as exc:
        st.error(str(exc))


def _reports_page(st: Any, workspace: Workspace, snapshot: dict[str, Any] | None) -> None:
    _render_header(
        st,
        "Reports",
        "Download bounded, checksum-verified artifacts from their allowlisted report directories.",
    )
    st.caption(
        f"In-app download limit: {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MiB per file. Larger outputs remain available at their local artifact path; the dashboard will not load them into memory."
    )
    if not snapshot:
        st.info("Refresh integrity checks to discover and verify reports.")
        return
    if snapshot.get("latest_valid_evaluation"):
        item = snapshot["latest_valid_evaluation"]
        report = item["report"]
        st.subheader("Latest validated evaluation")
        st.caption(
            f"{display_path(item['path'], workspace)} · dataset `{report.get('analysis_sha256')}` · protocol `{report.get('protocol_sha256')}`"
        )
        for filename, label, mime in (
            ("report.md", "Download evaluation report (Markdown)", "text/markdown"),
            ("metrics.json", "Download verified evaluation metrics", "application/json"),
            ("splits.json", "Download verified evaluation split", "application/json"),
        ):
            _safe_report_download(st, item["path"], filename, label, mime)
        st.caption(
            "Prediction CSV downloads are available only when each file is within the in-app size limit and is listed in the verified manifest."
        )
        for method, artifact in (report.get("prediction_files") or {}).items():
            relative = artifact.get("path") if isinstance(artifact, dict) else None
            if not isinstance(relative, str):
                continue
            _safe_report_download(
                st, item["path"], relative, f"Download test predictions — {method}", "text/csv"
            )
    else:
        st.info(
            "No complete evaluation with validated current provenance is available. Partial/failed reports are not exposed here as success downloads."
        )
    directory, bundle = _find_ground_truth_report(snapshot)
    if directory and bundle:
        st.subheader("Verified descriptive ground-truth report")
        _safe_report_download(
            st,
            directory,
            "ground_truth_report.json",
            "Download ground-truth report JSON",
            "application/json",
        )
        _safe_report_download(
            st, directory, "report.md", "Download ground-truth report Markdown", "text/markdown"
        )
        _safe_report_download(
            st, directory, "events.csv", "Download event intervals CSV", "text/csv"
        )
    verified_robustness = next(
        (
            (item["path"], item)
            for item in snapshot.get("robustness_reports", [])
            if item.get("status") in {"verified", "partial_failure"}
        ),
        None,
    )
    if verified_robustness:
        directory, bundle = verified_robustness
        st.subheader("Verified exploratory robustness report")
        st.caption(
            f"Recorded analysis outcome: {bundle['report'].get('status')} · current integrity verification is not scientific validation."
        )
        _safe_report_download(
            st, directory, "report.md", "Download robustness report Markdown", "text/markdown"
        )
        _safe_report_download(
            st, directory, "robustness.json", "Download robustness result JSON", "application/json"
        )
    st.caption(
        "All report payloads retain their own evidence status and limitations. A downloadable file is not an endorsement of scientific validity."
    )


def _ui_readiness(st: Any, workspace: Workspace) -> None:
    st.subheader("Detailed readiness check")
    if workspace.manifest_path is None or workspace.audit_report_path is None:
        st.warning(
            "A run manifest and audit report must both be configured before readiness can be checked."
        )
        return
    try:
        manifest = read_json(workspace.manifest_path)
        size = 0
        for entry in manifest.get("runs", []):
            if isinstance(entry, dict) and isinstance(entry.get("file"), str):
                size += safe_child(workspace.raw_root, entry["file"]).stat().st_size
        cap = int(os.environ.get("ACN_DASHBOARD_READINESS_MAX_INPUT_BYTES", str(32 * 1024 * 1024)))
    except (DashboardDataError, ValueError, OSError) as exc:
        st.error(f"Cannot safely estimate readiness input size: {exc}")
        return
    st.caption(
        f"Declared input CSV size: {size:,} bytes · in-UI parser limit: {cap:,} bytes. Existing readiness backend parses each run into memory; larger inputs should use the CLI after considering resource limits."
    )
    confirmed = st.checkbox(
        "I understand readiness re-parses each declared run and checks only software/data gates, not scientific validity.",
        key="acn-readiness-confirm",
    )
    if st.button("Run detailed readiness check", disabled=not confirmed or size > cap):
        try:
            with st.spinner(
                "Running the existing readiness validator; no progress estimate is available…"
            ):
                report = build_readiness_report(
                    raw_root=workspace.raw_root,
                    manifest_path=workspace.manifest_path,
                    audit_report_path=workspace.audit_report_path,
                    config_path=workspace.config_path,
                )
            st.session_state["acn_readiness_report"] = report
            st.session_state["acn_readiness_checked_at"] = datetime.now(timezone.utc).isoformat()
        except (ReadinessError, ProcessingError, OSError, ValueError) as exc:
            st.error(f"Readiness failed safely: {exc}")
    if size > cap:
        st.warning(
            "Input exceeds the configured in-UI size limit. No parse was started; use the established CLI deliberately."
        )
    report = st.session_state.get("acn_readiness_report")
    if report:
        st.caption(
            f"Last check: {st.session_state.get('acn_readiness_checked_at')} · {report.get('status')} · this is not scientific approval."
        )
        st.dataframe(report.get("checks", []), use_container_width=True, hide_index=True)


def run_app() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit(
            "Install the legacy UI dependencies with `python3 -m pip install -e '.[legacy-ui]'`."
        ) from exc

    st.set_page_config(
        page_title="ACN Research Console",
        page_icon="◫",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    theme = st.sidebar.selectbox("Appearance", ["Light", "Dark"], index=0, key="acn_theme")
    st.markdown(_theme_css(theme), unsafe_allow_html=True)
    st.sidebar.markdown(
        "<div class='acn-kicker'>ACN research console</div>", unsafe_allow_html=True
    )
    st.sidebar.caption(f"Software v{app_version()} · local artifacts only")
    page = st.sidebar.radio("Navigate", _NAVIGATION, key="acn_navigation")
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "No live traffic control, automatic alignment, label editing, protocol approval, or model training is exposed here."
    )

    try:
        workspace = resolve_workspace()
    except DashboardDataError as exc:
        _render_header(
            st,
            "Workspace configuration",
            "The console needs explicit, safe project/data roots before reading artifacts.",
        )
        st.error(str(exc))
        st.markdown(
            "Set `ACN_PROJECT_ROOT` and, if needed, `ACN_RAW_ROOT`, `ACN_PROCESSED_ROOT`, `ACN_REPORTS_ROOT`, `ACN_CONFIG_PATH`, `ACN_RUN_MANIFEST`, and `ACN_AUDIT_REPORT` in the launch environment. There is no filesystem browser or path upload."
        )
        return
    session_key = (
        str(workspace.project_root)
        + "|"
        + str(workspace.raw_root)
        + "|"
        + str(workspace.processed_root)
    )
    if st.session_state.get("acn_workspace_key") != session_key:
        st.session_state["acn_workspace_key"] = session_key
        st.session_state["acn_snapshot"] = None
        st.session_state.pop("acn_processed_verified", None)
        st.session_state.pop("acn_telemetry_request", None)
        st.session_state.pop("acn_inventory_report", None)
        st.session_state.pop("acn_inventory_checked_at", None)
        st.session_state.pop("acn_readiness_report", None)
        st.session_state.pop("acn_readiness_checked_at", None)
        clear_preview_cache()
    snapshot = st.session_state.get("acn_snapshot")

    if page == "Overview":
        _overview(st, workspace, snapshot)
        if snapshot:
            _ui_readiness(st, workspace)
    elif page == "Data & artifacts":
        _data_page(st, workspace, snapshot)
    elif page == "Telemetry":
        _telemetry_page(st, workspace, snapshot)
    elif page == "Ground truth & labels":
        _ground_truth_page(st, workspace, snapshot)
    elif page == "Protocol workspace":
        _protocol_page(st, workspace, snapshot)
    elif page == "Model evaluation":
        _evaluation_page(st, workspace, snapshot)
    elif page == "Robustness & provenance":
        _robustness_page(st, workspace, snapshot)
    elif page == "Reports":
        _reports_page(st, workspace, snapshot)
    st.markdown("<hr class='acn-rule'>", unsafe_allow_html=True)
    st.caption(
        "Research software only · displayed values come from checksum-linked artifacts · synthetic tests are not network evidence."
    )


if __name__ == "__main__":
    run_app()
