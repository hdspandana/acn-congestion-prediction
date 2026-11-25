# ACN AI-Based Network Congestion Prediction

Research software for evaluating whether **endpoint-only telemetry forecasts a future bottleneck queue event** at a useful lead time and predeclared false-alarm rate on configurations excluded from training. This repository is a reproducible analysis scaffold, **not evidence that an ML predictor works**.

## Current evidence and scope

The public repository initially contained only four tracked scaffold files. This workspace adds installable Python packaging, read-only artifact/CSV inspection and testbed preflight, explicit-manifest data preparation, a separate-stream aligner gated by hash-linked human approval, future-onset labels with censoring, leakage-resistant grouped/factor-level holdouts, rule baselines, optional sklearn models, event-level evaluation, uncertainty estimation, a descriptive ground-truth validation report, bounded exploratory robustness analyses, and checksum-linked reports.

The owner's unpushed Kali changes and raw pilot measurements are **not available in this workspace**. No data collection, privileged topology changes, network traffic tests, model training on real observations, or performance claims have been made. Synthetic records occur only in software tests. Configuration values that define the research question (queue-delay threshold, persistence, horizon, observation window, ground-truth gap, feature windows, alarm policy, and the feature temporal policy's maximum gap/minimum valid fraction/minimum temporal coverage) remain deliberately unset. `configs/experiment.yaml` records an owner-described but unverified topology; it is not proof of the current host state.

## Research constraints

- Router bottleneck-qdisc backlog is the intended ground truth. `backlog_bytes * 8 / configured_rate_bps` is a serialization-delay proxy, **not direct per-packet queueing delay**.
- Queue/backlog/counter/phase/label/group identifiers are blocked as endpoint predictors. Features use trailing observations at or before each prediction anchor.
- Future onset is distinct from detecting an event already active. Incomplete or gappy horizons are censored, not converted to negatives.
- The primary evaluation holds out whole configurations. Whole-run grouping is available as an explicitly weaker alternative; row-random primary splits are rejected.
- Thresholds are calibrated on validation groups under an event false-alarms-per-minute cap. Held-out event recall/precision, false-alarm rate, lead time, secondary window metrics, and group-level uncertainty are written with provenance hashes.
- A software test passing establishes software behavior only. It does not validate collector schemas, qdisc semantics, timing alignment, or prediction performance.

## Install and test

Python 3.10+ is required for the Python package. Node.js 20+ and npm are required to test and build the React frontend. The primary user application is the React/TypeScript + FastAPI local app; Python remains the source of truth for data validation, model lifecycle, inference, evaluation, provenance, and safeguards. See [the web application guide](docs/WEB_APP.md).

```bash
# Research CLI/tests and ML model families:
python3 -m pip install -e '.[web-ml,test]'
python3 -m pytest -q
# Test and build the React application, then serve it from FastAPI:
cd web
npm ci
npm test -- --run
npm run typecheck
npm run build
cd ..
acn-web
```

The legacy Streamlit research console remains in the source tree for compatibility, but it is not the supported operational UI and does not replace the React application. Existing research APIs and the `acn` CLI remain available.

For supported Python/Node versions, exact local and CI verification commands, Ruff exceptions, installed-wheel smoke checks, and release-hygiene classifications, see [development and release verification](docs/DEVELOPMENT.md).

The optional [`qa_screenshots/`](qa_screenshots/README.md) directory contains UI layout and empty/error-state captures. Files with `test-only` in their names use clearly labelled synthetic or mocked browser fixtures; none of the screenshots are telemetry, experimental results, model predictions, or scientific evidence.

The `acn` executable offers:

```text
acn --help
acn validate-config configs/experiment.yaml
acn preflight --config configs/experiment.yaml
acn audit <raw-root> --output <audit.json> --timestamp-column <observed-time-header> --timestamp-unit <seconds|milliseconds|microseconds|nanoseconds>
# Optional only for two verified separate streams; generated outputs remain review-gated:
acn align-streams --raw-root <preserved-raw-root> --spec <reviewed-alignment-spec.json> --output-dir <new-outside-raw>
acn inspect-alignment-report --report-dir <new-outside-raw> --raw-root <preserved-raw-root> --spec <reviewed-alignment-spec.json>
# Machine-check files, schema, units/timestamps, target availability, counts, and processed provenance:
acn readiness --raw-root <raw-root> --manifest <reviewed-runs.json> --audit-report <audit.json> --config configs/experiment.yaml --output <readiness.json>
# Before freeze, descriptive preparation only (all provisional values/paths are explicit):
acn prepare --exploratory --config configs/experiment.yaml --raw-root <raw-root> --manifest <reviewed-runs.json> --audit-report <audit.json> --output <new-prepared-dir> --queue-delay-threshold-ms <candidate> --consecutive-samples-required <candidate> --prediction-horizon-ms <candidate> --observation-window-ms <candidate> --max-ground-truth-gap-ms <candidate> --feature-window-seconds <candidate-seconds>
acn validate-target --processed-dir <exploratory-prepared-dir> --output reports/target-validation/<new-run>
# After the separate protocol freeze, re-prepare under frozen values:
acn prepare --config configs/experiment.yaml --audit-report reports/audit.json
acn validate-target --processed-dir data/processed --output reports/target-validation/<new-run>
acn evaluate --config configs/experiment.yaml
acn inspect-report --report-dir reports/evaluations/<new-run> --processed-dir data/processed --raw-root data/raw --audit-report reports/audit.json --run-manifest data/raw/runs.json
acn robustness --config configs/experiment.yaml --processed-dir data/processed --spec /path/to/predeclared-spec.json --output reports/robustness/<new-run>
acn inspect-robustness-report --report-dir reports/robustness/<new-run> --processed-dir data/processed --raw-root data/raw
```

The `.yaml` configuration intentionally uses **JSON syntax**, parsed with Python's `json` module. `validate-config` accepts the unresolved draft; `--analysis-ready` also requires a complete `protocol_freeze` record (protocol ID/version, timezone-aware date, approvers, decision-record path and SHA-256), a real matching decision file, top-level `status: "protocol_frozen"`, and all research-defining values including a manifest path and an explicitly resolved feature temporal policy. Do not change that status merely to make the CLI run. The full frozen config SHA-256 is bound to prepared data and reports; any config edit requires re-preparation. Confirmatory `evaluate` also rejects seed/group overrides that differ from the frozen primary policy. Before the freeze, `prepare --exploratory` permits a descriptive pass only when every provisional target/horizon/window value and raw/manifest/output path is supplied explicitly. The resulting manifest/report is marked exploratory; evaluation refuses that dataset, and confirmatory preparation must be repeated from raw sources after protocol freeze.

## Safe inspection and data pipeline

Before mapping or preparing real collector data, create a value-sparing inventory outside the raw tree:

```bash
acn inspect-artifacts /path/to/copied-kali-artifacts --output reports/kali-artifact-inspection.json
```

The JSON inventory fingerprints regular files, profiles CSV structure and timestamps without emitting measurement rows, extracts only bounded/allowlisted metadata, and marks endpoint/router/qdisc fields as name-based candidates. It never follows symlinks or aligns streams. The separate `acn align-streams` command is available only after explicit clock/evidence/policy specification, emits traceable provenance, and blocks preparation until a hash-bound JSON decision approves the current alignment fingerprint. The fingerprint covers spec/policy, mappings, original source and clock-evidence hashes, and derived aligned/provenance/exclusion hashes; preparation rechecks those artifacts and rejects stale approvals. Review [the alignment contract](docs/SEPARATE_STREAM_ALIGNMENT.md) and [collector integration](docs/COLLECTOR_INTEGRATION.md#read-only-artifact-discovery-before-adaptation) for limitations. No Kali artifacts were available in this workspace, so no real schema or clock facts have been established.

`acn preflight` is strictly read-only. It checks required executables, configured namespace presence, interface addresses, expected routes, router forwarding, and the bottleneck TBF rate/burst/latency when tools and namespaces exist. It does **not** create or alter namespaces, links, routes, sysctls, qdiscs, or firewall rules; it sends no ping probes and starts no load. `ready_for_manual_review` only means these read-only checks matched the draft topology. In the current Arena environment, preflight returned exit code 2 / `not_ready`: `tc` and `iperf3` are absent and the expected namespaces do not exist. No corrective action was attempted.

`acn audit` recursively reads CSVs and writes a new JSON report and Markdown summary outside the input tree. It records file hashes, schema, missing values, timestamp problems, duplicate rows/times, cadence/gaps, malformed rows, and structural errors. It never repairs or deletes raw files and refuses to overwrite outputs.

`acn prepare` supports **one aligned CSV per run**, with a JSON run manifest mapping canonical fields and explicit units to observed source headers. The CLI requires `--audit-report`; it verifies the report is complete, covers the same raw root, and hashes each manifest input identically. It never infers mappings. For separate endpoint/router files, `acn align-streams` accepts only a reviewed explicit spec and writes source-row provenance, exclusions, and hashes; its generated manifest is unusable for preparation/readiness until a separate copy is marked approved with a hash-linked human decision record. See [`docs/SEPARATE_STREAM_ALIGNMENT.md`](docs/SEPARATE_STREAM_ALIGNMENT.md). Neither alignment nor approval proves real clocks, qdisc truth, scientific validity, or performance. Derived `analysis.csv` and `processing_manifest.json` retain source/config/audit hashes, run metadata, mappings, missing/invalid measurement counts, label reasons, and allowed feature columns. Use new output directories; existing output is never overwritten.

`acn validate-target` builds a descriptive report from the checksummed prepared table: raw backlog and backlog-derived proxy timelines, derived event intervals, onset/duration bounds, cadence/gaps, missing/invalid target values, censoring, phase mismatches, feature coverage, and optional explicitly mapped per-sample qdisc drop/overlimit increments. It never selects target thresholds from model outcomes. The proxy is **not direct per-packet queueing delay**. `inspect-target-report` can verify report artifacts against prepared data and raw hashes.

`acn evaluate` verifies the processed-data checksum, creates grouped train/validation/test partitions, fits preprocessing only on training rows that pass the resolved feature temporal policy and retain at least one observed non-coverage endpoint measurement summary, chooses thresholds on validation groups only, and evaluates held-out event performance. Baselines and models share the same scored-anchor eligibility; temporally ineligible or fully featureless rows are not assigned an imputed warning. It reports per-run/per-configuration event metrics, exposure exclusions, and cluster-bootstrap limitations. It writes a new-only report directory containing split assignments, metrics, predictions, raw/audit/config hashes, model/dependency settings, and a Markdown limitations statement. If an optional method fails, that method is explicitly recorded; the report is marked partial/failed and the CLI returns nonzero rather than appearing fully successful. Fewer than three independent split groups blocks evaluation before reports are written.

### Run-manifest contract

The preparation manifest is JSON with `schema_version: 2` and a nonempty `runs` list. Each run entry needs a unique `run_id`, a `configuration_id`, a relative CSV `file` under the configured raw root, numeric `bottleneck_rate_bps`, explicit `timestamp_unit` (`seconds`, `milliseconds`, `microseconds`, or `nanoseconds`), `column_map`, and `column_units` for each mapped numeric field. `column_map` must map `timestamp` and `queue_backlog_bytes`, and at least one configured endpoint feature such as `rtt_ms`, `loss_fraction`, `goodput_bps`, or `tcp_retransmission_delta`. Units are validated and supported units are converted to canonical feature units before processing where a defined conversion exists. TCP retransmission increments are the exception to cross-unit conversion: every mapped run must declare the same exact `packets` or `bytes` unit, packets and bytes are never converted into one another, and that unit is bound into the v2 feature specification and digest. Numeric timestamp units must match the audited manifest source; timezone-bearing ISO timestamps use seconds, while timezone-naive ISO values are rejected pending explicit timezone evidence. Optional phase fields are metadata, not predictors. See [collector integration](docs/COLLECTOR_INTEGRATION.md) before adapting real files. This manifest is a reviewed adapter contract, not an invented claim about the unavailable collector.

Example (illustrative schema only; names/data are not inferred from real traces):

```json
{
  "schema_version": 2,
  "runs": [{
    "run_id": "replace-with-a-real-unique-run-id",
    "configuration_id": "replace-with-a-reviewed-configuration-id",
    "file": "replace-with-an-inspected-aligned.csv",
    "bottleneck_rate_bps": 5000000,
    "timestamp_unit": "milliseconds",
    "column_map": {
      "timestamp": "replace_with_observed_timestamp_header",
      "queue_backlog_bytes": "replace_with_observed_backlog_header",
      "rtt_ms": "replace_with_observed_rtt_header"
    },
    "column_units": {"queue_backlog_bytes": "bytes", "rtt_ms": "ms"},
    "metadata": {"source_review": "record actual provenance here"}
  }]
}
```

Do not paste this example unchanged into a real run. Confirm the actual headers, units, alignment, and provenance first.

## Unfinished / blocked work

1. Bring in the latest local Kali checkout, preserve its unpushed work and raw files, and compare it with this scaffold without destructive cleanup.
2. Inspect actual collector scripts/logs and map their true schemas. For separate streams, use the explicit adapter only after reviewing actual clock evidence and policies; complete the human hash-linked approval gate. This software does not infer offsets/drift or validate scientific alignment.
3. Audit pilot runs and verify qdisc backlog semantics, interface/rate settings, actual-versus-commanded phases, timestamp clocks, sampling cadence, and collection failures.
4. Justify and preregister the threshold/persistence/horizon/window/gap/alarm settings and sensitivity grid from operational constraints and pilot traces—not model scores.
5. Reassess whether there are enough independent configurations for a meaningful held-out configuration evaluation; then run real analysis and report actual outputs.

`acn robustness` supports explicit, checksummed exploratory sensitivity, feature-family ablation, within-run circular-shift negative-control, model-seed stability, and evidence-linked factor-level holdouts; it requires a user-supplied JSON specification and does not search for or select a best threshold/horizon/seed. Factor holdout enforces configuration-level isolation and one factor level per configuration, but its scientific definition and adequacy remain human responsibilities. See [execution status](docs/EXECUTION_STATUS.md), [academic submission outline](docs/ACADEMIC_SUBMISSION_OUTLINE.md), [Kali first-inspection runbook](docs/KALI_INSPECTION_RUNBOOK.md), [machine-learning pipeline](docs/ML_PIPELINE.md), [real-data pilot workflow](docs/REAL_DATA_PILOT.md), [robustness analysis](docs/ROBUSTNESS.md), [decision register](docs/DECISION_REGISTER.md), [machine-checkable readiness report](docs/COLLECTOR_INTEGRATION.md#machine-readable-readiness-check), [proposed configuration matrix](docs/CONFIGURATION_MATRIX.md), [experiment protocol](docs/EXPERIMENT_PROTOCOL.md), [data dictionary](docs/DATA_DICTIONARY.md), [collector integration contract](docs/COLLECTOR_INTEGRATION.md), [development log](docs/DEVELOPMENT_LOG.md), [initial literature review](docs/LITERATURE.md), and [research-gap audit](docs/RESEARCH_GAP_AUDIT.md). The research-gap audit is repository documentation only; it is not displayed as website content. Publication of this repository does not change those evidence limits: no real trained model, live collector integration, or validated predictive performance is claimed.
