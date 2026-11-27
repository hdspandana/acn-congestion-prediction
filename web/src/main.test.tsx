import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from './App'

const overview = {
  application: { name: 'ACN Network Intelligence', version: '0.2.0', mode: 'local' },
  telemetry_source: { status: 'not_configured', message: 'No source is configured.', csv_count: 0, checked_at_utc: '2026-10-26T00:00:00Z' },
  model_status: { status: 'unavailable', eligible_count: 0, active_model_id: null, message: 'No eligible model.' },
  latest_warning: null, history_count: 0,
  data_freshness: { status: 'unknown', message: 'No wall-clock source timestamp.' },
  blockers: ['No real telemetry source is configured.'],
}

function json(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } })
}

function installFetch(overrides: { models?: unknown; prediction?: unknown; datasets?: unknown; research?: unknown; groundTruthIndex?: unknown; groundTruthSeries?: unknown; evaluationDetail?: unknown; reportDetail?: unknown } = {}) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const path = String(input)
    if (path.endsWith('/api/v1/overview')) return json(overview)
    if (path.endsWith('/api/v1/models')) return json(overrides.models ?? { models: [], active_model_id: null })
    if (path.endsWith('/api/v1/datasets')) return json(overrides.datasets ?? {
      raw: { status: 'missing', csv_count: 0, message: 'No inputs.' },
      processed: { status: 'missing', row_count: null, run_count: null, protocol_status: null, message: 'No prepared dataset.' },
      audit: { status: 'not_configured', message: 'No audit.' },
    })
    if (path.endsWith('/api/v1/research')) return json(overrides.research ?? {
      protocol: { status: 'draft', message: 'No frozen protocol.', config_sha256: null },
      alignment: { status: 'not_required', message: 'No separate alignment.' }, evaluations: [], reports: [],
      operations: { audit_available: true, evaluation_available: false, message: 'Blocked.', prepared_status: 'missing', source_lineage_status: 'unverified', blockers: ['No verified data.'] },
    })
    if (path.endsWith('/api/v1/research/ground-truth')) return json(overrides.groundTruthIndex ?? {
      status: 'unavailable', message: 'Prepared analysis is unavailable.', blockers: ['No current verified prepared data.'],
      alignment_status: 'unverified', alignment_message: 'Unavailable.', runs: [],
    })
    if (path.includes('/api/v1/research/ground-truth/series')) return overrides.groundTruthSeries ? json(overrides.groundTruthSeries) : json({ error: { code: 'not_found', message: 'No verified run.' } }, 404)
    if (path.includes('/api/v1/evaluations/')) return overrides.evaluationDetail ? json(overrides.evaluationDetail) : json({ error: { code: 'not_found', message: 'No evaluation.' } }, 404)
    if (path.includes('/api/v1/reports/')) return overrides.reportDetail ? json(overrides.reportDetail) : json({ error: { code: 'not_found', message: 'No report.' } }, 404)
    if (path.endsWith('/api/v1/predictions/pilot?candidate=expanded')) return json({
      training_rows: 450, cutoff: 0.0991, predictions: [{ timestamp_seconds: 0.4, rtt_ms: 14, score: 0.10 }],
    })
    if (path.endsWith('/api/v1/predictions') && overrides.prediction) return json(overrides.prediction)
    if (path.includes('/api/v1/telemetry/inspect')) return json({
      status: 'structurally_valid', source_kind: 'uploaded_batch', received_at_utc: '2026-10-26T00:00:00Z',
      input_name: 'trace.csv', input_sha256: 'a'.repeat(64), row_count: 2,
      fields: ['timestamp_seconds', 'rtt_ms'],
      points: [{ timestamp_seconds: 0, rtt_ms: 10 }, { timestamp_seconds: 1, rtt_ms: null }],
      quality: { status: 'review_required', missing_fraction: { rtt_ms: .5 }, gap_count: null,
        max_gap_seconds: 1, timestamp_span_seconds: 1,
        wall_clock_freshness: 'Unknown · run-relative timestamp', notes: ['Batch only.'] },
      warnings: [],
    })
    if (path.includes('/api/v1/history')) return json({ predictions: [] })
    return json({ error: { code: 'not_found', message: 'Not found' } }, 404)
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

afterEach(() => { cleanup(); vi.unstubAllGlobals(); localStorage.clear() })

describe('ACN Network Intelligence operations UI', () => {
  it('shows a truthful empty operational state with no fabricated scores or warnings', async () => {
    installFetch()
    render(<App />)
    expect(await screen.findByText('Predictions are not available yet.')).toBeInTheDocument()
    expect(screen.getByText('No forecast history')).toBeInTheDocument()
    expect(screen.getByText('No telemetry source')).toBeInTheDocument()
    expect(screen.queryByText(/accuracy|congestion score|healthy network/i)).not.toBeInTheDocument()
  })

  it('shows only the pilot workflow when the registry is empty', async () => {
    installFetch()
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('Predictions are not available yet.')
    await user.click(screen.getByRole('button', { name: 'Predict congestion' }))
    expect(screen.getByRole('button', { name: 'Run bundled demo' })).toBeEnabled()
    expect(screen.queryByText('Registered models')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Model registry' }))
    expect(await screen.findByText('No registered model yet')).toBeInTheDocument()
  })

  it('scores an uploaded RTT trace from the pilot panel with an empty registry', async () => {
    installFetch()
    const user = userEvent.setup()
    render(<App />)
    await user.click(await screen.findByRole('button', { name: 'Predict congestion' }))
    await user.upload(screen.getByLabelText('Pilot RTT CSV'), new File(
      ['timestamp_seconds,rtt_ms\n0,12\n0.2,13\n0.4,14\n'], 'trace.csv', { type: 'text/csv' },
    ))
    await user.click(screen.getByRole('button', { name: 'Run pilot prediction' }))
    expect(await screen.findByText(/1 of 1 forecast timestamps/)).toBeInTheDocument()
    expect(screen.getByText('What happened on the router is unknown.')).toBeInTheDocument()
    expect(screen.getByRole('region', { name: 'Forecast timeline' })).toBeInTheDocument()
    expect(screen.getByText('Measured ping time (ms)')).toBeInTheDocument()
  })

  it('runs the bundled endpoint trace without asking for an upload', async () => {
    installFetch()
    const user = userEvent.setup()
    render(<App />)
    await user.click(await screen.findByRole('button', { name: 'Predict congestion' }))
    await user.click(screen.getByText('Optional lab examples with router measurements'))
    await user.click(screen.getByRole('button', { name: 'Run bundled demo' }))
    expect(await screen.findByText(/1 of 1 forecast timestamps/)).toBeInTheDocument()
  })

  it('renders a malformed ineligible model record without inventing its contract', async () => {
    installFetch({
      models: {
        models: [{
          model_id: 'model_20261026T000000Z_1234abcd', method: null, created_at_utc: null,
          eligible: false, status: 'ineligible',
          reason: 'malformed or unavailable model manifest: fixture',
          forecast_horizon_seconds: null, score_threshold: null,
          score_semantics: 'not_recorded', calibration_status: 'not_recorded',
          feature_columns: [], feature_spec_version: null,
          feature_specification_sha256: null, feature_specification_status: null,
          tcp_retransmission_unit: null, feature_temporal_policy: null,
          feature_temporal_policy_status: null, model_parameter_overrides: null,
          evaluation_id: null, protocol_sha256: null, analysis_sha256: null,
        }],
        active_model_id: null,
      },
    })
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('Predictions are not available yet.')
    await user.click(screen.getByRole('button', { name: 'Open Model Workspace' }))
    expect(await screen.findByText('Unknown method')).toBeInTheDocument()
    expect(screen.getByText(/evaluation not recorded/)).toBeInTheDocument()
    expect(screen.getByText(/malformed or unavailable model manifest/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Use for inference' })).not.toBeInTheDocument()
  })

  it('renders an actual inference response with its uncalibrated score and unresolved outcome', async () => {
    const model = {
      model_id: 'model_20261026T000000Z_1234abcd', method: 'logistic_regression',
      created_at_utc: '2026-10-26T00:00:00Z', eligible: true, status: 'eligible', reason: null,
      forecast_horizon_seconds: 2, score_threshold: .7, score_semantics: 'uncalibrated positive-class model score; not a calibrated probability',
      feature_columns: ['rtt_ms_1s_latest'], evaluation_id: 'evaluation-fixture',
      protocol_sha256: 'b'.repeat(64), analysis_sha256: 'c'.repeat(64),
    }
    const record = {
      prediction_id: 'pred_fixture_only', predicted_at_utc: '2026-10-26T00:00:00Z',
      model_id: model.model_id, model_method: model.method, input_name: 'trace.csv',
      input_sha256: 'a'.repeat(64), row_count: 1, forecast_horizon_seconds: 2,
      score: .82, score_threshold: .7, risk: 'warning', outcome_state: 'unresolved', timestamp_seconds: 1,
    }
    const input = {
      status: 'structurally_valid', source_kind: 'uploaded_batch', received_at_utc: '2026-10-26T00:00:00Z',
      input_name: 'trace.csv', input_sha256: 'a'.repeat(64), row_count: 1,
      fields: ['timestamp_seconds', 'rtt_ms'], points: [{ timestamp_seconds: 1, rtt_ms: 12 }],
      quality: { status: 'complete', missing_fraction: { rtt_ms: 0 }, gap_count: null,
        max_gap_seconds: null, timestamp_span_seconds: 0, wall_clock_freshness: 'Unknown · run-relative timestamp', notes: [] },
      warnings: [record],
    }
    installFetch({ models: { models: [model], active_model_id: null }, prediction: { predictions: [record], input, model } })
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('Forecasts are ready when your telemetry is.')
    await user.click(screen.getByRole('button', { name: 'Predict congestion' }))
    const select = await screen.findByRole('combobox', { name: 'Validated model' })
    await user.selectOptions(select, model.model_id)
    await user.upload(screen.getByLabelText('Choose telemetry CSV file'), new File(['timestamp_seconds,rtt_ms\\n1,12\\n'], 'trace.csv', { type: 'text/csv' }))
    await user.click(screen.getByRole('button', { name: /Run prediction/i }))
    expect(await screen.findByText('1 warning above the validation threshold')).toBeInTheDocument()
    expect(screen.getByText(/not a calibrated probability/i)).toBeInTheDocument()
    expect(screen.getByText('Recorded · unresolved outcome')).toBeInTheDocument()
  })

  it('inspects an uploaded batch and labels it as non-live with run-relative time', async () => {
    installFetch()
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('Predictions are not available yet.')
    await user.click(screen.getByRole('button', { name: /Batch monitor/i }))
    const input = screen.getByLabelText('Choose telemetry CSV file') as HTMLInputElement
    const file = new File(['timestamp_seconds,rtt_ms\n0,10\n1,\n'], 'trace.csv', { type: 'text/csv' })
    await user.upload(input, file)
    expect(await screen.findByText('trace.csv')).toBeInTheDocument()
    expect(screen.getByText(/Uploaded batch/)).toBeInTheDocument()
    expect(screen.getByText('Unknown · run-relative timestamp')).toBeInTheDocument()
    expect(screen.getByText('50.0% missing')).toBeInTheDocument()
  })

  it('keeps research empty states explicit and gates evaluation when evidence is missing', async () => {
    installFetch()
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('Predictions are not available yet.')
    await user.click(screen.getByRole('button', { name: 'Ground truth' }))
    expect(await screen.findByText('No verified prepared batch data')).toBeInTheDocument()
    expect(screen.queryByText(/Observed events\s*0/)).not.toBeInTheDocument()
    await user.click(screen.getAllByRole('button', { name: 'Evaluation' }).slice(-1)[0])
    expect(await screen.findByText('No verified evaluations')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Run frozen evaluation/i })).toBeDisabled()
    expect(screen.getAllByText(/No verified data/).length).toBeGreaterThan(0)
  })

  it('compares artifact-backed methods without converting non-estimable metrics to zero', async () => {
    const evaluationId = 'evaluation_fixture_only'
    const report = {
      report_id: evaluationId, kind: 'evaluation', status: 'complete', integrity_status: 'verified',
      provenance_status: 'verified', created_at_utc: '2026-10-26T00:00:00Z',
      protocol_sha256: 'a'.repeat(64), analysis_sha256: 'b'.repeat(64),
      processing_manifest_sha256: 'c'.repeat(64), source_manifest_sha256: 'd'.repeat(64), audit_sha256: 'e'.repeat(64),
      grouping: 'configuration_id', blockers: [], artifacts: ['report.md', 'metrics.json', 'splits.json'],
      summary: {
        methods: [
          { method: 'static_threshold', event_count: 3, event_precision: null, event_recall: .5,
            false_alarms_per_minute: 0, lead_time_median_seconds: 1.2, lead_time_event_count: 2,
            average_precision: .6, roc_auc: .7, scored_anchor_count: 30, threshold: 12,
            validation_false_alarms_per_minute: 0, adequacy_warnings: [],
            uncertainty: { event_recall: { estimable: true, interval: [.2, .8], cluster_count: 4, confidence: .95 } } },
          { method: 'ewma', event_count: 3, event_precision: .7, event_recall: .6,
            false_alarms_per_minute: 1.5, lead_time_median_seconds: null, lead_time_event_count: 0,
            average_precision: null, roc_auc: null, scored_anchor_count: null, threshold: .3,
            validation_false_alarms_per_minute: 1.2, adequacy_warnings: ['sparse_events'], uncertainty: {} },
          { method: 'logistic_regression', event_count: 3, event_precision: .8, event_recall: .7,
            false_alarms_per_minute: .4, lead_time_median_seconds: 2.1, lead_time_event_count: 2,
            average_precision: .75, roc_auc: .8, scored_anchor_count: 40, threshold: .45,
            validation_false_alarms_per_minute: .3, adequacy_warnings: [], uncertainty: {} },
        ],
        method_failures: { random_forest: 'software fixture failure' }, requested_methods: ['static_threshold','ewma','logistic_regression','random_forest'],
        split_seed: 2026, held_out_group_count: 2, held_out_groups: ['cfg-7','cfg-8'], held_out_groups_truncated: false,
        split_row_counts: {}, split_event_counts: {}, split_group_counts: {}, threshold_calibration_split: 'validation',
        threshold_calibration_rule: 'validation false-alarm cap', verified_artifacts: ['metrics.json'],
      },
    }
    installFetch({
      research: {
        protocol: { status: 'protocol_frozen', message: 'Frozen fixture.', config_sha256: 'a'.repeat(64) },
        alignment: { status: 'not_required', message: 'No separate alignment.' },
        evaluations: [{ evaluation_id: evaluationId, status: 'complete', provenance_status: 'verified', grouping: 'configuration_id', created_at_utc: '2026-10-26T00:00:00Z', methods: ['static_threshold','ewma','logistic_regression'], blockers: [] }],
        reports: [], operations: { audit_available: true, evaluation_available: true, message: 'Ready.', prepared_status: 'verified', source_lineage_status: 'verified', blockers: [] },
      },
      evaluationDetail: report,
    })
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('Predictions are not available yet.')
    await user.click(screen.getByRole('button', { name: 'Evaluation' }))
    expect(await screen.findByText('Held-out event comparison')).toBeInTheDocument()
    expect(screen.getAllByText('Not estimable').length).toBeGreaterThan(0)
    expect(screen.getAllByText('0.000').length).toBeGreaterThan(0)
    expect(screen.getAllByText('logistic regression').length).toBeGreaterThan(0)
    expect(screen.getByText(/software fixture failure/)).toBeInTheDocument()
    expect(screen.getByText(/Held-out test groups \(2\)/)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /metrics.json/ })).toHaveAttribute('href', expect.stringContaining('/files/metrics.json'))
  })

  it('supports keyboard focus, mobile navigation, and theme switching', async () => {
    installFetch()
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('Predictions are not available yet.')
    const open = screen.getByRole('button', { name: 'Open navigation' })
    open.focus()
    expect(document.activeElement).toBe(open)
    await user.keyboard('{Enter}')
    expect(document.querySelector('.app-shell')).toHaveClass('nav-open')
    await user.click(screen.getByRole('button', { name: 'Overview' }))
    expect(document.querySelector('.app-shell')).not.toHaveClass('nav-open')
    await user.click(screen.getByRole('button', { name: 'Switch to light theme' }))
    expect(document.documentElement.dataset.theme).toBe('light')
    expect(screen.getByRole('button', { name: 'Switch to dark theme' })).toBeInTheDocument()
  })


  it('renders prepared queue values with verified event boundaries, gaps, and proxy semantics', async () => {
    installFetch({
      groundTruthIndex: {
        status: 'ready', message: 'Verified prepared batch.', blockers: [],
        analysis_sha256: 'a'.repeat(64), protocol_sha256: 'b'.repeat(64), threshold_ms: 8,
        max_gap_seconds: 1.5, alignment_status: 'approved', alignment_message: 'Hash-linked review verified.',
        report_id: 'ground_truth_fixture', runs: [{ run_id: 'run-fixture', configuration_id: 'cfg-1', row_count: 3, event_count: 1, event_boundaries_available: true }],
      },
      groundTruthSeries: {
        status: 'verified_prepared_batch', message: 'Validated values.', run_id: 'run-fixture',
        analysis_sha256: 'a'.repeat(64), report_id: 'ground_truth_fixture', event_boundaries_available: true,
        threshold_ms: 8, max_gap_seconds: 1.5, original_point_count: 3, displayed_point_count: 3,
        sampling_note: 'All validated samples are shown.',
        points: [
          { timestamp_seconds: 0, queue_backlog_bytes: 0, qdisc_queue_delay_proxy_ms: 0, event_active: false, future_event_onset: false, segment_start: true },
          { timestamp_seconds: 1, queue_backlog_bytes: 1000, qdisc_queue_delay_proxy_ms: 8, event_active: true, future_event_onset: true, segment_start: false },
          { timestamp_seconds: 4, queue_backlog_bytes: null, qdisc_queue_delay_proxy_ms: null, event_active: null, future_event_onset: null, segment_start: true },
        ],
        events: [{ observed_start_seconds: 1, observed_end_seconds: 1, onset_interval_lower_exclusive_seconds: 0, onset_interval_upper_inclusive_seconds: 1, onset_interval_identified: true, left_censored_or_gap_ambiguous: false, right_censored_or_gap_ambiguous: true, sample_count: 1 }],
        limitations: ['A gap and right-censored interval are retained.'],
      },
    })
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('Predictions are not available yet.')
    await user.click(screen.getByRole('button', { name: 'Ground truth' }))
    await screen.findByText('Ground truth & queue proxy')
    expect(await screen.findByText('Verified observed intervals; onset bounds and censoring are retained')).toBeInTheDocument()
    expect(screen.getByText(/Backlog-derived serialization-delay proxy/)).toBeInTheDocument()
    expect(screen.getByText(/qdisc_queue_delay_proxy_ms = backlog_bytes/)).toBeInTheDocument()
    expect(screen.getByText(/Right censored \/ ambiguous/)).toBeInTheDocument()
    expect(screen.getByText(/gap and right-censored interval are retained/)).toBeInTheDocument()
  })

  it('shows report fingerprints, provenance blockers, and only allowlisted download links', async () => {
    const reportId = 'ground_truth_fixture'
    installFetch({
      research: {
        protocol: { status: 'draft', message: 'Draft protocol.', config_sha256: null },
        alignment: { status: 'not_required', message: 'No separate alignment.' }, evaluations: [],
        reports: [{ report_id: reportId, kind: 'ground_truth', status: 'integrity verified', created_at_utc: '2026-10-26T00:00:00Z' }],
        operations: { audit_available: true, evaluation_available: false, message: 'Blocked.', prepared_status: 'missing', source_lineage_status: 'unverified', blockers: [] },
      },
      reportDetail: {
        report_id: reportId, kind: 'ground_truth', status: 'complete', integrity_status: 'verified',
        provenance_status: 'integrity_only', created_at_utc: '2026-10-26T00:00:00Z',
        protocol_sha256: 'a'.repeat(64), analysis_sha256: 'b'.repeat(64),
        processing_manifest_sha256: 'c'.repeat(64), source_manifest_sha256: 'd'.repeat(64), audit_sha256: 'e'.repeat(64),
        blockers: ['Current source lineage has not been established.'],
        artifacts: ['ground_truth_report.json', 'events.csv', 'ground_truth_report.html'],
        summary: { evidence_status: 'descriptive software fixture', event_count_observed_in_trace: 1 },
      },
    })
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('Predictions are not available yet.')
    await user.click(screen.getByRole('button', { name: 'Reports & provenance' }))
    expect(await screen.findByText(/Current source lineage has not been established/)).toBeInTheDocument()
    expect(screen.getByText('e'.repeat(64))).toBeInTheDocument()
    const download = screen.getByRole('link', { name: /events.csv/ })
    expect(download).toHaveAttribute('href', `/api/v1/reports/${reportId}/files/events.csv`)
    expect(screen.getByText(/checksum alone is not scientific validation/)).toBeInTheDocument()
  })

})
