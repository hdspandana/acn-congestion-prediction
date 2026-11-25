export type SourceStatus = {
  status: string
  message: string
  csv_count?: number
  checked_at_utc?: string
}

export type OperationalOverview = {
  application: { name: string; version: string; mode: string }
  telemetry_source: SourceStatus
  model_status: { status: string; eligible_count: number; active_model_id: string | null; message: string }
  latest_warning: PredictionRecord | null
  history_count: number
  data_freshness: { status: string; message: string }
  blockers: string[]
}

export type ModelRecord = {
  model_id: string
  method: string | null
  created_at_utc: string | null
  eligible: boolean
  status: string
  reason: string | null
  forecast_horizon_seconds: number | null
  score_threshold: number | null
  score_semantics: string
  calibration_status: string
  feature_columns: string[]
  feature_spec_version?: string | null
  feature_specification_sha256?: string | null
  feature_specification_status?: string | null
  tcp_retransmission_unit?: string | null
  feature_temporal_policy?: Record<string, unknown> | null
  feature_temporal_policy_status?: string | null
  model_parameter_overrides?: Record<string, unknown> | null
  evaluation_id: string | null
  protocol_sha256: string | null
  analysis_sha256: string | null
}

export type PredictionRecord = {
  prediction_id: string
  predicted_at_utc: string
  model_id: string
  model_method: string
  input_name: string
  input_sha256: string
  row_count: number
  forecast_horizon_seconds: number
  score: number | null
  score_threshold: number | null
  risk: 'warning' | 'below_threshold' | 'not_scored'
  outcome_state: 'unresolved' | 'verified_event' | 'false_alarm'
  timestamp_seconds: number
  feature_spec_version?: string | null
  feature_specification_sha256?: string | null
  feature_specification_status?: string | null
  tcp_retransmission_unit?: string | null
  input_declared_tcp_retransmission_unit?: string | null
  feature_temporal_policy_status?: string | null
  feature_temporal_eligibility_status?: string | null
  feature_temporal_eligibility_reason?: string | null
}

export type TelemetrySeries = {
  status: string
  source_kind: 'uploaded_batch'
  received_at_utc: string
  input_name: string
  input_sha256: string
  row_count: number
  fields: string[]
  points: Array<Record<string, number | null>>
  quality: {
    status: string
    missing_fraction: Record<string, number>
    gap_count: number | null
    max_gap_seconds: number | null
    timestamp_span_seconds: number | null
    wall_clock_freshness: string
    notes: string[]
  }
  warnings: PredictionRecord[]
}

export type DatasetSummary = {
  raw: { status: string; csv_count: number; message: string }
  processed: { status: string; row_count: number | null; run_count: number | null; protocol_status: string | null; message: string }
  audit: { status: string; message: string }
}

export type ResearchSummary = {
  protocol: { status: string; message: string; config_sha256: string | null }
  alignment: { status: string; message: string }
  evaluations: Array<{ evaluation_id: string; status: string; provenance_status: string; grouping: string | null; created_at_utc: string | null; methods: string[]; blockers: string[] }>
  reports: Array<{ report_id: string; kind: string; status: string; created_at_utc: string | null }>
  operations: { audit_available: boolean; evaluation_available: boolean; message: string; prepared_status?: string; source_lineage_status?: string; feature_specification_status?: string; tcp_retransmission_unit?: string | null; feature_temporal_policy_status?: string; blockers?: string[] }
}

export type EvaluationMethodMetric = {
  method: string
  event_count: number | null
  event_precision: number | null
  event_recall: number | null
  false_alarms_per_minute: number | null
  lead_time_median_seconds: number | null
  lead_time_event_count: number | null
  average_precision: number | null
  roc_auc: number | null
  scored_anchor_count: number | null
  threshold: number | null
  validation_false_alarms_per_minute: number | null
  uncertainty: Record<string, {
    cluster_count?: number
    estimable?: boolean
    interval?: [number, number] | null
    reason?: string | null
    warning?: string | null
    confidence?: number
    unit_of_resampling?: string
    bootstrap_resamples_used?: number
  }>
  adequacy_warnings: string[]
}

export type EvaluationDetail = {
  report_id: string
  kind: 'evaluation'
  status: string
  integrity_status: string
  provenance_status: string
  created_at_utc: string | null
  protocol_sha256: string | null
  analysis_sha256: string | null
  processing_manifest_sha256: string | null
  source_manifest_sha256: string | null
  audit_sha256: string | null
  grouping: string | null
  blockers: string[]
  artifacts: string[]
  summary: {
    methods: EvaluationMethodMetric[]
    method_failures: Record<string, string>
    requested_methods: string[]
    split_seed: number | null
    held_out_group_count: number
    held_out_groups: string[]
    held_out_groups_truncated: boolean
    split_row_counts: Record<string, number>
    split_event_counts: Record<string, number>
    split_group_counts: Record<string, number>
    threshold_calibration_split: string | null
    threshold_calibration_rule: string | null
    verified_artifacts: string[]
  }
}

export type ReportDetail = {
  report_id: string
  kind: string
  status: string
  integrity_status: string
  provenance_status: string
  created_at_utc: string | null
  protocol_sha256: string | null
  analysis_sha256: string | null
  processing_manifest_sha256: string | null
  source_manifest_sha256: string | null
  audit_sha256: string | null
  method?: string | null
  grouping?: string | null
  blockers: string[]
  artifacts: string[]
  summary: Record<string, unknown>
}

export type GroundTruthRun = {
  run_id: string
  configuration_id: string | null
  row_count: number
  event_count: number | null
  event_boundaries_available: boolean
}

export type GroundTruthIndex = {
  status: string
  message: string
  blockers: string[]
  analysis_sha256: string | null
  protocol_sha256: string | null
  threshold_ms: number | null
  max_gap_seconds: number | null
  alignment_status: string
  alignment_message: string
  report_id: string | null
  runs: GroundTruthRun[]
}

export type GroundTruthEvent = {
  observed_start_seconds: number | null
  observed_end_seconds: number | null
  onset_interval_lower_exclusive_seconds: number | null
  onset_interval_upper_inclusive_seconds: number | null
  onset_interval_identified: boolean
  left_censored_or_gap_ambiguous: boolean
  right_censored_or_gap_ambiguous: boolean
  sample_count: number | null
}

export type GroundTruthSeries = {
  status: string
  message: string
  run_id: string
  analysis_sha256: string | null
  report_id: string | null
  event_boundaries_available: boolean
  threshold_ms: number | null
  max_gap_seconds: number | null
  original_point_count: number
  displayed_point_count: number
  sampling_note: string
  points: Array<{
    timestamp_seconds: number
    queue_backlog_bytes: number | null
    qdisc_queue_delay_proxy_ms: number | null
    event_active: boolean | null
    future_event_onset: boolean | null
    segment_start: boolean
  }>
  events: GroundTruthEvent[]
  limitations: string[]
}

export class ApiError extends Error {
  code: string
  status: number
  constructor(message: string, code: string, status: number) {
    super(message)
    this.code = code
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { ...init, headers: { ...(init?.headers ?? {}) } })
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    const error = payload?.error
    throw new ApiError(error?.message ?? `Request failed (${response.status})`, error?.code ?? 'request_failed', response.status)
  }
  return payload as T
}

export const reportFileHref = (reportId: string, filename: string) =>
  `/api/v1/reports/${encodeURIComponent(reportId)}/files/${encodeURIComponent(filename)}`

export const api = {
  overview: () => request<OperationalOverview>('/api/v1/overview'),
  models: () => request<{ models: ModelRecord[]; active_model_id: string | null }>('/api/v1/models'),
  history: () => request<{ predictions: PredictionRecord[] }>('/api/v1/history?limit=200'),
  datasets: () => request<DatasetSummary>('/api/v1/datasets'),
  research: () => request<ResearchSummary>('/api/v1/research'),
  evaluationDetail: (id: string) => request<EvaluationDetail>(`/api/v1/evaluations/${encodeURIComponent(id)}`),
  reportDetail: (id: string) => request<ReportDetail>(`/api/v1/reports/${encodeURIComponent(id)}`),
  groundTruthIndex: () => request<GroundTruthIndex>('/api/v1/research/ground-truth'),
  groundTruthSeries: (runId: string) => request<GroundTruthSeries>(`/api/v1/research/ground-truth/series?run_id=${encodeURIComponent(runId)}`),
  buildGroundTruth: () => request<{ status: string; report_id: string; message: string }>('/api/v1/research/ground-truth/run', { method: 'POST' }),
  inspectTelemetry: (file: File) => {
    const body = new FormData()
    body.append('file', file)
    return request<TelemetrySeries>('/api/v1/telemetry/inspect', { method: 'POST', body })
  },
  predict: (modelId: string, file: File, tcpRetransmissionUnit?: string) => {
    const body = new FormData()
    body.append('model_id', modelId)
    body.append('file', file)
    if (tcpRetransmissionUnit) body.append('tcp_retransmission_unit', tcpRetransmissionUnit)
    return request<{ predictions: PredictionRecord[]; input: TelemetrySeries; model: ModelRecord }>('/api/v1/predictions', { method: 'POST', body })
  },
  audit: () => request<{ status: string; output_id: string; message: string }>('/api/v1/audit', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ confirmed: true }),
  }),
  runEvaluation: () => request<{ status: string; evaluation_id: string; message: string }>('/api/v1/evaluations/run', { method: 'POST' }),
  train: (evaluationId: string, method: string) => request<{ status: string; model: ModelRecord; message: string }>('/api/v1/models/train', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ evaluation_id: evaluationId, method }),
  }),
  selectModel: (modelId: string) => request<{ active_model_id: string; message: string }>('/api/v1/models/active', {
    method: 'PUT', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId }),
  }),
}
