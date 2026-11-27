/// <reference types="vite/client" />
import pilotEvidence from './pilot-warning-evidence.json'
import freshEvidence from './fresh-test-evidence.json'
import freshRtt from './fresh-test-rtt.csv?raw'
import heldOutRtt from './held-out-rtt.csv?raw'
import React, { useEffect, useMemo, useState } from 'react'
import {
  Activity, ArrowRight, ArrowUpRight, Bell,
  Check, ChevronDown, CircleAlert, Clock3, Database, Download, FileSearch, Gauge,
  History, Layers3, LayoutDashboard, LoaderCircle, LockKeyhole, Menu,
  Moon, Network, Play, RefreshCw, Search, ShieldCheck, Sparkles, Sun,
  Upload, Waves, X,
} from 'lucide-react'
import {
  Bar, BarChart, CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { api, reportFileHref, type DatasetSummary, type EvaluationDetail, type GroundTruthIndex,
  type GroundTruthSeries, type ModelRecord, type OperationalOverview, type PredictionRecord,
  type ReportDetail, type ResearchSummary, type TelemetrySeries } from './api'

type PageId = 'overview' | 'monitor' | 'predict' | 'history' | 'workspace'
type WorkspaceTab = 'datasets' | 'models' | 'evaluation' | 'protocol' | 'ground-truth' | 'robustness' | 'reports'
type Theme = 'dark' | 'light'

const endpointColors: Record<string, string> = {
  rtt_ms: '#43E3F5', goodput_bps: '#2DD4BF', loss_fraction: '#A899FF',
  tcp_retransmission_delta: '#FFB454',
}
const endpointLabels: Record<string, string> = {
  rtt_ms: 'RTT', goodput_bps: 'Throughput', loss_fraction: 'Packet loss',
  tcp_retransmission_delta: 'Retransmission changes',
}

function formatTime(value?: string | null) {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.valueOf()) ? value : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date)
}
function formatScore(value: number | null | undefined) {
  return value == null || !Number.isFinite(value) ? 'Not scored' : value.toFixed(3)
}
function unitLabel(field: string) {
  return field === 'rtt_ms' ? 'ms' : field === 'goodput_bps' ? 'bit/s' : field === 'loss_fraction' ? 'fraction' : field === 'tcp_retransmission_delta' ? 'source unit not declared in CSV' : 's'
}
function ErrorNotice({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return <div className="notice notice-danger" role="alert"><CircleAlert size={18}/><div><strong>We couldn't complete that request.</strong><p>{message}</p>{onRetry && <button className="text-button" onClick={onRetry}>Try again</button>}</div></div>
}
function LoadingLine({ label = 'Loading verified workspace data' }: { label?: string }) {
  return <div className="loading-line" role="status"><LoaderCircle className="spin" size={17}/>{label}…</div>
}
function SectionHeading({ eyebrow, title, description, action }: { eyebrow?: string; title: string; description?: string; action?: React.ReactNode }) {
  return <div className="section-heading"><div>{eyebrow && <div className="eyebrow">{eyebrow}</div>}<h1>{title}</h1>{description && <p>{description}</p>}</div>{action && <div className="heading-action">{action}</div>}</div>
}
function StatusPill({ label, tone = 'neutral' }: { label: string; tone?: 'success' | 'warning' | 'danger' | 'neutral' | 'cyan' }) {
  return <span className={`status-pill ${tone}`}><i aria-hidden="true"/>{label}</span>
}
function EmptyPanel({ icon: Icon, title, body, action, compact = false }: { icon: React.ElementType; title: string; body: string; action?: React.ReactNode; compact?: boolean }) {
  return <div className={`empty-panel ${compact ? 'compact' : ''}`}><div className="empty-icon"><Icon size={21}/></div><h3>{title}</h3><p>{body}</p>{action}</div>
}
function Dropzone({ onFile, accept = '.csv,text/csv', title = 'Drop a telemetry CSV here', hint = 'CSV · up to 10 MB · local batch inspection only' }: { onFile: (file: File) => void; accept?: string; title?: string; hint?: string }) {
  const [dragging, setDragging] = useState(false)
  const [name, setName] = useState('')
  const choose = (file?: File) => { if (file) { setName(file.name); onFile(file) } }
  return <label className={`dropzone ${dragging ? 'dragging' : ''}`} onDragOver={(e) => { e.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={(e) => { e.preventDefault(); setDragging(false); choose(e.dataTransfer.files[0]) }}>
    <input type="file" accept={accept} aria-label="Choose telemetry CSV file" onChange={(e) => choose(e.target.files?.[0])}/>
    <div className="upload-icon"><Upload size={19}/></div><span className="drop-title">{name ? name : title}</span><span className="drop-hint">{name ? 'Select another file to replace this input' : hint}</span><span className="browse-link">Browse files <ArrowRight size={14}/></span>
  </label>
}

function App() {
  const [page, setPage] = useState<PageId>('overview')
  const [workspaceTab, setWorkspaceTab] = useState<WorkspaceTab>('datasets')
  const [workspaceOpen, setWorkspaceOpen] = useState(true)
  const [theme, setTheme] = useState<Theme>(() => (localStorage.getItem('acn-theme') as Theme) || 'dark')
  const [overview, setOverview] = useState<OperationalOverview | null>(null)
  const [models, setModels] = useState<ModelRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [mobileNav, setMobileNav] = useState(false)

  const refresh = async () => {
    setLoading(true); setError('')
    try {
      const [summary, catalog] = await Promise.all([api.overview(), api.models()])
      setOverview(summary); setModels(catalog.models)
    } catch (e) { setError(e instanceof Error ? e.message : 'Network service is unavailable') }
    finally { setLoading(false) }
  }
  useEffect(() => { void refresh() }, [])
  useEffect(() => { document.documentElement.dataset.theme = theme; localStorage.setItem('acn-theme', theme) }, [theme])
  const openWorkspace = (tab: WorkspaceTab = 'datasets') => { setWorkspaceTab(tab); setPage('workspace'); setMobileNav(false) }
  const go = (next: PageId) => { setPage(next); setMobileNav(false) }

  return <div className={`app-shell ${mobileNav ? 'nav-open' : ''}`}>
    <aside className="sidebar" aria-label="Main navigation">
      <div className="brand"><div className="brand-mark"><Network size={19}/><span/></div><div><b>ACN</b><small>NETWORK INTELLIGENCE</small></div><button className="mobile-close icon-button" aria-label="Close navigation" onClick={() => setMobileNav(false)}><X size={18}/></button></div>
      <div className="workspace-select"><span className="workspace-dot"/><div><small>WORKSPACE</small><strong>Local operations node</strong></div><ChevronDown size={15}/></div>
      <div className="nav-group-label">OPERATIONS</div>
      <nav className="nav-list">
        <NavItem id="overview" active={page === 'overview'} icon={LayoutDashboard} label="Overview" onClick={() => go('overview')}/>
        <NavItem id="monitor" active={page === 'monitor'} icon={Activity} label="Batch monitor" onClick={() => go('monitor')} trailing={<span className="nav-tag">BATCH</span>}/>
        <NavItem id="predict" active={page === 'predict'} icon={Sparkles} label="Predict congestion" onClick={() => go('predict')} trailing={models.some(m => m.eligible) ? <span className="nav-ready"/> : undefined}/>
        <NavItem id="history" active={page === 'history'} icon={History} label="Warning history" onClick={() => go('history')}/>
      </nav>
      <button className="research-toggle" aria-expanded={workspaceOpen} onClick={() => setWorkspaceOpen(!workspaceOpen)}><span><Layers3 size={16}/> MODEL OPERATIONS</span><ChevronDown size={15} className={workspaceOpen ? '' : 'rotate'}/></button>
      {workspaceOpen && <nav className="nav-list research-nav">
        <WorkspaceNav tab="datasets" active={page === 'workspace' && workspaceTab === 'datasets'} icon={Database} label="Data & sources" onClick={openWorkspace}/>
        <WorkspaceNav tab="models" active={page === 'workspace' && workspaceTab === 'models'} icon={Gauge} label="Model registry" onClick={openWorkspace}/>
        <WorkspaceNav tab="evaluation" active={page === 'workspace' && workspaceTab === 'evaluation'} icon={ShieldCheck} label="Evaluation" onClick={openWorkspace}/>
        <WorkspaceNav tab="protocol" active={page === 'workspace' && workspaceTab === 'protocol'} icon={LockKeyhole} label="Protocol & alignment" onClick={openWorkspace}/>
        <WorkspaceNav tab="ground-truth" active={page === 'workspace' && workspaceTab === 'ground-truth'} icon={Waves} label="Ground truth" onClick={openWorkspace}/>
        <WorkspaceNav tab="robustness" active={page === 'workspace' && workspaceTab === 'robustness'} icon={Activity} label="Robustness" onClick={openWorkspace}/>
        <WorkspaceNav tab="reports" active={page === 'workspace' && workspaceTab === 'reports'} icon={FileSearch} label="Reports & provenance" onClick={openWorkspace}/>
      </nav>}
      <div className="sidebar-bottom">
        <div className="service-indicator"><span className={`service-light ${loading ? 'pending' : overview?.telemetry_source.status === 'available' ? 'online' : 'offline'}`}/><div><strong>{loading ? 'Connecting' : overview?.telemetry_source.status === 'available' ? 'Service ready' : 'No telemetry source'}</strong><small>{loading ? 'Checking local API' : 'No live collector configured'}</small></div></div>
        <button className="theme-button" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}>{theme === 'dark' ? <Sun size={16}/> : <Moon size={16}/>}<span>{theme === 'dark' ? 'Light appearance' : 'Dark appearance'}</span></button>
        <div className="sidebar-foot"><span>ACN · LOCAL INSTANCE</span><span>v{overview?.application.version ?? '—'}</span></div>
      </div>
    </aside>
    {mobileNav && <button className="nav-scrim" aria-label="Close navigation" onClick={() => setMobileNav(false)}/>}
    <main className="main-area">
      <header className="topbar"><button className="mobile-menu icon-button" onClick={() => setMobileNav(true)} aria-label="Open navigation"><Menu size={20}/></button><div className="crumb"><span>ACN</span><span className="crumb-sep">/</span><strong>{page === 'workspace' ? 'Model operations' : pageTitle(page)}</strong></div><div className="top-actions"><span className="local-tag"><span/>LOCAL INSTANCE</span><button className="icon-button" title="Refresh current status" aria-label="Refresh status" onClick={() => void refresh()}><RefreshCw size={16}/></button><div className="avatar">NW</div></div></header>
      {error && <div className="page-wrap"><ErrorNotice message={error} onRetry={() => void refresh()}/></div>}
      {page === 'overview' && <OverviewPage summary={overview} models={models} loading={loading} error={error} onRetry={refresh} onNavigate={go} onWorkspace={openWorkspace}/>}
      {page === 'monitor' && <MonitorPage/>}
      {page === 'predict' && <PredictPage models={models} onPrediction={refresh}/>}
      {page === 'history' && <HistoryPage/>}
      {page === 'workspace' && <WorkspacePage tab={workspaceTab} onTab={setWorkspaceTab} onModels={async () => { const result = await api.models(); setModels(result.models) }} onRefresh={refresh}/>}
    </main>
  </div>
}

function pageTitle(page: PageId) { return ({ overview: 'Overview', monitor: 'Batch monitor', predict: 'Predict congestion', history: 'Warning history', workspace: 'Model operations' })[page] }
function methodLabel(method: string | null | undefined) { return method ? method.replaceAll('_', ' ') : 'Unknown method' }
function horizonLabel(value: number | null | undefined) { return value == null ? 'Not recorded' : `${value}s` }
function NavItem({ active, icon: Icon, label, onClick, trailing }: { id: string; active: boolean; icon: React.ElementType; label: string; onClick: () => void; trailing?: React.ReactNode }) {
  return <button className={`nav-item ${active ? 'active' : ''}`} aria-current={active ? 'page' : undefined} onClick={onClick}><Icon size={17}/><span>{label}</span>{trailing}</button>
}
function WorkspaceNav({ tab, active, icon: Icon, label, onClick }: { tab: WorkspaceTab; active: boolean; icon: React.ElementType; label: string; onClick: (tab: WorkspaceTab) => void }) {
  return <button className={`nav-item sub-item ${active ? 'active' : ''}`} aria-current={active ? 'page' : undefined} onClick={() => onClick(tab)}><Icon size={16}/><span>{label}</span></button>
}

function OverviewPage({ summary, models, loading, error, onRetry, onNavigate, onWorkspace }: { summary: OperationalOverview | null; models: ModelRecord[]; loading: boolean; error: string; onRetry: () => void; onNavigate: (page: PageId) => void; onWorkspace: (tab?: WorkspaceTab) => void }) {
  if (loading && !summary) return <div className="page-wrap"><LoadingLine/></div>
  if (error && !summary) return <div className="page-wrap"><ErrorNotice message={error} onRetry={onRetry}/></div>
  const eligible = models.filter(model => model.eligible)
  const sourceReady = summary?.telemetry_source.status === 'available'
  return <div className="page-wrap">
    <SectionHeading eyebrow="NETWORK INTELLIGENCE · OVERVIEW" title="Know before the queue builds." description="Endpoint telemetry and validated forecasts, in one operational view." action={<button className="button button-quiet" onClick={onRetry}><RefreshCw size={15}/>Refresh</button>}/>
    <div className="overview-banner">
      <div className="banner-noise" aria-hidden="true"/><div className="banner-copy"><div className="banner-kicker"><span className="signal-bars"><i/><i/><i/></span> PREDICTIVE NETWORK MONITORING</div><h2>{eligible.length ? 'Forecasts are ready when your telemetry is.' : 'Predictions are not available yet.'}</h2><p>{eligible.length ? `${eligible.length} model${eligible.length === 1 ? '' : 's'} passed current provenance and evaluation checks. Load a batch to create a real forecast.` : 'Connect reviewed telemetry and validate a model before any congestion warning can be issued.'}</p><div className="banner-actions"><button className="button button-primary" onClick={() => onNavigate(eligible.length ? 'predict' : 'monitor')}><Sparkles size={16}/>{eligible.length ? 'Make a prediction' : 'Inspect telemetry'}<ArrowRight size={16}/></button><button className="button button-ghost" onClick={() => onWorkspace('models')}>Open Model Workspace</button></div></div>
      <div className="signal-visual" aria-hidden="true"><div className="signal-orbit orbit-a"/><div className="signal-orbit orbit-b"/><div className="signal-core"><Activity size={27}/></div><span className="signal-node node-a"/><span className="signal-node node-b"/><span className="signal-node node-c"/><div className="signal-label">QUEUE EVENT<br/><b>FORECAST</b></div></div>
    </div>
    <div className="section-line"><div><span className="eyebrow">SYSTEM STATE</span><h2>Operational readiness</h2></div><span className="updated-note">Checked {formatTime(summary?.telemetry_source.checked_at_utc)}</span></div>
    <div className="state-grid">
      <StateCard icon={Network} label="Telemetry source" status={sourceReady ? 'Available' : 'Not connected'} tone={sourceReady ? 'success' : 'warning'} detail={summary?.telemetry_source.message ?? 'No verified source is configured.'} onClick={() => onNavigate('monitor')}/>
      <StateCard icon={ShieldCheck} label="Prediction model" status={eligible.length ? `${eligible.length} eligible` : 'Unavailable'} tone={eligible.length ? 'success' : 'neutral'} detail={summary?.model_status.message ?? 'No model has passed the eligibility checks.'} onClick={() => onWorkspace('models')}/>
      <StateCard icon={Clock3} label="Input freshness" status={summary?.data_freshness.status === 'available' ? 'Observed' : 'Unknown'} tone={summary?.data_freshness.status === 'available' ? 'cyan' : 'neutral'} detail={summary?.data_freshness.message ?? 'No wall-clock source timestamps are available.'} onClick={() => onNavigate('monitor')}/>
      <StateCard icon={Bell} label="Latest warning" status={summary?.latest_warning ? 'Forecast recorded' : 'No forecasts'} tone={summary?.latest_warning ? (summary.latest_warning.risk === 'warning' ? 'warning' : 'neutral') : 'neutral'} detail={summary?.latest_warning ? `${formatTime(summary.latest_warning.predicted_at_utc)} · ${summary.latest_warning.outcome_state.replace('_', ' ')}` : 'Warnings appear here only after genuine model inference.'} onClick={() => onNavigate('history')}/>
    </div>
    {summary?.blockers?.length ? <div className="blocker-strip"><ShieldCheck size={17}/><div><strong>Before forecasts can be trusted</strong><span>{summary.blockers.slice(0, 2).join(' ')}</span></div><button className="text-button" onClick={() => onWorkspace('protocol')}>Review requirements <ArrowRight size={14}/></button></div> : null}
    <div className="overview-lower">
      <section className="panel forecast-preview"><div className="panel-head"><div><span className="eyebrow">FORECAST ACTIVITY</span><h3>Recent warnings</h3></div><button className="text-button" onClick={() => onNavigate('history')}>View history <ArrowRight size={14}/></button></div>
        {summary?.latest_warning ? <div className="latest-forecast"><div className={`forecast-symbol ${summary.latest_warning.risk}`}><Bell size={19}/></div><div className="forecast-main"><strong>{summary.latest_warning.risk === 'warning' ? 'Threshold warning' : 'Below warning threshold'}</strong><span>{summary.latest_warning.input_name} · {summary.latest_warning.model_method}</span></div><div className="forecast-time">{formatTime(summary.latest_warning.predicted_at_utc)}<small>{summary.latest_warning.forecast_horizon_seconds}s horizon</small></div></div> : <EmptyPanel compact icon={Bell} title="No forecast history" body="When an eligible model predicts from uploaded telemetry, the real result and its input reference will appear here." action={<button className="button button-secondary button-small" onClick={() => onNavigate('predict')}>Go to prediction <ArrowRight size={14}/></button>}/>}
      </section>
      <section className="panel source-preview"><div className="panel-head"><div><span className="eyebrow">INPUT CHANNEL</span><h3>Telemetry source</h3></div><StatusPill label={sourceReady ? 'Configured' : 'Batch only'} tone={sourceReady ? 'success' : 'warning'}/></div><div className="source-callout"><div className="source-icon"><Upload size={18}/></div><div><strong>Batch CSV inspection</strong><span>No streaming collector is configured. Uploaded files are analyzed as batches, never labeled “live”.</span></div></div><button className="button button-secondary full-width" onClick={() => onNavigate('monitor')}>Open monitor <ArrowRight size={15}/></button></section>
    </div>
  </div>
}
function StateCard({ icon: Icon, label, status, tone, detail, onClick }: { icon: React.ElementType; label: string; status: string; tone: 'success' | 'warning' | 'neutral' | 'cyan'; detail: string; onClick: () => void }) {
  return <button className="state-card" onClick={onClick}><div className="state-card-top"><span className="state-card-icon"><Icon size={17}/></span><span className="state-label">{label}</span><ArrowUpRight className="state-link" size={15}/></div><strong className={`state-status ${tone}`}>{status}</strong><span className="state-detail">{detail}</span></button>
}

function MonitorPage() {
  const [telemetry, setTelemetry] = useState<TelemetrySeries | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [visible, setVisible] = useState<string[]>([])
  const inspect = async (file: File) => {
    setBusy(true); setError(''); setTelemetry(null)
    try { const result = await api.inspectTelemetry(file); setTelemetry(result); setVisible(result.fields.filter(f => f !== 'timestamp_seconds')) }
    catch (e) { setError(e instanceof Error ? e.message : 'Telemetry inspection failed') }
    finally { setBusy(false) }
  }
  return <div className="page-wrap">
    <SectionHeading
      eyebrow="OPERATIONS · TELEMETRY"
      title="Batch monitor"
      description="Inspect endpoint signals, sample coverage, and real warnings. Batch uploads are not a live feed."
      action={<StatusPill label="No live collector" tone="warning" />}
    />
    <div className="monitor-layout">
      <section className="panel monitor-chart-panel"><div className="panel-head"><div><span className="eyebrow">MEASURED INPUT</span><h3>Endpoint telemetry</h3></div>{telemetry && <StatusPill label="Uploaded batch" tone="cyan"/>}</div>
        {!telemetry && !busy && <Dropzone onFile={inspect} title="Inspect an endpoint telemetry CSV" hint="Normalized CSV · timestamp_seconds plus canonical endpoint columns · 10 MB max"/>}
        {busy && <div className="chart-loading"><LoaderCircle size={22} className="spin"/><strong>Inspecting uploaded batch</strong><span>Validating columns, units, timestamps, and missing intervals.</span></div>}
        {error && <ErrorNotice message={error} onRetry={() => {setError('')}}/>}
        {telemetry && <><div className="telemetry-meta"><div><span>FILE</span><strong>{telemetry.input_name}</strong></div><div><span>ROWS</span><strong>{telemetry.row_count.toLocaleString()}</strong></div><div><span>RECEIVED</span><strong>{formatTime(telemetry.received_at_utc)}</strong></div><button className="icon-button" aria-label="Remove uploaded batch" onClick={() => setTelemetry(null)}><X size={17}/></button></div>
          {visible.length > 0 ? <div className="metric-charts">{visible.map(field => <section className="metric-chart-card" key={field}><div className="metric-chart-title"><strong>{endpointLabels[field] ?? field}</strong><span>{unitLabel(field)}</span></div><div className="chart-wrap"><TelemetryChart data={telemetry} field={field}/></div></section>)}</div> : <EmptyPanel compact icon={Activity} title="No signals selected" body="Choose a measurement from the legend to show its timeline."/>}
          <div className="legend-row" aria-label="Select measured signals">{telemetry.fields.filter(f => f !== 'timestamp_seconds').map(field => <button key={field} className={`legend-item ${visible.includes(field) ? '' : 'dimmed'}`} aria-pressed={visible.includes(field)} onClick={() => setVisible(v => v.includes(field) ? v.filter(x => x !== field) : [...v, field])}><i style={{background: endpointColors[field]}}/>{endpointLabels[field] ?? field}<small>{unitLabel(field)}</small></button>)}</div>
          {telemetry.warnings.length > 0 && <div className="chart-warning-note"><Bell size={15}/>{telemetry.warnings.length} actual forecast warning{telemetry.warnings.length === 1 ? '' : 's'} match this uploaded file fingerprint.</div>}
        </>}
      </section>
      <aside className="monitor-side">
        <div className="panel quality-panel"><div className="panel-head"><div><span className="eyebrow">INPUT VALIDATION</span><h3>Data quality</h3></div><StatusPill label={telemetry?.quality.status ?? 'Not checked'} tone={telemetry ? (telemetry.quality.status === 'complete' ? 'success' : 'warning') : 'neutral'}/></div>
          {telemetry ? <><div className="quality-list">{Object.entries(telemetry.quality.missing_fraction).map(([field, fraction]) => <div className="quality-row" key={field}><span>{endpointLabels[field] ?? field}<small>{unitLabel(field)}</small></span><strong>{(fraction * 100).toFixed(1)}% missing</strong><div className="quality-track"><i style={{width: `${Math.min(100, fraction * 100)}%`}}/></div></div>)}</div><div className="quality-footer"><div><span>Sampling gaps</span><strong>{telemetry.quality.gap_count}</strong></div><div><span>Largest gap</span><strong>{telemetry.quality.max_gap_seconds == null ? '—' : `${telemetry.quality.max_gap_seconds.toFixed(2)} s`}</strong></div></div><div className="freshness-note"><Clock3 size={15}/><span>{telemetry.quality.wall_clock_freshness}</span></div>{telemetry.quality.notes.map((n, i) => <p className="quality-note" key={i}>{n}</p>)}</> : <EmptyPanel compact icon={FileSearch} title="No batch inspected" body="Upload a normalized telemetry CSV to review its exact fields, missingness, and timestamp gaps."/>}
        </div>
        <div className="panel signal-key"><span className="eyebrow">SIGNAL DEFINITIONS</span><div><strong>Measured predictors</strong><p>RTT, throughput, packet loss and retransmission changes are endpoint measurements, not direct queue measurements. Jitter is not in the verified canonical contract and is not inferred.</p></div><div><strong>Congestion ground truth</strong><p>Router qdisc backlog is used for outcome labels only. It is not accepted as an inference feature.</p></div></div>
      </aside>
    </div>
  </div>
}
function formatTelemetryAxisTick(field: string, value: number) {
  if (field === 'goodput_bps') {
    if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toLocaleString(undefined, { maximumFractionDigits: 1 })}M`
    if (Math.abs(value) >= 1_000) return `${(value / 1_000).toLocaleString(undefined, { maximumFractionDigits: 1 })}k`
  }
  return value.toLocaleString(undefined, { maximumFractionDigits: field === 'loss_fraction' ? 3 : 1 })
}

function secondsTickPrecision(timestamps: number[]) {
  const intervals = timestamps.slice(1).map((value, index) => value - timestamps[index]).filter(value => value > 0)
  const smallest = intervals.length ? Math.min(...intervals) : 1
  return smallest < 1 ? Math.min(3, Math.max(1, Math.ceil(-Math.log10(smallest)))) : 0
}

function TelemetryChart({ data, field }: { data: TelemetrySeries; field: string }) {
  const points = data.points
  if (!points.length) return <EmptyPanel compact icon={Activity} title="No plottable rows" body="The file had no valid timestamped telemetry rows."/>
  const timestamps = points.flatMap(point => typeof point.timestamp_seconds === 'number' ? [point.timestamp_seconds] : [])
  const timePrecision = secondsTickPrecision(timestamps)
  return <ResponsiveContainer width="100%" height="100%"><ComposedChart data={points} margin={{ top: 10, right: 12, left: -14, bottom: 4 }}>
    <CartesianGrid stroke="var(--grid)" strokeDasharray="3 5" vertical={false}/><XAxis dataKey="timestamp_seconds" tickLine={false} axisLine={false} tick={{fill:'var(--muted)',fontSize:11}} tickFormatter={(v) => `${Number(v).toFixed(timePrecision)}s`} minTickGap={35}/><YAxis tickLine={false} axisLine={false} tick={{fill:'var(--muted)',fontSize:11}} width={62} tickFormatter={v => formatTelemetryAxisTick(field, Number(v))}/><Tooltip contentStyle={{background:'var(--tooltip)',border:'1px solid var(--border)',borderRadius:8,color:'var(--text)'}} labelFormatter={(v) => `Run time ${Number(v).toFixed(3)} s`} formatter={(value, name) => [value == null ? 'Gap / missing' : String(value), endpointLabels[String(name)] ?? String(name)]}/>
    <Line type="linear" dataKey={field} name={field} stroke={endpointColors[field]} strokeWidth={2} dot={false} connectNulls={false} isAnimationActive={false} activeDot={{r:3,strokeWidth:0}}/>
    {data.warnings.map((warning, index) => <ReferenceLine key={`${warning.prediction_id}-${index}`} x={warning.timestamp_seconds} stroke="#FF5D73" strokeDasharray="4 4" label={{value:'forecast warning',fill:'#FF8D9C',fontSize:9,position:'insideTopRight'}}/>)}
  </ComposedChart></ResponsiveContainer>
}

function PilotPrediction() {
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<{training_rows: number; coverage?: {input_rows: number; missing_rtt_rows: number; sampling_gaps: number}; cutoff?: number | null; predictions: {timestamp_seconds: number; rtt_ms?: number; score: number}[]} | null>(null)
  const [model, setModel] = useState('expanded')
  const shortWindow = model !== 'original'
  const [bundled, setBundled] = useState<'bundled' | 'fresh' | null>(null)
  const [inputName, setInputName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  async function predict(input: File, replay: typeof bundled = null) {
    setBundled(replay)
    setInputName(input.name)
    setBusy(true)
    setError('')
    setResult(null)
    const body = new FormData()
    body.append('file', input)
    try {
      const response = await fetch('/api/v1/predictions/pilot' + (model === 'original' ? '' :
        model === 'short' ? '?short_window=true' : `?candidate=${model}`), {method: 'POST', body})
      const data = await response.json()
      if (!response.ok) throw new Error(data.error?.message ?? 'Pilot prediction failed.')
      setResult(data)
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Pilot prediction failed.')
    } finally {
      setBusy(false)
    }
  }

  const firstTime = result?.predictions[0]?.timestamp_seconds ?? 0
  const cutoff = shortWindow ? result?.cutoff ?? null : pilotEvidence.threshold
  const observed = (bundled === 'fresh' ? freshEvidence.observed : pilotEvidence.observed) as Record<string, number | undefined>
  const rows = (result?.predictions ?? []).map(row => ({
    ...row,
    elapsed: row.timestamp_seconds - firstTime,
    warning: cutoff !== null && row.score >= cutoff,
    warningScore: cutoff !== null && row.score >= cutoff ? row.score : null,
    actual: bundled ? observed[String(Math.round(row.timestamp_seconds * 1000))] : undefined,
  }))
  const warnings = rows.filter(row => row.warning)
  const confirmed = warnings.find(row => row.actual === 1)
  const hasOnset = rows.some(row => row.actual === 1)
  const falseWarnings = warnings.filter(row => row.actual === 0).length
  const unknownWarnings = warnings.filter(row => row.actual === undefined).length
  const warningCount = warnings.length
  const rttRange = rows.reduce<{min: number; max: number} | null>((range, row) => {
    if (row.rtt_ms == null) return range
    return range ? {min: Math.min(range.min, row.rtt_ms), max: Math.max(range.max, row.rtt_ms)} :
      {min: row.rtt_ms, max: row.rtt_ms}
  }, null)
  const freshOnset = freshEvidence.onset.map(time => time - firstTime)
  const freshQueue = freshEvidence.queue.map(([time, delay]) => ({elapsed: time - firstTime, delay}))
  const timelineDomain = [Math.min(0, freshQueue[0].elapsed), Math.max(rows.at(-1)?.elapsed ?? 0, freshQueue.at(-1)!.elapsed)]
  const test = pilotEvidence.held_out
  const coverage = result?.coverage

  function downloadForecasts() {
    const header = 'timestamp_seconds,elapsed_seconds,rtt_ms,model,score,warning_cutoff,forecast_next_second,observed_outcome'
    const csv = [header, ...rows.map(row => [
      row.timestamp_seconds, row.elapsed, row.rtt_ms, model, row.score, cutoff,
      cutoff === null ? 'score_only' : row.warning ? 'possible_queue_buildup' : 'no_warning',
      row.actual === 1 ? 'congestion_began' : row.actual === 0 ? 'no_new_congestion' : 'unknown',
    ].join(','))].join('\r\n')
    const url = URL.createObjectURL(new Blob([csv], {type: 'text/csv;charset=utf-8'}))
    const link = document.createElement('a')
    link.href = url
    link.download = 'acn-forecast-results.csv'
    link.click()
    window.setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  return <section className="panel predict-flow pilot-prediction" aria-label="Kali pilot prediction">
    <div className="step-content">
      <h3>Kali pilot model</h3>
      <p>Forecast sustained queue buildup one second ahead using endpoint RTT.</p>
      <label>
        Model{' '}
        <select disabled={busy} value={model}
          onChange={event => {setModel(event.target.value); setResult(null); setError('')}}>
          <option value="original">Original logistic regression</option>
          <option value="short">Three-reading logistic regression (6 training runs)</option>
          <option value="expanded">Three-reading logistic regression (11 training runs)</option>
          <option value="relative">Baseline-relative logistic regression (11 training runs)</option>
        </select>
      </label>
      <details>
        <summary>Fresh 9 Mbit/s evaluation</summary>
        <p>Cutoffs chosen at 3 Mbit/s. The five test runs were held out from training and cutoff selection.</p>
        <div className="prediction-table-wrap"><table className="prediction-table">
          <thead><tr><th>Model</th><th>Onsets caught</th><th>False warnings</th></tr></thead>
          <tbody>
            <tr><td>Three readings, 6 training runs</td><td>2 / 5</td><td>20 / 184</td></tr>
            <tr><td>Three readings, 11 training runs</td><td>3 / 5</td><td>31 / 184</td></tr>
            <tr><td>Baseline-relative, 11 training runs</td><td>0 / 5</td><td>10 / 184</td></tr>
          </tbody>
        </table></div>
      </details>
      <div className="pilot-upload">
        <label className="pilot-file">
          <span>Endpoint RTT CSV</span>
          <input type="file" accept=".csv" aria-label="Pilot RTT CSV" disabled={busy}
            onChange={event => {setFile(event.target.files?.[0] ?? null); setResult(null); setError('')}} />
        </label>
        <button className="button button-primary" disabled={!file || busy} onClick={() => { if (file) void predict(file) }}>
          {busy ? 'Scoring…' : 'Run pilot prediction'}
        </button>
      </div>
      <details>
        <summary>Optional lab examples with router measurements</summary>
        <p>Use these recordings to see how warnings are checked against measured queue buildup.</p>
        <button className="button" type="button" disabled={busy}
          onClick={() => void predict(new File([heldOutRtt], 'endpoint_ping.csv', {type: 'text/csv'}), 'bundled')}>
          Run bundled demo
        </button>
        <button className="button" type="button" disabled={busy}
          onClick={() => void predict(new File([freshRtt], '9mbit-test-rtt.csv', {type: 'text/csv'}), 'fresh')}>
          Run 9 Mbit/s test example
        </button>
      </details>
      <p className="pilot-hint">Use <code>timestamp_seconds</code> and <code>rtt_ms</code> with readings
        at most 0.5 seconds apart. The three-reading models need three consecutive valid readings to start scoring.</p>
      {error && <div className="pilot-error" role="alert">{error}
      </div>}
      {result && <div className="pilot-results" aria-live="polite">
        <h4>{cutoff === null ? 'Scores only' : warningCount ? 'The model warned of possible queue buildup' : 'The model raised no warnings'}</h4>
        <p><strong>{bundled === 'fresh' ? '9 Mbit/s test example' : bundled ? '7 Mbit/s bundled replay' : inputName}</strong>
          {warningCount > 0 && ` · First warning at ${warnings[0].elapsed.toFixed(3)} s after the first forecast. Each warning looks one second ahead.`}
        </p>
        {bundled ? <>
          <p><strong>{hasOnset ? confirmed ? 'The model warned in time for the measured onset.' : cutoff === null ? 'The router recorded one onset. This model produced scores only.' : 'The router recorded one onset that the model missed.' : 'No onset could be matched to a scored forecast window.'}</strong>
            {cutoff !== null && ` False warnings: ${falseWarnings}. Warnings without enough evidence: ${unknownWarnings}.`}
          </p>
          {bundled === 'fresh' && confirmed && <p>The first matching warning came {(freshEvidence.onset[0] - confirmed.timestamp_seconds).toFixed(2)}–{(freshEvidence.onset[1] - confirmed.timestamp_seconds).toFixed(2)} s before the measured onset interval.</p>}
        </> : <p><strong>What happened on the router is unknown.</strong> This upload contains endpoint RTT only. A matching router queue trace is needed to confirm whether congestion began. A quiet forecast can still miss an onset.</p>}
        {rttRange && <p>Across the scored readings, ping round-trip time ranged from {rttRange.min.toFixed(2)} to {rttRange.max.toFixed(2)} ms.</p>}
        <p>{warningCount} of {result.predictions.length} forecast timestamps raised a warning. Nearby warnings may refer to the same possible event.</p>
        {coverage && <p><strong>File coverage:</strong> Scored {rows.length} of {coverage.input_rows} readings.
          {rows.length < coverage.input_rows && ' A reading needs enough uninterrupted RTT history before it can be scored.'}
          {coverage.missing_rtt_rows > 0 && ` Missing RTT values: ${coverage.missing_rtt_rows}.`}
          {coverage.sampling_gaps > 0 && ` Gaps over 0.5 seconds: ${coverage.sampling_gaps}.`}
        </p>}
        <button className="button" type="button" onClick={downloadForecasts}><Download size={15}/>Download forecast CSV</button>
        <details><summary>Model details</summary>
        <p>Model fitted on {result.training_rows} labeled timestamps.
          {cutoff === null ? 'No validation cutoff met the warning budget with an active warning.'
            : ` Provisional cutoff ${cutoff.toFixed(6)}, selected on 3 Mbit/s validation data.`}</p>
        {shortWindow ? <p>The cutoff limits warnings on negative validation timestamps to 10%.
          See the fresh evaluation above for measured test performance.</p> : <p><strong>High false-warning rate.</strong> On the previously examined {test.rate_mbit} Mbit/s test,
          the cutoff warned ahead of {test.onsets_caught}/{test.onsets_total} measured onsets and
          warned at {test.false_warning_timestamps}/{test.negative_timestamps} negative timestamps.</p>}
        </details>
        <section aria-label="Forecast timeline"><h4>Timeline for this file</h4>
          <p>Elapsed seconds from the first scored reading. {bundled === 'fresh' && 'Shading marks the measured onset interval. The vertical dashed line marks the first matching warning, when present.'}</p>
          {rttRange && <>
            <p><strong>Measured ping time (ms)</strong> · RTT readings at the scored timestamps.</p>
            <div className="chart-wrap"><ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={rows}>
                <XAxis type="number" dataKey="elapsed" domain={bundled === 'fresh' ? timelineDomain : ['dataMin', 'dataMax']} tickFormatter={value => `${Number(value).toFixed(1)} s`}/>
                <YAxis width={72}/>
                <Tooltip labelFormatter={value => `${Number(value).toFixed(3)} s`}/>
                <Line dataKey="rtt_ms" name="Measured RTT (ms)" stroke="#60A5FA" dot={rows.length === 1} isAnimationActive={false}/>
              </ComposedChart>
            </ResponsiveContainer></div>
          </>}
          <p><strong>Model score</strong> · Amber dots mark warnings. The dashed horizontal line is the warning cutoff.</p>
          <div className="chart-wrap"><ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={rows}>
              <XAxis type="number" dataKey="elapsed" domain={bundled === 'fresh' ? timelineDomain : ['dataMin', 'dataMax']} tickFormatter={value => `${Number(value).toFixed(1)} s`}/>
              <YAxis width={72} tickFormatter={value => Number(value).toFixed(3)}/>
              <Tooltip labelFormatter={value => `${Number(value).toFixed(3)} s`}/>
              {cutoff !== null && <ReferenceLine y={cutoff} stroke="#FFB454" strokeDasharray="5 4"/>}
              {bundled === 'fresh' && <ReferenceArea x1={freshOnset[0]} x2={freshOnset[1]} fill="#2DD4BF" fillOpacity={0.22}/>}
              <Line dataKey="score" name="Model score" stroke="#2DD4BF" dot={rows.length === 1} isAnimationActive={false}/>
              <Line dataKey="warningScore" name="Warning score" stroke="none" dot={{r: 3, fill: "#FFB454", stroke: "none"}} isAnimationActive={false}/>
            </ComposedChart>
          </ResponsiveContainer></div>
          {bundled === 'fresh' && <div className="chart-wrap"><ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={freshQueue}>
              <XAxis type="number" dataKey="elapsed" domain={timelineDomain}/>
              <YAxis width={72}/>
              <Tooltip labelFormatter={value => `${Number(value).toFixed(3)} s`}/>
              <ReferenceArea x1={freshOnset[0]} x2={freshOnset[1]} fill="#2DD4BF" fillOpacity={0.22}/>
              {confirmed && <ReferenceLine x={confirmed.elapsed} stroke="#FFB454" strokeDasharray="5 4"/>}
              <Line dataKey="delay" name="Router queue delay proxy (ms)" stroke="#60A5FA" dot={false} isAnimationActive={false}/>
            </ComposedChart>
          </ResponsiveContainer></div>}
        </section>
        <details><summary>View forecast scores and decisions</summary>
          <div className="prediction-table-wrap"><table className="prediction-table"><thead><tr>
            <th>Elapsed time (s)</th><th>Model score</th><th>Forecast within 1 s</th>
            {bundled && <th>Observed afterward</th>}
          </tr></thead><tbody>
            {rows.map(row => {
              const {actual, warning} = row
              return <tr key={row.timestamp_seconds}>
                <td>{(row.timestamp_seconds - firstTime).toFixed(3)}</td>
                <td>{row.score.toFixed(6)}</td>
                <td>{cutoff === null ? 'Score only' : warning ? 'Possible queue buildup' : 'No warning'}</td>
                {bundled && <td>{actual === 1 ? 'Congestion began' : actual === 0 ? 'No new congestion' : 'Outcome unknown'}</td>}
              </tr>
            })}
          </tbody></table></div>
        </details>
      </div>}
    </div>
  </section>
}
function PredictPage({ models, onPrediction }: { models: ModelRecord[]; onPrediction: () => void }) {
  const eligible = models.filter(model => model.eligible)
  const [modelId, setModelId] = useState('')
  const [tcpUnit, setTcpUnit] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<{predictions: PredictionRecord[]; input: TelemetrySeries; model: ModelRecord} | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => { if (!eligible.some(m => m.model_id === modelId)) setModelId(eligible[0]?.model_id ?? '') }, [models])
  const selected = eligible.find(model => model.model_id === modelId)
  const requiresTcpUnit = Boolean(selected?.feature_columns.some(column => column.startsWith('tcp_retransmission_delta_')))
  const run = async () => {
    if (!file || !selected || (requiresTcpUnit && !tcpUnit)) return
    setBusy(true); setError(''); setResult(null)
    try { const response = await api.predict(selected.model_id, file, requiresTcpUnit ? tcpUnit : undefined); setResult(response); onPrediction() }
    catch (e) { setError(e instanceof Error ? e.message : 'Inference failed') }
    finally { setBusy(false) }
  }
  return <div className="page-wrap">
    <SectionHeading eyebrow="OPERATIONS · INFERENCE" title="Predict congestion"
      description="Upload endpoint telemetry to generate forecast scores."/>
    <PilotPrediction/>
    {eligible.length > 0 && <details className="registry-prediction" open>
      <summary>Registered models <span>{eligible.length} eligible</span></summary>
    <div className="predict-layout"><section className="panel predict-flow">
      <div className="workflow-step"><div className="step-number">01</div><div className="step-content"><h3>Select an eligible model</h3><p>Only models linked to a complete configuration-held-out evaluation are available.</p><select aria-label="Validated model" className="select-control" value={modelId} onChange={e => {setModelId(e.target.value); setTcpUnit('')}}>{eligible.map(m => <option key={m.model_id} value={m.model_id}>{methodLabel(m.method)} · {m.model_id.slice(0,12)}</option>)}</select></div></div>
      <div className="workflow-step"><div className="step-number">02</div><div className="step-content"><h3>Provide endpoint telemetry</h3><p>Canonical fields only. The service validates timestamps, units, gaps, missing values and model feature compatibility.</p><Dropzone onFile={f => {setFile(f); setResult(null); setError('')}} title="Choose a telemetry CSV" hint="timestamp_seconds · RTT (ms) · throughput (bit/s) · loss (fraction) · retransmission increments (declare packets/sample or bytes/sample when the model uses them)"/>{requiresTcpUnit && <label className="run-select-label">INPUT TCP RETRANSMISSION UNIT<select aria-label="Input TCP retransmission unit" className="select-control" value={tcpUnit} onChange={e => setTcpUnit(e.target.value)}><option value="">Declare the CSV unit…</option><option value="packets">packets per sample</option><option value="bytes">bytes per sample</option></select><span>Declare the CSV’s actual unit. The selected model requires {selected?.tcp_retransmission_unit ?? 'an exact unit'}; packets and bytes are not interchangeable.</span></label>}</div></div>
      <div className="workflow-step"><div className="step-number">03</div><div className="step-content"><h3>Execute inference</h3><p>Model pipeline preprocessing is loaded from the verified artifact; no substitute model is selected automatically.</p><button className="button button-primary" disabled={!selected || !file || (requiresTcpUnit && !tcpUnit) || busy} onClick={() => void run()}>{busy ? <LoaderCircle size={16} className="spin"/> : <Play size={16}/>} {busy ? 'Running model…' : 'Run prediction'}<ArrowRight size={15}/></button></div></div>
      {error && <ErrorNotice message={error}/>}
      {result && <PredictionResults predictions={result.predictions} model={result.model} input={result.input}/>}
    </section><aside className="predict-side">
      <div className="panel selected-model-card"><span className="eyebrow">MODEL CONTRACT</span><h3>{selected ? methodLabel(selected.method) : 'Waiting for eligible model'}</h3>{selected ? <><div className="model-details"><div><span>Forecast horizon</span><strong>{horizonLabel(selected.forecast_horizon_seconds)}</strong></div><div><span>Score threshold</span><strong>{selected.score_threshold?.toPrecision(4) ?? '—'}</strong></div><div><span>Features</span><strong>{selected.feature_columns.length} endpoint windows</strong></div><div><span>Validation</span><strong>Configuration-held-out</strong></div><div><span>TCP retransmission unit</span><strong>{selected.tcp_retransmission_unit ?? 'Not applicable'}</strong></div><div><span>Feature-gap policy</span><strong>{selected.feature_temporal_policy_status ?? 'Not recorded'}</strong></div></div><div className="score-note"><CircleAlert size={15}/><span>{selected.score_semantics}. This is not a calibrated probability.</span></div><div className="fingerprint">PROTOCOL <code>{selected.protocol_sha256 ? `${selected.protocol_sha256.slice(0,16)}…` : 'Not recorded'}</code></div></> : <EmptyPanel compact icon={ShieldCheck} title="Eligibility is strict" body="Frozen protocol, current source hashes, complete grouped evaluation, and an intact model bundle are required."/>}</div>
      <div className="panel inference-limit"><div className="limit-head"><ShieldCheck size={17}/><strong>Inference boundary</strong></div><ul><li>Endpoint-only predictors are accepted.</li><li>Queue backlog and qdisc counters are excluded.</li><li>Run-relative timestamps do not prove wall-clock freshness.</li><li>Warnings are forecasts, not verified congestion events.</li></ul></div>
    </aside></div>
    </details>}
  </div>
}
function PredictionResults({ predictions, model, input }: { predictions: PredictionRecord[]; model: ModelRecord; input: TelemetrySeries }) {
  const warningCount = predictions.filter(p => p.risk === 'warning').length
  const scoredCount = predictions.filter(p => p.score != null).length
  const last = predictions[predictions.length - 1]
  return <section className="result-panel" aria-live="polite"><div className="result-heading"><div><span className="eyebrow">INFERENCE COMPLETE · {input.row_count.toLocaleString()} INPUT ROWS</span><h3>Forecast output</h3></div><StatusPill label="Recorded · unresolved outcome" tone="warning"/></div><div className="result-summary"><div className={`result-glyph ${last?.risk === 'warning' ? 'warn' : ''}`}><Activity size={22}/></div><div><strong>{warningCount ? `${warningCount} warning${warningCount === 1 ? '' : 's'} above the validation threshold` : scoredCount ? 'No scored row crossed the validation warning threshold' : 'No prediction score was produced for this input'}</strong><span>{scoredCount.toLocaleString()} of {predictions.length.toLocaleString()} rows scored · {model.forecast_horizon_seconds == null ? 'horizon not recorded' : `${model.forecast_horizon_seconds}s ahead`} · {methodLabel(model.method)} · {model.model_id.slice(0,12)}</span></div></div><div className="prediction-table-wrap"><table className="prediction-table"><thead><tr><th>Run time</th><th>Model score</th><th>Threshold</th><th>Forecast</th></tr></thead><tbody>{predictions.slice(-8).map(p => <tr key={p.prediction_id}><td>{p.timestamp_seconds.toFixed(2)} s</td><td>{formatScore(p.score)}</td><td>{formatScore(p.score_threshold)}</td><td><StatusPill label={p.risk === 'not_scored' && p.feature_temporal_eligibility_reason ? p.feature_temporal_eligibility_reason.replaceAll('_',' ') : p.risk.replace('_',' ')} tone={p.risk === 'warning' ? 'warning' : p.risk === 'not_scored' ? 'neutral' : 'success'}/></td></tr>)}</tbody></table></div><p className="result-footnote">Scores are model outputs, not calibrated probabilities. Forecasts remain unresolved until later ground-truth review. Exact input and model fingerprints are recorded in warning history.</p></section>
}

function HistoryPage() {
  const [records, setRecords] = useState<PredictionRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [query, setQuery] = useState('')
  const [risk, setRisk] = useState('all')
  const refresh = async () => { setLoading(true); setError(''); try { setRecords((await api.history()).predictions) } catch(e) { setError(e instanceof Error ? e.message : 'History could not be loaded') } finally { setLoading(false) } }
  useEffect(() => { void refresh() }, [])
  const shown = useMemo(() => records.filter(r => (risk === 'all' || r.risk === risk) && `${r.input_name} ${r.model_method} ${r.model_id}`.toLowerCase().includes(query.toLowerCase())), [records, risk, query])
  return <div className="page-wrap"><SectionHeading eyebrow="OPERATIONS · FORECAST LOG" title="Warning history" description="Every row is an actual model inference. Forecasts are not verified events." action={<button className="button button-quiet" onClick={() => void refresh()}><RefreshCw size={15}/>Refresh</button>}/>
    <div className="history-summary"><div><span>RECORDED PREDICTIONS</span><strong>{records.length.toLocaleString()}</strong></div><div><span>OUTCOME VERIFIED</span><strong>{records.filter(r => r.outcome_state !== 'unresolved').length.toLocaleString()}</strong></div><div><span>UNRESOLVED FORECASTS</span><strong>{records.filter(r => r.outcome_state === 'unresolved').length.toLocaleString()}</strong></div><div className="history-summary-note"><ShieldCheck size={16}/>No forecast is treated as a confirmed congestion event without linked ground truth.</div></div>
    <section className="panel history-panel"><div className="history-tools"><label className="search-box"><Search size={16}/><input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search input, model, or ID" aria-label="Search prediction history"/></label><select aria-label="Filter by forecast" className="filter-select" value={risk} onChange={e => setRisk(e.target.value)}><option value="all">All outcomes</option><option value="warning">Threshold warnings</option><option value="below_threshold">Below threshold</option><option value="not_scored">Not scored</option></select></div>
      {loading ? <LoadingLine label="Loading prediction history"/> : error ? <ErrorNotice message={error} onRetry={refresh}/> : shown.length === 0 ? <EmptyPanel compact icon={History} title={records.length ? 'No matching forecasts' : 'No forecasts recorded'} body={records.length ? 'Adjust the filters to see other model runs.' : 'Once a validated model has run against a telemetry batch, its actual scores and unresolved outcome state will be recorded here.'}/> : <div className="table-scroll"><table className="data-table"><thead><tr><th>Predicted at</th><th>Forecast</th><th>Model version</th><th>Input / run</th><th>Score · threshold</th><th>Outcome state</th></tr></thead><tbody>{shown.map(record => <tr key={record.prediction_id}><td>{formatTime(record.predicted_at_utc)}</td><td><span>{record.forecast_horizon_seconds}s</span></td><td><strong>{record.model_method.replaceAll('_',' ')}</strong><small>{record.model_id.slice(0,14)}</small></td><td><strong>{record.input_name}</strong><small>SHA {record.input_sha256.slice(0,12)}…</small></td><td>{formatScore(record.score)} <span className="muted-inline">/ {formatScore(record.score_threshold)}</span></td><td><StatusPill label={record.outcome_state.replaceAll('_',' ')} tone={record.outcome_state === 'unresolved' ? 'warning' : record.outcome_state === 'verified_event' ? 'success' : 'danger'}/></td></tr>)}</tbody></table></div>}
    </section>
  </div>
}

function WorkspacePage({ tab, onTab, onModels, onRefresh }: { tab: WorkspaceTab; onTab: (tab: WorkspaceTab) => void; onModels: () => Promise<void>; onRefresh: () => Promise<void> }) {
  const [datasets, setDatasets] = useState<DatasetSummary | null>(null)
  const [research, setResearch] = useState<ResearchSummary | null>(null)
  const [models, setModels] = useState<ModelRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const refresh = async () => { setLoading(true); setError(''); try { const [d,r,m] = await Promise.all([api.datasets(),api.research(),api.models()]); setDatasets(d);setResearch(r);setModels(m.models) } catch(e) { setError(e instanceof Error ? e.message : 'Workspace data unavailable') } finally { setLoading(false) } }
  useEffect(() => { void refresh() }, [])
  const action = async (kind: 'audit' | 'evaluation') => {
    setBusy(kind);setError('');setNotice('')
    try { const result = kind === 'audit' ? await api.audit() : await api.runEvaluation(); setNotice(result.message); await refresh(); if (kind === 'evaluation') onTab('evaluation') }
    catch(e) { setError(e instanceof Error ? e.message : `${kind} failed`) }
    finally { setBusy('') }
  }
  const train = async (evaluationId: string, method: string) => {
    setBusy('train');setError('');setNotice('')
    try { const result = await api.train(evaluationId,method); setNotice(result.message); await refresh(); await onModels(); onTab('models') }
    catch(e) { setError(e instanceof Error ? e.message : 'Model training failed') }
    finally { setBusy('') }
  }
  const select = async (modelId: string) => {
    setBusy(`select-${modelId}`);setError('')
    try { const result = await api.selectModel(modelId); setNotice(result.message); await refresh(); await onModels() }
    catch(e) { setError(e instanceof Error ? e.message : 'Model selection failed') }
    finally { setBusy('') }
  }
  const subtabs: {id:WorkspaceTab;label:string}[] = [{id:'datasets',label:'Data & sources'},{id:'models',label:'Model registry'},{id:'evaluation',label:'Evaluation'},{id:'protocol',label:'Protocol'},{id:'ground-truth',label:'Ground truth'},{id:'robustness',label:'Robustness'},{id:'reports',label:'Reports'}]
  return <div className="page-wrap"><SectionHeading eyebrow="MODEL OPERATIONS" title={tabTitle(tab)} description="Model controls remain separate from operational inference; every consequential action uses the validated Python pipeline." action={<button className="button button-quiet" onClick={() => {void refresh();void onRefresh()}}><RefreshCw size={15}/>Refresh workspace</button>}/>
    <nav className="workspace-tabs" aria-label="Model operations sections">{subtabs.map(t => <button key={t.id} aria-current={tab===t.id?'page':undefined} className={tab===t.id?'selected':''} onClick={() => onTab(t.id)}>{t.label}</button>)}</nav>
    {busy && <div className="action-progress" role="status"><LoaderCircle className="spin" size={16}/>{busy === 'evaluation' ? 'Running grouped evaluation synchronously. The request remains open until the Python workflow returns.' : busy === 'train' ? 'Fitting and registering the candidate from the validated training split…' : busy === 'audit' ? 'Auditing the configured source tree…' : 'Updating selected model…'}</div>}
    {notice && <div className="notice notice-success"><Check size={17}/><span>{notice}</span></div>}{error && <ErrorNotice message={error}/>}
    {loading && !datasets ? <LoadingLine label="Checking data and model artifacts"/> : tab === 'datasets' ? <DatasetsTab data={datasets} busy={!!busy} onAudit={() => void action('audit')}/> : tab === 'models' ? <ModelsTab models={models} research={research} busy={busy} onTrain={train} onSelect={select}/> : tab === 'evaluation' ? <EvaluationTab data={research} datasets={datasets} busy={!!busy} onRun={() => void action('evaluation')}/> : tab === 'protocol' ? <ProtocolTab data={research} datasets={datasets}/> : tab === 'ground-truth' ? <GroundTruthTab datasets={datasets} data={research} onBuilt={refresh}/> : tab === 'robustness' ? <RobustnessTab data={research}/> : <ReportsTab data={research}/>}
  </div>
}
function tabTitle(tab: WorkspaceTab) { return ({ datasets:'Data & sources', models:'Model registry', evaluation:'Grouped evaluation', protocol:'Protocol & alignment', 'ground-truth':'Ground truth & labels', robustness:'Robustness analysis', reports:'Reports & provenance' })[tab] }

function DatasetsTab({data,busy,onAudit}:{data:DatasetSummary|null;busy:boolean;onAudit:()=>void}) {
  return <div className="workspace-grid"><section className="panel"><div className="panel-head"><div><span className="eyebrow">INPUT INVENTORY</span><h3>Configured datasets</h3></div><StatusPill label={data?.raw.status ?? 'Unknown'} tone={data?.raw.status==='available'?'success':'warning'}/></div>{data ? <><div className="dataset-count"><Database size={19}/><strong>{data.raw.csv_count.toLocaleString()}</strong><span>CSV file{data.raw.csv_count===1?'':'s'} discovered under the configured raw root</span></div><p className="subtle-copy">{data.raw.message}</p><div className="divider"/><div className="detail-row"><span>Processed dataset</span><strong>{data.processed.status}</strong></div><div className="detail-row"><span>Prepared rows</span><strong>{data.processed.row_count?.toLocaleString() ?? '—'}</strong></div><div className="detail-row"><span>Runs · configurations</span><strong>{data.processed.run_count ?? '—'}</strong></div><div className="detail-row"><span>Protocol marker</span><strong>{data.processed.protocol_status ?? 'not verified'}</strong></div><p className="subtle-copy">{data.processed.message}</p><div className="divider"/><div className="detail-row"><span>Audit</span><StatusPill label={data.audit.status} tone={data.audit.status==='passed'?'success':'warning'}/></div><p className="subtle-copy">{data.audit.message}</p></> : <EmptyPanel compact icon={Database} title="No dataset inventory" body="The API has not returned configured sources."/>}</section>
    <section className="panel action-panel"><span className="eyebrow">SAFE OPERATIONS</span><h3>Inspect before you train</h3><p>Audit reads the declared raw tree and writes a new, uniquely named report outside the inputs. It does not edit collector files or change protocol status.</p><button className="button button-secondary" disabled={busy} onClick={onAudit}>{busy?<LoaderCircle size={15} className="spin"/>:<FileSearch size={15}/>}Run read-only audit</button><div className="callout-small"><ShieldCheck size={15}/><span>Preparation, alignment approval, and labels continue to require the existing CLI and human evidence gates.</span></div></section></div>
}
function ModelsTab({models,research,busy,onTrain,onSelect}:{models:ModelRecord[];research:ResearchSummary|null;busy:string;onTrain:(evaluationId:string,method:string)=>void;onSelect:(modelId:string)=>void}) {
  const evaluations=(research?.evaluations??[]).filter(e=>e.status==='complete'&&e.provenance_status==='verified'&&e.grouping==='configuration_id')
  const methods=[...new Set(evaluations.flatMap(e=>e.methods).filter(m=>['logistic_regression','random_forest','hist_gradient_boosting'].includes(m)))]
  return <div className="model-workspace"><section className="panel"><div className="panel-head"><div><span className="eyebrow">REGISTERED CANDIDATES</span><h3>Model catalog</h3></div><span className="count-chip">{models.length} artifact{models.length===1?'':'s'}</span></div>{models.length===0?<EmptyPanel icon={Layers3} title="No registered model yet" body="Complete a configuration-held-out evaluation to register a candidate. The Kali pilot can score RTT files from Predict congestion."/>:<div className="model-list">{models.map(model=><div className="model-row" key={model.model_id}><div className="model-icon"><Gauge size={17}/></div><div className="model-info"><strong>{methodLabel(model.method)}</strong><span>{model.model_id} · evaluation {model.evaluation_id ?? 'not recorded'}</span><small>{model.reason ?? `${model.feature_columns.length} features · ${horizonLabel(model.forecast_horizon_seconds)} horizon · score cutoff ${formatScore(model.score_threshold)}`}</small></div><StatusPill label={model.eligible?'Eligible':model.status} tone={model.eligible?'success':'warning'}/>{model.eligible&&<button className="button button-small button-secondary" disabled={busy===`select-${model.model_id}`} onClick={()=>onSelect(model.model_id)}>{busy===`select-${model.model_id}`?<LoaderCircle size={14} className="spin"/>:'Use for inference'}</button>}</div>)}</div>}</section>
    <section className="model-stage-note" aria-label="Model and outcome lifecycle"><div><span>01</span><strong>Candidate</strong><small>Train from the recorded training split only.</small></div><ArrowRight size={15}/><div><span>02</span><strong>Eligible registered model</strong><small>Current protocol, provenance, and artifact gates must pass.</small></div><ArrowRight size={15}/><div><span>03</span><strong>Actual forecast</strong><small>Produced only after an uploaded batch is inferred.</small></div><ArrowRight size={15}/><div><span>04</span><strong>Verified outcome</strong><small>Not implied by a score or forecast warning.</small></div></section>
    <section className="panel train-panel"><span className="eyebrow">REGISTER CANDIDATE</span><h3>Use an evaluated training split</h3><p>Candidate training reuses the exact train partition, feature list, seed and sklearn Pipeline used by a verified evaluation. The held-out test is never used for fitting.</p>{evaluations.length===0?<div className="locked-callout"><LockKeyhole size={17}/><div><strong>Evaluation evidence required</strong><span>Run a complete, provenance-verified grouped evaluation before registering a model.</span></div></div>:<div className="train-options">{evaluations.map(ev=><div key={ev.evaluation_id} className="train-evaluation"><div><strong>{ev.evaluation_id}</strong><small>configuration-held-out · {ev.methods.join(', ')}</small></div>{methods.map(method=><button key={method} className="button button-small button-secondary" disabled={!!busy} onClick={()=>onTrain(ev.evaluation_id,method)}>Train {method.replaceAll('_',' ')}</button>)}</div>)}</div>}<div className="callout-small"><CircleAlert size={15}/><span>Registration does not claim empirical accuracy. Selection requires the frozen protocol and current provenance checks on every request.</span></div></section>
  </div>
}
function EvaluationTab({data,datasets,busy,onRun}:{data:ResearchSummary|null;datasets:DatasetSummary|null;busy:boolean;onRun:()=>void}) {
  const evaluations=useMemo(()=>data?.evaluations??[],[data])
  const [selectedId,setSelectedId]=useState('')
  const [detail,setDetail]=useState<EvaluationDetail|null>(null)
  const [detailBusy,setDetailBusy]=useState(false)
  const [detailError,setDetailError]=useState('')
  useEffect(()=>{
    if(!evaluations.length){setSelectedId('');setDetail(null);return}
    if(!evaluations.some(item=>item.evaluation_id===selectedId))setSelectedId(evaluations[0].evaluation_id)
  },[evaluations,selectedId])
  useEffect(()=>{
    if(!selectedId){setDetail(null);return}
    let current=true;setDetailBusy(true);setDetailError('')
    api.evaluationDetail(selectedId).then(value=>{if(current)setDetail(value)}).catch(e=>{if(current){setDetail(null);setDetailError(e instanceof Error?e.message:'Evaluation details could not be verified')}}).finally(()=>{if(current)setDetailBusy(false)})
    return ()=>{current=false}
  },[selectedId])
  const eligible=evaluations.filter(e=>e.status==='complete'&&e.provenance_status==='verified'&&e.grouping==='configuration_id')
  return <div className="research-page-stack">
    <section className="panel evaluation-intro"><div className="panel-head"><div><span className="eyebrow">HELD-OUT REPORTS</span><h3>Grouped evaluation</h3></div><button className="button button-primary button-small" disabled={busy||!data?.operations.evaluation_available} onClick={onRun}>{busy?<LoaderCircle size={14} className="spin"/>:<Play size={14}/>}Run frozen evaluation</button></div><p className="subtle-copy">The existing Python evaluator uses grouped train/validation/test splits, train-only preprocessing, validation-only threshold selection, endpoint features, event metrics, and checksum-linked artifacts. It writes a new-only report directory; this screen never recomputes evaluation metrics.</p>{!data?.operations.evaluation_available&&<div className="blocker-strip"><LockKeyhole size={17}/><div><strong>Evaluation is gated</strong><span>{data?.operations.blockers?.join(' ')||'The current frozen protocol, prepared data, audit, and source lineage have not all verified.'}</span></div></div>}</section>
    <div className="evaluation-layout">
      <aside className="panel evaluation-index"><div className="panel-head"><div><span className="eyebrow">EVALUATION INVENTORY</span><h3>Saved runs</h3></div><span className="count-chip">{evaluations.length}</span></div>{evaluations.length?evaluations.map(item=><button key={item.evaluation_id} className={`evaluation-select ${selectedId===item.evaluation_id?'active':''}`} aria-pressed={selectedId===item.evaluation_id} onClick={()=>setSelectedId(item.evaluation_id)}><strong>{item.evaluation_id}</strong><span>{item.grouping??'grouping unknown'} · {item.methods.join(', ')||'no successful methods'}</span><StatusPill label={`${item.status} · ${item.provenance_status}`} tone={item.status==='complete'&&item.provenance_status==='verified'?'success':'warning'}/></button>):<EmptyPanel compact icon={ShieldCheck} title="No verified evaluations" body="A complete evaluation is required before model registration or inference eligibility."/>}</aside>
      <section className="panel evaluation-detail-panel">{detailBusy?<LoadingLine label="Verifying report checksums and current lineage"/>:detailError?<ErrorNotice message={detailError}/>:detail?<EvaluationDetailView detail={detail}/>:<EmptyPanel compact icon={FileSearch} title="Select an evaluation" body="Choose a saved report to inspect held-out outcomes, uncertainty, failures, and provenance."/>}</section>
    </div>
    <section className="panel"><span className="eyebrow">ELIGIBILITY GATES</span><h3>Current blockers and checks</h3><GateList data={data} datasets={datasets}/></section>
    {eligible.length>0&&<p className="subtle-copy">{eligible.length} evaluation{eligible.length===1?' is':'s are'} complete, provenance-verified, and configuration-grouped. Model registration still re-checks every gate and fits only the recorded training partition.</p>}
  </div>
}
function metricValue(value:number|null|undefined,digits=3){return value==null||!Number.isFinite(value)?'Not estimable':value.toFixed(digits)}
function EvaluationDetailView({detail}:{detail:EvaluationDetail}){
  const methods=detail.summary.methods
  return <div className="evaluation-detail-content">
    <div className="panel-head evaluation-detail-head"><div><span className="eyebrow">CHECKSUM-VERIFIED REPORT · {formatTime(detail.created_at_utc)}</span><h3>{detail.report_id}</h3></div><StatusPill label={`${detail.integrity_status} · ${detail.provenance_status}`} tone={detail.provenance_status==='verified'?'success':'warning'}/></div>
    {detail.blockers.length>0&&<div className="notice notice-warning"><CircleAlert size={17}/><div><strong>Interpret provenance cautiously</strong><p>{detail.blockers.join(' ')}</p></div></div>}
    <div className="fingerprint-grid"><Fingerprint label="Protocol SHA-256" value={detail.protocol_sha256}/><Fingerprint label="Analysis SHA-256" value={detail.analysis_sha256}/><Fingerprint label="Processing manifest" value={detail.processing_manifest_sha256}/><Fingerprint label="Source manifest" value={detail.source_manifest_sha256}/><Fingerprint label="Audit report" value={detail.audit_sha256}/></div>
    <div className="detail-summary-grid"><div><span>Outcome</span><strong>{detail.status}</strong></div><div><span>Grouping</span><strong>{detail.grouping??'Not recorded'}</strong></div><div><span>Held-out groups</span><strong>{detail.summary.held_out_group_count}</strong></div><div><span>Split seed</span><strong>{detail.summary.split_seed??'Not recorded'}</strong></div><div><span>Threshold-selection split</span><strong>{detail.summary.threshold_calibration_split??'Not recorded'}</strong></div></div>
    {methods.length>0?<><div className="evaluation-chart-section"><div><h4>Held-out event comparison</h4><p>Only estimable event precision/recall values are plotted. Missing values remain absent, never coerced to zero.</p></div><div className="evaluation-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={methods} margin={{top:10,right:18,left:0,bottom:12}}><CartesianGrid stroke="var(--grid)" strokeDasharray="3 5" vertical={false}/><XAxis dataKey="method" tick={{fill:'var(--muted)',fontSize:11}} tickLine={false} axisLine={false} tickFormatter={v=>String(v).replaceAll('_',' ')}/><YAxis domain={[0,1]} tick={{fill:'var(--muted)',fontSize:11}} tickLine={false} axisLine={false} width={36}/><Tooltip contentStyle={{background:'var(--tooltip)',border:'1px solid var(--border)',borderRadius:8,color:'var(--text)'}} formatter={(value,name)=>[value==null?'Not estimable':`${(Number(value)*100).toFixed(1)}%`,name==='event_precision'?'Event precision':'Event recall']}/><Bar dataKey="event_precision" name="Event precision" fill="#43E3F5" radius={[3,3,0,0]} isAnimationActive={false}/><Bar dataKey="event_recall" name="Event recall" fill="#A899FF" radius={[3,3,0,0]} isAnimationActive={false}/></BarChart></ResponsiveContainer></div></div>
    <div className="table-scroll"><table className="data-table evaluation-metrics-table"><thead><tr><th>Method</th><th>Test events</th><th>Event precision</th><th>Event recall</th><th>False alarms / min</th><th>Median lead time</th><th>Average precision*</th><th>Scored anchors</th><th>Validation threshold</th></tr></thead><tbody>{methods.map(m=><tr key={m.method}><td>{m.method.replaceAll('_',' ')}</td><td>{m.event_count==null?'Not reported':m.event_count.toLocaleString()}</td><td>{m.event_precision==null?'Not estimable':`${(m.event_precision*100).toFixed(1)}%`}</td><td>{m.event_recall==null?'Not estimable':`${(m.event_recall*100).toFixed(1)}%`}</td><td>{m.false_alarms_per_minute==null?'Not estimable':m.false_alarms_per_minute.toFixed(3)}</td><td>{m.lead_time_median_seconds==null?'Not estimable':`${m.lead_time_median_seconds.toFixed(2)} s (${m.lead_time_event_count??0} matched)`}</td><td>{metricValue(m.average_precision)}</td><td>{m.scored_anchor_count==null?'Not reported':m.scored_anchor_count.toLocaleString()}</td><td>{metricValue(m.threshold)}</td></tr>)}</tbody></table></div>
    <p className="metric-footnote">* Average precision is the evaluator's secondary anchor-level PR summary, not an event metric. It is undefined when supported classes are absent. Event metrics use held-out event/exposure denominators. A zero is shown only when the verified artifact reports a numeric zero.</p>
    <h4 className="subsection-title">Uncertainty and adequacy</h4><div className="table-scroll"><table className="data-table"><thead><tr><th>Method</th><th>Metric</th><th>Configuration clusters</th><th>Reported interval</th><th>Interpretation</th></tr></thead><tbody>{methods.flatMap(m=>Object.entries(m.uncertainty).map(([metric,ci])=><tr key={`${m.method}-${metric}`}><td>{m.method.replaceAll('_',' ')}</td><td>{metric.replaceAll('_',' ')}</td><td>{ci.cluster_count??'Not reported'}</td><td>{ci.estimable&&ci.interval?`${ci.confidence==null?'Unspecified':`${(ci.confidence*100).toFixed(0)}%`} [${ci.interval[0].toFixed(3)}, ${ci.interval[1].toFixed(3)}]`:'Not estimable'}</td><td>{ci.reason??ci.warning??(ci.estimable?`Bootstrap interval over ${ci.unit_of_resampling??'recorded clusters'}.`:'Uncertainty interval unavailable.')}</td></tr>))}{methods.every(m=>Object.keys(m.uncertainty).length===0)&&<tr><td colSpan={5}>Uncertainty intervals are not present in this verified artifact; none are inferred.</td></tr>}</tbody></table></div>
    {methods.some(m=>m.adequacy_warnings.length>0)&&<div className="notice notice-warning"><CircleAlert size={17}/><div><strong>Outcome adequacy warnings</strong><p>{methods.flatMap(m=>m.adequacy_warnings.map(w=>`${m.method}: ${w.replaceAll('_',' ')}`)).join(' · ')}</p></div></div>}
    {Object.keys(detail.summary.method_failures).length>0&&<div className="method-failures"><h4>Method failures (not omitted from the record)</h4>{Object.entries(detail.summary.method_failures).map(([method,reason])=><div key={method}><strong>{method.replaceAll('_',' ')}</strong><span>{reason}</span></div>)}</div>}
    <details className="held-out-groups"><summary>Held-out test groups ({detail.summary.held_out_group_count})</summary><p>{detail.summary.held_out_groups.join(', ')||'No held-out group identifiers were recorded.'}{detail.summary.held_out_groups_truncated?' … (display limited to first 500 groups)':''}</p></details>
    <div className="detail-downloads"><strong>Verified report artifacts</strong><ArtifactLinks reportId={detail.report_id} artifacts={detail.artifacts}/></div>
    <p className="subtle-copy">No single-metric ranking is applied. Thresholds are selected on validation data only; test metrics remain held-out. A mechanically complete report does not itself establish scientific sufficiency.</p>
  </>:<><EmptyPanel compact icon={CircleAlert} title="No successful methods" body="The integrity-verified report contains no successful method results."/>{Object.keys(detail.summary.method_failures).length>0&&<div className="method-failures">{Object.entries(detail.summary.method_failures).map(([method,reason])=><div key={method}><strong>{method}</strong><span>{reason}</span></div>)}</div>}<ArtifactLinks reportId={detail.report_id} artifacts={detail.artifacts}/></>}
  </div>
}
function Fingerprint({label,value}:{label:string;value:string|null|undefined}){return <div className="fingerprint-item"><span>{label}</span><code title={value??'Not recorded'}>{value??'Not recorded'}</code></div>}
function ArtifactLinks({reportId,artifacts}:{reportId:string;artifacts:string[]}){return <div className="artifact-links">{artifacts.map(file=><a key={file} className="button button-small button-secondary" href={reportFileHref(reportId,file)} download><Download size={13}/>{file}</a>)}</div>}
function GateList({data,datasets}:{data:ResearchSummary|null;datasets:DatasetSummary|null}) {
  const operations=data?.operations
  const gates:[string,boolean,string][]=[
    ['Frozen protocol',data?.protocol.status==='protocol_frozen',data?.protocol.status??'Not checked'],
    ['Alignment review',data?.alignment.status==='not_required'||data?.alignment.status==='approved',data?.alignment.status??'Not checked'],
    ['Prepared artifact integrity',operations?.prepared_status==='verified',operations?.prepared_status??datasets?.processed.status??'Not checked'],
    ['Exact feature contract & temporal policy',operations?.feature_specification_status==='current_contract_complete'&&operations?.feature_temporal_policy_status==='resolved',operations?.feature_specification_status??'Not checked'],
    ['Current audit and source lineage',operations?.source_lineage_status==='verified',operations?.source_lineage_status??datasets?.audit.status??'Not checked'],
    ['Complete configuration-held-out report',data?.evaluations.some(e=>e.status==='complete'&&e.provenance_status==='verified'&&e.grouping==='configuration_id')??false,'Requires current complete report'],
  ]
  return <div><div className="gate-list">{gates.map(([name,ok,status])=><div key={name}><span className={`gate-check ${ok?'pass':''}`}>{ok?<Check size={13}/>:<X size={13}/>}</span><span>{name}</span><strong>{ok?'Pass':status==='unknown'?'Unknown':'Blocked'}</strong></div>)}</div>{operations?.blockers?.length? <ul className="gate-blockers">{operations.blockers.map((blocker,i)=><li key={i}>{blocker}</li>)}</ul>:null}</div>
}
function ProtocolTab({data,datasets}:{data:ResearchSummary|null;datasets:DatasetSummary|null}) {return <div className="workspace-grid"><section className="panel"><span className="eyebrow">QUALITY GATE</span><h3>Protocol state</h3><div className="protocol-status"><StatusPill label={data?.protocol.status??'not checked'} tone={data?.protocol.status==='protocol_frozen'?'success':'warning'}/><p>{data?.protocol.message??'Protocol evidence is unavailable.'}</p></div><div className="detail-row"><span>Config fingerprint</span><code>{data?.protocol.config_sha256?.slice(0,24)??'—'}</code></div><div className="divider"/><div className="detail-row"><span>Alignment</span><strong>{data?.alignment.status??'not checked'}</strong></div><p className="subtle-copy">{data?.alignment.message}</p></section><section className="panel"><span className="eyebrow">HUMAN EVIDENCE REQUIRED</span><h3>No approval shortcuts</h3><p className="subtle-copy">This web interface does not modify protocol files or approve time alignment. Continue with the existing decision record and CLI verification; UI workflows re-check the resulting hashes before training or inference.</p><div className="callout-small"><LockKeyhole size={15}/><span>Current prepared status: {datasets?.processed.protocol_status??'unavailable'}. Draft or exploratory outputs are not confirmatory.</span></div></section></div>}
function GroundTruthTab({datasets,onBuilt}:{datasets:DatasetSummary|null;data:ResearchSummary|null;onBuilt:()=>Promise<void>}){
  const [index,setIndex]=useState<GroundTruthIndex|null>(null)
  const [selectedRun,setSelectedRun]=useState('')
  const [series,setSeries]=useState<GroundTruthSeries|null>(null)
  const [loading,setLoading]=useState(true)
  const [seriesLoading,setSeriesLoading]=useState(false)
  const [building,setBuilding]=useState(false)
  const [error,setError]=useState('')
  const [buildNotice,setBuildNotice]=useState('')
  const refresh=async()=>{setLoading(true);setError('');try{const next=await api.groundTruthIndex();setIndex(next);if(!next.runs.some(r=>r.run_id===selectedRun))setSelectedRun(next.runs[0]?.run_id??'')}catch(e){setError(e instanceof Error?e.message:'Ground-truth index could not be loaded')}finally{setLoading(false)}}
  useEffect(()=>{void refresh()},[])
  useEffect(()=>{if(!selectedRun){setSeries(null);return}let current=true;setSeriesLoading(true);setError('');api.groundTruthSeries(selectedRun).then(value=>{if(current)setSeries(value)}).catch(e=>{if(current){setSeries(null);setError(e instanceof Error?e.message:'Verified run timeline could not be loaded')}}).finally(()=>{if(current)setSeriesLoading(false)});return()=>{current=false}},[selectedRun])
  const build=async()=>{setBuilding(true);setError('');setBuildNotice('');try{const result=await api.buildGroundTruth();setBuildNotice(result.message);await refresh();await onBuilt()}catch(e){setError(e instanceof Error?e.message:'Descriptive ground-truth report failed')}finally{setBuilding(false)}}
  return <div className="research-page-stack">
    <section className="panel ground-truth-overview"><div className="panel-head"><div><span className="eyebrow">PREPARED TARGET · BATCH ONLY</span><h3>Ground truth & queue proxy</h3></div><div className="ground-truth-actions"><StatusPill label={index?.status??(loading?'Checking':'Unavailable')} tone={index?.status==='ready'?'success':'warning'}/><button className="button button-secondary button-small" onClick={()=>void refresh()} aria-label="Refresh ground-truth artifacts"><RefreshCw size={14}/>Refresh</button></div></div><p className="subtle-copy">This view reads only the current integrity- and lineage-verified prepared dataset. The descriptive report uses the existing Python target logic; the browser does not derive event metrics or manufacture labels.</p>
      {loading&&!index?<LoadingLine label="Checking prepared data, audit lineage, and report fingerprints"/>:index?.status==='ready'?<><div className="ground-truth-meta"><div><span>Prepared rows</span><strong>{index.runs.reduce((sum,run)=>sum+run.row_count,0).toLocaleString()}</strong></div><div><span>Runs</span><strong>{index.runs.length}</strong></div><div><span>Observed events</span><strong>{index.report_id?index.runs.reduce((sum,run)=>sum+(run.event_count??0),0).toLocaleString():'Not reported'}</strong></div><div><span>Target threshold</span><strong>{index.threshold_ms==null?'Not recorded':`${index.threshold_ms} ms`}</strong></div><div><span>Max target gap</span><strong>{index.max_gap_seconds==null?'Not recorded':`${index.max_gap_seconds} s`}</strong></div></div><div className="fingerprint-grid"><Fingerprint label="Analysis SHA-256" value={index.analysis_sha256}/><Fingerprint label="Protocol SHA-256" value={index.protocol_sha256}/></div><div className="alignment-callout"><ShieldCheck size={16}/><div><strong>Alignment: {index.alignment_status}</strong><span>{index.alignment_message}</span></div></div>{index.blockers.length>0&&<div className="notice notice-warning"><CircleAlert size={16}/><div><strong>Event boundaries unavailable</strong><p>{index.blockers.join(' ')}</p></div></div>}</>:<EmptyPanel icon={Database} title="No verified prepared batch data" body={index?.message??'The configured workspace has no integrity- and lineage-verified processed data.'} action={datasets&&<span className="subtle-copy">Prepared dataset state: {datasets.processed.status}. Source status: {datasets.audit.status}.</span>}/>}
      {error&&<ErrorNotice message={error} onRetry={()=>void refresh()}/>}{buildNotice&&<div className="notice notice-success"><Check size={16}/><p>{buildNotice}</p></div>}
      {index?.status==='ready'&&<div className="ground-truth-build-row"><div><strong>{index.report_id?'Current report verified':'No current event report'}</strong><span>{index.report_id?`Report ${index.report_id} is linked to these prepared-data fingerprints.`:'Generate the descriptive report to expose event boundaries, censoring, and per-run quality summaries.'}</span></div><button className="button button-primary button-small" disabled={building} onClick={()=>void build()}>{building?<LoaderCircle size={14} className="spin"/>:<Play size={14}/>} {building?'Building report…':index.report_id?'Rebuild descriptive report':'Build ground-truth report'}</button></div>}
    </section>
    {index?.status==='ready'&&<section className="panel ground-truth-timeline"><div className="panel-head"><div><span className="eyebrow">VALIDATED RUN TIMELINE</span><h3>Queue backlog & proxy over time</h3></div><label className="run-select-label">RUN<select className="select-control" aria-label="Select prepared run" value={selectedRun} onChange={e=>setSelectedRun(e.target.value)}>{index.runs.map(run=><option key={run.run_id} value={run.run_id}>{run.run_id}{run.configuration_id?` · ${run.configuration_id}`:''}</option>)}</select></label></div>
      {!index.report_id&&<p className="proxy-note"><CircleAlert size={15}/>Prepared measurements and label columns are verified, but no current report links event intervals; no event count, boundary overlay, or censoring interpretation is asserted.</p>}
      {seriesLoading?<LoadingLine label="Loading validated run values"/>:series&&<><p className="subtle-copy">{series.sampling_note} Run-relative timestamps only; this is not a live feed.</p><div className="ground-truth-chart-grid"><GroundTruthChart series={series} field="queue_backlog_bytes" title="Raw qdisc queue backlog" unit="bytes" color="#43E3F5"/><GroundTruthChart series={series} field="qdisc_queue_delay_proxy_ms" title="Backlog-derived serialization-delay proxy" unit="ms" color="#A899FF" threshold={series.threshold_ms}/></div><div className="proxy-note"><CircleAlert size={15}/><span><strong>Proxy semantics:</strong> qdisc_queue_delay_proxy_ms = backlog_bytes × 8 ÷ declared bottleneck rate. It is a serialization-delay proxy, not direct packet queueing delay. Event bands come only from the current checksum-verified target report; dashed threshold is the threshold stored in the preparation manifest.</span></div>
      {series.event_boundaries_available?<div className="table-scroll"><table className="data-table"><caption>Verified observed intervals; onset bounds and censoring are retained</caption><thead><tr><th>Observed start</th><th>Observed end</th><th>Onset interval</th><th>Active samples</th><th>Censoring / gaps</th></tr></thead><tbody>{series.events.length?series.events.map((event,i)=><tr key={`${event.observed_start_seconds}-${i}`}><td>{event.observed_start_seconds==null?'Not identified':`${event.observed_start_seconds.toFixed(3)} s`}</td><td>{event.observed_end_seconds==null?'Not identified':`${event.observed_end_seconds.toFixed(3)} s`}</td><td>{event.onset_interval_identified&&event.onset_interval_lower_exclusive_seconds!=null&&event.onset_interval_upper_inclusive_seconds!=null?`(${event.onset_interval_lower_exclusive_seconds.toFixed(3)}, ${event.onset_interval_upper_inclusive_seconds.toFixed(3)}] s`:'Unidentified / gap-ambiguous'}</td><td>{event.sample_count??'Not reported'}</td><td>{event.left_censored_or_gap_ambiguous?'Left censored / ambiguous':''}{event.left_censored_or_gap_ambiguous&&event.right_censored_or_gap_ambiguous?' · ':''}{event.right_censored_or_gap_ambiguous?'Right censored / ambiguous':''}{!event.left_censored_or_gap_ambiguous&&!event.right_censored_or_gap_ambiguous?'No censoring flagged':''}</td></tr>):<tr><td colSpan={5}>The verified report observed no event intervals in this run.</td></tr>}</tbody></table></div>:null}
      <ul className="limitations-list">{series.limitations.map((item,i)=><li key={i}>{item}</li>)}</ul></>}
    </section>}
    <section className="panel"><span className="eyebrow">INTERPRETATION BOUNDARIES</span><h3>Keep uncertainty visible</h3><ul className="explain-list"><li>Threshold crossings are sample-interval observations, not exact onset times.</li><li>Gapped or incomplete horizons remain censored or ambiguous, not negatives.</li><li>Queue backlog and qdisc counters are ground-truth/training data only; predictors remain endpoint-only.</li><li>Changing alignment approval or input fingerprints invalidates the linkage and requires re-verification.</li></ul></section>
  </div>
}
function GroundTruthChart({series,field,title,unit,color,threshold}:{series:GroundTruthSeries;field:'queue_backlog_bytes'|'qdisc_queue_delay_proxy_ms';title:string;unit:string;color:string;threshold?:number|null}){
  const valueKey=field==='queue_backlog_bytes'?'backlog_plot':'proxy_plot'
  const timePrecision=secondsTickPrecision(series.points.map(point=>point.timestamp_seconds))
  const points=series.points.flatMap((point,index)=>{
    const observed={timestamp_seconds:point.timestamp_seconds,[valueKey]:point[field]}
    if(index===0||!point.segment_start)return [observed]
    // Insert a null plotting break at the same observed timestamp; retain the real boundary sample.
    return [{timestamp_seconds:point.timestamp_seconds,[valueKey]:null},observed]
  })
  const eventAreas=series.event_boundaries_available?series.events.filter(event=>event.observed_start_seconds!=null&&event.observed_end_seconds!=null&&event.observed_end_seconds>=event.observed_start_seconds):[]
  const hasValues=series.points.some(point=>point[field]!=null)
  return <section className="ground-chart-card"><div className="ground-chart-title"><div><strong>{title}</strong><span>{unit}</span></div><small>{series.event_boundaries_available?'teal shading = verified observed event':'no verified event overlay'}</small></div>{hasValues?<div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><ComposedChart data={points} margin={{top:14,right:16,left:4,bottom:6}}><CartesianGrid stroke="var(--grid)" strokeDasharray="3 5" vertical={false}/><XAxis dataKey="timestamp_seconds" type="number" domain={['dataMin','dataMax']} tickLine={false} axisLine={false} tick={{fill:'var(--muted)',fontSize:11}} tickFormatter={v=>`${Number(v).toFixed(timePrecision)} s`} minTickGap={35}/><YAxis tickLine={false} axisLine={false} tick={{fill:'var(--muted)',fontSize:11}} width={56} tickFormatter={v=>Number(v).toLocaleString()}/><Tooltip contentStyle={{background:'var(--tooltip)',border:'1px solid var(--border)',borderRadius:8,color:'var(--text)'}} labelFormatter={v=>`Run time ${Number(v).toFixed(3)} s`} formatter={value=>[value==null?'Gap / missing':`${Number(value).toLocaleString()} ${unit}`,title]}/>{eventAreas.map((event,i)=><ReferenceArea key={`${event.observed_start_seconds}-${i}`} x1={event.observed_start_seconds!} x2={event.observed_end_seconds!} fill="#2DD4BF" fillOpacity={0.11} strokeOpacity={0}/>)}<Line type="linear" dataKey={valueKey} name={title} stroke={color} strokeWidth={2} dot={false} connectNulls={false} isAnimationActive={false}/>{field==='qdisc_queue_delay_proxy_ms'&&threshold!=null&&<ReferenceLine y={threshold} stroke="#FFB454" strokeDasharray="5 4" label={{value:`manifest threshold ${threshold} ms`,fill:'#FFB454',fontSize:10,position:'insideTopRight'}}/>}</ComposedChart></ResponsiveContainer></div>:<EmptyPanel compact icon={Activity} title="No observed values" body={`The verified prepared run has no plottable ${unit} values; no substitute series is shown.`}/>}</section>
}
function RobustnessTab({data}:{data:ResearchSummary|null}) {const items=data?.reports.filter(r=>r.kind==='robustness')??[];return <div className="panel"><div className="panel-head"><div><span className="eyebrow">SENSITIVITY & LIMITATIONS</span><h3>Robustness artifacts</h3></div><span className="count-chip">{items.length} reports</span></div><p className="subtle-copy">Existing CLI-backed robustness analyses are read-only here. They are exploratory unless the analysis plan was frozen before outcomes were inspected.</p>{items.length?items.map(item=><div className="eval-row" key={item.report_id}><strong>{item.report_id}</strong><StatusPill label={item.status} tone="warning"/></div>):<EmptyPanel compact icon={Activity} title="No robustness report" body="No verified robustness artifact was found in the configured report root."/>}</div>}
function ReportsTab({data}:{data:ResearchSummary|null}){
  const reportItems=useMemo(()=>[...(data?.reports??[]),...(data?.evaluations??[]).map(e=>({report_id:e.evaluation_id,kind:'evaluation',status:e.status,created_at_utc:e.created_at_utc}))],[data])
  const [selectedId,setSelectedId]=useState('')
  const [detail,setDetail]=useState<ReportDetail|null>(null)
  const [loading,setLoading]=useState(false)
  const [error,setError]=useState('')
  useEffect(()=>{if(!reportItems.length){setSelectedId('');setDetail(null);return}if(!reportItems.some(item=>item.report_id===selectedId))setSelectedId(reportItems[0].report_id)},[reportItems,selectedId])
  useEffect(()=>{if(!selectedId){setDetail(null);return}let current=true;setLoading(true);setError('');api.reportDetail(selectedId).then(item=>{if(current)setDetail(item)}).catch(e=>{if(current){setDetail(null);setError(e instanceof Error?e.message:'Report detail could not be verified')}}).finally(()=>{if(current)setLoading(false)});return()=>{current=false}},[selectedId])
  return <div className="reports-layout"><section className="panel reports-index"><div className="panel-head"><div><span className="eyebrow">ARTIFACT INDEX</span><h3>Reports & provenance</h3></div><span className="count-chip">{reportItems.length} found</span></div><p className="subtle-copy">Report identifiers are resolved only within the configured reports root. Details appear after artifact checks; file downloads use manifest allowlists.</p>{reportItems.length?reportItems.map(item=><button className={`report-select ${selectedId===item.report_id?'active':''}`} key={`${item.kind}-${item.report_id}`} aria-pressed={selectedId===item.report_id} onClick={()=>setSelectedId(item.report_id)}><strong>{item.report_id}</strong><span>{item.kind.replaceAll('_',' ')} · {formatTime(item.created_at_utc)}</span><StatusPill label={item.status} tone={item.status.includes('verified')||item.status==='complete'?'success':'warning'}/></button>):<EmptyPanel compact icon={FileSearch} title="No reports discovered" body="Reports appear after existing audit, target-validation, evaluation, or robustness workflows run."/>}</section>
    <section className="panel report-detail-panel">{loading?<LoadingLine label="Verifying checksums and loading report provenance"/>:error?<ErrorNotice message={error}/>:detail?<><div className="panel-head"><div><span className="eyebrow">{detail.kind.replaceAll('_',' ').toUpperCase()} · {formatTime(detail.created_at_utc)}</span><h3>{detail.report_id}</h3></div><StatusPill label={`${detail.integrity_status} · ${detail.provenance_status}`} tone={detail.provenance_status.includes('verified')?'success':'warning'}/></div><div className="detail-summary-grid"><div><span>Outcome</span><strong>{detail.status}</strong></div><div><span>Kind</span><strong>{detail.kind.replaceAll('_',' ')}</strong></div><div><span>Method / grouping</span><strong>{detail.method??detail.grouping??'See verified manifest'}</strong></div><div><span>Created</span><strong>{formatTime(detail.created_at_utc)}</strong></div></div>{detail.blockers.length>0&&<div className="notice notice-warning"><CircleAlert size={17}/><div><strong>Lineage or eligibility limitation</strong><p>{detail.blockers.join(' ')}</p></div></div>}<div className="fingerprint-grid"><Fingerprint label="Protocol SHA-256" value={detail.protocol_sha256}/><Fingerprint label="Analysis SHA-256" value={detail.analysis_sha256}/><Fingerprint label="Processing manifest" value={detail.processing_manifest_sha256}/><Fingerprint label="Source manifest" value={detail.source_manifest_sha256}/><Fingerprint label="Audit report" value={detail.audit_sha256}/></div><div className="detail-downloads"><strong>Manifest-allowlisted downloads (max 16 MiB each)</strong><ArtifactLinks reportId={detail.report_id} artifacts={detail.artifacts}/></div><details className="held-out-groups"><summary>Verified report summary</summary><pre className="summary-json">{JSON.stringify(detail.summary,null,2)}</pre></details><p className="subtle-copy">Integrity verification confirms saved bytes match their manifest. Provenance status separately reflects current input/protocol linkage; a verified checksum alone is not scientific validation.</p></>:<EmptyPanel compact icon={FileSearch} title="Select a report" body="Choose an artifact to inspect its verified input fingerprints, method/status, provenance blockers, and safe downloads."/>}</section>
  </div>
}

export default App
