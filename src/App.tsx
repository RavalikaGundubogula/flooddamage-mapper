import { useState, useEffect } from 'react'
import './App.css'
import {
  checkBackendHealth,
  fetchAnalysisJob,
  startNewAnalysis,
  type AnalysisJob,
  API_BASE_URL,
} from './api'

const navItems = [
  { icon: '▦', label: 'Dashboard' },
  { icon: '＋', label: 'New Analysis' },
  { icon: '⌖', label: 'GIS Explorer' },
  { icon: '▤', label: 'Flood Results' },
  { icon: '⌁', label: 'Infrastructure' },
  { icon: '⇄', label: 'Connectivity' },
  { icon: '▧', label: 'Reports' },
  { icon: '◷', label: 'Job History' },
]

// Preset areas with valid coordinate polygons
const PRESET_AOIS: Record<string, number[][][]> = {
  Nepal: [
    [
      [85.25, 27.65],
      [85.38, 27.65],
      [85.38, 27.76],
      [85.25, 27.76],
      [85.25, 27.65],
    ],
  ],
  'Kathmandu Valley': [
    [
      [85.28, 27.67],
      [85.36, 27.67],
      [85.36, 27.73],
      [85.28, 27.73],
      [85.28, 27.67],
    ],
  ],
  'Terai Region': [
    [
      [85.0, 26.9],
      [85.5, 26.9],
      [85.5, 27.2],
      [85.0, 27.2],
      [85.0, 26.9],
    ],
  ],
}

function App() {
  const [active, setActive] = useState('Dashboard')
  const [area, setArea] = useState('Nepal')
  const [date, setDate] = useState('2026-08-26')
  const [language, setLanguage] = useState('en')
  const [notice, setNotice] = useState('')
  const [backendStatus, setBackendStatus] = useState<{
    connected: boolean
    version?: string
    checking: boolean
  }>({
    connected: false,
    checking: true,
  })
  const [currentJob, setCurrentJob] = useState<AnalysisJob | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [layerFilter, setLayerFilter] = useState<'all' | 'water' | 'roads' | 'settlements'>('all')

  // Check backend health on mount
  useEffect(() => {
    let mounted = true
    async function checkHealth() {
      const res = await checkBackendHealth()
      if (mounted) {
        setBackendStatus({
          connected: res.connected,
          version: res.info?.version,
          checking: false,
        })
        if (res.connected) {
          // Attempt to load the real Nepal reference case study
          try {
            const defaultJob = await fetchAnalysisJob('job_nepal_2026_case')
            if (mounted && defaultJob) {
              setCurrentJob(defaultJob)
            }
          } catch {
            // No default job yet, leave empty state
          }
        }
      }
    }
    checkHealth()
    return () => {
      mounted = false
    }
  }, [])

  async function handleRunAnalysis() {
    setIsLoading(true)
    setNotice('')
    try {
      const coords = PRESET_AOIS[area] || PRESET_AOIS['Nepal']
      const job = await startNewAnalysis({
        event_date: date,
        coordinates: coords,
        language,
        mode: 'sync',
      })
      setCurrentJob(job)
      setActive('Flood Results')
      setNotice(`Analysis completed successfully for ${area} (Job: ${job.analysis_id})`)
    } catch (err: any) {
      setNotice(`Analysis request error: ${err.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  // Derive display statistics dynamically from live backend job or fallback gracefully
  const summary = currentJob?.summary
  const stats = [
    {
      label: 'Flooded area',
      value: summary ? `${summary.flooded_area_km2.toFixed(2)} km²` : '—',
      note: summary ? `Confidence: ${(summary.confidence_score * 100).toFixed(0)}%` : 'Awaiting analysis',
      icon: '◈',
      color: 'blue',
    },
    {
      label: 'Buildings exposed',
      value: summary ? `${summary.buildings_in_flood_zone.toLocaleString()}` : '—',
      note: summary ? `Total detected: ${summary.total_buildings_detected.toLocaleString()}` : 'Awaiting analysis',
      icon: '⌂',
      color: 'orange',
    },
    {
      label: 'Roads potentially affected',
      value: summary ? `${summary.roads_affected.toLocaleString()}` : '—',
      note: summary ? `Blocked segments: ${summary.roads_blocked.toLocaleString()}` : 'Awaiting analysis',
      icon: '↗',
      color: 'teal',
    },
    {
      label: 'Isolated settlements',
      value: summary ? `${summary.isolated_settlements.toLocaleString()}` : '—',
      note: summary ? `Hospital access lost: ${summary.settlements_lost_hospital_access.toLocaleString()}` : 'Awaiting analysis',
      icon: '◎',
      color: 'purple',
    },
  ]

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">≈</div>
          <div>
            <strong>Flood<span>Scope</span></strong>
            <small>FLOOD DAMAGE MAPPER</small>
          </div>
        </div>

        <div className="workspace-label">WORKSPACE</div>
        <nav className="nav-list">
          {navItems.map((item) => (
            <button
              key={item.label}
              className={`nav-item ${active === item.label ? 'active' : ''}`}
              onClick={() => {
                setActive(item.label)
                setNotice('')
              }}
            >
              <span className="nav-icon">{item.icon}</span>
              <span>{item.label}</span>
              {item.label === 'Job History' && <span className="nav-arrow">›</span>}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div
            className="status-dot"
            style={{
              background: backendStatus.connected ? '#10b981' : '#f59e0b',
            }}
          />
          <div>
            <strong>{backendStatus.connected ? 'Backend connected' : 'Backend offline'}</strong>
            <small>
              {backendStatus.connected
                ? `v${backendStatus.version || '0.8.0'} @ ${API_BASE_URL.replace('http://', '')}`
                : 'Run FastAPI on :8000'}
            </small>
          </div>
        </div>
        <div className="profile">
          <div className="avatar">FM</div>
          <div><strong>Mapping workspace</strong><small>Track B · Multimodal AI</small></div>
          <span className="profile-menu">···</span>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumbs">Workspace <span>/</span> <strong>{active}</strong></div>
          <div className="top-actions">
            <span
              className="demo-pill"
              style={{
                background: backendStatus.connected ? '#ecfdf5' : '#fff7e8',
                borderColor: backendStatus.connected ? '#a7f3d0' : '#f2e2bd',
                color: backendStatus.connected ? '#047857' : '#9b6b20',
              }}
            >
              <span
                style={{
                  background: backendStatus.connected ? '#10b981' : '#d69b39',
                }}
              />{' '}
              {backendStatus.connected ? 'API READY' : 'STANDALONE MODE'}
            </span>
            <button
              className="help-button"
              onClick={() =>
                setNotice('FloodScope connects to FastAPI at ' + API_BASE_URL + ' for Sentinel SAR/optical flood mapping.')
              }
            >
              ? Help
            </button>
          </div>
        </header>

        <div className="page-content">
          <section className="welcome-row">
            <div>
              <div className="eyebrow">FLOOD MONITORING & SPATIAL INTELLIGENCE</div>
              <h1>{active === 'Dashboard' ? 'Flood intelligence dashboard' : active}</h1>
              <p className="subtitle">
                Monitor flood extent, explore potential infrastructure exposure, and review AI situation reports.
              </p>
            </div>
            <button
              className="primary-button"
              onClick={() => {
                setActive('New Analysis')
                setNotice('')
              }}
            >
              <span>＋</span> New analysis
            </button>
          </section>

          {notice && (
            <div className="notice">
              <span>ⓘ</span>
              {notice}
              <button onClick={() => setNotice('')}>×</button>
            </div>
          )}

          <section
            className="status-banner"
            style={{
              background: backendStatus.connected ? '#f0fdf4' : '#fff8eb',
              borderColor: backendStatus.connected ? '#bbf7d0' : '#f0dfbc',
            }}
          >
            <div
              className="status-symbol"
              style={{
                color: backendStatus.connected ? '#166534' : '#a87b2e',
                borderColor: backendStatus.connected ? '#86efac' : '#d8b879',
              }}
            >
              {backendStatus.connected ? '✓' : 'i'}
            </div>
            <div>
              <strong>
                {backendStatus.connected
                  ? currentJob
                    ? `Active Analysis: ${currentJob.analysis_id} (${currentJob.status})`
                    : 'Backend connected — Ready for new satellite analysis jobs'
                  : 'Backend not connected — Running in client-side preview mode'}
              </strong>
              <p>
                {backendStatus.connected
                  ? currentJob?.summary
                    ? `Observed ${currentJob.summary.flooded_area_km2.toFixed(2)} km² inundation with ${currentJob.summary.roads_blocked} blocked roads and ${currentJob.summary.isolated_settlements} isolated settlements.`
                    : 'FastAPI service is active. Use "Configure analysis" below to submit real-time jobs.'
                  : 'Connect your local or hosted FastAPI backend to retrieve Sentinel observations and verified job results.'}
              </p>
            </div>
            <span
              className="status-label"
              style={{
                background: backendStatus.connected ? '#dcfce7' : '#f8edcf',
                color: backendStatus.connected ? '#15803d' : '#9c7838',
              }}
            >
              {backendStatus.connected ? 'ONLINE' : 'OFFLINE'}
            </span>
          </section>

          <section className="stats-grid">
            {stats.map((stat) => (
              <article className="stat-card" key={stat.label}>
                <div className={`stat-icon ${stat.color}`}>{stat.icon}</div>
                <div className="stat-label">{stat.label}</div>
                <div className="stat-value">{stat.value}</div>
                <div className="stat-note">{stat.note}</div>
              </article>
            ))}
          </section>

          <section className="content-grid">
            <article className="panel map-panel">
              <div className="panel-heading">
                <div>
                  <h2>Geospatial explorer</h2>
                  <p>
                    {currentJob ? `Layers for Job ${currentJob.analysis_id}` : 'Area of interest and map layers'}
                  </p>
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button
                    className="subtle-button"
                    onClick={() => setLayerFilter(layerFilter === 'all' ? 'water' : 'all')}
                  >
                    Filter: {layerFilter}
                  </button>
                  <button className="subtle-button" onClick={() => setActive('GIS Explorer')}>
                    Open map ↗
                  </button>
                </div>
              </div>

              <div className="map-placeholder">
                <div className="map-grid" />
                <div className="map-water water-one" />
                <div className="map-water water-two" />
                <div className="map-land land-one" />
                <div className="map-land land-two" />
                <div className="map-pin pin-one">●</div>
                <div className="map-pin pin-two">●</div>
                <div className="map-label label-one">
                  {currentJob ? `AOI: ${area}` : 'Area of interest'}
                </div>
                <div className="map-label label-two">
                  {currentJob?.summary ? `${currentJob.summary.flooded_area_km2.toFixed(1)} km² Flood` : 'Sample location'}
                </div>
                <div className="map-disclaimer">
                  {currentJob
                    ? `Verified Sentinel-1 & OSM Layers · Job ${currentJob.analysis_id}`
                    : 'Illustrative map · Connect backend for satellite overlays'}
                </div>
                <div className="map-zoom">
                  <button aria-label="Zoom in">+</button>
                  <button aria-label="Zoom out">−</button>
                </div>
              </div>

              <div className="map-legend">
                <span><i className="legend-dot blue-dot" /> Flood Inundation</span>
                <span><i className="legend-dot green-dot" /> Accessible Roads</span>
                <span><i className="legend-dot red-dot" /> Blocked Roads / Assets</span>
              </div>
            </article>

            <article className="panel analysis-panel">
              <div className="panel-heading">
                <div>
                  <h2>Configure analysis</h2>
                  <p>Define your area and observation date</p>
                </div>
                <span className="step-badge">01 / 03</span>
              </div>

              <form
                onSubmit={(event) => {
                  event.preventDefault()
                  handleRunAnalysis()
                }}
              >
                <label htmlFor="area">Area of interest</label>
                <select
                  id="area"
                  value={area}
                  onChange={(event) => setArea(event.target.value)}
                >
                  <option value="Nepal">Nepal (2026 Reference Case)</option>
                  <option value="Kathmandu Valley">Kathmandu Valley</option>
                  <option value="Terai Region">Terai Region</option>
                </select>

                <label htmlFor="date">Observation date</label>
                <input
                  id="date"
                  type="date"
                  value={date}
                  onChange={(event) => setDate(event.target.value)}
                />

                <label htmlFor="lang">Report Language</label>
                <select
                  id="lang"
                  value={language}
                  onChange={(event) => setLanguage(event.target.value)}
                >
                  <option value="en">English (en)</option>
                  <option value="ne">Nepali (ne)</option>
                </select>

                <label>Data sources</label>
                <div className="source-option">
                  <div className="source-logo sentinel">S1</div>
                  <div className="source-copy">
                    <strong>Sentinel-1 SAR</strong>
                    <small>Radar · Cloud-independent · Active</small>
                  </div>
                  <span className="source-state" style={{ color: '#047857' }}>Ready</span>
                </div>
                <div className="source-option">
                  <div className="source-logo optical">S2</div>
                  <div className="source-copy">
                    <strong>Sentinel-2 optical</strong>
                    <small>Surface reflectance · Optional</small>
                  </div>
                  <span className="source-state">Configured</span>
                </div>

                <button
                  className="primary-button submit-button"
                  type="submit"
                  disabled={isLoading}
                >
                  {isLoading ? 'Processing Pipeline...' : 'Run Analysis'} <span>→</span>
                </button>
                <p className="form-footnote">
                  {backendStatus.connected
                    ? 'Submits job to backend pipeline with automated OSM & SAR inference.'
                    : 'Backend offline: run uvicorn backend.app.main:app --port 8000.'}
                </p>
              </form>
            </article>
          </section>

          <section className="bottom-grid">
            <article className="panel">
              <div className="panel-heading">
                <div>
                  <h2>Analysis workflow</h2>
                  <p>From satellite discovery to impact reporting</p>
                </div>
              </div>
              <div className="workflow">
                {[
                  ['01', 'Satellite discovery', 'Find available observations', currentJob ? 'Complete' : 'Pending'],
                  ['02', 'Flood segmentation', 'Generate flood extent mask', currentJob ? 'Complete' : 'Pending'],
                  ['03', 'Exposure assessment', 'Intersect with infrastructure', currentJob ? 'Complete' : 'Pending'],
                  ['04', 'Connectivity analysis', 'Estimate network disruption', currentJob ? 'Complete' : 'Pending'],
                ].map(([number, title, description, status]) => (
                  <div className="workflow-step" key={number}>
                    <span className="workflow-number">{number}</span>
                    <div><strong>{title}</strong><small>{description}</small></div>
                    <span
                      className="pending-tag"
                      style={{
                        background: status === 'Complete' ? '#dcfce7' : '#f3f4f6',
                        color: status === 'Complete' ? '#15803d' : '#6b7280',
                      }}
                    >
                      {status}
                    </span>
                  </div>
                ))}
              </div>
            </article>

            <article className="panel outputs-panel">
              <div className="panel-heading">
                <div>
                  <h2>Recent activity</h2>
                  <p>Processing jobs and reports</p>
                </div>
                <button className="text-button" onClick={() => setActive('Job History')}>
                  View history →
                </button>
              </div>
              <div className="empty-state">
                <div className="empty-icon">{currentJob ? '✓' : '◷'}</div>
                <strong>
                  {currentJob
                    ? `Job ${currentJob.analysis_id} — ${currentJob.status}`
                    : 'No analysis jobs yet'}
                </strong>
                <p>
                  {currentJob
                    ? `Created: ${new Date(currentJob.created_at).toLocaleString()}`
                    : 'Submit an analysis above to see live geospatial outputs and situation reports.'}
                </p>
                {currentJob && (
                  <button
                    className="subtle-button"
                    onClick={() => setActive('Reports')}
                  >
                    View Situation Report
                  </button>
                )}
              </div>
            </article>
          </section>

          <footer className="footer">
            <span>FloodScope · Flood Damage Mapper (Multimodal AI 2026)</span>
            <span>
              Backend: {API_BASE_URL} <i /> Status:{' '}
              {backendStatus.connected ? 'Connected' : 'Offline'}
            </span>
          </footer>
        </div>
      </main>
    </div>
  )
}

export default App
