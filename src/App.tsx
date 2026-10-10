
import { useState } from 'react'
import './App.css'

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

const stats = [
  { label: 'Flooded area', value: '—', note: 'Awaiting analysis', icon: '◈', color: 'blue' },
  { label: 'Buildings exposed', value: '—', note: 'Awaiting analysis', icon: '⌂', color: 'orange' },
  { label: 'Roads potentially affected', value: '—', note: 'Awaiting analysis', icon: '↗', color: 'teal' },
  { label: 'Isolated settlements', value: '—', note: 'Awaiting analysis', icon: '◎', color: 'purple' },
]

function App() {
  const [active, setActive] = useState('Dashboard')
  const [area, setArea] = useState('Nepal')
  const [date, setDate] = useState('2026-10-10')
  const [notice, setNotice] = useState('')

  function startAnalysis() {
    setActive('New Analysis')
    setNotice('Demo interface only — connect the analysis API to submit a real job.')
  }

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
          <div className="status-dot" />
          <div><strong>Frontend ready</strong><small>Backend not connected</small></div>
        </div>
        <div className="profile">
          <div className="avatar">FM</div>
          <div><strong>Mapping workspace</strong><small>Project dashboard</small></div>
          <span className="profile-menu">···</span>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumbs">Workspace <span>/</span> <strong>{active}</strong></div>
          <div className="top-actions">
            <span className="demo-pill"><span /> DEMO MODE</span>
            <button className="help-button" onClick={() => setNotice('FloodScope displays analysis results when a compatible backend is connected.')}>? Help</button>
          </div>
        </header>

        <div className="page-content">
          <section className="welcome-row">
            <div>
              <div className="eyebrow">FLOOD MONITORING & SPATIAL INTELLIGENCE</div>
              <h1>{active === 'Dashboard' ? 'Flood intelligence dashboard' : active}</h1>
              <p className="subtitle">
                Monitor flood extent, explore potential infrastructure exposure, and review analysis outputs.
              </p>
            </div>
            <button className="primary-button" onClick={startAnalysis}>
              <span>＋</span> New analysis
            </button>
          </section>

          {notice && <div className="notice"><span>ⓘ</span>{notice}<button onClick={() => setNotice('')}>×</button></div>}

          <section className="status-banner">
            <div className="status-symbol">i</div>
            <div>
              <strong>Demo workspace — no live flood analysis yet</strong>
              <p>Connect your team's backend to retrieve satellite observations and verified job results. Figures below are intentionally left blank.</p>
            </div>
            <span className="status-label">NOT CONNECTED</span>
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
                  <p>Area of interest and map layers</p>
                </div>
                <button className="subtle-button" onClick={() => setActive('GIS Explorer')}>Open map ↗</button>
              </div>

              <div className="map-placeholder">
                <div className="map-grid" />
                <div className="map-water water-one" />
                <div className="map-water water-two" />
                <div className="map-land land-one" />
                <div className="map-land land-two" />
                <div className="map-pin pin-one">●</div>
                <div className="map-pin pin-two">●</div>
                <div className="map-label label-one">Area of interest</div>
                <div className="map-label label-two">Sample location</div>
                <div className="map-disclaimer">Illustrative map · Not real flood data</div>
                <div className="map-zoom"><button aria-label="Zoom in">+</button><button aria-label="Zoom out">−</button></div>
              </div>

              <div className="map-legend">
                <span><i className="legend-dot blue-dot" /> Water / rivers</span>
                <span><i className="legend-dot green-dot" /> Land</span>
                <span><i className="legend-dot red-dot" /> Sample marker</span>
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

              <form onSubmit={(event) => { event.preventDefault(); startAnalysis() }}>
                <label htmlFor="area">Area of interest</label>
                <select id="area" value={area} onChange={(event) => setArea(event.target.value)}>
                  <option value="Nepal">Nepal</option>
                  <option value="Kathmandu Valley">Kathmandu Valley</option>
                  <option value="Terai Region">Terai Region</option>
                  <option value="Custom AOI">Custom area (API required)</option>
                </select>

                <label htmlFor="date">Observation date</label>
                <input id="date" type="date" value={date} onChange={(event) => setDate(event.target.value)} />

                <label>Data sources</label>
                <div className="source-option">
                  <div className="source-logo sentinel">S1</div>
                  <div className="source-copy"><strong>Sentinel-1 SAR</strong><small>Radar · Cloud-independent</small></div>
                  <span className="source-state">API needed</span>
                </div>
                <div className="source-option">
                  <div className="source-logo optical">S2</div>
                  <div className="source-copy"><strong>Sentinel-2 optical</strong><small>Surface reflectance</small></div>
                  <span className="source-state">API needed</span>
                </div>

                <button className="primary-button submit-button" type="submit">Configure analysis <span>→</span></button>
                <p className="form-footnote">This demo does not submit a real processing job.</p>
              </form>
            </article>
          </section>

          <section className="bottom-grid">
            <article className="panel">
              <div className="panel-heading">
                <div><h2>Analysis workflow</h2><p>From satellite discovery to impact reporting</p></div>
              </div>
              <div className="workflow">
                {[
                  ['01', 'Satellite discovery', 'Find available observations'],
                  ['02', 'Flood segmentation', 'Generate flood extent mask'],
                  ['03', 'Exposure assessment', 'Intersect with infrastructure'],
                  ['04', 'Connectivity analysis', 'Estimate network disruption'],
                ].map(([number, title, description]) => (
                  <div className="workflow-step" key={number}>
                    <span className="workflow-number">{number}</span>
                    <div><strong>{title}</strong><small>{description}</small></div>
                    <span className="pending-tag">Pending</span>
                  </div>
                ))}
              </div>
            </article>

            <article className="panel outputs-panel">
              <div className="panel-heading">
                <div><h2>Recent activity</h2><p>Processing jobs and reports</p></div>
                <button className="text-button" onClick={() => setActive('Job History')}>View history →</button>
              </div>
              <div className="empty-state">
                <div className="empty-icon">◷</div>
                <strong>No analysis jobs yet</strong>
                <p>Once a backend is connected, your recent jobs and reports will appear here.</p>
                <button className="subtle-button" onClick={startAnalysis}>Start an analysis</button>
              </div>
            </article>
          </section>

          <footer className="footer">
            <span>FloodScope · Flood Damage Mapper</span>
            <span>Frontend prototype <i /> Real data requires backend integration</span>
          </footer>
        </div>
      </main>
    </div>
  )
}

export default App
