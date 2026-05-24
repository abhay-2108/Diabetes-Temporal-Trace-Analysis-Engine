import { useState, useEffect } from 'react'

const API_BASE = 'http://localhost:8000'

function App() {
  const [patients, setPatients] = useState([])
  const [selectedPatient, setSelectedPatient] = useState(null)
  const [analysis, setAnalysis] = useState(null)
  const [loading, setLoading] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  
  // Custom Query States
  const [customMode, setCustomMode] = useState(false)
  const [customNote, setCustomNote] = useState('')

  useEffect(() => {
    fetch(`${API_BASE}/api/patients`)
      .then(res => res.json())
      .then(data => setPatients(data))
      .catch(err => console.error('Failed to load patients:', err))
  }, [])

  const handleAnalyze = async () => {
    if (!selectedPatient) return
    setLoading(true)
    setAnalysis(null)
    setCustomMode(false)
    try {
      const res = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patient_nbr: selectedPatient.patient_nbr }),
      })
      const data = await res.json()
      setAnalysis(data)
    } catch (err) {
      console.error('Analysis failed:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleSelectCustomMode = () => {
    setSelectedPatient(null)
    setAnalysis(null)
    setCustomNote('')
    setCustomMode(true)
  }

  const handleAnalyzeCustom = async () => {
    if (!customNote.trim()) return
    setLoading(true)
    setAnalysis(null)
    try {
      const res = await fetch(`${API_BASE}/api/analyze_custom`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notes: customNote }),
      })
      const data = await res.json()
      setAnalysis(data)
    } catch (err) {
      console.error('Custom note analysis failed:', err)
    } finally {
      setLoading(false)
    }
  }

  const filteredPatients = patients.filter(p =>
    String(p.patient_nbr).includes(searchTerm) ||
    (p.race || '').toLowerCase().includes(searchTerm.toLowerCase()) ||
    (p.gender || '').toLowerCase().includes(searchTerm.toLowerCase())
  )

  const getEgfrColor = (val) => {
    if (val >= 60) return '#10b981'
    if (val >= 30) return '#f59e0b'
    return '#ef4444'
  }

  const isHighA1c = (val) => {
    return val && (val.includes('>7') || val.includes('>8') || val === '8' || val === '7')
  }

  // Parses guideline page citations [Page X]
  const parseCitations = (text) => {
    if (!text) return ''
    const parts = text.split(/(\[Page \d+\])/g)
    return parts.map((part, i) => {
      if (part.match(/\[Page \d+\]/)) {
        return <span key={i} className="page-citation">{part}</span>
      }
      return part
    })
  }

  // Parses markdown double asterisks **bold** and single *bold* and removes leading bullet points
  const renderHighlightedText = (text) => {
    if (!text) return ''
    
    // Remove leading bullet points (* or -)
    let cleanText = text.replace(/^[\*\-]\s+/, '')
    
    // Split by **text** or *text*
    const parts = cleanText.split(/(\*\*.*?\*\*|\*.*?\*)/g)
    return parts.map((part, idx) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return (
          <strong key={idx} style={{ fontWeight: '700', color: '#1a202c' }}>
            {parseCitations(part.slice(2, -2))}
          </strong>
        )
      } else if (part.startsWith('*') && part.endsWith('*')) {
        return (
          <strong key={idx} style={{ fontWeight: '600', color: '#2d3748' }}>
            {parseCitations(part.slice(1, -1))}
          </strong>
        )
      }
      return <span key={idx}>{parseCitations(part)}</span>
    })
  }

  return (
    <div className="app-layout">
      {/* ── Sidebar ─────────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo-container">
            <div className="logo-icon">🧬</div>
            <div className="logo-text">
              <h1>DiaTrace.AI</h1>
              <p>Clinical Decision Engine</p>
            </div>
          </div>
        </div>

        <div className="patient-search">
          <div className="search-wrapper">
            <span className="search-icon">🔍</span>
            <input
              id="patient-search-input"
              type="text"
              className="search-input"
              placeholder="Search patients..."
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
            />
          </div>
        </div>

        {/* ── Custom Unstructured Query Option ────────── */}
        <div style={{ padding: '0 16px 12px 16px' }}>
          <button
            id="custom-note-toggle-btn"
            className="custom-query-toggle-btn"
            onClick={handleSelectCustomMode}
            style={{
              width: '100%',
              padding: '10px 14px',
              borderRadius: '8px',
              border: customMode ? '2px solid #5a67d8' : '2px dashed #cbd5e0',
              background: customMode ? '#ebf8ff' : '#f7fafc',
              color: customMode ? '#5a67d8' : '#4a5568',
              fontWeight: '600',
              fontSize: '0.88rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              boxShadow: customMode ? '0 0 0 3px rgba(90, 103, 216, 0.15)' : 'none',
              transition: 'all 0.2s ease',
            }}
            onMouseOver={e => {
              if (!customMode) {
                e.currentTarget.style.border = '2px solid #5a67d8';
                e.currentTarget.style.color = '#5a67d8';
              }
            }}
            onMouseOut={e => {
              if (!customMode) {
                e.currentTarget.style.border = '2px dashed #cbd5e0';
                e.currentTarget.style.color = '#4a5568';
              }
            }}
          >
            ✍️ New Custom Note Query
          </button>
        </div>

        <div className="patient-list">
          {filteredPatients.map(p => (
            <div
              key={p.patient_nbr}
              id={`patient-${p.patient_nbr}`}
              className={`patient-item ${(!customMode && selectedPatient?.patient_nbr === p.patient_nbr) ? 'selected' : ''}`}
              onClick={() => {
                setCustomMode(false)
                setSelectedPatient(p)
                setAnalysis(null)
              }}
            >
              <div className="patient-avatar">
                {String(p.patient_nbr).slice(-2)}
              </div>
              <div className="patient-info">
                <h3>Patient #{p.patient_nbr}</h3>
                <p>{p.age || '?'} · {p.gender || '?'} · {p.race || '?'}</p>
              </div>
              <div className="patient-meta">
                <span className="visit-badge">{p.visit_count} visits</span>
              </div>
            </div>
          ))}
        </div>

        <button
          id="analyze-btn"
          className="analyze-btn"
          disabled={customMode || !selectedPatient || loading}
          onClick={handleAnalyze}
        >
          {loading ? '⏳ Analyzing...' : '🚀 Run Clinical Analysis'}
        </button>
      </aside>

      {/* ── Main Content ────────────────────────────── */}
      <main className="main-content">
        {/* Custom Note Query Input Form (Empty State - Custom Mode) */}
        {!analysis && !loading && customMode && (
          <div className="custom-note-form-container" style={{ padding: '40px', maxWidth: '800px', margin: '0 auto' }}>
            <div className="card" style={{ padding: '32px', borderRadius: '16px', border: '1px solid #e2e8f0', boxShadow: '0 4px 20px rgba(0,0,0,0.05)', background: '#fff' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '24px' }}>
                <div style={{ fontSize: '2.5rem', background: '#ebf8ff', padding: '12px', borderRadius: '12px', color: '#3182ce' }}>✍️</div>
                <div>
                  <h3 style={{ fontSize: '1.4rem', fontWeight: '700', color: '#2d3748', margin: 0 }}>Clinical Note Auditor</h3>
                  <p style={{ fontSize: '0.9rem', color: '#718096', margin: '4px 0 0 0' }}>Paste raw unstructured SOAP notes to run live multi-agent guideline audits.</p>
                </div>
              </div>
              
              <textarea
                id="custom-note-textarea"
                value={customNote}
                onChange={e => setCustomNote(e.target.value)}
                placeholder="Paste clinician's unstructured SOAP progress note here... (e.g. 'Patient complains of burning feet tingling, vision blurry. Actively taking Metformin 1000mg BID...')"
                style={{
                  width: '100%',
                  height: '240px',
                  padding: '16px',
                  borderRadius: '12px',
                  border: '1px solid #cbd5e0',
                  fontSize: '0.95rem',
                  lineHeight: '1.6',
                  fontFamily: 'inherit',
                  resize: 'vertical',
                  outline: 'none',
                  boxSizing: 'border-box',
                  transition: 'border-color 0.2s',
                  marginBottom: '20px'
                }}
                onFocus={e => e.target.style.borderColor = '#3182ce'}
                onBlur={e => e.target.style.borderColor = '#cbd5e0'}
              />
              
              <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <button
                  id="custom-analyze-btn"
                  onClick={handleAnalyzeCustom}
                  disabled={!customNote.trim()}
                  style={{
                    padding: '12px 28px',
                    background: customNote.trim() ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' : '#cbd5e0',
                    color: '#fff',
                    fontWeight: '600',
                    fontSize: '0.95rem',
                    border: 'none',
                    borderRadius: '8px',
                    cursor: customNote.trim() ? 'pointer' : 'not-allowed',
                    boxShadow: customNote.trim() ? '0 4px 14px rgba(102, 126, 234, 0.4)' : 'none',
                    transition: 'all 0.2s ease',
                  }}
                  onMouseOver={e => { if(customNote.trim()) e.currentTarget.style.transform = 'translateY(-1px)'; }}
                  onMouseOut={e => { e.currentTarget.style.transform = 'translateY(0)'; }}
                >
                  ⚡ Execute Multi-Agent Audit
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Regular Empty State (Select Patient/Custom Note) */}
        {!analysis && !loading && !customMode && (
          <div className="empty-state">
            <div className="empty-icon">🩺</div>
            <h3>Select a Patient or Custom Note</h3>
            <p>Choose a patient from the sidebar, or click &quot;New Custom Note Query&quot; to enter raw unstructured progress text and audit guideline compliance.</p>
          </div>
        )}

        {loading && (
          <div className="loading-overlay">
            <div className="spinner"></div>
            <p className="loading-text">Running multi-agent clinical analysis pipeline...</p>
          </div>
        )}

        {analysis && !loading && (
          <>
            <div className="main-header">
              <h2>Clinical Decision Report — {customMode ? 'Custom Clinical Note' : `Patient #${analysis.patient_nbr}`}</h2>
              <p>Generated by DiaTrace.AI Multi-Agent Pipeline</p>
            </div>

            <div className="report-grid">
              {/* ── Patient Profile Card ──────────── */}
              <div className="card" id="card-profile">
                <div className="card-header">
                  <div className="card-icon purple">👤</div>
                  <div>
                    <div className="card-title">Patient Profile</div>
                    <div className="card-subtitle">Demographics & Overview</div>
                  </div>
                </div>
                <div className="stat-row">
                  <div className="stat-chip">
                    <div className="value">{analysis.demographics.age || '?'}</div>
                    <div className="label">Age Group</div>
                  </div>
                  <div className="stat-chip">
                    <div className="value">{analysis.demographics.gender || '?'}</div>
                    <div className="label">Gender</div>
                  </div>
                  <div className="stat-chip">
                    <div className="value">{analysis.visit_count}</div>
                    <div className="label">Visits</div>
                  </div>
                </div>
              </div>

              {/* ── Biomarker Trajectory Card ────── */}
              <div className="card" id="card-biomarkers">
                <div className="card-header">
                  <div className="card-icon teal">📊</div>
                  <div>
                    <div className="card-title">Biomarker Trajectory</div>
                    <div className="card-subtitle">eGFR Slope & HbA1c History</div>
                  </div>
                </div>
                <div className="stat-row">
                  <div className="stat-chip">
                    <div className="value" style={{ color: analysis.biomarkers.egfr_slope < -5 ? '#ef4444' : analysis.biomarkers.egfr_slope < -1 ? '#f59e0b' : '#10b981' }}>
                      {analysis.biomarkers.egfr_slope}
                    </div>
                    <div className="label">eGFR Slope (mL/min/yr)</div>
                  </div>
                  <div className="stat-chip">
                    <div className="value" style={{ color: getEgfrColor(analysis.biomarkers.latest_egfr || 0) }}>
                      {analysis.biomarkers.latest_egfr ?? 'N/A'}
                    </div>
                    <div className="label">Latest eGFR</div>
                  </div>
                </div>

                {analysis.biomarkers.egfr_trajectory && analysis.biomarkers.egfr_trajectory.length > 0 && (
                  <div className="egfr-chart" style={{ marginTop: '16px' }}>
                    {analysis.biomarkers.egfr_trajectory.map((val, i) => {
                      const maxEgfr = Math.max(...analysis.biomarkers.egfr_trajectory, 100)
                      const heightPct = (val / maxEgfr) * 100
                      return (
                        <div
                          key={i}
                          className="egfr-bar"
                          style={{
                            height: `${heightPct}%`,
                            background: getEgfrColor(val),
                          }}
                        >
                          <span className="bar-label">{val}</span>
                        </div>
                      )
                    })}
                  </div>
                )}

                {analysis.biomarkers.a1c_values && analysis.biomarkers.a1c_values.length > 0 && (
                  <div className="a1c-row">
                    <span style={{ fontSize: '0.72rem', color: '#718096', fontWeight: 500 }}>HbA1c: </span>
                    {analysis.biomarkers.a1c_values.map((val, i) => (
                      <span key={i} className={`a1c-badge ${isHighA1c(val) ? 'high' : 'normal'}`}>
                        {val}
                      </span>
                    ))}
                  </div>
                )}
              </div>

              {/* ── Complications Card ───────────── */}
              <div className="card" id="card-complications">
                <div className="card-header">
                  <div className="card-icon orange">⚠️</div>
                  <div>
                    <div className="card-title">Detected Complications</div>
                    <div className="card-subtitle">Microvascular & Metabolic Flags</div>
                  </div>
                </div>
                <div className="badge-list">
                  {analysis.complications.length === 0 && (
                    <span className="badge success">✅ No complications detected</span>
                  )}
                  {analysis.complications.map((c, i) => (
                    <span key={i} className={`badge ${c.toLowerCase().includes('severe') ? 'danger' : 'warning'}`}>
                      {c.toLowerCase().includes('severe') ? '🔴' : '🟡'} {c}
                    </span>
                  ))}
                </div>
              </div>

              {/* ── ADA Guideline Clashes Card ──── */}
              <div className="card" id="card-clashes">
                <div className="card-header">
                  <div className="card-icon red">🚨</div>
                  <div>
                    <div className="card-title">ADA Guideline Clashes</div>
                    <div className="card-subtitle">Contraindications & Safety Alerts</div>
                  </div>
                </div>
                {analysis.ada_clashes.length === 0 && (
                  <div style={{ color: '#10b981', fontSize: '0.85rem', fontWeight: 500 }}>
                    ✅ No guideline clashes detected.
                  </div>
                )}
                {analysis.ada_clashes.map((clash, i) => (
                  <div key={i} className="clash-item">
                    <span className="clash-icon">⚡</span>
                    <span>{renderHighlightedText(clash)}</span>
                  </div>
                ))}
              </div>

              {/* ── Evidence Chain Card ──────────── */}
              <div className="card full-width" id="card-evidence">
                <div className="card-header">
                  <div className="card-icon blue">🔗</div>
                  <div>
                    <div className="card-title">Causal Evidence Chain</div>
                    <div className="card-subtitle">Human-Auditable Reasoning Trace with Source Citations</div>
                  </div>
                </div>
                <div className="evidence-timeline">
                  {analysis.evidence_chain.map((ev, i) => (
                    <div key={i} className="evidence-node">
                      <div className="evidence-dot"></div>
                      <div className="evidence-text">
                        {renderHighlightedText(ev)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* ── Recommendations Card ─────────── */}
              <div className="card full-width" id="card-recommendations">
                <div className="card-header">
                  <div className="card-icon pink">💊</div>
                  <div>
                    <div className="card-title">Guideline-Aligned Recommendations</div>
                    <div className="card-subtitle">Actionable Clinical Interventions</div>
                  </div>
                </div>
                {analysis.recommendations.length === 0 && (
                  <div style={{ color: '#718096', fontSize: '0.85rem' }}>
                    No specific recommendations generated. Review the raw report below.
                  </div>
                )}
                {analysis.recommendations.map((rec, i) => (
                  <div key={i} className="recommendation-item">
                    <div className="rec-number">{i + 1}</div>
                    <span>{renderHighlightedText(rec)}</span>
                  </div>
                ))}
              </div>

            </div>

            {/* Custom Mode Reset Button */}
            {customMode && (
              <div style={{ display: 'flex', justifyContent: 'center', marginTop: '32px', marginBottom: '48px' }}>
                <button
                  onClick={() => {
                    setAnalysis(null)
                    setCustomNote('')
                  }}
                  style={{
                    padding: '12px 32px',
                    background: '#edf2f7',
                    color: '#4a5568',
                    fontWeight: '600',
                    border: 'none',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
                    transition: 'background 0.2s',
                  }}
                  onMouseOver={e => e.currentTarget.style.background = '#e2e8f0'}
                  onMouseOut={e => e.currentTarget.style.background = '#edf2f7'}
                >
                  🧹 Clear & New Query
                </button>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  )
}

export default App
