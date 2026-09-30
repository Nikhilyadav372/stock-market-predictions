import React, { useEffect, useState } from 'react'
import { predictionsApi } from '../api/client'
import { useStock } from '../context/StockContext'
import { EmptyState, Disclaimer } from '../components/ui'
import { format } from 'date-fns'

export const PredictionHistory: React.FC = () => {
  const { selectedSymbol } = useStock()
  const [predictions, setPredictions] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    predictionsApi.list(selectedSymbol)
      .then(({ data }) => setPredictions(data))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [selectedSymbol])

  return (
    <div className="animate-in" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div>
        <h2 style={{ fontSize: 24, fontWeight: 800, color: '#e2e8f0', fontFamily: 'Plus Jakarta Sans, sans-serif' }}>
          Prediction <span className="gradient-text">History</span>
        </h2>
        <p style={{ color: '#64748b', fontSize: 13 }}>All saved predictions for {selectedSymbol}</p>
      </div>

      {loading ? (
        <div className="skeleton" style={{ height: 300 }} />
      ) : predictions.length === 0 ? (
        <EmptyState message="No predictions saved yet. Go to Forecast and generate predictions first." />
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {predictions.map((p: any) => (
            <div key={p.id} className="glass-card glass-card-hover" style={{ padding: 20 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
                <div>
                  <p style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 4 }}>#{p.id} — {p.model_name}</p>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <span className="badge badge-info">{p.task}</span>
                    <span className="badge badge-neutral">Horizon: {p.horizon}d</span>
                    <span className="badge badge-info">{p.symbol}</span>
                  </div>
                </div>
                <p style={{ fontSize: 12, color: '#64748b' }}>
                  {p.created_at ? format(new Date(p.created_at), 'MMM d, yyyy HH:mm') : '—'}
                </p>
              </div>

              {/* Forecast values */}
              {p.forecast && p.forecast.length > 0 && (
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  {p.forecast.map((f: any) => (
                    <div key={f.period} style={{ background: 'rgba(99,102,241,0.08)', borderRadius: 8, padding: '6px 12px', textAlign: 'center' }}>
                      <p style={{ fontSize: 10, color: '#64748b', marginBottom: 2 }}>Day +{f.period}</p>
                      <p style={{ fontWeight: 700, color: f.direction === 'UP' ? '#22c55e' : f.direction === 'DOWN' ? '#ef4444' : '#e2e8f0', fontSize: 13 }}>
                        {p.task === 'regression' ? `$${f.value?.toFixed(2)}` : (f.direction || (f.value > 0.5 ? 'UP' : 'DOWN'))}
                      </p>
                    </div>
                  ))}
                </div>
              )}

              {/* Metrics */}
              {p.model_metrics && Object.keys(p.model_metrics).length > 0 && (
                <div style={{ marginTop: 10, display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                  {Object.entries(p.model_metrics).filter(([, v]) => v != null).map(([k, v]: [string, any]) => (
                    <div key={k} style={{ fontSize: 11, color: '#64748b' }}>
                      <span style={{ color: '#475569' }}>{k.replace(/_/g, ' ')}: </span>
                      <span style={{ color: '#94a3b8' }}>{typeof v === 'number' ? (v < 1 ? `${(v * 100).toFixed(1)}%` : v.toFixed(4)) : v}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      <Disclaimer />
    </div>
  )
}
