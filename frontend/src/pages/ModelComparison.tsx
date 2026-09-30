import React, { useEffect, useState, useCallback } from 'react'
import { modelsApi } from '../api/client'
import { useStock } from '../context/StockContext'
import { EmptyState, Disclaimer } from '../components/ui'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

export const ModelComparison: React.FC = () => {
  const { selectedSymbol } = useStock()
  const [models, setModels] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const { data } = await modelsApi.list(selectedSymbol)
      setModels(data)
    } catch {} finally { setLoading(false) }
  }, [selectedSymbol])

  useEffect(() => { load() }, [load])

  const regressionModels = models.filter((m) => m.task === 'regression' && m.test_mae != null)
  const classificationModels = models.filter((m) => m.task === 'classification' && m.test_accuracy != null)

  return (
    <div className="animate-in" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div>
        <h2 style={{ fontSize: 24, fontWeight: 800, color: '#e2e8f0', fontFamily: 'Plus Jakarta Sans, sans-serif' }}>
          Model <span className="gradient-text">Comparison</span>
        </h2>
        <p style={{ color: '#64748b', fontSize: 13 }}>Compare performance across all trained models for {selectedSymbol}</p>
      </div>

      {loading && <div className="skeleton" style={{ height: 300 }} />}

      {!loading && models.length === 0 && (
        <EmptyState message="No models trained yet. Go to Forecast → Train Model to get started." />
      )}

      {/* Regression Model Table */}
      {regressionModels.length > 0 && (
        <div className="glass-card" style={{ padding: 24 }}>
          <h3 style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 16 }}>📉 Regression Models</h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr>
                  {['Model', 'Type', 'Feature Set', 'MAE', 'RMSE', 'R²', 'Dir. Accuracy'].map((h) => (
                    <th key={h} style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600, fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.06em', borderBottom: '1px solid rgba(99,102,241,0.12)' }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {regressionModels.map((m, i) => (
                  <tr key={m.id} style={{ background: i % 2 === 0 ? 'rgba(99,102,241,0.03)' : 'transparent' }}>
                    <td style={{ padding: '10px 12px', color: '#e2e8f0', fontWeight: 600 }}>{m.name}</td>
                    <td style={{ padding: '10px 12px', color: '#818cf8' }}>{m.model_type}</td>
                    <td style={{ padding: '10px 12px', color: '#64748b' }}>{m.feature_set}</td>
                    <td style={{ padding: '10px 12px', color: '#e2e8f0' }}>{m.test_mae?.toFixed(4) ?? '—'}</td>
                    <td style={{ padding: '10px 12px', color: '#e2e8f0' }}>{m.test_rmse?.toFixed(4) ?? '—'}</td>
                    <td style={{ padding: '10px 12px', color: m.test_r2 > 0 ? '#22c55e' : '#ef4444' }}>{m.test_r2?.toFixed(4) ?? '—'}</td>
                    <td style={{ padding: '10px 12px' }}>
                      {m.directional_accuracy ? (
                        <span className={m.directional_accuracy > 0.55 ? 'badge badge-positive' : 'badge badge-neutral'}>
                          {(m.directional_accuracy * 100)?.toFixed(1)}%
                        </span>
                      ) : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* MAE Bar Chart */}
          <div style={{ marginTop: 24 }}>
            <p style={{ fontWeight: 600, color: '#e2e8f0', marginBottom: 12 }}>MAE Comparison (lower is better)</p>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={regressionModels.map((m) => ({ name: m.model_type, mae: m.test_mae }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.08)" />
                <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 12 }} />
                <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
                <Tooltip contentStyle={{ background: '#0f0f1a', border: '1px solid rgba(99,102,241,0.3)', borderRadius: 8, color: '#e2e8f0' }} />
                <Bar dataKey="mae" fill="#6366f1" radius={[4, 4, 0, 0]} name="Test MAE" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Classification Model Table */}
      {classificationModels.length > 0 && (
        <div className="glass-card" style={{ padding: 24 }}>
          <h3 style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 16 }}>↕️ Classification Models</h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr>
                  {['Model', 'Type', 'Accuracy', 'F1 Score', 'ROC AUC', 'Dir. Accuracy'].map((h) => (
                    <th key={h} style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600, fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.06em', borderBottom: '1px solid rgba(99,102,241,0.12)' }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {classificationModels.map((m, i) => (
                  <tr key={m.id} style={{ background: i % 2 === 0 ? 'rgba(99,102,241,0.03)' : 'transparent' }}>
                    <td style={{ padding: '10px 12px', color: '#e2e8f0', fontWeight: 600 }}>{m.name}</td>
                    <td style={{ padding: '10px 12px', color: '#818cf8' }}>{m.model_type}</td>
                    <td style={{ padding: '10px 12px', color: m.test_accuracy > 0.55 ? '#22c55e' : '#f59e0b' }}>
                      {m.test_accuracy ? `${(m.test_accuracy * 100)?.toFixed(1)}%` : '—'}
                    </td>
                    <td style={{ padding: '10px 12px', color: '#e2e8f0' }}>{m.test_f1?.toFixed(4) ?? '—'}</td>
                    <td style={{ padding: '10px 12px', color: '#e2e8f0' }}>—</td>
                    <td style={{ padding: '10px 12px' }}>
                      {m.directional_accuracy ? (
                        <span className={m.directional_accuracy > 0.55 ? 'badge badge-positive' : 'badge badge-neutral'}>
                          {(m.directional_accuracy * 100)?.toFixed(1)}%
                        </span>
                      ) : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Info about metrics */}
      <div className="glass-card" style={{ padding: 20 }}>
        <h3 style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 12 }}>📖 Metric Definitions</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 12 }}>
          {[
            { m: 'MAE', d: 'Mean Absolute Error — average absolute difference between predicted and actual prices' },
            { m: 'RMSE', d: 'Root Mean Square Error — penalizes large errors more heavily than MAE' },
            { m: 'R²', d: 'Coefficient of determination — how much variance the model explains (1.0 = perfect)' },
            { m: 'Dir. Accuracy', d: 'Directional Accuracy — fraction of times the model correctly predicted UP or DOWN' },
            { m: 'Accuracy', d: 'Classification accuracy — fraction of correct UP/DOWN predictions' },
            { m: 'F1 Score', d: 'Harmonic mean of precision and recall (important for imbalanced datasets)' },
          ].map(({ m, d }) => (
            <div key={m} style={{ background: 'rgba(99,102,241,0.06)', borderRadius: 8, padding: '10px 14px' }}>
              <p style={{ fontWeight: 700, color: '#818cf8', fontSize: 13, marginBottom: 4 }}>{m}</p>
              <p style={{ fontSize: 11, color: '#64748b', lineHeight: 1.5 }}>{d}</p>
            </div>
          ))}
        </div>
      </div>

      <Disclaimer />
    </div>
  )
}
