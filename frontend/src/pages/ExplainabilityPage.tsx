import React, { useState, useEffect } from 'react'
import { predictionsApi } from '../api/client'
import { useStock } from '../context/StockContext'
import { EmptyState, Disclaimer } from '../components/ui'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import toast from 'react-hot-toast'

export const ExplainabilityPage: React.FC = () => {
  const { selectedSymbol } = useStock()
  const [predictions, setPredictions] = useState<any[]>([])
  const [selectedPred, setSelectedPred] = useState<number | ''>('')
  const [explanation, setExplanation] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    predictionsApi.list(selectedSymbol).then(async ({ data }) => {
      let list = data.filter((p: any) => p.id)
      if (list.length === 0) {
        try {
          const resAll = await predictionsApi.list()
          list = resAll.data.filter((p: any) => p.id)
        } catch {}
      }
      setPredictions(list)
      if (list.length > 0) setSelectedPred(list[0].id)
      else setSelectedPred('')
    }).catch(() => {})
  }, [selectedSymbol])

  const handleFetchExplanation = async () => {
    if (!selectedPred) return toast.error('Select a prediction first')
    setLoading(true)
    try {
      const { data } = await predictionsApi.explain(+selectedPred)
      setExplanation(data)
    } catch (err: any) {
      const msg = err.response?.data?.detail || 'Failed to load explanation'
      toast.error(msg)
      setExplanation(null)
    } finally { setLoading(false) }
  }

  const posFeatures = explanation?.top_positive_features?.slice(0, 10) ?? []
  const negFeatures = explanation?.top_negative_features?.slice(0, 10) ?? []

  const chartData = [
    ...posFeatures.map((f: any) => ({
      feature: f.feature.replace(/_/g, ' '),
      shap_value: f.shap_value,
      color: '#22c55e',
    })),
    ...negFeatures.map((f: any) => ({
      feature: f.feature.replace(/_/g, ' '),
      shap_value: f.shap_value,
      color: '#ef4444',
    })),
  ].sort((a, b) => b.shap_value - a.shap_value)

  const importanceData = explanation?.feature_importance
    ? Object.entries(explanation.feature_importance as Record<string, number>)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 15)
        .map(([k, v]) => ({ feature: k.replace(/_/g, ' '), importance: v }))
    : []

  return (
    <div className="animate-in" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div>
        <h2 style={{ fontSize: 24, fontWeight: 800, color: '#e2e8f0', fontFamily: 'Plus Jakarta Sans, sans-serif' }}>
          Model <span className="gradient-text">Explainability</span>
        </h2>
        <p style={{ color: '#64748b', fontSize: 13 }}>
          SHAP-based feature attribution for tree & linear models — understand what drives each prediction
        </p>
      </div>

      {/* Controls */}
      <div className="glass-card" style={{ padding: 20, display: 'flex', alignItems: 'flex-end', gap: 16 }}>
        <div style={{ flex: 1 }}>
          <label style={{ fontSize: 12, color: '#64748b', marginBottom: 6, display: 'block' }}>Select Prediction</label>
          <select
            value={selectedPred}
            onChange={(e) => { setSelectedPred(e.target.value ? +e.target.value : ''); setExplanation(null) }}
            id="prediction-select"
            style={{ width: '100%', background: '#141428', border: '1px solid rgba(99,102,241,0.25)', borderRadius: 8, padding: '10px 12px', color: '#e2e8f0', fontSize: 13 }}
          >
            {predictions.length === 0 && <option value="">No predictions found in database</option>}
            {predictions.map((p) => (
              <option key={p.id} value={p.id}>
                #{p.id} — {p.symbol || p.stock_symbol || selectedSymbol} — {p.model_name || 'Model'} ({p.task}) — {p.created_at?.slice(0, 10)}
              </option>
            ))}
          </select>
        </div>
        <button
          className="btn-primary"
          onClick={handleFetchExplanation}
          disabled={loading || !selectedPred}
          id="load-explanation-btn"
        >
          {loading ? '⏳ Loading...' : '🔍 Load Explanation'}
        </button>
      </div>

      {predictions.length === 0 && (
        <div className="glass-card" style={{ padding: 24, textAlign: 'center' }}>
          <p style={{ fontSize: 16, fontWeight: 700, color: '#e2e8f0', marginBottom: 8 }}>
            No Saved Predictions Found
          </p>
          <p style={{ color: '#64748b', fontSize: 13, marginBottom: 16 }}>
            To view SHAP feature explainability, please first train a model and click <strong>"Generate Forecast"</strong> under the <strong>Forecast → Predict</strong> tab.
          </p>
        </div>
      )}

      {!explanation && !loading && (
        <EmptyState
          message="Select a prediction and click 'Load Explanation' to view SHAP feature attributions.
Note: Explanations are available for XGBoost, Random Forest, and Linear models."
        />
      )}

      {explanation && (
        <>
          {/* Explanation Text */}
          <div className="glass-card" style={{ padding: 20 }}>
            <p style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 8 }}>💡 Model Explanation</p>
            <p style={{ color: '#94a3b8', lineHeight: 1.7, fontSize: 14 }}>{explanation.explanation_text}</p>
          </div>

          <div className="disclaimer-banner">
            <span>ℹ️</span>
            <span>{explanation.disclaimer}</span>
          </div>

          {/* SHAP Values Chart */}
          {chartData.length > 0 && (
            <div className="glass-card" style={{ padding: 24 }}>
              <p style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 16 }}>SHAP Feature Contributions</p>
              <p style={{ fontSize: 12, color: '#64748b', marginBottom: 16 }}>
                Positive values push the prediction higher; negative values push it lower
              </p>
              <ResponsiveContainer width="100%" height={Math.max(200, chartData.length * 30)}>
                <BarChart data={chartData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.08)" />
                  <XAxis type="number" tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={(v) => v?.toFixed(4)} />
                  <YAxis type="category" dataKey="feature" tick={{ fill: '#94a3b8', fontSize: 11 }} width={160} />
                  <Tooltip
                    formatter={(v: any) => v?.toFixed(6)}
                    contentStyle={{ background: '#0f0f1a', border: '1px solid rgba(99,102,241,0.3)', borderRadius: 8, color: '#e2e8f0' }}
                  />
                  <Bar dataKey="shap_value" name="SHAP Value" radius={[0, 4, 4, 0]}
                    fill="#6366f1"
                    label={{ position: 'right', fill: '#64748b', fontSize: 10 }}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Feature Importance */}
          {importanceData.length > 0 && (
            <div className="glass-card" style={{ padding: 24 }}>
              <p style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 16 }}>Mean |SHAP| Feature Importance</p>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={importanceData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.08)" />
                  <XAxis type="number" tick={{ fill: '#64748b', fontSize: 11 }} />
                  <YAxis type="category" dataKey="feature" tick={{ fill: '#94a3b8', fontSize: 11 }} width={160} />
                  <Tooltip contentStyle={{ background: '#0f0f1a', border: '1px solid rgba(99,102,241,0.3)', borderRadius: 8, color: '#e2e8f0' }} />
                  <Bar dataKey="importance" fill="#818cf8" radius={[0, 4, 4, 0]} name="Mean |SHAP|" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}

      <Disclaimer />
    </div>
  )
}
