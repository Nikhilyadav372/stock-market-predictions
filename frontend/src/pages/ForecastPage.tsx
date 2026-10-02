import React, { useState, useEffect, useCallback } from 'react'
import { Brain, Play, Clock, CheckCircle, XCircle } from 'lucide-react'
import { modelsApi, predictionsApi } from '../api/client'
import { useStock } from '../context/StockContext'
import { EmptyState, Spinner, StatusBadge, Disclaimer } from '../components/ui'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'
import toast from 'react-hot-toast'

const MODEL_TYPES = [
  { value: 'naive', label: 'Naive Baseline', desc: 'Previous close as next prediction' },
  { value: 'linear', label: 'Linear Regression', desc: 'Ridge regression on feature matrix' },
  { value: 'random_forest', label: 'Random Forest', desc: '100 decision trees, no shuffle' },
  { value: 'xgboost', label: 'XGBoost', desc: 'Gradient boosting with early stopping' },
  { value: 'lstm', label: 'LSTM', desc: 'Long Short-Term Memory (PyTorch)' },
  { value: 'gru', label: 'GRU', desc: 'Gated Recurrent Unit (PyTorch)' },
]

export const ForecastPage: React.FC = () => {
  const { selectedSymbol } = useStock()
  const [models, setModels] = useState<any[]>([])
  const [selectedModel, setSelectedModel] = useState<number | null>(null)
  const [horizon, setHorizon] = useState(5)
  const [trainConfig, setTrainConfig] = useState({
    model_type: 'xgboost',
    task: 'regression',
    feature_set: 'market',
    lookback_window: 60,
  })
  const [training, setTraining] = useState(false)
  const [runId, setRunId] = useState<number | null>(null)
  const [runStatus, setRunStatus] = useState<any>(null)
  const [predicting, setPredicting] = useState(false)
  const [forecast, setForecast] = useState<any>(null)
  const [loadingModels, setLoadingModels] = useState(false)
  const [activeTab, setActiveTab] = useState<'train' | 'predict'>('train')

  const loadModels = useCallback(async () => {
    setLoadingModels(true)
    try {
      const { data } = await modelsApi.list(selectedSymbol)
      const valid = data.filter((m: any) => m.has_artifact)
      setModels(valid)
      if (valid.length > 0) {
        setSelectedModel((prev) => (valid.some((m: any) => m.id === prev) ? prev : valid[0].id))
      } else {
        setSelectedModel(null)
      }
    } catch {} finally { setLoadingModels(false) }
  }, [selectedSymbol])

  useEffect(() => { loadModels() }, [loadModels])

  // Poll training status
  useEffect(() => {
    if (!runId) return
    const interval = setInterval(async () => {
      try {
        const { data } = await modelsApi.getRunStatus(runId)
        setRunStatus(data)
        if (data.status === 'done' || data.status === 'failed') {
          clearInterval(interval)
          setTraining(false)
          if (data.status === 'done') {
            toast.success('Model training complete!')
            loadModels()
          } else {
            toast.error(`Training failed: ${data.error_message}`)
          }
        }
      } catch {}
    }, 3000)
    return () => clearInterval(interval)
  }, [runId, loadModels])

  const handleTrain = async () => {
    setTraining(true)
    setRunStatus(null)
    try {
      const { data } = await modelsApi.train({ symbol: selectedSymbol, ...trainConfig })
      setRunId(data.run_id)
      toast('Training started in background...', { icon: '🚀' })
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to start training')
      setTraining(false)
    }
  }

  const handlePredict = async () => {
    if (!selectedModel) return toast.error('Select a trained model first')
    setPredicting(true)
    try {
      const { data } = await predictionsApi.predict({
        symbol: selectedSymbol,
        model_id: selectedModel,
        horizon,
      })
      setForecast(data)
      toast.success('Forecast generated!')
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Prediction failed')
    } finally { setPredicting(false) }
  }

  return (
    <div className="animate-in" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div>
        <h2 style={{ fontSize: 24, fontWeight: 800, color: '#e2e8f0', fontFamily: 'Plus Jakarta Sans, sans-serif' }}>
          <span className="gradient-text">AI Forecast</span> Engine
        </h2>
        <p style={{ color: '#64748b', fontSize: 13 }}>Train ML models and generate price forecasts for {selectedSymbol}</p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, background: 'rgba(20,20,40,0.6)', padding: 4, borderRadius: 12, border: '1px solid rgba(99,102,241,0.15)', width: 'fit-content' }}>
        {['train', 'predict'].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab as any)}
            style={{
              padding: '8px 20px',
              borderRadius: 9,
              fontSize: 13,
              fontWeight: 600,
              border: 'none',
              cursor: 'pointer',
              background: activeTab === tab ? 'linear-gradient(135deg, #4f46e5, #6366f1)' : 'transparent',
              color: activeTab === tab ? 'white' : '#64748b',
              transition: 'all 0.2s',
            }}
          >
            {tab === 'train' ? '🧠 Train Model' : '📈 Generate Forecast'}
          </button>
        ))}
      </div>

      {activeTab === 'train' && (
        <div className="grid-responsive-2" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
          {/* Train Config */}
          <div className="glass-card" style={{ padding: 24 }}>
            <h3 style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 20, display: 'flex', alignItems: 'center', gap: 8 }}>
              <Brain size={18} style={{ color: '#818cf8' }} /> Training Configuration
            </h3>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <div>
                <label style={{ fontSize: 12, color: '#64748b', marginBottom: 6, display: 'block' }}>Model Type</label>
                <select
                  value={trainConfig.model_type}
                  onChange={(e) => setTrainConfig({ ...trainConfig, model_type: e.target.value })}
                  id="model-type-select"
                  style={{ width: '100%', background: '#141428', border: '1px solid rgba(99,102,241,0.25)', borderRadius: 8, padding: '10px 12px', color: '#e2e8f0', fontSize: 13 }}
                >
                  {MODEL_TYPES.map((m) => (
                    <option key={m.value} value={m.value}>{m.label}</option>
                  ))}
                </select>
                <p style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>
                  {MODEL_TYPES.find((m) => m.value === trainConfig.model_type)?.desc}
                </p>
              </div>

              <div>
                <label style={{ fontSize: 12, color: '#64748b', marginBottom: 6, display: 'block' }}>Prediction Task</label>
                <div style={{ display: 'flex', gap: 8 }}>
                  {['regression', 'classification'].map((t) => (
                    <button
                      key={t}
                      onClick={() => setTrainConfig({ ...trainConfig, task: t })}
                      style={{
                        flex: 1, padding: '8px', borderRadius: 8, fontSize: 12, fontWeight: 600,
                        border: '1px solid rgba(99,102,241,0.25)', cursor: 'pointer',
                        background: trainConfig.task === t ? 'rgba(99,102,241,0.25)' : 'rgba(99,102,241,0.05)',
                        color: trainConfig.task === t ? '#818cf8' : '#64748b',
                      }}
                    >
                      {t === 'regression' ? '📉 Regression' : '↕️ Classification'}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label style={{ fontSize: 12, color: '#64748b', marginBottom: 6, display: 'block' }}>Feature Set</label>
                <div style={{ display: 'flex', gap: 8 }}>
                  {['market', 'market+sentiment'].map((fs) => (
                    <button
                      key={fs}
                      onClick={() => setTrainConfig({ ...trainConfig, feature_set: fs })}
                      style={{
                        flex: 1, padding: '8px', borderRadius: 8, fontSize: 12, fontWeight: 600,
                        border: '1px solid rgba(99,102,241,0.25)', cursor: 'pointer',
                        background: trainConfig.feature_set === fs ? 'rgba(99,102,241,0.25)' : 'rgba(99,102,241,0.05)',
                        color: trainConfig.feature_set === fs ? '#818cf8' : '#64748b',
                      }}
                    >
                      {fs === 'market' ? '📊 Market Only' : '📰 + Sentiment'}
                    </button>
                  ))}
                </div>
              </div>

              {['lstm', 'gru'].includes(trainConfig.model_type) && (
                <div>
                  <label style={{ fontSize: 12, color: '#64748b', marginBottom: 6, display: 'block' }}>
                    Lookback Window: {trainConfig.lookback_window} days
                  </label>
                  <input
                    type="range" min={20} max={120} step={10}
                    value={trainConfig.lookback_window}
                    onChange={(e) => setTrainConfig({ ...trainConfig, lookback_window: +e.target.value })}
                    style={{ width: '100%', accentColor: '#6366f1' }}
                  />
                </div>
              )}

              <button
                className="btn-primary"
                onClick={handleTrain}
                disabled={training}
                id="train-model-btn"
                style={{ marginTop: 8 }}
              >
                {training ? <><Spinner size={14} /> Training...</> : <><Play size={14} /> Start Training</>}
              </button>
            </div>
          </div>

          {/* Training Status */}
          <div className="glass-card" style={{ padding: 24 }}>
            <h3 style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 20 }}>Training Status</h3>
            {!runStatus && !training && (
              <div style={{ textAlign: 'center', padding: 40, color: '#64748b' }}>
                <Brain size={40} style={{ margin: '0 auto 12px', opacity: 0.3 }} />
                <p>Configure and start training to see status</p>
              </div>
            )}
            {training && !runStatus && (
              <div style={{ textAlign: 'center', padding: 40 }}>
                <Spinner size={32} />
                <p style={{ color: '#818cf8', marginTop: 12, fontWeight: 600 }}>Initializing training...</p>
              </div>
            )}
            {runStatus && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  {runStatus.status === 'running' && <Spinner size={18} />}
                  {runStatus.status === 'done' && <CheckCircle size={18} style={{ color: '#22c55e' }} />}
                  {runStatus.status === 'failed' && <XCircle size={18} style={{ color: '#ef4444' }} />}
                  <StatusBadge status={runStatus.status} />
                  {runStatus.duration_seconds && (
                    <span style={{ fontSize: 12, color: '#64748b', marginLeft: 'auto' }}>
                      <Clock size={12} style={{ display: 'inline', marginRight: 4 }} />
                      {runStatus.duration_seconds?.toFixed(1)}s
                    </span>
                  )}
                </div>

                {runStatus.status === 'done' && (
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 8 }}>
                    {[
                      { label: 'Test MAE', value: runStatus.test_mae?.toFixed(4) },
                      { label: 'Test RMSE', value: runStatus.test_rmse?.toFixed(4) },
                      { label: 'Test R²', value: runStatus.test_r2?.toFixed(4) },
                      { label: 'Dir. Accuracy', value: runStatus.directional_accuracy ? `${(runStatus.directional_accuracy * 100)?.toFixed(1)}%` : null },
                      { label: 'Accuracy', value: runStatus.test_accuracy ? `${(runStatus.test_accuracy * 100)?.toFixed(1)}%` : null },
                      { label: 'F1 Score', value: runStatus.test_f1?.toFixed(4) },
                    ].filter((m) => m.value != null).map((m) => (
                      <div key={m.label} style={{ background: 'rgba(99,102,241,0.08)', borderRadius: 8, padding: '10px 14px' }}>
                        <p style={{ fontSize: 10, color: '#64748b', marginBottom: 4 }}>{m.label}</p>
                        <p style={{ fontWeight: 700, color: '#818cf8', fontSize: 16 }}>{m.value}</p>
                      </div>
                    ))}
                  </div>
                )}

                {runStatus.error_message && (
                  <div className="sample-notice">Error: {runStatus.error_message}</div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'predict' && (
        <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: 20 }}>
          {/* Predict Controls */}
          <div className="glass-card" style={{ padding: 24 }}>
            <h3 style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 20 }}>Forecast Settings</h3>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <div>
                <label style={{ fontSize: 12, color: '#64748b', marginBottom: 6, display: 'block' }}>Trained Model</label>
                {loadingModels ? (
                  <div className="skeleton" style={{ height: 40 }} />
                ) : models.length === 0 ? (
                  <p style={{ fontSize: 12, color: '#f87171', padding: '8px 0' }}>
                    No trained models found. Train a model first.
                  </p>
                ) : (
                  <select
                    value={selectedModel ?? ''}
                    onChange={(e) => setSelectedModel(+e.target.value)}
                    id="model-select"
                    style={{ width: '100%', background: '#141428', border: '1px solid rgba(99,102,241,0.25)', borderRadius: 8, padding: '10px 12px', color: '#e2e8f0', fontSize: 13 }}
                  >
                    {models.map((m) => (
                      <option key={m.id} value={m.id}>{m.name} ({m.task})</option>
                    ))}
                  </select>
                )}
              </div>

              <div>
                <label style={{ fontSize: 12, color: '#64748b', marginBottom: 6, display: 'block' }}>
                  Forecast Horizon: {horizon} periods
                </label>
                <div style={{ display: 'flex', gap: 6 }}>
                  {[1, 5, 10, 20].map((h) => (
                    <button
                      key={h}
                      onClick={() => setHorizon(h)}
                      style={{
                        flex: 1, padding: '8px 4px', borderRadius: 8, fontSize: 12, fontWeight: 600,
                        border: '1px solid rgba(99,102,241,0.25)', cursor: 'pointer',
                        background: horizon === h ? 'rgba(99,102,241,0.3)' : 'rgba(99,102,241,0.05)',
                        color: horizon === h ? '#818cf8' : '#64748b',
                      }}
                    >
                      {h}d
                    </button>
                  ))}
                </div>
              </div>

              <button
                className="btn-primary"
                onClick={handlePredict}
                disabled={predicting || !selectedModel}
                id="generate-forecast-btn"
              >
                {predicting ? <><Spinner size={14} /> Generating...</> : <>🔮 Generate Forecast</>}
              </button>
            </div>
          </div>

          {/* Forecast Chart */}
          <div className="glass-card" style={{ padding: 24 }}>
            <h3 style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 8 }}>Forecast Results</h3>
            {forecast ? (
              <>
                <div style={{ display: 'flex', gap: 16, marginBottom: 16 }}>
                  <div>
                    <span className="badge badge-info">{forecast.model_name}</span>
                  </div>
                  <span style={{ fontSize: 12, color: '#64748b' }}>
                    Horizon: {forecast.horizon} periods
                  </span>
                </div>
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={forecast.forecast}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.08)" />
                    <XAxis dataKey="period" tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={(v) => `Day +${v}`} />
                    <YAxis tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={(v) => `$${v?.toFixed(0)}`} width={65} />
                    <Tooltip formatter={(v: any) => `$${v?.toFixed(4)}`} contentStyle={{ background: '#0f0f1a', border: '1px solid rgba(99,102,241,0.3)', borderRadius: 8, color: '#e2e8f0' }} />
                    <Legend />
                    <Line type="monotone" dataKey="value" stroke="#6366f1" strokeWidth={2} dot={{ fill: '#6366f1', r: 5 }} name="Forecast" />
                    <Line type="monotone" dataKey="upper_bound" stroke="#22c55e" strokeWidth={1} strokeDasharray="4 2" dot={false} name="Upper Bound" />
                    <Line type="monotone" dataKey="lower_bound" stroke="#ef4444" strokeWidth={1} strokeDasharray="4 2" dot={false} name="Lower Bound" />
                  </LineChart>
                </ResponsiveContainer>

                {/* Metrics */}
                {forecast.model_metrics && Object.keys(forecast.model_metrics).length > 0 && (
                  <div style={{ marginTop: 16, display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8 }}>
                    {Object.entries(forecast.model_metrics).filter(([, v]) => v != null).map(([k, v]: [string, any]) => (
                      <div key={k} style={{ background: 'rgba(99,102,241,0.08)', borderRadius: 8, padding: '8px 12px', textAlign: 'center' }}>
                        <p style={{ fontSize: 10, color: '#64748b', marginBottom: 2 }}>{k.replace(/_/g, ' ').toUpperCase()}</p>
                        <p style={{ fontWeight: 700, color: '#818cf8', fontSize: 14 }}>
                          {typeof v === 'number' ? (v < 1 && v > 0 ? `${(v * 100).toFixed(1)}%` : v.toFixed(4)) : v}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <EmptyState message="Configure a model and horizon, then click Generate Forecast" />
            )}
          </div>
        </div>
      )}

      <Disclaimer />
    </div>
  )
}
