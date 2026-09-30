import React, { useState } from 'react'
import { backtestApi, modelsApi } from '../api/client'
import { useStock } from '../context/StockContext'
import { Spinner, EmptyState } from '../components/ui'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts'
import toast from 'react-hot-toast'
import { format, subMonths, subYears } from 'date-fns'

export const BacktestPage: React.FC = () => {
  const { selectedSymbol } = useStock()
  const [models, setModels] = React.useState<any[]>([])
  const [selectedModel, setSelectedModel] = useState<number | ''>('')
  const [startDate, setStartDate] = useState(format(subMonths(new Date(), 6), 'yyyy-MM-dd'))
  const [endDate, setEndDate] = useState(format(new Date(), 'yyyy-MM-dd'))
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<any>(null)

  React.useEffect(() => {
    modelsApi.list(selectedSymbol).then(({ data }) => {
      const ready = data.filter((m: any) => m.has_artifact)
      setModels(ready)
      if (ready.length > 0) setSelectedModel(ready[0].id)
    }).catch(() => {})
  }, [selectedSymbol])

  const handleRun = async () => {
    if (!selectedModel) return toast.error('Select a model first')
    setRunning(true)
    try {
      const { data } = await backtestApi.run({
        symbol: selectedSymbol,
        model_id: +selectedModel,
        start_date: startDate,
        end_date: endDate,
      })
      setResult(data)
      toast.success('Backtest complete!')
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Backtest failed')
    } finally { setRunning(false) }
  }

  const metrics = result ? [
    { label: 'Total Trades', value: result.total_trades, unit: '' },
    { label: 'Win Rate', value: result.win_rate != null ? `${(result.win_rate * 100).toFixed(1)}%` : '—', unit: '' },
    { label: 'Cumulative Return', value: result.cumulative_return != null ? `${result.cumulative_return.toFixed(2)}%` : '—', unit: '', color: result.cumulative_return >= 0 ? '#22c55e' : '#ef4444' },
    { label: 'Max Drawdown', value: result.max_drawdown != null ? `${result.max_drawdown.toFixed(2)}%` : '—', unit: '', color: '#ef4444' },
    { label: 'Sharpe Ratio', value: result.sharpe_ratio != null ? result.sharpe_ratio.toFixed(3) : '—', unit: '' },
  ] : []

  return (
    <div className="animate-in" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div>
        <h2 style={{ fontSize: 24, fontWeight: 800, color: '#e2e8f0', fontFamily: 'Plus Jakarta Sans, sans-serif' }}>
          Historical <span className="gradient-text">Backtesting</span>
        </h2>
        <p style={{ color: '#64748b', fontSize: 13 }}>Simulate a directional trading strategy using trained ML predictions</p>
      </div>

      {/* Disclaimer */}
      <div className="disclaimer-banner">
        <span>⚠️</span>
        <span>
          <strong>Educational Backtest Only</strong> — This simulation does not account for
          transaction costs, slippage, bid-ask spread, taxes, or real-world execution constraints.
          Past simulated performance does NOT predict future real-world returns.
        </span>
      </div>

      <div className="grid-responsive-2" style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: 20 }}>
        {/* Config Panel */}
        <div className="glass-card" style={{ padding: 24 }}>
          <h3 style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 20 }}>Configuration</h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div>
              <label style={{ fontSize: 12, color: '#64748b', marginBottom: 6, display: 'block' }}>Model</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(+e.target.value)}
                id="backtest-model-select"
                style={{ width: '100%', background: '#141428', border: '1px solid rgba(99,102,241,0.25)', borderRadius: 8, padding: '10px 12px', color: '#e2e8f0', fontSize: 13 }}
              >
                {models.length === 0 && <option value="">No trained models</option>}
                {models.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
              </select>
            </div>

            <div>
              <label style={{ fontSize: 12, color: '#64748b', marginBottom: 6, display: 'block' }}>Start Date</label>
              <input
                type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)}
                id="backtest-start-date"
                style={{ width: '100%', background: '#141428', border: '1px solid rgba(99,102,241,0.25)', borderRadius: 8, padding: '10px 12px', color: '#e2e8f0', fontSize: 13 }}
              />
            </div>

            <div>
              <label style={{ fontSize: 12, color: '#64748b', marginBottom: 6, display: 'block' }}>End Date</label>
              <input
                type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)}
                id="backtest-end-date"
                style={{ width: '100%', background: '#141428', border: '1px solid rgba(99,102,241,0.25)', borderRadius: 8, padding: '10px 12px', color: '#e2e8f0', fontSize: 13 }}
              />
            </div>

            {/* Quick range buttons */}
            <div>
              <label style={{ fontSize: 12, color: '#64748b', marginBottom: 6, display: 'block' }}>Quick Range</label>
              <div style={{ display: 'flex', gap: 6 }}>
                {[
                  { label: '3M', months: 3 }, { label: '6M', months: 6 },
                  { label: '1Y', years: 1 }, { label: '2Y', years: 2 },
                ].map((r) => (
                  <button
                    key={r.label}
                    onClick={() => {
                      const end = new Date()
                      const start = r.months ? subMonths(end, r.months) : subYears(end, r.years!)
                      setStartDate(format(start, 'yyyy-MM-dd'))
                      setEndDate(format(end, 'yyyy-MM-dd'))
                    }}
                    style={{ flex: 1, padding: '6px', borderRadius: 6, fontSize: 11, fontWeight: 600, border: '1px solid rgba(99,102,241,0.2)', background: 'rgba(99,102,241,0.08)', color: '#64748b', cursor: 'pointer' }}
                  >
                    {r.label}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label style={{ fontSize: 12, color: '#64748b', marginBottom: 6, display: 'block' }}>Strategy</label>
              <div style={{ background: 'rgba(99,102,241,0.08)', border: '1px solid rgba(99,102,241,0.15)', borderRadius: 8, padding: '10px 14px', fontSize: 13, color: '#94a3b8' }}>
                📈 Directional (Long-Flat)
                <p style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>Buy when model predicts UP, stay flat when DOWN</p>
              </div>
            </div>

            <button
              className="btn-primary"
              onClick={handleRun}
              disabled={running || !selectedModel}
              id="run-backtest-btn"
            >
              {running ? <><Spinner size={14} /> Running...</> : <>▶ Run Backtest</>}
            </button>
          </div>
        </div>

        {/* Results Panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {result ? (
            <>
              {/* Metrics */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12 }}>
                {metrics.map((m) => (
                  <div key={m.label} className="glass-card" style={{ padding: 16, textAlign: 'center' }}>
                    <p style={{ fontSize: 10, color: '#64748b', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{m.label}</p>
                    <p style={{ fontWeight: 800, fontSize: 20, color: (m as any).color || '#e2e8f0', fontFamily: 'Plus Jakarta Sans, sans-serif' }}>{m.value}</p>
                  </div>
                ))}
              </div>

              {/* Equity Curve */}
              <div className="glass-card" style={{ padding: 24 }}>
                <p style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 16 }}>Equity Curve (Starting Capital: $10,000)</p>
                <ResponsiveContainer width="100%" height={280}>
                  <AreaChart data={result.equity_curve}>
                    <defs>
                      <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.08)" />
                    <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 11 }} interval="preserveStartEnd" />
                    <YAxis tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={(v) => `$${v.toLocaleString()}`} width={70} />
                    <Tooltip
                      formatter={(v: any) => `$${v?.toLocaleString()}`}
                      contentStyle={{ background: '#0f0f1a', border: '1px solid rgba(99,102,241,0.3)', borderRadius: 8, color: '#e2e8f0' }}
                    />
                    <ReferenceLine y={10000} stroke="rgba(99,102,241,0.3)" strokeDasharray="3 3" label={{ value: 'Start', fill: '#64748b', fontSize: 10 }} />
                    <Area type="monotone" dataKey="equity" stroke="#6366f1" strokeWidth={2} fill="url(#equityGrad)" dot={false} name="equity" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </>
          ) : (
            <EmptyState message="Configure parameters and click Run Backtest to simulate the strategy" />
          )}
        </div>
      </div>
    </div>
  )
}
