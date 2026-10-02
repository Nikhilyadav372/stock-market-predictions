import React, { useEffect, useState, useCallback } from 'react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts'
import { TrendingUp, TrendingDown, DollarSign, Activity, BarChart2, Target } from 'lucide-react'
import { stocksApi } from '../api/client'
import { useStock } from '../context/StockContext'
import { MetricCard, ErrorState, Disclaimer, SampleDataNotice } from '../components/ui'
import { format } from 'date-fns'

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload?.length) {
    return (
      <div style={{ background: '#0f0f1a', border: '1px solid rgba(99,102,241,0.3)', borderRadius: 10, padding: '10px 14px' }}>
        <p style={{ color: '#64748b', fontSize: 11, marginBottom: 4 }}>{label}</p>
        <p style={{ color: '#818cf8', fontWeight: 700, fontSize: 16 }}>
          ${payload[0]?.value?.toFixed(2)}
        </p>
        {payload[1] && (
          <p style={{ color: '#a78bfa', fontSize: 12 }}>SMA20: ${payload[1]?.value?.toFixed(2)}</p>
        )}
      </div>
    )
  }
  return null
}

export const Dashboard: React.FC = () => {
  const { selectedSymbol } = useStock()
  const [history, setHistory] = useState<any>(null)
  const [indicators, setIndicators] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isSample] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [histRes, indRes] = await Promise.all([
        stocksApi.history(selectedSymbol),
        stocksApi.indicators(selectedSymbol),
      ])
      setHistory(histRes.data)
      setIndicators(indRes.data)
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }, [selectedSymbol])

  useEffect(() => { load() }, [load])

  const prices = history?.prices || []
  const latest = prices[prices.length - 1]
  const prev = prices[prices.length - 2]
  const dailyChange = latest && prev ? ((latest.close - prev.close) / prev.close) * 100 : null
  const weekHigh = prices.length > 0 ? Math.max(...prices.slice(-252).map((p: any) => p.high)) : null
  const weekLow = prices.length > 0 ? Math.min(...prices.slice(-252).map((p: any) => p.low)) : null

  // Build chart data (last 180 days)
  const indData = indicators?.indicators || []
  const indMap = new Map(indData.map((row: any) => [row.date, row]))
  const chartData = (prices.length > 0 ? prices.slice(-180) : indData.slice(-180)).map((p: any) => {
    const ind: any = indMap.get(p.date) || {}
    return {
      date: p.date,
      close: p.close,
      sma20: ind.sma_20 ?? p.sma_20,
      rsi: ind.rsi ?? p.rsi,
    }
  })

  if (error) return (
    <div>
      <ErrorState message={error} onRetry={load} />
    </div>
  )

  return (
    <div className="animate-in" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h2 style={{ fontSize: 24, fontWeight: 800, color: '#e2e8f0', fontFamily: 'Plus Jakarta Sans, sans-serif' }}>
            {selectedSymbol} <span className="gradient-text">Overview</span>
          </h2>
          <p style={{ color: '#64748b', fontSize: 13 }}>
            {loading ? 'Loading...' : `${prices.length} trading days of data`}
          </p>
        </div>
      </div>

      {isSample && <SampleDataNotice />}

      {/* Metric Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 16 }}>
        <MetricCard
          label="Latest Close"
          value={latest ? `$${latest.close?.toFixed(2)}` : '—'}
          change={dailyChange ?? undefined}
          icon={<DollarSign size={16} style={{ color: '#818cf8' }} />}
          loading={loading}
        />
        <MetricCard
          label="Daily Change"
          value={dailyChange !== null ? `${dailyChange >= 0 ? '+' : ''}${dailyChange?.toFixed(2)}%` : '—'}
          sub={prev ? `vs $${prev.close?.toFixed(2)}` : undefined}
          icon={dailyChange !== null && dailyChange >= 0
            ? <TrendingUp size={16} style={{ color: '#22c55e' }} />
            : <TrendingDown size={16} style={{ color: '#ef4444' }} />}
          loading={loading}
          color={dailyChange !== null && dailyChange >= 0 ? '#22c55e' : '#ef4444'}
        />
        <MetricCard
          label="Volume"
          value={latest ? `${(latest.volume / 1e6)?.toFixed(1)}M` : '—'}
          icon={<BarChart2 size={16} style={{ color: '#f59e0b' }} />}
          loading={loading}
          color="#f59e0b"
        />
        <MetricCard
          label="52W High"
          value={weekHigh ? `$${weekHigh?.toFixed(2)}` : '—'}
          icon={<Activity size={16} style={{ color: '#22c55e' }} />}
          loading={loading}
          color="#22c55e"
        />
        <MetricCard
          label="52W Low"
          value={weekLow ? `$${weekLow?.toFixed(2)}` : '—'}
          icon={<Target size={16} style={{ color: '#ef4444' }} />}
          loading={loading}
          color="#ef4444"
        />
        <MetricCard
          label="RSI (14)"
          value={!loading && indData.length ? (indData[indData.length - 1]?.rsi?.toFixed(1) ?? '—') : '—'}
          sub={!loading && indData.length
            ? indData[indData.length - 1]?.rsi > 70 ? 'Overbought'
              : indData[indData.length - 1]?.rsi < 30 ? 'Oversold' : 'Neutral'
            : undefined}
          icon={<Activity size={16} style={{ color: '#818cf8' }} />}
          loading={loading}
        />
      </div>

      {/* Price Chart */}
      <div className="glass-card" style={{ padding: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <div>
            <p style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 2 }}>Price History</p>
            <p style={{ fontSize: 12, color: '#64748b' }}>Close price with 20-day moving average</p>
          </div>
          <span className="badge badge-info">Last 180 Days</span>
        </div>
        {loading
          ? <div className="skeleton" style={{ height: 280 }} />
          : chartData.length > 0
            ? (
              <ResponsiveContainer width="100%" height={280}>
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="colorClose" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.08)" />
                  <XAxis
                    dataKey="date"
                    tick={{ fill: '#64748b', fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v) => format(new Date(v), 'MMM d')}
                    interval={29}
                  />
                  <YAxis
                    tick={{ fill: '#64748b', fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v) => `$${v}`}
                    width={60}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Area
                    type="monotone"
                    dataKey="close"
                    stroke="#6366f1"
                    strokeWidth={2}
                    fill="url(#colorClose)"
                    dot={false}
                  />
                  <Area
                    type="monotone"
                    dataKey="sma20"
                    stroke="#a78bfa"
                    strokeWidth={1.5}
                    fill="none"
                    dot={false}
                    strokeDasharray="4 2"
                  />
                </AreaChart>
              </ResponsiveContainer>
            )
            : <div style={{ height: 280, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b' }}>No data available. Click "Refresh Data" to load.</div>
        }
      </div>

      {/* RSI + MACD charts */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        {/* RSI */}
        <div className="glass-card" style={{ padding: 20 }}>
          <p style={{ fontWeight: 600, color: '#e2e8f0', marginBottom: 4 }}>RSI (14)</p>
          <p style={{ fontSize: 11, color: '#64748b', marginBottom: 12 }}>Relative Strength Index — overbought &gt;70, oversold &lt;30</p>
          {loading
            ? <div className="skeleton" style={{ height: 160 }} />
            : (
              <ResponsiveContainer width="100%" height={160}>
                <AreaChart data={chartData.slice(-90)}>
                  <defs>
                    <linearGradient id="rsiGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.2} />
                      <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.06)" />
                  <XAxis dataKey="date" hide />
                  <YAxis domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 10 }} tickLine={false} axisLine={false} width={30} />
                  <Tooltip formatter={(v: any) => v?.toFixed(1)} contentStyle={{ background: '#0f0f1a', border: '1px solid rgba(99,102,241,0.3)', borderRadius: 8, color: '#e2e8f0' }} />
                  <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="3 3" />
                  <ReferenceLine y={30} stroke="#22c55e" strokeDasharray="3 3" />
                  <Area type="monotone" dataKey="rsi" stroke="#f59e0b" strokeWidth={1.5} fill="url(#rsiGrad)" dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            )}
        </div>

        {/* MACD */}
        <div className="glass-card" style={{ padding: 20 }}>
          <p style={{ fontWeight: 600, color: '#e2e8f0', marginBottom: 4 }}>MACD</p>
          <p style={{ fontSize: 11, color: '#64748b', marginBottom: 12 }}>Moving Average Convergence Divergence (12, 26, 9)</p>
          {loading
            ? <div className="skeleton" style={{ height: 160 }} />
            : (
              <ResponsiveContainer width="100%" height={160}>
                <AreaChart data={indData.slice(-90).map((r: any) => ({ date: r.date, macd: r.macd, signal: r.macd_signal }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.06)" />
                  <XAxis dataKey="date" hide />
                  <YAxis tick={{ fill: '#64748b', fontSize: 10 }} tickLine={false} axisLine={false} width={40} />
                  <Tooltip formatter={(v: any) => v?.toFixed(4)} contentStyle={{ background: '#0f0f1a', border: '1px solid rgba(99,102,241,0.3)', borderRadius: 8, color: '#e2e8f0' }} />
                  <ReferenceLine y={0} stroke="rgba(99,102,241,0.3)" />
                  <Area type="monotone" dataKey="macd" stroke="#818cf8" strokeWidth={1.5} fill="rgba(129,140,248,0.1)" dot={false} />
                  <Area type="monotone" dataKey="signal" stroke="#f59e0b" strokeWidth={1.5} fill="none" dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            )}
        </div>
      </div>

      <Disclaimer />
    </div>
  )
}
