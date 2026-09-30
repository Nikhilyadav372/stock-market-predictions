import React, { useEffect, useState, useCallback } from 'react'
import { sentimentApi } from '../api/client'
import { useStock } from '../context/StockContext'
import { EmptyState, SampleDataNotice, Spinner, Disclaimer } from '../components/ui'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend
} from 'recharts'
import toast from 'react-hot-toast'

export const SentimentPage: React.FC = () => {
  const { selectedSymbol } = useStock()
  const [sentiment, setSentiment] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)

  const loadExisting = useCallback(async () => {
    setLoading(true)
    try {
      const { data } = await sentimentApi.get(selectedSymbol)
      setSentiment(data)
    } catch {} finally { setLoading(false) }
  }, [selectedSymbol])

  useEffect(() => { loadExisting() }, [loadExisting])

  const handleAnalyze = async () => {
    setAnalyzing(true)
    try {
      const { data } = await sentimentApi.analyze({ symbol: selectedSymbol, days: 7 })
      setSentiment(data)
      if (data.is_sample) {
        toast('Using labeled sample news data — configure NEWS_API_KEY for live headlines', { icon: '⚠️' })
      } else {
        toast.success(`Analyzed ${data.headlines.length} articles for ${selectedSymbol}`)
      }
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Sentiment analysis failed')
    } finally { setAnalyzing(false) }
  }

  const overallScore = sentiment?.overall_score ?? 0
  const scoreColor = overallScore > 0.05 ? '#22c55e' : overallScore < -0.05 ? '#ef4444' : '#f59e0b'
  const label = sentiment?.overall_label ?? 'neutral'

  const pieData = sentiment ? [
    { name: 'Positive', value: sentiment.daily_sentiment.length > 0 ? parseFloat((sentiment.daily_sentiment.reduce((a: number, d: any) => a + d.positive, 0) / sentiment.daily_sentiment.length * 100).toFixed(1)) : 0, color: '#22c55e' },
    { name: 'Neutral', value: sentiment.daily_sentiment.length > 0 ? parseFloat((sentiment.daily_sentiment.reduce((a: number, d: any) => a + d.neutral, 0) / sentiment.daily_sentiment.length * 100).toFixed(1)) : 0, color: '#f59e0b' },
    { name: 'Negative', value: sentiment.daily_sentiment.length > 0 ? parseFloat((sentiment.daily_sentiment.reduce((a: number, d: any) => a + d.negative, 0) / sentiment.daily_sentiment.length * 100).toFixed(1)) : 0, color: '#ef4444' },
  ] : []

  return (
    <div className="animate-in" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h2 style={{ fontSize: 24, fontWeight: 800, color: '#e2e8f0', fontFamily: 'Plus Jakarta Sans, sans-serif' }}>
            News <span className="gradient-text">Sentiment</span>
          </h2>
          <p style={{ color: '#64748b', fontSize: 13 }}>
            Financial news sentiment analysis powered by FinBERT for {selectedSymbol}
          </p>
        </div>
        <button
          className="btn-primary"
          onClick={handleAnalyze}
          disabled={analyzing}
          id="analyze-sentiment-btn"
        >
          {analyzing ? <><Spinner size={14} /> Analyzing...</> : <>📰 Analyze News</>}
        </button>
      </div>

      {sentiment?.is_sample && <SampleDataNotice />}

      {/* Overall Score */}
      {sentiment && (
        <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr', gap: 20 }}>
          <div className="glass-card" style={{ padding: 24, textAlign: 'center' }}>
            <p style={{ fontSize: 12, color: '#64748b', marginBottom: 12, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Overall Sentiment</p>
            <div style={{
              width: 100, height: 100, borderRadius: '50%', margin: '0 auto 16px',
              background: `conic-gradient(${scoreColor} ${Math.abs(overallScore) * 100}%, rgba(99,102,241,0.1) 0)`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              position: 'relative',
            }}>
              <div style={{ width: 70, height: 70, borderRadius: '50%', background: '#0f0f1a', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <span style={{ fontSize: 18, fontWeight: 800, color: scoreColor }}>
                  {overallScore >= 0 ? '+' : ''}{overallScore.toFixed(2)}
                </span>
              </div>
            </div>
            <span className={`badge ${label === 'positive' ? 'badge-positive' : label === 'negative' ? 'badge-negative' : 'badge-neutral'}`} style={{ fontSize: 13, padding: '4px 16px' }}>
              {label.toUpperCase()}
            </span>
            <p style={{ fontSize: 11, color: '#64748b', marginTop: 12 }}>
              Based on {sentiment.headlines.length} recent headlines
            </p>
          </div>

          {/* Pie Chart */}
          <div className="glass-card" style={{ padding: 24 }}>
            <p style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 16 }}>Sentiment Distribution</p>
            <ResponsiveContainer width="100%" height={180}>
              <PieChart>
                <Pie data={pieData} cx="50%" cy="50%" outerRadius={70} dataKey="value" label={({ name, value }) => `${name}: ${value}%`} labelLine={false}>
                  {pieData.map((entry, i) => <Cell key={i} fill={entry.color} />)}
                </Pie>
                <Tooltip formatter={(v: any) => `${v}%`} contentStyle={{ background: '#0f0f1a', border: '1px solid rgba(99,102,241,0.3)', borderRadius: 8 }} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Daily Trend */}
      {sentiment && sentiment.daily_sentiment.length > 1 && (
        <div className="glass-card" style={{ padding: 24 }}>
          <p style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 16 }}>Daily Sentiment Trend</p>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={sentiment.daily_sentiment}>
              <defs>
                <linearGradient id="sentGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.08)" />
              <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 11 }} />
              <YAxis domain={[-1, 1]} tick={{ fill: '#64748b', fontSize: 11 }} />
              <Tooltip contentStyle={{ background: '#0f0f1a', border: '1px solid rgba(99,102,241,0.3)', borderRadius: 8, color: '#e2e8f0' }} />
              <Area type="monotone" dataKey="compound_score" stroke="#6366f1" strokeWidth={2} fill="url(#sentGrad)" dot={false} name="Sentiment Score" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Headlines */}
      {sentiment && sentiment.headlines.length > 0 && (
        <div className="glass-card" style={{ padding: 24 }}>
          <p style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 16 }}>Recent Headlines</p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {sentiment.headlines.map((h: any, i: number) => (
              <div
                key={i}
                style={{
                  padding: '12px 16px',
                  borderRadius: 10,
                  background: 'rgba(99,102,241,0.05)',
                  border: '1px solid rgba(99,102,241,0.1)',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 12,
                }}
              >
                <span className={`badge ${h.sentiment === 'positive' ? 'badge-positive' : h.sentiment === 'negative' ? 'badge-negative' : 'badge-neutral'}`} style={{ flexShrink: 0, marginTop: 1 }}>
                  {h.sentiment}
                </span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <p style={{ fontSize: 13, color: '#e2e8f0', lineHeight: 1.5, marginBottom: 4 }}>{h.title}</p>
                  <div style={{ display: 'flex', gap: 12, fontSize: 11, color: '#64748b' }}>
                    {h.source && <span>{h.source}</span>}
                    {h.is_sample && <span style={{ color: '#f59e0b' }}>⚠️ SAMPLE</span>}
                    <span>Score: {h.score?.toFixed(3)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {!sentiment && !loading && (
        <EmptyState
          message="Click 'Analyze News' to fetch and analyze financial headlines"
          action={
            <button className="btn-primary" onClick={handleAnalyze}>📰 Analyze Now</button>
          }
        />
      )}

      <Disclaimer />
    </div>
  )
}
