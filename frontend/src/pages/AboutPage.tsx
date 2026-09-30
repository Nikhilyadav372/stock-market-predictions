import React from 'react'
import { Disclaimer } from '../components/ui'
import { Shield, BookOpen, Cpu } from 'lucide-react'

export const AboutPage: React.FC = () => {
  const techStack = {
    Frontend: ['React 18', 'TypeScript', 'Vite', 'Tailwind CSS', 'Recharts'],
    Backend: ['Python 3.11', 'FastAPI', 'Pydantic v2', 'Uvicorn'],
    'Machine Learning': ['scikit-learn', 'XGBoost', 'PyTorch (LSTM/GRU)', 'SHAP'],
    'NLP/Sentiment': ['FinBERT (HuggingFace)', 'Transformers'],
    Database: ['PostgreSQL', 'SQLAlchemy 2.x', 'Alembic'],
    Infrastructure: ['Docker', 'Docker Compose'],
  }

  const mlModels = [
    { name: 'Naive Baseline', desc: 'Uses previous close as prediction. No training required.', complexity: 'O(1)' },
    { name: 'Linear Regression', desc: 'Ridge regression on the full feature matrix. Fast and interpretable.', complexity: 'O(nd)' },
    { name: 'Random Forest', desc: '100 decision trees trained in parallel on time-series features.', complexity: 'O(n log n · T)' },
    { name: 'XGBoost', desc: 'Gradient boosting with early stopping on validation set.', complexity: 'O(n log n · T)' },
    { name: 'LSTM', desc: 'Long Short-Term Memory RNN. 60-period lookback window. PyTorch.', complexity: 'O(n · L · H)' },
    { name: 'GRU', desc: 'Gated Recurrent Unit. Lighter than LSTM, often comparable accuracy.', complexity: 'O(n · L · H)' },
  ]

  return (
    <div className="animate-in" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div>
        <h2 style={{ fontSize: 24, fontWeight: 800, color: '#e2e8f0', fontFamily: 'Plus Jakarta Sans, sans-serif' }}>
          About <span className="gradient-text">This Project</span>
        </h2>
        <p style={{ color: '#64748b', fontSize: 13 }}>AI Stock Forecasting & Market Sentiment Analysis Platform</p>
      </div>

      {/* Project Overview */}
      <div className="glass-card" style={{ padding: 24 }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16 }}>
          <div style={{ width: 48, height: 48, borderRadius: 12, background: 'linear-gradient(135deg, #4f46e5, #818cf8)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
            <Cpu size={24} color="white" />
          </div>
          <div>
            <h3 style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 8 }}>Project Overview</h3>
            <p style={{ color: '#94a3b8', lineHeight: 1.7, fontSize: 14 }}>
              This platform demonstrates end-to-end machine learning for financial time-series forecasting.
              It combines classical ML (Linear Regression, Random Forest, XGBoost) with deep learning
              (LSTM, GRU), financial-news sentiment analysis using FinBERT, SHAP-based explainability,
              and historical backtesting — all in a production-quality monorepo architecture.
            </p>
            <p style={{ color: '#94a3b8', lineHeight: 1.7, fontSize: 14, marginTop: 10 }}>
              Built as a portfolio project for MCA AI/ML students to demonstrate real engineering
              practices: proper time-series validation, data-leakage prevention, reproducible
              ML pipelines, REST API design, and full-stack development.
            </p>
          </div>
        </div>
      </div>

      {/* Tech Stack */}
      <div className="glass-card" style={{ padding: 24 }}>
        <h3 style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
          <BookOpen size={18} style={{ color: '#818cf8' }} /> Technology Stack
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 12 }}>
          {Object.entries(techStack).map(([category, items]) => (
            <div key={category} style={{ background: 'rgba(99,102,241,0.06)', borderRadius: 10, padding: '14px 16px' }}>
              <p style={{ fontSize: 11, fontWeight: 700, color: '#6366f1', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{category}</p>
              {items.map((item) => (
                <div key={item} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                  <div style={{ width: 4, height: 4, borderRadius: '50%', background: '#6366f1', flexShrink: 0 }} />
                  <span style={{ fontSize: 13, color: '#94a3b8' }}>{item}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* ML Models */}
      <div className="glass-card" style={{ padding: 24 }}>
        <h3 style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 16 }}>🧠 Implemented Models</h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {mlModels.map((m) => (
            <div key={m.name} style={{ display: 'flex', alignItems: 'flex-start', gap: 16, padding: '12px 16px', background: 'rgba(99,102,241,0.05)', borderRadius: 10 }}>
              <div style={{ minWidth: 140 }}>
                <p style={{ fontWeight: 600, color: '#818cf8', fontSize: 13 }}>{m.name}</p>
                <code style={{ fontSize: 10, color: '#4338ca' }}>{m.complexity}</code>
              </div>
              <p style={{ fontSize: 13, color: '#94a3b8', lineHeight: 1.5 }}>{m.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Key ML Concepts */}
      <div className="glass-card" style={{ padding: 24 }}>
        <h3 style={{ fontWeight: 700, color: '#e2e8f0', marginBottom: 16 }}>📐 Key Engineering Decisions</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          {[
            { title: 'Chronological Split (70/15/15)', desc: 'Time-series data is never randomly shuffled to prevent look-ahead bias. Training always uses the oldest 70% of data.' },
            { title: 'Scaler Fitted on Train Only', desc: 'StandardScaler is fitted exclusively on training data and applied to val/test. Prevents statistical leakage.' },
            { title: 'No Future Lag Features', desc: 'All lag and rolling features use shift(n) with n≥1 to ensure no future information contaminates the feature matrix.' },
            { title: 'Background Training', desc: 'ML model training runs as a FastAPI background task, allowing the UI to stay responsive while training large models.' },
            { title: 'Provider Abstraction', desc: 'Market data provider is configurable via env var. yfinance, Alpha Vantage, and Polygon are supported with automatic fallback.' },
            { title: 'FinBERT Sentiment', desc: 'Uses ProsusAI/finbert — a BERT model trained specifically on financial text for domain-appropriate sentiment scoring.' },
          ].map(({ title, desc }) => (
            <div key={title} style={{ background: 'rgba(99,102,241,0.06)', borderRadius: 10, padding: '14px 16px' }}>
              <p style={{ fontWeight: 600, color: '#818cf8', fontSize: 13, marginBottom: 6 }}>{title}</p>
              <p style={{ fontSize: 12, color: '#64748b', lineHeight: 1.6 }}>{desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Disclaimer */}
      <div className="glass-card" style={{ padding: 24, borderColor: 'rgba(245,158,11,0.2)' }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
          <Shield size={20} style={{ color: '#f59e0b', flexShrink: 0 }} />
          <div>
            <p style={{ fontWeight: 700, color: '#fbbf24', marginBottom: 8 }}>Important Disclaimer</p>
            <p style={{ color: '#94a3b8', fontSize: 13, lineHeight: 1.7 }}>
              This application is built for <strong style={{ color: '#e2e8f0' }}>educational and research purposes only</strong>.
              All predictions are experimental machine-learning forecasts and should <strong style={{ color: '#ef4444' }}>not</strong> be
              used as financial advice. Stock prices are influenced by countless factors that no ML model can fully capture.
              The backtesting module does not account for transaction costs, slippage, taxes, or real-world execution constraints.
              Past simulated performance does not guarantee future results.
            </p>
          </div>
        </div>
      </div>

      <Disclaimer />
    </div>
  )
}
