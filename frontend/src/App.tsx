import { useState } from 'react'
import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { Sidebar } from './components/Sidebar'
import { Topbar } from './components/Topbar'
import { StockProvider } from './context/StockContext'
import { Dashboard } from './pages/Dashboard'
import { ForecastPage } from './pages/ForecastPage'
import { ModelComparison } from './pages/ModelComparison'
import { SentimentPage } from './pages/SentimentPage'
import { BacktestPage } from './pages/BacktestPage'
import { ExplainabilityPage } from './pages/ExplainabilityPage'
import { PredictionHistory } from './pages/PredictionHistory'
import { AboutPage } from './pages/AboutPage'

const PAGE_TITLES: Record<string, string> = {
  '/': 'Dashboard',
  '/analysis': 'Stock Analysis',
  '/forecast': 'AI Forecast',
  '/compare': 'Model Comparison',
  '/sentiment': 'Sentiment Analysis',
  '/backtest': 'Backtesting',
  '/explain': 'Explainability',
  '/history': 'Prediction History',
  '/about': 'About',
}

function App() {
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <BrowserRouter>
      <StockProvider>
        <div style={{ display: 'flex', minHeight: '100vh', width: '100%' }}>
          <Sidebar mobileOpen={mobileOpen} onMobileClose={() => setMobileOpen(false)} />
          <div className="main-content-wrapper">
            <AppContent onMenuClick={() => setMobileOpen(true)} />
          </div>
        </div>

        <Toaster
          position="top-right"
          toastOptions={{
            style: {
              background: '#1a1a35',
              color: '#e2e8f0',
              border: '1px solid rgba(99,102,241,0.3)',
              borderRadius: 10,
              fontSize: 13,
            },
            success: { iconTheme: { primary: '#22c55e', secondary: '#0f0f1a' } },
            error: { iconTheme: { primary: '#ef4444', secondary: '#0f0f1a' } },
          }}
        />
      </StockProvider>
    </BrowserRouter>
  )
}

function AppContent({ onMenuClick }: { onMenuClick: () => void }) {
  const { pathname } = useLocation()
  const title = PAGE_TITLES[pathname] || 'Stock AI Platform'

  return (
    <>
      <Topbar title={title} onMenuClick={onMenuClick} />
      <main style={{ flex: 1, padding: 24, maxWidth: 1400, width: '100%' }}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/analysis" element={<Dashboard />} />
          <Route path="/forecast" element={<ForecastPage />} />
          <Route path="/compare" element={<ModelComparison />} />
          <Route path="/sentiment" element={<SentimentPage />} />
          <Route path="/backtest" element={<BacktestPage />} />
          <Route path="/explain" element={<ExplainabilityPage />} />
          <Route path="/history" element={<PredictionHistory />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </>
  )
}

export default App
