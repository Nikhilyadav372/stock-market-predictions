import React from 'react'
import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard, TrendingUp, Brain, BarChart3, MessageSquare,
  FlaskConical, Lightbulb, History, Settings, Zap, Shield, X
} from 'lucide-react'

const NAV_ITEMS = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/analysis', icon: TrendingUp, label: 'Stock Analysis' },
  { to: '/forecast', icon: Brain, label: 'Forecast' },
  { to: '/compare', icon: BarChart3, label: 'Model Comparison' },
  { to: '/sentiment', icon: MessageSquare, label: 'Sentiment' },
  { to: '/backtest', icon: FlaskConical, label: 'Backtesting' },
  { to: '/explain', icon: Lightbulb, label: 'Explainability' },
  { to: '/history', icon: History, label: 'Prediction History' },
  { to: '/about', icon: Settings, label: 'About' },
]

interface SidebarProps {
  mobileOpen?: boolean
  onMobileClose?: () => void
}

export const Sidebar: React.FC<SidebarProps> = ({ mobileOpen, onMobileClose }) => {
  return (
    <>
      {/* Mobile Backdrop Overlay */}
      {mobileOpen && (
        <div
          onClick={onMobileClose}
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.65)',
            backdropFilter: 'blur(4px)',
            zIndex: 90,
          }}
        />
      )}

      <aside
        className={`app-sidebar ${mobileOpen ? 'mobile-open' : ''}`}
        style={{
          width: 240,
          minHeight: '100vh',
          height: '100vh',
          background: '#0c0c16',
          borderRight: '1px solid rgba(99,102,241,0.12)',
          display: 'flex',
          flexDirection: 'column',
          padding: '20px 12px',
          position: 'fixed',
          top: 0,
          left: 0,
          zIndex: 100,
          transition: 'transform 0.3s ease',
        }}
      >
        {/* Logo & Close Button */}
        <div style={{ padding: '8px 8px 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div
              style={{
                width: 36,
                height: 36,
                borderRadius: 10,
                background: 'linear-gradient(135deg, #4f46e5, #818cf8)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Zap size={18} color="white" />
            </div>
            <div>
              <p style={{ fontSize: 14, fontWeight: 800, color: '#e2e8f0', fontFamily: 'Plus Jakarta Sans, sans-serif', lineHeight: 1.2 }}>
                StockAI
              </p>
              <p style={{ fontSize: 10, color: '#64748b', fontWeight: 500 }}>ML Platform</p>
            </div>
          </div>

          <button
            onClick={onMobileClose}
            className="mobile-close-btn"
            style={{
              background: 'rgba(99,102,241,0.1)',
              border: '1px solid rgba(99,102,241,0.2)',
              borderRadius: 8,
              color: '#94a3b8',
              cursor: 'pointer',
              padding: 6,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <X size={18} />
          </button>
        </div>

        {/* Navigation */}
        <nav style={{ flex: 1, overflowY: 'auto' }}>
          <p style={{ fontSize: 10, fontWeight: 600, color: '#475569', letterSpacing: '0.08em', padding: '0 8px 8px', textTransform: 'uppercase' }}>
            Navigation
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                onClick={onMobileClose}
                className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
                id={`nav-${label.toLowerCase().replace(/\s/g, '-')}`}
              >
                <Icon size={16} />
                <span>{label}</span>
              </NavLink>
            ))}
          </div>
        </nav>

        {/* Disclaimer */}
        <div style={{ marginTop: 'auto', padding: '10px 8px 0' }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8 }}>
            <Shield size={12} style={{ color: '#f59e0b', marginTop: 2, flexShrink: 0 }} />
            <p style={{ fontSize: 10, color: '#64748b', lineHeight: 1.5 }}>
              Educational & research use only. Not financial advice.
            </p>
          </div>
        </div>
      </aside>
    </>
  )
}
