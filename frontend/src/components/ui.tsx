import React from 'react'

// Skeleton loader blocks for various UI shapes
export const SkeletonLine: React.FC<{ width?: string; height?: number }> = ({
  width = '100%', height = 16,
}) => (
  <div className="skeleton" style={{ width, height, marginBottom: 8 }} />
)

export const SkeletonCard: React.FC<{ height?: number }> = ({ height = 120 }) => (
  <div className="glass-card" style={{ height, padding: 20 }}>
    <SkeletonLine width="40%" height={12} />
    <SkeletonLine width="70%" height={32} />
    <SkeletonLine width="50%" height={10} />
  </div>
)

export const SkeletonChart: React.FC = () => (
  <div className="glass-card" style={{ height: 300, padding: 20 }}>
    <SkeletonLine width="30%" height={14} />
    <div className="skeleton" style={{ height: 240, marginTop: 12 }} />
  </div>
)

// Metric card
interface MetricCardProps {
  label: string
  value: string | number
  sub?: string
  change?: number
  icon?: React.ReactNode
  loading?: boolean
  color?: string
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label, value, sub, change, icon, loading, color,
}) => {
  if (loading) return <SkeletonCard />
  const isPositive = change !== undefined && change >= 0
  return (
    <div className="glass-card glass-card-hover animate-in" style={{ padding: 20 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
        <p style={{ fontSize: 12, color: '#64748b', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{label}</p>
        {icon && (
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: color ? `${color}20` : 'rgba(99,102,241,0.15)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            {icon}
          </div>
        )}
      </div>
      <p style={{ fontSize: 26, fontWeight: 800, color: '#e2e8f0', fontFamily: 'Plus Jakarta Sans, sans-serif', lineHeight: 1 }}>
        {value}
      </p>
      {(sub || change !== undefined) && (
        <div style={{ marginTop: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
          {change !== undefined && (
            <span className={isPositive ? 'badge badge-positive' : 'badge badge-negative'}>
              {isPositive ? '+' : ''}{typeof change === 'number' ? change.toFixed(2) : change}%
            </span>
          )}
          {sub && <span style={{ fontSize: 11, color: '#64748b' }}>{sub}</span>}
        </div>
      )}
    </div>
  )
}

// Error state
export const ErrorState: React.FC<{ message: string; onRetry?: () => void }> = ({ message, onRetry }) => (
  <div
    className="glass-card"
    style={{ padding: 40, textAlign: 'center', borderColor: 'rgba(239,68,68,0.2)' }}
  >
    <p style={{ fontSize: 32, marginBottom: 12 }}>⚠️</p>
    <p style={{ color: '#f87171', fontWeight: 600, marginBottom: 6 }}>Something went wrong</p>
    <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>{message}</p>
    {onRetry && (
      <button className="btn-secondary" onClick={onRetry}>Try Again</button>
    )}
  </div>
)

// Empty state
export const EmptyState: React.FC<{ message: string; action?: React.ReactNode }> = ({ message, action }) => (
  <div className="glass-card" style={{ padding: 60, textAlign: 'center' }}>
    <p style={{ fontSize: 40, marginBottom: 12 }}>📊</p>
    <p style={{ color: '#64748b', fontSize: 14 }}>{message}</p>
    {action && <div style={{ marginTop: 20 }}>{action}</div>}
  </div>
)

// Loading spinner
export const Spinner: React.FC<{ size?: number }> = ({ size = 20 }) => (
  <svg
    width={size} height={size}
    viewBox="0 0 24 24"
    style={{ animation: 'spin 0.8s linear infinite' }}
  >
    <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    <circle cx="12" cy="12" r="10" stroke="rgba(99,102,241,0.3)" strokeWidth="3" fill="none" />
    <path d="M12 2a10 10 0 0 1 10 10" stroke="#818cf8" strokeWidth="3" fill="none" strokeLinecap="round" />
  </svg>
)

// Status badge for model training
export const StatusBadge: React.FC<{ status: string }> = ({ status }) => {
  const config: Record<string, { label: string; cls: string }> = {
    pending: { label: 'Pending', cls: 'badge badge-neutral' },
    running: { label: 'Training...', cls: 'badge badge-info' },
    done: { label: 'Ready', cls: 'badge badge-positive' },
    failed: { label: 'Failed', cls: 'badge badge-negative' },
  }
  const c = config[status] || { label: status, cls: 'badge' }
  return <span className={c.cls}>{c.label}</span>
}

// Sample data notice
export const SampleDataNotice: React.FC = () => (
  <div className="sample-notice" style={{ marginBottom: 12 }}>
    ⚠️ <strong>SAMPLE DATA</strong> — This is synthetic demo data, not real market data.
    Configure a real provider in <code>.env</code> for live data.
  </div>
)

// Disclaimer
export const Disclaimer: React.FC = () => (
  <div className="disclaimer-banner" style={{ marginTop: 16 }}>
    <span>⚡</span>
    <span>
      <strong>Educational Disclaimer:</strong> Predictions are experimental machine-learning
      forecasts for educational and research purposes only and are{' '}
      <strong>not financial advice</strong>. Past performance does not guarantee future results.
    </span>
  </div>
)
