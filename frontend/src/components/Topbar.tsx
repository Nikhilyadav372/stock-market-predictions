import React from 'react'
import { RefreshCw, Menu } from 'lucide-react'
import { StockSelector } from './StockSelector'
import { useStock } from '../context/StockContext'
import { stocksApi } from '../api/client'
import toast from 'react-hot-toast'

interface TopbarProps {
  title: string
  onMenuClick?: () => void
}

export const Topbar: React.FC<TopbarProps> = ({ title, onMenuClick }) => {
  const { selectedSymbol } = useStock()
  const [refreshing, setRefreshing] = React.useState(false)

  const handleRefresh = async () => {
    setRefreshing(true)
    try {
      const { data } = await stocksApi.refreshData({ symbol: selectedSymbol })
      if (data.is_sample) {
        toast('Using sample data — configure a real provider in .env', { icon: '⚠️' })
      } else {
        toast.success(`Data refreshed: +${data.records_added} new records for ${selectedSymbol}`)
      }
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to refresh data')
    } finally {
      setRefreshing(false)
    }
  }

  return (
    <header
      className="topbar-header"
      style={{
        height: 64,
        background: 'rgba(10,10,20,0.9)',
        backdropFilter: 'blur(12px)',
        borderBottom: '1px solid rgba(99,102,241,0.12)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 24px',
        position: 'sticky',
        top: 0,
        zIndex: 40,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <button
          onClick={onMenuClick}
          className="mobile-menu-btn"
          style={{
            background: 'rgba(99,102,241,0.1)',
            border: '1px solid rgba(99,102,241,0.2)',
            borderRadius: 8,
            color: '#818cf8',
            cursor: 'pointer',
            padding: '8px',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          title="Open Menu"
        >
          <Menu size={18} />
        </button>

        <h1 className="topbar-title" style={{ fontSize: 18, fontWeight: 700, color: '#e2e8f0', fontFamily: 'Plus Jakarta Sans, sans-serif' }}>
          {title}
        </h1>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <StockSelector />

        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="btn-secondary"
          style={{ padding: '8px 12px', fontSize: 13 }}
          id="topbar-refresh-btn"
          title="Refresh market data"
        >
          <RefreshCw size={14} className={refreshing ? 'animate-spin' : ''} />
          <span className="topbar-refresh-text">{refreshing ? 'Refreshing...' : 'Refresh'}</span>
        </button>
      </div>
    </header>
  )
}
