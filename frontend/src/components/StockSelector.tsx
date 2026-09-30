import React, { useState, useRef, useEffect } from 'react'
import { Search, TrendingUp, ChevronDown, ArrowRight } from 'lucide-react'
import { stocksApi } from '../api/client'
import { useStock } from '../context/StockContext'

const QUICK_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'TATAMOTORS.NS', 'RELIANCE.NS', 'TCS.NS']

export const StockSelector: React.FC = () => {
  const { selectedSymbol, setSelectedSymbol } = useStock()
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const [results, setResults] = useState<any[]>([])
  const inputRef = useRef<HTMLInputElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const handleSearch = async (q: string) => {
    setQuery(q)
    if (q.length < 1) { setResults([]); return }
    try {
      const { data } = await stocksApi.list()
      const filtered = data.filter((s: any) =>
        s.symbol.toLowerCase().includes(q.toLowerCase()) ||
        (s.name || '').toLowerCase().includes(q.toLowerCase())
      )
      setResults(filtered.slice(0, 8))
    } catch { setResults([]) }
  }

  const selectStock = (symbol: string) => {
    if (!symbol) return
    setSelectedSymbol(symbol.trim().toUpperCase())
    setQuery('')
    setOpen(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && query.trim()) {
      selectStock(query.trim())
    }
  }

  return (
    <div className="relative" ref={containerRef}>
      <button
        onClick={() => { setOpen(!open); setTimeout(() => inputRef.current?.focus(), 100) }}
        className="flex items-center gap-2 px-4 py-2 rounded-xl"
        style={{
          background: 'rgba(20,20,40,0.8)',
          border: '1px solid rgba(99,102,241,0.3)',
          color: '#e2e8f0',
          cursor: 'pointer',
          minWidth: 160,
        }}
        id="stock-selector-btn"
      >
        <TrendingUp size={16} className="text-indigo-400" />
        <span style={{ fontWeight: 700, letterSpacing: '0.04em' }}>{selectedSymbol}</span>
        <ChevronDown size={14} style={{ marginLeft: 'auto', color: '#64748b' }} />
      </button>

      {open && (
        <div
          className="animate-in"
          style={{
            position: 'absolute',
            top: 'calc(100% + 8px)',
            left: 0,
            zIndex: 100,
            minWidth: 300,
            background: '#0f0f1a',
            border: '1px solid rgba(99,102,241,0.25)',
            borderRadius: 12,
            padding: 12,
            boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
          }}
        >
          {/* Search Input */}
          <div style={{ position: 'relative', marginBottom: 10 }}>
            <Search size={14} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: '#64748b' }} />
            <input
              ref={inputRef}
              value={query}
              onChange={(e) => handleSearch(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type symbol (e.g. TATAMOTORS.NS, NVDA) & press Enter..."
              id="stock-search-input"
              style={{
                width: '100%',
                background: 'rgba(99,102,241,0.08)',
                border: '1px solid rgba(99,102,241,0.2)',
                borderRadius: 8,
                padding: '8px 10px 8px 30px',
                color: '#e2e8f0',
                fontSize: 13,
                outline: 'none',
              }}
            />
          </div>

          {/* Custom query action button */}
          {query.trim() !== '' && (
            <button
              onClick={() => selectStock(query.trim())}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '8px 12px',
                borderRadius: 8,
                cursor: 'pointer',
                background: 'rgba(99,102,241,0.15)',
                border: '1px solid rgba(99,102,241,0.3)',
                color: '#818cf8',
                fontWeight: 600,
                fontSize: 12,
                marginBottom: 8,
                textAlign: 'left',
              }}
            >
              <Search size={14} />
              <span>Load <strong>"{query.trim().toUpperCase()}"</strong></span>
              <ArrowRight size={14} style={{ marginLeft: 'auto' }} />
            </button>
          )}

          {/* Quick access */}
          {!query && (
            <div>
              <p style={{ fontSize: 11, color: '#64748b', marginBottom: 6, paddingLeft: 4 }}>POPULAR STOCKS</p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {QUICK_STOCKS.map((s) => (
                  <button
                    key={s}
                    onClick={() => selectStock(s)}
                    style={{
                      padding: '4px 10px',
                      borderRadius: 6,
                      fontSize: 11,
                      fontWeight: 600,
                      cursor: 'pointer',
                      background: s === selectedSymbol ? 'rgba(99,102,241,0.3)' : 'rgba(99,102,241,0.1)',
                      border: '1px solid rgba(99,102,241,0.2)',
                      color: s === selectedSymbol ? '#818cf8' : '#94a3b8',
                    }}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Search results */}
          {results.length > 0 && (
            <div style={{ marginTop: 8 }}>
              {results.map((r) => (
                <button
                  key={r.symbol}
                  onClick={() => selectStock(r.symbol)}
                  style={{
                    width: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                    padding: '8px 10px',
                    borderRadius: 8,
                    cursor: 'pointer',
                    background: 'transparent',
                    border: 'none',
                    color: '#e2e8f0',
                    textAlign: 'left',
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = 'rgba(99,102,241,0.1)')}
                  onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                >
                  <span style={{ fontWeight: 700, fontSize: 13, color: '#818cf8', minWidth: 60 }}>{r.symbol}</span>
                  <span style={{ fontSize: 12, color: '#94a3b8', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.name || ''}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
