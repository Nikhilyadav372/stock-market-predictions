// Centralized API client using Axios with base URL from environment variables
import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60000, // 60s timeout for ML training requests
  headers: { 'Content-Type': 'application/json' },
})

// Global error interceptor — logs errors but lets components handle display
api.interceptors.response.use(
  (res) => res,
  (err) => {
    const msg = err.response?.data?.detail || err.message || 'Unknown error'
    console.error('[API Error]', err.config?.url, msg)
    return Promise.reject(err)
  }
)

// ─── Stock API ────────────────────────────────────────────────────────────────

export const stocksApi = {
  list: () => api.get<any[]>('/api/stocks'),
  get: (symbol: string) => api.get<any>(`/api/stocks/${symbol}`),
  history: (symbol: string, startDate?: string, endDate?: string) =>
    api.get<any>(`/api/stocks/${symbol}/history`, {
      params: { start_date: startDate, end_date: endDate },
    }),
  indicators: (symbol: string, startDate?: string, endDate?: string) =>
    api.get<any>(`/api/stocks/${symbol}/indicators`, {
      params: { start_date: startDate, end_date: endDate },
    }),
  refreshData: (payload: { symbol: string; force?: boolean; start_date?: string }) =>
    api.post<any>('/api/data/refresh', payload),
}

// ─── Models API ───────────────────────────────────────────────────────────────

export const modelsApi = {
  train: (payload: {
    symbol: string
    model_type: string
    task: string
    feature_set: string
    lookback_window?: number
    hyperparameters?: Record<string, any>
  }) => api.post<any>('/api/models/train', payload),

  getRunStatus: (runId: number) => api.get<any>(`/api/models/runs/${runId}`),

  list: (symbol?: string) =>
    api.get<any[]>('/api/models', { params: symbol ? { symbol } : {} }),

  get: (modelId: number) => api.get<any>(`/api/models/${modelId}`),
}

// ─── Predictions API ──────────────────────────────────────────────────────────

export const predictionsApi = {
  predict: (payload: { symbol: string; model_id: number; horizon: number }) =>
    api.post<any>('/api/predict', payload),

  list: (symbol?: string) =>
    api.get<any[]>('/api/predictions', { params: symbol ? { symbol } : {} }),

  explain: (predictionId: number) => api.get<any>(`/api/explain/${predictionId}`),
}

// ─── Sentiment API ────────────────────────────────────────────────────────────

export const sentimentApi = {
  analyze: (payload: { symbol: string; days?: number }) =>
    api.post<any>('/api/sentiment/analyze', payload),

  get: (symbol: string) => api.get<any>(`/api/sentiment/${symbol}`),
}

// ─── Backtest API ─────────────────────────────────────────────────────────────

export const backtestApi = {
  run: (payload: {
    symbol: string
    model_id: number
    start_date: string
    end_date: string
    strategy?: string
  }) => api.post<any>('/api/backtest', payload),

  get: (backtestId: number) => api.get<any>(`/api/backtest/${backtestId}`),
}

// ─── Health ───────────────────────────────────────────────────────────────────

export const healthApi = {
  check: () => api.get<any>('/api/health'),
}
