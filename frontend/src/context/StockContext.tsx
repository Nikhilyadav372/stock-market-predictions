import React, { createContext, useContext, useState } from 'react'

interface StockContextType {
  selectedSymbol: string
  setSelectedSymbol: (s: string) => void
}

const StockContext = createContext<StockContextType>({
  selectedSymbol: 'AAPL',
  setSelectedSymbol: () => {},
})

export const StockProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL')
  return (
    <StockContext.Provider value={{ selectedSymbol, setSelectedSymbol }}>
      {children}
    </StockContext.Provider>
  )
}

export const useStock = () => useContext(StockContext)
