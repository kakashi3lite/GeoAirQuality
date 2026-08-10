import { createContext, useContext, useMemo, useState, type ReactNode } from 'react'

interface AppState {
  userId: string
  location: { lat: number; lon: number } | null
  setLocation: (l: { lat: number; lon: number }) => void
  destination: { lat: number; lon: number } | null
  setDestination: (d: { lat: number; lon: number } | null) => void
  isLocationPermitted: boolean
  setLocationPermitted: (b: boolean) => void
}

const AppContext = createContext<AppState | null>(null)

function loadUserId(): string {
  const key = 'gaq_user_id'
  let id = localStorage.getItem(key)
  if (!id) {
    id = crypto.randomUUID()
    localStorage.setItem(key, id)
  }
  return id
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [userId] = useState(loadUserId)
  const [location, setLocation] = useState<{ lat: number; lon: number } | null>(null)
  const [destination, setDestination] = useState<{ lat: number; lon: number } | null>(null)
  const [isLocationPermitted, setLocationPermitted] = useState(false)

  const value = useMemo(
    () => ({
      userId,
      location,
      setLocation,
      destination,
      setDestination,
      isLocationPermitted,
      setLocationPermitted,
    }),
    [userId, location, destination, isLocationPermitted],
  )

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

export function useApp(): AppState {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useApp must be used within AppProvider')
  return ctx
}
