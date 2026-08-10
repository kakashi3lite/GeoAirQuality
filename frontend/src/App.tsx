import { Suspense, lazy } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { AmbientBackground } from '@/components/layout/AmbientBackground'
import { Header } from '@/components/layout/Header'
import { BottomNav } from '@/components/layout/BottomNav'
import { useApp } from '@/context/AppContext'
import { LocationGate } from '@/components/layout/LocationGate'

const HomePage = lazy(() => import('@/components/home/HomePage'))
const MapPage = lazy(() => import('@/components/map/MapPage'))
const LogPage = lazy(() => import('@/components/log/LogPage'))
const InsightsPage = lazy(() => import('@/components/insights/InsightsPage'))

function Shell() {
  const { location } = useApp()
  if (!location) return <LocationGate />

  return (
    <div className="relative min-h-full">
      <AmbientBackground />
      <div className="relative z-10 mx-auto max-w-3xl px-4 pb-28 pt-4 sm:px-6 lg:px-8">
        <Header />
        <Suspense
          fallback={
            <div className="mt-8 animate-pulse space-y-4">
              <div className="glass h-52" />
              <div className="glass h-32" />
              <div className="glass h-32" />
            </div>
          }
        >
          <Routes>
            <Route path="/" element={<Navigate to="/home" replace />} />
            <Route path="/home" element={<HomePage />} />
            <Route path="/map" element={<MapPage />} />
            <Route path="/log" element={<LogPage />} />
            <Route path="/insights" element={<InsightsPage />} />
            <Route path="*" element={<Navigate to="/home" replace />} />
          </Routes>
        </Suspense>
      </div>
      <BottomNav />
    </div>
  )
}

export default function App() {
  return <Shell />
}
