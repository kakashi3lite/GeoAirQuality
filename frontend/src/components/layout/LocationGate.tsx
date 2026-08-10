import { useState } from 'react'
import { LocateFixed } from 'lucide-react'
import { useApp } from '@/context/AppContext'

export function LocationGate() {
  const { setLocation, setLocationPermitted } = useApp()
  const [status, setStatus] = useState<'idle' | 'denied'>('idle')

  function request() {
    if (!('geolocation' in navigator)) {
      setStatus('denied')
      return
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setLocation({ lat: pos.coords.latitude, lon: pos.coords.longitude })
        setLocationPermitted(true)
      },
      () => setStatus('denied'),
      { enableHighAccuracy: false, timeout: 8000 },
    )
  }

  return (
    <div className="relative flex min-h-full items-center justify-center px-6">
      <div className="ambient-sky fixed inset-0 -z-0" aria-hidden="true" />
      <div className="glass w-full max-w-sm p-8 text-center">
        <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-full bg-sky-500/15 text-sky-300 animate-float-slow">
          <LocateFixed className="h-8 w-8" />
        </div>
        <h1 className="text-2xl font-semibold gradient-text">Breathe</h1>
        <p className="mt-2 text-sm text-slate-300">
          Personal air safety for respiratory health. We'll use your location
          to tell you if it's safe — for you — right now.
        </p>
        <button
          onClick={request}
          className="mt-6 w-full rounded-2xl bg-sky-500 px-6 py-4 text-base font-semibold text-white shadow-glow transition-transform hover:scale-[1.02] active:scale-[0.98]"
          style={{ '--glow': 'rgba(56,189,248,.45)' } as React.CSSProperties}
        >
          Use my location
        </button>
        {status === 'denied' && (
          <p className="mt-4 text-sm text-amber-300">
            Location is needed for a personal safety score. Please allow access
            and try again.
          </p>
        )}
        <p className="mt-4 text-xs text-slate-500">
          No account needed. Location stays on your device.
        </p>
      </div>
    </div>
  )
}
