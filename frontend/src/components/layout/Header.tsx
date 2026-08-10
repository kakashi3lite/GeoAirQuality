import { format } from 'date-fns'
import { Cloud, MapPin } from 'lucide-react'
import { useApp } from '@/context/AppContext'
import { useSafetyAssessment } from '@/hooks/useApi'
import { DataHonestyBanner } from '@/components/shared/DataHonestyBanner'

export function Header() {
  const { location } = useApp()
  const { data } = useSafetyAssessment()

  return (
    <header className="mb-5">
      <DataHonestyBanner status={data?.data_status ?? 'available'} />
      <div className="mt-4 flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-sky-500/15 text-sky-300">
            <Cloud className="h-6 w-6" />
          </div>
          <div>
            <h1 className="text-lg font-semibold leading-tight gradient-text">
              Breathe
            </h1>
            <p className="text-sm text-slate-400">
              {data ? `Updated ${format(new Date(data.generated_at), 'h:mm a')}` : '…'}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-sm text-slate-300">
          <MapPin className="h-3.5 w-3.5 text-sky-300" />
          {location ? `${location.lat.toFixed(2)}, ${location.lon.toFixed(2)}` : 'locating…'}
        </div>
      </div>
    </header>
  )
}
