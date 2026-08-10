import { motion } from 'framer-motion'
import { AlertTriangle, Info } from 'lucide-react'

export function DataHonestyBanner({ status }: { status: 'available' | 'partial' | 'unavailable' }) {
  if (status === 'available') return null

  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex items-start gap-2 rounded-2xl border px-4 py-3 text-sm ${
        status === 'unavailable'
          ? 'border-amber-400/30 bg-amber-400/10 text-amber-200'
          : 'border-sky-400/20 bg-sky-400/10 text-sky-200'
      }`}
      role={status === 'unavailable' ? 'alert' : 'note'}
    >
      {status === 'unavailable' ? (
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      ) : (
        <Info className="mt-0.5 h-4 w-4 shrink-0" />
      )}
      <span>
        {status === 'unavailable'
          ? 'No monitoring data is available for your area right now. Use caution and check official local guidance.'
          : 'Some pollutant readings are missing in your area — this estimate may be less precise than usual.'}
      </span>
    </motion.div>
  )
}
