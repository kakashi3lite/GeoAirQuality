import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { ArrowRight, Bell } from 'lucide-react'
import { CategoryIcon } from '@/components/shared/CategoryIcon'
import type { NewsEvent } from '@/types/api'

export function NearbyAlertsStrip({ events }: { events: NewsEvent[] }) {
  if (events.length === 0) return null

  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
      className="glass p-5"
    >
      <h2 className="flex items-center gap-2 font-semibold text-slate-100">
        <Bell className="h-4 w-4 text-amber-300" />
        Near you
      </h2>
      <ul className="mt-4 space-y-2">
        {events.slice(0, 3).map((ev, i) => (
          <motion.li
            key={i}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.35 + i * 0.08 }}
            className="glass-inset flex items-center gap-3 p-3.5"
          >
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-white/5">
              <CategoryIcon category={ev.category} />
            </span>
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium text-slate-100">{ev.title}</p>
              <p className="text-xs text-slate-400">
                {ev.distance_km.toFixed(1)} km away · severity {ev.severity}/100
              </p>
            </div>
            <Link
              to="/map"
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-white/5 text-slate-300 transition-colors hover:bg-white/10"
              aria-label="View on map"
            >
              <ArrowRight className="h-4 w-4" />
            </Link>
          </motion.li>
        ))}
      </ul>
    </motion.section>
  )
}
