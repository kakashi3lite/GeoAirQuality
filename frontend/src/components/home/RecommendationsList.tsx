import { motion } from 'framer-motion'
import { Shield, MapPin, Clock, Route, Info } from 'lucide-react'
import type { Recommendation } from '@/types/api'

const ICONS: Record<Recommendation['type'], typeof Info> = {
  precaution: Shield,
  location: MapPin,
  timing: Clock,
  route: Route,
  general: Info,
}

export function RecommendationsList({ recommendations }: { recommendations: Recommendation[] }) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.18 }}
      className="glass p-5"
    >
      <h2 className="font-semibold text-slate-100">What to do</h2>
      <ul className="mt-4 space-y-2">
        {recommendations.map((rec, i) => {
          const Icon = ICONS[rec.type] ?? Info
          return (
            <motion.li
              key={i}
              initial={{ opacity: 0, x: 16 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.25 + i * 0.05 }}
              className="glass-inset flex items-start gap-3 p-3.5"
            >
              <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-sky-500/15 text-sky-300">
                <Icon className="h-4 w-4" />
              </span>
              <p className="text-sm leading-relaxed text-slate-200">{rec.text}</p>
            </motion.li>
          )
        })}
      </ul>
    </motion.section>
  )
}
