import { motion } from 'framer-motion'
import { Route, ShieldAlert } from 'lucide-react'
import { RISK_STYLES, type RouteRisk } from '@/types/api'

export function RouteRiskPreview({ risk }: { risk: RouteRisk }) {
  const color = risk.route_risk_score >= 80 ? '#00e400' : risk.route_risk_score >= 60 ? '#ffd400' : '#ff7e00'
  const style = color === '#00e400' ? RISK_STYLES.low : color === '#ffd400' ? RISK_STYLES.moderate : RISK_STYLES.high

  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.24 }}
      className="glass p-5"
    >
      <h2 className="flex items-center gap-2 font-semibold text-slate-100">
        <Route className="h-4 w-4 text-sky-300" />
        Your route
      </h2>
      <div className="mt-3 flex items-center justify-between">
        <span className="text-sm text-slate-300">
          Safety <span className="font-semibold" style={{ color }}>{risk.route_risk_score}/100</span>
        </span>
        <span className="text-xs text-slate-400">
          Best: {risk.safest_segment} · Worst: {risk.worst_segment}
        </span>
      </div>
      <div className="mt-3 flex gap-2">
        {risk.segments.map((seg) => (
          <div key={seg.point} className="glass-inset flex-1 px-3 py-2 text-center">
            <p className="text-[11px] capitalize text-slate-400">{seg.point}</p>
            <p className="text-sm font-semibold" style={{ color: style.bar }}>
              {seg.safety_score}
            </p>
          </div>
        ))}
      </div>
      {risk.news_notes.length > 0 && (
        <div className="mt-3 flex items-center gap-2 rounded-xl border border-amber-400/20 bg-amber-400/10 px-3 py-2 text-xs text-amber-200">
          <ShieldAlert className="h-4 w-4 shrink-0" />
          {risk.news_notes[0].text}
        </div>
      )}
    </motion.section>
  )
}
