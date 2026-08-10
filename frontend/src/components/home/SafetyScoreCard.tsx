import { motion } from 'framer-motion'
import { RiskBadge } from '@/components/shared/RiskBadge'
import { ScoreGauge } from '@/components/shared/ScoreGauge'
import { RISK_STYLES, type SafetyAssessment } from '@/types/api'

const COMPONENT_LABELS: Record<string, string> = {
  aqi: 'AQI',
  weather: 'Weather',
  news: 'News',
  history: 'History',
}

export function SafetyScoreCard({ data }: { data: SafetyAssessment }) {
  const risk = RISK_STYLES[data.risk_level]

  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45 }}
      className="glass relative overflow-hidden p-6"
      style={{ boxShadow: `0 8px 32px rgba(2,8,23,.35), 0 0 60px -20px ${risk.color}55` }}
    >
      {/* soft accent wash */}
      <div
        className="pointer-events-none absolute -right-20 -top-20 h-64 w-64 rounded-full opacity-20 blur-3xl"
        style={{ background: risk.color }}
        aria-hidden="true"
      />

      <div className="flex flex-col items-center gap-5 sm:flex-row sm:items-center">
        <ScoreGauge score={data.safety_score} color={risk.color} />
        <div className="flex-1 text-center sm:text-left">
          <RiskBadge level={data.risk_level} />
          <p className="mt-3 text-base leading-relaxed text-slate-200">{data.summary}</p>
        </div>
      </div>

      <div className="mt-5 grid grid-cols-2 gap-2 sm:grid-cols-4">
        {Object.entries(data.component_scores).map(([key, value]) => (
          <div key={key} className="glass-inset px-3 py-2">
            <div className="flex items-baseline justify-between">
              <span className="text-xs text-slate-400">{COMPONENT_LABELS[key] ?? key}</span>
              <span className="text-xs font-semibold text-slate-200">{Math.round(value)}</span>
            </div>
            <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-white/10">
              <div
                className="h-full rounded-full"
                style={{ width: `${value}%`, background: risk.bar }}
              />
            </div>
          </div>
        ))}
      </div>
    </motion.section>
  )
}
