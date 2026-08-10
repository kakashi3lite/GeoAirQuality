import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { TrendingDown, TrendingUp, Minus, Clock, Activity } from 'lucide-react'
import { useInsights } from '@/hooks/useApi'

export default function InsightsPage() {
  const { data, isLoading } = useInsights()

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="glass h-40 animate-pulse" />
        <div className="glass h-40 animate-pulse" />
      </div>
    )
  }

  if (!data) {
    return (
      <div className="glass p-8 text-center">
        <p className="text-slate-300">We couldn't load your insights.</p>
      </div>
    )
  }

  const maxCorr = Math.max(...data.top_triggers.map((t) => Math.abs(t.correlation)), 0.1)
  const trendDown = data.recent_trend.includes('down')
  const TrendIcon = trendDown ? TrendingDown : data.recent_trend.includes('up') ? TrendingUp : Minus
  const trendColor = trendDown ? 'text-emerald-300' : 'text-amber-300'

  return (
    <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
      {data.top_triggers.length === 0 ? (
        <div className="glass p-8 text-center">
          <Activity className="mx-auto h-10 w-10 text-sky-400" />
          <h2 className="mt-3 text-lg font-semibold text-slate-100">Your triggers start here</h2>
          <p className="mx-auto mt-2 max-w-xs text-sm text-slate-400">
            You haven't logged symptoms yet. Once you do, Breathe will show what
            correlates with how you feel.
          </p>
          <Link
            to="/log"
            className="mt-5 inline-block rounded-2xl bg-sky-500 px-6 py-3 font-semibold text-white"
          >
            Log a symptom
          </Link>
        </div>
      ) : (
        <>
          <div className="glass p-5">
            <h2 className="flex items-center gap-2 font-semibold text-slate-100">
              <Activity className="h-4 w-4 text-sky-300" />
              Your triggers
            </h2>
            <ul className="mt-4 space-y-3">
              {data.top_triggers.map((t, i) => (
                <li key={t.factor}>
                  <div className="flex items-baseline justify-between text-sm">
                    <span className="text-slate-200">{t.factor}</span>
                    <span className="text-xs text-slate-400">
                      r={t.correlation.toFixed(2)} · {t.occurrences}×
                    </span>
                  </div>
                  <div className="mt-1 h-2 overflow-hidden rounded-full bg-white/10">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${(Math.abs(t.correlation) / maxCorr) * 100}%` }}
                      transition={{ delay: 0.2 + i * 0.08, duration: 0.6 }}
                      className="h-full rounded-full bg-gradient-to-r from-sky-400 to-emerald-400"
                    />
                  </div>
                </li>
              ))}
            </ul>
          </div>

          <div className="glass p-5">
            <h2 className="flex items-center gap-2 font-semibold text-slate-100">
              <Clock className="h-4 w-4 text-sky-300" />
              Safest vs riskiest times
            </h2>
            <div className="mt-4 grid grid-cols-2 gap-3">
              <div className="glass-inset p-3">
                <p className="text-xs font-medium text-emerald-300">Safest</p>
                <ul className="mt-2 space-y-1.5">
                  {data.safest_times.map((b) => (
                    <li key={b.hour} className="flex justify-between text-sm text-slate-200">
                      <span>{formatHour(b.hour)}</span>
                      <span className="text-slate-400">avg {b.avg_severity.toFixed(1)}</span>
                    </li>
                  ))}
                </ul>
              </div>
              <div className="glass-inset p-3">
                <p className="text-xs font-medium text-amber-300">Riskiest</p>
                <ul className="mt-2 space-y-1.5">
                  {data.riskiest_times.map((b) => (
                    <li key={b.hour} className="flex justify-between text-sm text-slate-200">
                      <span>{formatHour(b.hour)}</span>
                      <span className="text-slate-400">avg {b.avg_severity.toFixed(1)}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          <div className="glass p-5">
            <h2 className="flex items-center gap-2 font-semibold text-slate-100">
              <TrendIcon className={`h-4 w-4 ${trendColor}`} />
              {data.period_days}-day trend
            </h2>
            <p className="mt-3 text-sm text-slate-300">{data.recent_trend}</p>
          </div>
        </>
      )}
    </motion.section>
  )
}

function formatHour(hour: number): string {
  const suffix = hour >= 12 ? 'PM' : 'AM'
  const h = hour % 12 === 0 ? 12 : hour % 12
  return `${h} ${suffix}`
}
