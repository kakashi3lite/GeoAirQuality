import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Check, Send } from 'lucide-react'
import { useApp } from '@/context/AppContext'
import { useLogSymptom } from '@/hooks/useApi'

const SYMPTOMS = [
  { key: 'coughing', label: 'Coughing', emoji: '🫁' },
  { key: 'wheezing', label: 'Wheezing', emoji: '💨' },
  { key: 'shortness_of_breath', label: 'Short of breath', emoji: '😮‍💨' },
  { key: 'chest_tightness', label: 'Chest tightness', emoji: '🤒' },
  { key: 'eye_irritation', label: 'Eye irritation', emoji: '👁️' },
  { key: 'fatigue', label: 'Fatigue', emoji: '😴' },
]

export default function LogPage() {
  const { location } = useApp()
  const [type, setType] = useState<string | null>(null)
  const [severity, setSeverity] = useState(3)
  const log = useLogSymptom()

  const canSubmit = type !== null && !!location

  return (
    <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
      <div className="glass p-6">
        <h1 className="text-xl font-semibold text-slate-100">How are you feeling right now?</h1>
        <p className="mt-1 text-sm text-slate-400">
          Your answer helps Breathe learn what triggers your symptoms.
        </p>

        <div className="mt-5 grid grid-cols-2 gap-2 sm:grid-cols-3">
          {SYMPTOMS.map((s) => (
            <button
              key={s.key}
              onClick={() => setType(s.key)}
              aria-pressed={type === s.key}
              className={`flex min-h-[64px] flex-col items-center justify-center gap-1 rounded-2xl border px-3 py-3 text-sm transition-all ${
                type === s.key
                  ? 'border-sky-400/60 bg-sky-500/15 text-sky-200'
                  : 'border-white/10 bg-white/5 text-slate-300 hover:bg-white/10'
              }`}
            >
              <span className="text-2xl">{s.emoji}</span>
              {s.label}
            </button>
          ))}
        </div>

        <div className="mt-6">
          <div className="flex items-center justify-between">
            <span className="text-sm text-slate-300">How bad is it?</span>
            <span className="rounded-full bg-white/5 px-3 py-0.5 text-sm font-semibold text-sky-300">
              {severity}/5
            </span>
          </div>
          <input
            type="range"
            min={1}
            max={5}
            value={severity}
            onChange={(e) => setSeverity(Number(e.target.value))}
            className="mt-3 w-full accent-sky-400"
            aria-label="Severity from 1 (mild) to 5 (severe)"
          />
          <div className="mt-1 flex justify-between text-[11px] text-slate-500">
            <span>mild</span>
            <span>severe</span>
          </div>
        </div>

        {location && (
          <p className="mt-5 text-xs text-slate-400">
            📍 Logging at {location.lat.toFixed(3)}, {location.lon.toFixed(3)} — the
            current conditions (AQI, PM2.5, humidity) are captured automatically.
          </p>
        )}

        <button
          disabled={!canSubmit || log.isPending}
          onClick={() => type && log.mutate({ symptom_type: type, severity })}
          className="mt-6 flex w-full items-center justify-center gap-2 rounded-2xl bg-sky-500 px-6 py-4 text-base font-semibold text-white shadow-glow transition-transform enabled:hover:scale-[1.01] enabled:active:scale-[0.98] disabled:opacity-40"
          style={{ '--glow': 'rgba(56,189,248,.45)' } as React.CSSProperties}
        >
          {log.isSuccess ? <Check className="h-5 w-5" /> : <Send className="h-5 w-5" />}
          {log.isSuccess ? 'Logged — insights updated' : log.isPending ? 'Logging…' : 'Save'}
        </button>

        <AnimatePresence>
          {log.isSuccess && (
            <motion.p
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="mt-3 text-center text-sm text-emerald-300"
            >
              ✓ Thank you — your personal patterns are getting smarter.
            </motion.p>
          )}
          {log.isError && (
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="mt-3 text-center text-sm text-rose-300"
            >
              Couldn't save — please try again.
            </motion.p>
          )}
        </AnimatePresence>
      </div>
    </motion.section>
  )
}
