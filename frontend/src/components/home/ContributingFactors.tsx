import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, Sparkles } from 'lucide-react'
import type { SafetyAssessment } from '@/types/api'

export function ContributingFactors({ data }: { data: SafetyAssessment }) {
  const [open, setOpen] = useState(true)

  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 }}
      className="glass p-5"
    >
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between"
        aria-expanded={open}
      >
        <span className="flex items-center gap-2 font-semibold text-slate-100">
          <Sparkles className="h-4 w-4 text-sky-300" />
          Why this score?
        </span>
        <ChevronDown
          className={`h-5 w-5 text-slate-400 transition-transform ${open ? 'rotate-180' : ''}`}
        />
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.ul
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="mt-4 space-y-3 overflow-hidden"
          >
            {data.contributing_factors.map((f) => (
              <li key={f.factor} className="glass-inset p-4">
                <div className="flex items-start justify-between gap-3">
                  <p className="text-sm leading-snug text-slate-200">{f.detail}</p>
                  {f.threshold !== null && (
                    <span className="shrink-0 rounded-full bg-white/5 px-2 py-0.5 text-[11px] text-slate-400">
                      {f.unit ?? ''}
                    </span>
                  )}
                </div>
              </li>
            ))}
            {data.contributing_factors.length === 0 && (
              <li className="text-sm text-slate-400">
                Conditions look favorable for you right now — no factors are raising your risk.
              </li>
            )}
          </motion.ul>
        )}
      </AnimatePresence>
    </motion.section>
  )
}
