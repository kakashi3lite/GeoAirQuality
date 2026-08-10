import { RISK_STYLES, type RiskLevel } from '@/types/api'

export function RiskBadge({ level }: { level: RiskLevel }) {
  const s = RISK_STYLES[level]
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold"
      style={{ background: s.soft, color: s.color }}
    >
      <span className="h-2 w-2 rounded-full" style={{ background: s.color }} aria-hidden="true" />
      {s.label}
    </span>
  )
}
