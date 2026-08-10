export function ScoreGauge({ score, color }: { score: number; color: string }) {
  const r = 54
  const c = 2 * Math.PI * r
  const filled = (score / 100) * c
  return (
    <div className="score-glow relative h-32 w-32" style={{ '--glow': `${color}66` } as React.CSSProperties}>
      <svg viewBox="0 0 128 128" className="h-full w-full -rotate-90">
        <circle cx="64" cy="64" r={r} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="10" />
        <circle
          cx="64"
          cy="64"
          r={r}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={`${filled} ${c}`}
          style={{ transition: 'stroke-dasharray 800ms cubic-bezier(.4,0,.2,1)' }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-4xl font-bold tabular-nums" style={{ color }}>
          {score}
        </span>
        <span className="text-[10px] uppercase tracking-widest text-slate-400">of 100</span>
      </div>
    </div>
  )
}
