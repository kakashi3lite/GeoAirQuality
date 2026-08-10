import { useMemo } from 'react'

const PARTICLES = [
  { left: '8%', size: 5, dur: 26, delay: 0 },
  { left: '18%', size: 3, dur: 32, delay: 4 },
  { left: '30%', size: 4, dur: 24, delay: 8 },
  { left: '42%', size: 2, dur: 38, delay: 2 },
  { left: '55%', size: 5, dur: 29, delay: 6 },
  { left: '66%', size: 3, dur: 35, delay: 10 },
  { left: '78%', size: 4, dur: 27, delay: 3 },
  { left: '88%', size: 3, dur: 33, delay: 7 },
  { left: '96%', size: 2, dur: 41, delay: 5 },
  { left: '48%', size: 3, dur: 31, delay: 12 },
]

export function AmbientBackground() {
  const particles = useMemo(
    () =>
      PARTICLES.map((p, i) => (
        <span
          key={i}
          className="particle"
          style={{
            left: p.left,
            bottom: '-40px',
            width: p.size * 2,
            height: p.size * 2,
            animationDuration: `${p.dur}s`,
            animationDelay: `${p.delay}s`,
          }}
        />
      )),
    [],
  )

  return (
    <div className="ambient-sky fixed inset-0 -z-0" aria-hidden="true">
      {particles}
      {/* soft horizon glow */}
      <div
        className="absolute inset-x-0 top-1/3 h-72 opacity-60"
        style={{
          background:
            'radial-gradient(600px 220px at 50% 0%, rgba(125,211,252,0.10), transparent 70%)',
        }}
      />
    </div>
  )
}
