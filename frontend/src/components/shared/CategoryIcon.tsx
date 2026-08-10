import { Flame, Factory, Car, Sprout, ShieldAlert, Wind, Info } from 'lucide-react'

const MAP: Record<string, { icon: typeof Flame; color: string }> = {
  wildfire: { icon: Flame, color: '#fb923c' },
  industrial: { icon: Factory, color: '#a78bfa' },
  traffic: { icon: Car, color: '#60a5fa' },
  pollen: { icon: Sprout, color: '#4ade80' },
  dust_storm: { icon: Wind, color: '#fbbf24' },
  health_advisory: { icon: ShieldAlert, color: '#f87171' },
}

export function CategoryIcon({ category, className = 'h-5 w-5' }: { category: string; className?: string }) {
  const entry = MAP[category]
  const Icon = entry?.icon ?? Info
  return <Icon className={className} style={{ color: entry?.color ?? '#94a3b8' }} />
}
