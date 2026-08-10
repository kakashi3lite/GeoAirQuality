import { NavLink } from 'react-router-dom'
import { Home, Map as MapIcon, PenLine, LineChart } from 'lucide-react'

const TABS = [
  { to: '/home', label: 'Home', icon: Home },
  { to: '/map', label: 'Map', icon: MapIcon },
  { to: '/log', label: 'Log', icon: PenLine },
  { to: '/insights', label: 'Insights', icon: LineChart },
]

export function BottomNav() {
  return (
    <nav
      aria-label="Primary"
      className="fixed inset-x-0 bottom-0 z-30 border-t border-white/10 bg-abyss-900/80 backdrop-blur-xl"
      style={{ paddingBottom: 'env(safe-area-inset-bottom)' }}
    >
      <div className="mx-auto grid max-w-md grid-cols-4">
        {TABS.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/home'}
            className={({ isActive }) =>
              `flex min-h-[64px] flex-col items-center justify-center gap-1 text-[11px] transition-colors ${
                isActive ? 'text-sky-300' : 'text-slate-400 hover:text-slate-200'
              }`
            }
          >
            {({ isActive }) => (
              <>
                <Icon className={`h-6 w-6 ${isActive ? 'drop-shadow-[0_0_8px_rgba(56,189,248,.6)]' : ''}`} />
                <span className={isActive ? 'font-medium' : ''}>{label}</span>
              </>
            )}
          </NavLink>
        ))}
      </div>
    </nav>
  )
}
