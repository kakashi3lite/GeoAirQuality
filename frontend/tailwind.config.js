/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        abyss: { 950: '#0b1220', 900: '#0f1a2e', 800: '#16233c', 700: '#1e3050' },
        air: {
          good: '#00e400',
          moderate: '#ffd400',
          unhealthy: '#ff7e00',
          veryUnhealthy: '#ff3b30',
          hazardous: '#a052c7',
        },
      },
      fontFamily: {
        display: ['system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
      },
      boxShadow: {
        glow: '0 0 40px -8px var(--glow, rgba(0,228,100,.45))',
        glass: '0 8px 32px rgba(2, 8, 23, 0.35)',
      },
      animation: {
        'float-slow': 'float 18s ease-in-out infinite',
        'drift': 'drift 30s linear infinite',
        'pulse-soft': 'pulseSoft 3s ease-in-out infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-12px)' },
        },
        drift: {
          '0%': { transform: 'translateX(0)' },
          '100%': { transform: 'translateX(-2000px)' },
        },
        pulseSoft: {
          '0%, 100%': { opacity: '0.55' },
          '50%': { opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}
