import { useEffect, useRef, useState } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { useApp } from '@/context/AppContext'
import { useNewsNearby } from '@/hooks/useApi'
import { CategoryIcon } from '@/components/shared/CategoryIcon'
import type { NewsArticle } from '@/types/api'

export default function MapPage() {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const { location } = useApp()
  const { data: events } = useNewsNearby()

  const [selected, setSelected] = useState<NewsArticle | null>(null)

  useEffect(() => {
    if (!containerRef.current || !location || mapRef.current) return
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: {
        version: 8,
        sources: {
          osm: {
            type: 'raster',
            tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
            tileSize: 256,
            attribution: '© OpenStreetMap',
          },
        },
        layers: [
          {
            id: 'osm',
            type: 'raster',
            source: 'osm',
            paint: { 'raster-opacity': 0.75, 'raster-saturation': -0.6, 'raster-contrast': 0.1 },
          },
        ],
      },
      center: [location.lon, location.lat],
      zoom: 11,
      attributionControl: false,
    })
    map.addControl(new maplibregl.AttributionControl({ compact: true }), 'bottom-right')
    mapRef.current = map
    return () => {
      map.remove()
      mapRef.current = null
    }
  }, [location])

  // event markers
  useEffect(() => {
    const map = mapRef.current
    if (!map || !events) return
    const markers = events
      .filter((e) => e.latitude && e.longitude)
      .map((e) => {
        const el = document.createElement('div')
        el.className =
          'flex h-9 w-9 items-center justify-center rounded-full border border-white/30 bg-abyss-800/85 shadow-glow cursor-pointer'
        el.innerHTML = ''
        const marker = new maplibregl.Marker({ element: el })
          .setLngLat([e.longitude!, e.latitude!])
          .addTo(map)
        el.addEventListener('click', () => setSelected(e))
        return marker
      })
    return () => markers.forEach((m) => m.remove())
  }, [events, location])

  return (
    <div className="glass overflow-hidden p-2">
      <div ref={containerRef} className="h-[52vh] w-full overflow-hidden rounded-2xl" />
      {selected && (
        <div className="glass-inset mt-3 p-4">
          <div className="flex items-start gap-3">
            <CategoryIcon category={selected.event_category} className="mt-0.5 h-6 w-6" />
            <div className="flex-1">
              <p className="font-medium text-slate-100">{selected.title}</p>
              <p className="mt-1 text-xs text-slate-400">
                {selected.distance_km !== null ? `${selected.distance_km.toFixed(1)} km away · ` : ''}
                severity {selected.severity}/100
              </p>
              {selected.url && (
                <a
                  href={selected.url}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-2 inline-block text-xs font-medium text-sky-300 underline-offset-2 hover:underline"
                >
                  View source article
                </a>
              )}
            </div>
            <button
              onClick={() => setSelected(null)}
              className="rounded-full px-2 text-slate-400 hover:text-slate-200"
              aria-label="Close"
            >
              ✕
            </button>
          </div>
        </div>
      )}
      <p className="mt-3 px-2 pb-1 text-center text-xs text-slate-500">
        Tap an icon to see event details. Map tiles © OpenStreetMap contributors.
      </p>
    </div>
  )
}
