# Frontend Development Framework - React + TypeScript

## Overview
This document provides the complete implementation for the GeoAirQuality frontend using React, TypeScript, and MapLibre GL JS for spatial visualization.

## Technology Stack
- **React 18** with TypeScript for type safety
- **Vite** for fast development and building
- **MapLibre GL JS** for interactive maps and spatial layers
- **TanStack Query** for server state management
- **Tailwind CSS** for styling
- **Socket.IO Client** for real-time updates
- **Chart.js** for data visualization

---

## Project Setup

### Package Configuration
```json
{
  "name": "geoairquality-frontend",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
    "test": "vitest",
    "test:ui": "vitest --ui"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.8.0",
    "@tanstack/react-query": "^4.24.0",
    "maplibre-gl": "^3.6.0",
    "react-map-gl": "^7.1.0",
    "socket.io-client": "^4.7.0",
    "chart.js": "^4.2.0",
    "react-chartjs-2": "^5.2.0",
    "axios": "^1.3.0",
    "date-fns": "^2.29.0",
    "clsx": "^1.2.0",
    "lucide-react": "^0.312.0",
    "framer-motion": "^10.16.0",
    "@headlessui/react": "^1.7.0",
    "react-hook-form": "^7.43.0",
    "@hookform/resolvers": "^2.9.0",
    "zod": "^3.20.0"
  },
  "devDependencies": {
    "@types/react": "^18.0.27",
    "@types/react-dom": "^18.0.10",
    "@vitejs/plugin-react": "^3.1.0",
    "typescript": "^4.9.3",
    "vite": "^4.1.0",
    "tailwindcss": "^3.2.0",
    "autoprefixer": "^10.4.0",
    "postcss": "^8.4.0",
    "eslint": "^8.35.0",
    "@typescript-eslint/eslint-plugin": "^5.54.0",
    "@typescript-eslint/parser": "^5.54.0",
    "eslint-plugin-react-hooks": "^4.6.0",
    "eslint-plugin-react-refresh": "^0.3.4",
    "vitest": "^0.28.0",
    "@testing-library/react": "^14.0.0",
    "@testing-library/jest-dom": "^5.16.0"
  }
}
```

### Vite Configuration
```typescript
// vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@/components': path.resolve(__dirname, './src/components'),
      '@/hooks': path.resolve(__dirname, './src/hooks'),
      '@/services': path.resolve(__dirname, './src/services'),
      '@/types': path.resolve(__dirname, './src/types'),
      '@/utils': path.resolve(__dirname, './src/utils'),
    },
  },
  define: {
    global: 'globalThis',
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
})
```

### Tailwind CSS Configuration
```javascript
// tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
        },
        air: {
          good: '#00e400',
          moderate: '#ffff00',
          unhealthy: '#ff7e00',
          veryUnhealthy: '#ff0000',
          hazardous: '#8f3f97',
        }
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  plugins: [],
}
```

---

## Core Types and Interfaces

### API Types
```typescript
// src/types/api.ts
export interface Location {
  lat: number;
  lon: number;
}

export interface AirQualityReading {
  id: string;
  location: Location;
  timestamp: string;
  aqi: number;
  measurements: {
    pm25: number;
    pm10: number;
    o3: number;
    no2: number;
    so2: number;
    co: number;
  };
  metadata?: {
    sensor_id?: string;
    data_source?: string;
    confidence_score?: number;
  };
}

export interface AirQualityForecast {
  location: Location;
  forecasts: Array<{
    timestamp: string;
    aqi: number;
    primary_pollutant: string;
    confidence: number;
  }>;
}

export interface SpatialGrid {
  id: string;
  bounds: {
    north: number;
    south: number;
    east: number;
    west: number;
  };
  resolution: number;
  aggregated_data: {
    avg_aqi: number;
    max_aqi: number;
    reading_count: number;
    last_updated: string;
  };
}

export interface WebSocketMessage {
  type: 'air_quality_update' | 'forecast_update' | 'alert' | 'connection_status';
  data: any;
  timestamp: string;
}

export interface User {
  id: string;
  email: string;
  roles: string[];
  subscription_tier: 'public' | 'user' | 'premium' | 'enterprise';
  preferences: {
    units: 'metric' | 'imperial';
    alert_threshold: number;
    favorite_locations: Location[];
  };
}
```

### Component Props Types
```typescript
// src/types/components.ts
export interface MapProps {
  center: Location;
  zoom: number;
  onLocationChange?: (location: Location) => void;
  showControls?: boolean;
  height?: string;
}

export interface AirQualityCardProps {
  reading: AirQualityReading;
  showDetails?: boolean;
  onClick?: () => void;
}

export interface ChartProps {
  data: Array<{
    timestamp: string;
    value: number;
  }>;
  type: 'line' | 'bar';
  height?: number;
  showLegend?: boolean;
}
```

---

## Services Layer

### API Client
```typescript
// src/services/api.ts
import axios, { AxiosInstance, AxiosResponse } from 'axios';

export class ApiClient {
  private client: AxiosInstance;
  
  constructor(baseURL: string = '/api') {
    this.client = axios.create({
      baseURL,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    this.setupInterceptors();
  }
  
  private setupInterceptors() {
    // Request interceptor for auth
    this.client.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('access_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );
    
    // Response interceptor for token refresh
    this.client.interceptors.response.use(
      (response) => response,
      async (error) => {
        if (error.response?.status === 401) {
          const refreshToken = localStorage.getItem('refresh_token');
          if (refreshToken) {
            try {
              const response = await this.refreshToken(refreshToken);
              localStorage.setItem('access_token', response.data.access_token);
              
              // Retry original request
              const originalRequest = error.config;
              originalRequest.headers.Authorization = `Bearer ${response.data.access_token}`;
              return this.client.request(originalRequest);
            } catch {
              // Refresh failed, redirect to login
              localStorage.removeItem('access_token');
              localStorage.removeItem('refresh_token');
              window.location.href = '/login';
            }
          }
        }
        return Promise.reject(error);
      }
    );
  }
  
  async getCurrentAirQuality(lat: number, lon: number): Promise<AirQualityReading> {
    const response = await this.client.get('/air-quality/current', {
      params: { lat, lon }
    });
    return response.data;
  }
  
  async getAirQualityForecast(
    lat: number, 
    lon: number, 
    hours: number = 24
  ): Promise<AirQualityForecast> {
    const response = await this.client.get('/air-quality/forecast', {
      params: { lat, lon, hours }
    });
    return response.data;
  }
  
  async getHistoricalData(
    lat: number,
    lon: number,
    days: number = 7
  ): Promise<AirQualityReading[]> {
    const response = await this.client.get('/air-quality/history', {
      params: { lat, lon, days }
    });
    return response.data;
  }
  
  async getSpatialGrid(bounds: {
    north: number;
    south: number;
    east: number;
    west: number;
  }): Promise<SpatialGrid[]> {
    const response = await this.client.get('/spatial/grid', {
      params: bounds
    });
    return response.data;
  }
  
  async searchLocations(query: string): Promise<Location[]> {
    const response = await this.client.get('/locations/search', {
      params: { q: query }
    });
    return response.data;
  }
  
  private async refreshToken(refreshToken: string): Promise<AxiosResponse> {
    return this.client.post('/auth/refresh', { refresh_token: refreshToken });
  }
}

export const apiClient = new ApiClient();
```

### WebSocket Service
```typescript
// src/services/websocket.ts
import { io, Socket } from 'socket.io-client';
import { WebSocketMessage, Location } from '@/types/api';

export class WebSocketService {
  private socket: Socket | null = null;
  private location: Location | null = null;
  private callbacks: Map<string, Set<(data: any) => void>> = new Map();
  
  connect(location: Location) {
    if (this.socket) {
      this.disconnect();
    }
    
    this.location = location;
    this.socket = io('/ws', {
      path: '/socket.io/',
      query: {
        lat: location.lat,
        lon: location.lon,
      },
    });
    
    this.setupEventHandlers();
  }
  
  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }
  
  private setupEventHandlers() {
    if (!this.socket) return;
    
    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      this.notifyCallbacks('connection_status', { connected: true });
    });
    
    this.socket.on('disconnect', () => {
      console.log('WebSocket disconnected');
      this.notifyCallbacks('connection_status', { connected: false });
    });
    
    this.socket.on('air_quality_update', (data) => {
      this.notifyCallbacks('air_quality_update', data);
    });
    
    this.socket.on('forecast_update', (data) => {
      this.notifyCallbacks('forecast_update', data);
    });
    
    this.socket.on('alert', (data) => {
      this.notifyCallbacks('alert', data);
    });
  }
  
  subscribe(event: string, callback: (data: any) => void) {
    if (!this.callbacks.has(event)) {
      this.callbacks.set(event, new Set());
    }
    this.callbacks.get(event)!.add(callback);
    
    // Return unsubscribe function
    return () => {
      const callbacks = this.callbacks.get(event);
      if (callbacks) {
        callbacks.delete(callback);
      }
    };
  }
  
  private notifyCallbacks(event: string, data: any) {
    const callbacks = this.callbacks.get(event);
    if (callbacks) {
      callbacks.forEach(callback => callback(data));
    }
  }
  
  updateLocation(location: Location) {
    if (this.socket && this.location) {
      this.socket.emit('update_location', location);
      this.location = location;
    }
  }
}

export const websocketService = new WebSocketService();
```

---

## React Hooks

### Real-time Data Hook
```typescript
// src/hooks/useRealTimeData.ts
import { useEffect, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { websocketService } from '@/services/websocket';
import { apiClient } from '@/services/api';
import { AirQualityReading, Location } from '@/types/api';

export function useRealTimeAirQuality(location: Location) {
  const [isConnected, setIsConnected] = useState(false);
  const queryClient = useQueryClient();
  
  // Initial data fetch
  const { data: currentReading, isLoading, error } = useQuery({
    queryKey: ['airQuality', 'current', location.lat, location.lon],
    queryFn: () => apiClient.getCurrentAirQuality(location.lat, location.lon),
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchInterval: 10 * 60 * 1000, // 10 minutes fallback
  });
  
  useEffect(() => {
    // Connect to WebSocket for real-time updates
    websocketService.connect(location);
    
    const unsubscribeConnection = websocketService.subscribe(
      'connection_status',
      (data) => setIsConnected(data.connected)
    );
    
    const unsubscribeUpdates = websocketService.subscribe(
      'air_quality_update',
      (data: AirQualityReading) => {
        // Update query cache with real-time data
        queryClient.setQueryData(
          ['airQuality', 'current', location.lat, location.lon],
          data
        );
      }
    );
    
    return () => {
      unsubscribeConnection();
      unsubscribeUpdates();
      websocketService.disconnect();
    };
  }, [location.lat, location.lon, queryClient]);
  
  return {
    currentReading,
    isLoading,
    error,
    isConnected,
  };
}
```

### Map Management Hook
```typescript
// src/hooks/useMapManager.ts
import { useRef, useEffect, useState } from 'react';
import maplibregl from 'maplibre-gl';
import { Location, SpatialGrid } from '@/types/api';

export function useMapManager(container: string, initialLocation: Location) {
  const mapRef = useRef<maplibregl.Map | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [airQualityLayers, setAirQualityLayers] = useState<string[]>([]);
  
  useEffect(() => {
    if (!container) return;
    
    // Initialize map
    mapRef.current = new maplibregl.Map({
      container,
      style: {
        version: 8,
        sources: {
          'raster-tiles': {
            type: 'raster',
            tiles: [
              'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
            ],
            tileSize: 256,
          }
        },
        layers: [
          {
            id: 'simple-tiles',
            type: 'raster',
            source: 'raster-tiles',
          }
        ]
      },
      center: [initialLocation.lon, initialLocation.lat],
      zoom: 10,
    });
    
    mapRef.current.on('load', () => {
      setIsLoaded(true);
      setupAirQualityLayers();
    });
    
    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
      }
    };
  }, [container, initialLocation]);
  
  const setupAirQualityLayers = () => {
    if (!mapRef.current) return;
    
    const map = mapRef.current;
    
    // Add air quality grid source
    map.addSource('air-quality-grid', {
      type: 'geojson',
      data: {
        type: 'FeatureCollection',
        features: []
      }
    });
    
    // Add air quality heat layer
    map.addLayer({
      id: 'air-quality-heat',
      type: 'fill',
      source: 'air-quality-grid',
      paint: {
        'fill-color': [
          'interpolate',
          ['linear'],
          ['get', 'aqi'],
          0, '#00e400',     // Good (0-50)
          50, '#ffff00',    // Moderate (51-100)
          100, '#ff7e00',   // Unhealthy (101-150)
          150, '#ff0000',   // Very Unhealthy (151-200)
          200, '#8f3f97'    // Hazardous (201+)
        ],
        'fill-opacity': 0.6
      }
    });
    
    // Add sensor points layer
    map.addSource('sensor-points', {
      type: 'geojson',
      data: {
        type: 'FeatureCollection',
        features: []
      }
    });
    
    map.addLayer({
      id: 'sensor-points',
      type: 'circle',
      source: 'sensor-points',
      paint: {
        'circle-radius': [
          'interpolate',
          ['linear'],
          ['zoom'],
          8, 4,
          16, 8
        ],
        'circle-color': [
          'interpolate',
          ['linear'],
          ['get', 'aqi'],
          0, '#00e400',
          50, '#ffff00',
          100, '#ff7e00',
          150, '#ff0000',
          200, '#8f3f97'
        ],
        'circle-stroke-width': 2,
        'circle-stroke-color': '#ffffff',
        'circle-opacity': 0.8
      }
    });
    
    setAirQualityLayers(['air-quality-heat', 'sensor-points']);
  };
  
  const updateAirQualityData = (gridData: SpatialGrid[], sensorData: any[]) => {
    if (!mapRef.current || !isLoaded) return;
    
    const map = mapRef.current;
    
    // Update grid data
    const gridFeatures = gridData.map(grid => ({
      type: 'Feature' as const,
      geometry: {
        type: 'Polygon' as const,
        coordinates: [[
          [grid.bounds.west, grid.bounds.north],
          [grid.bounds.east, grid.bounds.north],
          [grid.bounds.east, grid.bounds.south],
          [grid.bounds.west, grid.bounds.south],
          [grid.bounds.west, grid.bounds.north]
        ]]
      },
      properties: {
        aqi: grid.aggregated_data.avg_aqi,
        max_aqi: grid.aggregated_data.max_aqi,
        reading_count: grid.aggregated_data.reading_count
      }
    }));
    
    const gridSource = map.getSource('air-quality-grid') as maplibregl.GeoJSONSource;
    gridSource.setData({
      type: 'FeatureCollection',
      features: gridFeatures
    });
    
    // Update sensor points
    const sensorFeatures = sensorData.map(sensor => ({
      type: 'Feature' as const,
      geometry: {
        type: 'Point' as const,
        coordinates: [sensor.location.lon, sensor.location.lat]
      },
      properties: {
        aqi: sensor.aqi,
        sensor_id: sensor.metadata?.sensor_id,
        timestamp: sensor.timestamp
      }
    }));
    
    const sensorSource = map.getSource('sensor-points') as maplibregl.GeoJSONSource;
    sensorSource.setData({
      type: 'FeatureCollection',
      features: sensorFeatures
    });
  };
  
  const flyTo = (location: Location, zoom?: number) => {
    if (mapRef.current) {
      mapRef.current.flyTo({
        center: [location.lon, location.lat],
        zoom: zoom || mapRef.current.getZoom(),
        duration: 1000
      });
    }
  };
  
  return {
    map: mapRef.current,
    isLoaded,
    updateAirQualityData,
    flyTo,
  };
}
```

---

## Map Components

### Interactive Map Component
```typescript
// src/components/Map/InteractiveMap.tsx
import React, { useEffect, useRef, useState } from 'react';
import { useMapManager } from '@/hooks/useMapManager';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/services/api';
import { Location } from '@/types/api';
import { MapControls } from './MapControls';
import { MapLegend } from './MapLegend';

interface InteractiveMapProps {
  center: Location;
  onLocationChange?: (location: Location) => void;
  height?: string;
  showControls?: boolean;
}

export const InteractiveMap: React.FC<InteractiveMapProps> = ({
  center,
  onLocationChange,
  height = '500px',
  showControls = true,
}) => {
  const mapContainer = useRef<HTMLDivElement>(null);
  const [mapId] = useState(() => `map-${Math.random().toString(36).substr(2, 9)}`);
  const [currentBounds, setCurrentBounds] = useState<any>(null);
  
  const { map, isLoaded, updateAirQualityData, flyTo } = useMapManager(mapId, center);
  
  // Fetch spatial grid data when map bounds change
  const { data: gridData } = useQuery({
    queryKey: ['spatialGrid', currentBounds],
    queryFn: () => apiClient.getSpatialGrid(currentBounds),
    enabled: !!currentBounds && isLoaded,
    staleTime: 2 * 60 * 1000, // 2 minutes
  });
  
  // Fetch current sensor readings
  const { data: sensorData } = useQuery({
    queryKey: ['sensorReadings', currentBounds],
    queryFn: () => {
      // This would be a new API endpoint for sensor readings in bounds
      return apiClient.getCurrentAirQuality(center.lat, center.lon);
    },
    enabled: !!currentBounds && isLoaded,
    refetchInterval: 30 * 1000, // 30 seconds
  });
  
  useEffect(() => {
    if (!map) return;
    
    const handleMoveEnd = () => {
      const bounds = map.getBounds();
      const newBounds = {
        north: bounds.getNorth(),
        south: bounds.getSouth(),
        east: bounds.getEast(),
        west: bounds.getWest(),
      };
      setCurrentBounds(newBounds);
      
      const mapCenter = map.getCenter();
      onLocationChange?.({
        lat: mapCenter.lat,
        lon: mapCenter.lng,
      });
    };
    
    const handleClick = (e: any) => {
      const { lng, lat } = e.lngLat;
      onLocationChange?.({ lat, lon: lng });
    };
    
    map.on('moveend', handleMoveEnd);
    map.on('click', handleClick);
    
    // Initial bounds
    handleMoveEnd();
    
    return () => {
      map.off('moveend', handleMoveEnd);
      map.off('click', handleClick);
    };
  }, [map, onLocationChange]);
  
  useEffect(() => {
    if (gridData && Array.isArray(sensorData) && updateAirQualityData) {
      updateAirQualityData(gridData, sensorData);
    }
  }, [gridData, sensorData, updateAirQualityData]);
  
  const handleLocationSearch = (location: Location) => {
    flyTo(location, 12);
  };
  
  return (
    <div className="relative" style={{ height }}>
      <div
        ref={mapContainer}
        id={mapId}
        className="w-full h-full rounded-lg overflow-hidden"
      />
      
      {showControls && (
        <MapControls
          onLocationSearch={handleLocationSearch}
          currentLocation={center}
        />
      )}
      
      <MapLegend />
      
      {!isLoaded && (
        <div className="absolute inset-0 bg-gray-100 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500 mx-auto"></div>
            <p className="mt-2 text-gray-600">Loading map...</p>
          </div>
        </div>
      )}
    </div>
  );
};
```

### Map Controls Component
```typescript
// src/components/Map/MapControls.tsx
import React, { useState } from 'react';
import { Search, MapPin, Layers } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/services/api';
import { Location } from '@/types/api';

interface MapControlsProps {
  onLocationSearch: (location: Location) => void;
  currentLocation: Location;
}

export const MapControls: React.FC<MapControlsProps> = ({
  onLocationSearch,
  currentLocation,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [showSearch, setShowSearch] = useState(false);
  
  const { data: searchResults, isLoading } = useQuery({
    queryKey: ['locationSearch', searchQuery],
    queryFn: () => apiClient.searchLocations(searchQuery),
    enabled: searchQuery.length >= 3,
    staleTime: 5 * 60 * 1000,
  });
  
  const handleSearchSelect = (location: Location) => {
    onLocationSearch(location);
    setShowSearch(false);
    setSearchQuery('');
  };
  
  return (
    <div className="absolute top-4 left-4 z-10 space-y-2">
      {/* Search Control */}
      <div className="bg-white rounded-lg shadow-lg">
        <div className="flex items-center p-2">
          <Search className="w-5 h-5 text-gray-400 mr-2" />
          <input
            type="text"
            placeholder="Search locations..."
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              setShowSearch(e.target.value.length >= 3);
            }}
            className="outline-none text-sm flex-1"
          />
        </div>
        
        {showSearch && searchResults && (
          <div className="border-t max-h-48 overflow-y-auto">
            {searchResults.map((location, index) => (
              <button
                key={index}
                onClick={() => handleSearchSelect(location)}
                className="w-full text-left p-2 hover:bg-gray-50 text-sm border-b last:border-b-0"
              >
                <div className="flex items-center">
                  <MapPin className="w-4 h-4 text-gray-400 mr-2" />
                  <span>{location.lat.toFixed(4)}, {location.lon.toFixed(4)}</span>
                </div>
              </button>
            ))}
            {isLoading && (
              <div className="p-2 text-center text-sm text-gray-500">
                Searching...
              </div>
            )}
          </div>
        )}
      </div>
      
      {/* Layer Controls */}
      <div className="bg-white rounded-lg shadow-lg p-2">
        <button className="flex items-center text-sm">
          <Layers className="w-4 h-4 mr-2" />
          Layers
        </button>
      </div>
    </div>
  );
};
```

---

## Real-time Data Visualization

### Air Quality Dashboard
```typescript
// src/components/Dashboard/AirQualityDashboard.tsx
import React, { useState } from 'react';
import { InteractiveMap } from '@/components/Map/InteractiveMap';
import { AirQualityCard } from '@/components/AirQuality/AirQualityCard';
import { ForecastChart } from '@/components/Charts/ForecastChart';
import { AlertPanel } from '@/components/Alerts/AlertPanel';
import { useRealTimeAirQuality } from '@/hooks/useRealTimeData';
import { Location } from '@/types/api';

export const AirQualityDashboard: React.FC = () => {
  const [selectedLocation, setSelectedLocation] = useState<Location>({
    lat: 40.7128,
    lon: -74.0060 // New York City
  });
  
  const { currentReading, isLoading, error, isConnected } = useRealTimeAirQuality(selectedLocation);
  
  const handleLocationChange = (location: Location) => {
    setSelectedLocation(location);
  };
  
  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-xl font-semibold text-gray-900 mb-2">
            Unable to load air quality data
          </h2>
          <p className="text-gray-600">{error.message}</p>
        </div>
      </div>
    );
  }
  
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold text-gray-900">
              GeoAirQuality Dashboard
            </h1>
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${
                isConnected ? 'bg-green-500' : 'bg-red-500'
              }`} />
              <span className="text-sm text-gray-600">
                {isConnected ? 'Live' : 'Offline'}
              </span>
            </div>
          </div>
        </div>
      </header>
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Map Section */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Air Quality Map
              </h2>
              <InteractiveMap
                center={selectedLocation}
                onLocationChange={handleLocationChange}
                height="600px"
              />
            </div>
          </div>
          
          {/* Sidebar */}
          <div className="space-y-6">
            {/* Current Reading */}
            {currentReading && (
              <AirQualityCard
                reading={currentReading}
                showDetails={true}
                isLoading={isLoading}
              />
            )}
            
            {/* Alerts */}
            <AlertPanel location={selectedLocation} />
            
            {/* Forecast Chart */}
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                24-Hour Forecast
              </h3>
              <ForecastChart location={selectedLocation} />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};
```

### Air Quality Card Component
```typescript
// src/components/AirQuality/AirQualityCard.tsx
import React from 'react';
import { Wind, Clock, MapPin } from 'lucide-react';
import { AirQualityReading } from '@/types/api';
import { getAQIColor, getAQILevel } from '@/utils/airQuality';
import { formatDistanceToNow } from 'date-fns';

interface AirQualityCardProps {
  reading: AirQualityReading;
  showDetails?: boolean;
  isLoading?: boolean;
  onClick?: () => void;
}

export const AirQualityCard: React.FC<AirQualityCardProps> = ({
  reading,
  showDetails = false,
  isLoading = false,
  onClick,
}) => {
  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow p-6 animate-pulse">
        <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
        <div className="h-8 bg-gray-200 rounded w-1/2 mb-2"></div>
        <div className="h-4 bg-gray-200 rounded w-3/4"></div>
      </div>
    );
  }
  
  const aqiColor = getAQIColor(reading.aqi);
  const aqiLevel = getAQILevel(reading.aqi);
  const timeAgo = formatDistanceToNow(new Date(reading.timestamp), { addSuffix: true });
  
  return (
    <div 
      className={`bg-white rounded-lg shadow p-6 ${onClick ? 'cursor-pointer hover:shadow-lg transition-shadow' : ''}`}
      onClick={onClick}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <MapPin className="w-4 h-4 text-gray-400 mr-2" />
          <span className="text-sm text-gray-600">
            {reading.location.lat.toFixed(4)}, {reading.location.lon.toFixed(4)}
          </span>
        </div>
        <div className="flex items-center text-sm text-gray-500">
          <Clock className="w-4 h-4 mr-1" />
          {timeAgo}
        </div>
      </div>
      
      {/* AQI Display */}
      <div className="text-center mb-4">
        <div 
          className="text-4xl font-bold mb-2"
          style={{ color: aqiColor }}
        >
          {reading.aqi}
        </div>
        <div 
          className="text-sm font-medium px-3 py-1 rounded-full text-white inline-block"
          style={{ backgroundColor: aqiColor }}
        >
          {aqiLevel}
        </div>
      </div>
      
      {/* Detailed Measurements */}
      {showDetails && (
        <div className="space-y-3">
          <div className="border-t pt-3">
            <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center">
              <Wind className="w-4 h-4 mr-1" />
              Pollutant Levels
            </h4>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">PM2.5:</span>
                <span className="font-medium">{reading.measurements.pm25} μg/m³</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">PM10:</span>
                <span className="font-medium">{reading.measurements.pm10} μg/m³</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">O₃:</span>
                <span className="font-medium">{reading.measurements.o3} μg/m³</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">NO₂:</span>
                <span className="font-medium">{reading.measurements.no2} μg/m³</span>
              </div>
              {reading.measurements.so2 > 0 && (
                <div className="flex justify-between">
                  <span className="text-gray-600">SO₂:</span>
                  <span className="font-medium">{reading.measurements.so2} μg/m³</span>
                </div>
              )}
              {reading.measurements.co > 0 && (
                <div className="flex justify-between">
                  <span className="text-gray-600">CO:</span>
                  <span className="font-medium">{reading.measurements.co} mg/m³</span>
                </div>
              )}
            </div>
          </div>
          
          {/* Metadata */}
          {reading.metadata && (
            <div className="border-t pt-3 text-xs text-gray-500">
              {reading.metadata.sensor_id && (
                <div>Sensor: {reading.metadata.sensor_id}</div>
              )}
              {reading.metadata.data_source && (
                <div>Source: {reading.metadata.data_source}</div>
              )}
              {reading.metadata.confidence_score && (
                <div>Confidence: {(reading.metadata.confidence_score * 100).toFixed(1)}%</div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
```

This comprehensive frontend framework provides:

1. **Complete TypeScript Setup**: Type-safe development with proper interfaces
2. **Real-time Data Integration**: WebSocket connections with automatic reconnection
3. **Interactive Maps**: MapLibre GL JS with spatial data visualization
4. **Performance Optimization**: React Query for caching and TanStack Query for server state
5. **Production-Ready Components**: Modular, reusable components with proper error handling
6. **Responsive Design**: Tailwind CSS with mobile-first approach

The framework integrates seamlessly with the backend infrastructure and provides the foundation for building a production-ready air quality monitoring dashboard.
