// Exact TypeScript mirrors of the API's Pydantic response models.

export interface Location {
  lat: number
  lon: number
}

export interface ContributingFactor {
  factor: string
  value: number | string | null
  threshold: number | string | null
  unit: string | null
  detail: string
}

export interface Recommendation {
  type: 'general' | 'precaution' | 'location' | 'timing' | 'route'
  text: string
}

export interface NewsEvent {
  title: string
  category: string
  severity: number
  respiratory_relevance: number
  distance_km: number
  latitude?: number | null
  longitude?: number | null
}

export interface RouteSegment {
  point: string
  safety_score: number
  aqi: number | null
  pm25: number | null
}

export interface RouteRisk {
  route_risk_score: number
  worst_segment: string
  safest_segment: string
  segments: RouteSegment[]
  news_notes: { type: string; text: string }[]
}

export interface CurrentConditions {
  aqi: number | null
  pm25: number | null
  pm10: number | null
  o3: number | null
  no2: number | null
  temperature: number | null
  humidity: number | null
  wind_speed: number | null
  wind_direction: string | null
  reading_count: number
  aq_reading_count: number
  source_timestamp: string | null
  grid_id: string | null
}

export interface SafetyAssessment {
  user_id: string
  safety_score: number
  risk_level: 'low' | 'moderate' | 'high' | 'very_high'
  summary: string
  data_status: 'available' | 'partial' | 'unavailable'
  component_scores: { aqi: number; weather: number; news: number; history: number }
  contributions: { aqi: number; weather: number; news: number; history: number }
  contributing_factors: ContributingFactor[]
  current_conditions: CurrentConditions
  recommendations: Recommendation[]
  nearby_events: NewsEvent[]
  route_risk: RouteRisk | null
  generated_at: string
}

export interface NewsArticle {
  id: number
  title: string
  summary: string | null
  url: string | null
  source_name: string
  source_type: string
  event_category: string
  severity: number
  respiratory_relevance: number
  distance_km: number | null
  latitude?: number | null
  longitude?: number | null
  published_at: string
}

export interface SymptomResponse {
  id: number
  symptom_type: string
  severity: number
  weather_snapshot: Record<string, number | null>
  logged_at: string
}

export interface Trigger {
  factor: string
  correlation: number
  occurrences: number
}

export interface TimeBucket {
  hour: number
  avg_severity: number
}

export interface Insights {
  top_triggers: Trigger[]
  safest_times: TimeBucket[]
  riskiest_times: TimeBucket[]
  recent_trend: string
  period_days: number
}

// ---- Risk → color/visual helpers ----
export type RiskLevel = SafetyAssessment['risk_level']

export const RISK_STYLES: Record<
  RiskLevel,
  { color: string; soft: string; label: string; bar: string }
> = {
  low: {
    color: '#00e400',
    soft: 'rgba(0,228,100,0.14)',
    label: 'Low risk',
    bar: '#00e400',
  },
  moderate: {
    color: '#ffd400',
    soft: 'rgba(255,212,0,0.14)',
    label: 'Moderate risk',
    bar: '#ffd400',
  },
  high: {
    color: '#ff7e00',
    soft: 'rgba(255,126,0,0.16)',
    label: 'High risk',
    bar: '#ff7e00',
  },
  very_high: {
    color: '#ff3b30',
    soft: 'rgba(255,59,48,0.16)',
    label: 'Very high risk',
    bar: '#ff3b30',
  },
}
