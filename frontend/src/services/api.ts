import axios from 'axios'
import type {
  Insights,
  NewsArticle,
  SafetyAssessment,
  SymptomResponse,
} from '@/types/api'

const http = axios.create({ baseURL: '/api', timeout: 12000 })

export const api = {
  async safetyAssessment(
    userId: string,
    lat: number,
    lon: number,
    dest?: { lat: number; lon: number },
  ): Promise<SafetyAssessment> {
    const { data } = await http.get<SafetyAssessment>(
      `/v1/patients/${userId}/safety-assessment`,
      {
        params: {
          lat,
          lon,
          ...(dest ? { dest_lat: dest.lat, dest_lon: dest.lon } : {}),
        },
      },
    )
    return data
  },

  async newsNearby(
    lat: number,
    lon: number,
    radiusKm = 25,
    limit = 20,
  ): Promise<NewsArticle[]> {
    const { data } = await http.get<NewsArticle[]>('/v1/news/nearby', {
      params: { lat, lon, radius_km: radiusKm, limit },
    })
    return data
  },

  async insights(userId: string, days = 30): Promise<Insights> {
    const { data } = await http.get<Insights>(`/v1/patients/${userId}/insights`, {
      params: { days },
    })
    return data
  },

  async logSymptom(
    userId: string,
    body: { symptom_type: string; severity: number; lat: number; lon: number },
  ): Promise<SymptomResponse> {
    const { data } = await http.post<SymptomResponse>(
      `/v1/patients/${userId}/symptoms`,
      body,
    )
    return data
  },
}
