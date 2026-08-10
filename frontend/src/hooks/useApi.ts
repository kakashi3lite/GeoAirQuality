import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/services/api'
import { useApp } from '@/context/AppContext'

export function useSafetyAssessment() {
  const { userId, location, destination } = useApp()
  return useQuery({
    queryKey: ['safety', userId, location?.lat, location?.lon, destination?.lat, destination?.lon],
    queryFn: () =>
      api.safetyAssessment(userId, location!.lat, location!.lon, destination ?? undefined),
    enabled: !!location,
    staleTime: 5 * 60 * 1000,
    refetchInterval: 5 * 60 * 1000,
  })
}

export function useNewsNearby() {
  const { location } = useApp()
  return useQuery({
    queryKey: ['news', location?.lat, location?.lon],
    queryFn: () => api.newsNearby(location!.lat, location!.lon),
    enabled: !!location,
    staleTime: 10 * 60 * 1000,
  })
}

export function useInsights() {
  const { userId } = useApp()
  return useQuery({
    queryKey: ['insights', userId],
    queryFn: () => api.insights(userId),
    staleTime: 15 * 60 * 1000,
  })
}

export function useLogSymptom() {
  const { userId, location } = useApp()
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: { symptom_type: string; severity: number }) =>
      api.logSymptom(userId, { ...body, lat: location!.lat, lon: location!.lon }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['safety'] })
      qc.invalidateQueries({ queryKey: ['insights'] })
    },
  })
}
