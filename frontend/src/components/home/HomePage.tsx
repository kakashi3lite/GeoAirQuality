import { motion } from 'framer-motion'
import { useSafetyAssessment } from '@/hooks/useApi'
import { SafetyScoreCard } from '@/components/home/SafetyScoreCard'
import { ContributingFactors } from '@/components/home/ContributingFactors'
import { RecommendationsList } from '@/components/home/RecommendationsList'
import { NearbyAlertsStrip } from '@/components/home/NearbyAlertsStrip'
import { RouteRiskPreview } from '@/components/home/RouteRiskPreview'

export default function HomePage() {
  const { data, isLoading, isError, refetch } = useSafetyAssessment()

  if (isLoading) return <HomeSkeleton />
  if (isError || !data) {
    return (
      <div className="glass p-8 text-center">
        <p className="text-slate-300">We couldn't reach the safety service.</p>
        <button
          onClick={() => refetch()}
          className="mt-4 rounded-2xl bg-sky-500 px-6 py-3 font-semibold text-white"
        >
          Try again
        </button>
      </div>
    )
  }

  return (
    <motion.section initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
      <SafetyScoreCard data={data} />
      <ContributingFactors data={data} />
      <RecommendationsList recommendations={data.recommendations} />
      {data.route_risk && <RouteRiskPreview risk={data.route_risk} />}
      <NearbyAlertsStrip events={data.nearby_events} />
    </motion.section>
  )
}

function HomeSkeleton() {
  return (
    <div className="space-y-4">
      <div className="glass flex h-52 animate-pulse items-center justify-center">
        <div className="h-32 w-32 rounded-full bg-white/5" />
      </div>
      <div className="glass h-36 animate-pulse" />
      <div className="glass h-28 animate-pulse" />
    </div>
  )
}

