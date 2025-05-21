

'use client'

import { useEffect, useState } from 'react'
import { fetchFromAPI } from '@/lib/api'
import { MusicalNoteIcon } from '@heroicons/react/24/solid'
import { useSession } from 'next-auth/react'
import { useRouter } from 'next/navigation'

interface Recommendation {
  song_name: string
  artist_name: string
  album_name: string
  genre: string
}

export default function DashboardPage() {
  const { data: session, status } = useSession()
  const router = useRouter()
  const [recommendations, setRecommendations] = useState<Recommendation[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (status === 'unauthenticated') {
      router.push('/login')
    }
  }, [status])

  useEffect(() => {
    const fetchRecommendations = async () => {
      if (session) {
        setLoading(true)
        try {
          const data = await fetchFromAPI(`/recommend/user?model_name=graphsage`, {
            method: 'POST',
            body: JSON.stringify({
              user_id: session.user.id,
              num_recommendations: 10,
            }),
          })
          setRecommendations(data.recommendations)
        } catch (err) {
          console.error('Failed to fetch recommendations:', err)
        } finally {
          setLoading(false)
        }
      }
    }

    fetchRecommendations()
  }, [session])

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-300 via-gray-700 to-black text-white flex flex-col items-center p-8 sm:p-20">
      <header className="flex flex-col items-center gap-4 animate-fadeIn">
        <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-gray-100 to-gray-400 animate-slideUp">
          Your Dashboard
        </h1>
        <p className="text-lg text-gray-300 animate-fadeInDelay">
          Discover songs curated just for you.
        </p>
      </header>

      <main className="flex flex-col items-center gap-8 mt-10 w-full max-w-5xl">
        {loading ? (
          <p className="text-gray-400 text-center">Loading your recommendations...</p>
        ) : recommendations.length > 0 ? (
          <ul className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {recommendations.map((rec, i) => (
              <li
                key={i}
                className="bg-gray-800 p-6 rounded-lg hover:shadow-lg hover:scale-105 transition transform hover:-translate-y-1"
              >
                <MusicalNoteIcon className="w-10 h-10 text-red-500 mb-4 hover:rotate-12 transition-transform" />
                <div className="font-semibold text-xl text-white">
                  {rec.song_name}
                </div>
                <div className="text-sm text-gray-400 mt-2">
                  {rec.artist_name}, {rec.album_name}, {rec.genre}
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-gray-400 text-center">
            No recommendations available. Try listening to more music!
          </p>
        )}
      </main>
    </div>
  )
}