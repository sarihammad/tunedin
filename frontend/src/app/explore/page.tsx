

'use client'
import { useState, useEffect } from 'react'
import { fetchFromAPI } from '@/lib/api'
import { MusicalNoteIcon } from '@heroicons/react/24/solid'

interface Recommendation {
  song_name: string;
  artist_name: string;
  album_name: string;
  genre: string;
}

export default function ExplorePage() {
  const [recommendations, setRecommendations] = useState<Recommendation[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const fetchRecommendations = async () => {
      setLoading(true)
      try {
        const data = await fetchFromAPI(`/recommend/song?model_name=graphsage`, {
          method: 'POST',
          body: JSON.stringify({
            song_id: 'song_0',
            num_recommendations: 12,
          }),
        })
        setRecommendations(data.recommendations)
      } catch (err: unknown) {
        if (err instanceof Error) {
          alert(err.message)
        } else {
          alert('An unexpected error occurred')
        }
      } finally {
        setLoading(false)
      }
    }

    fetchRecommendations()
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-300 via-gray-700 to-black text-white flex flex-col items-center p-8 sm:p-20">
      <header className="flex flex-col items-center gap-4 animate-fadeIn">
        <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-gray-100 to-gray-400 animate-slideUp">
          Explore Music
        </h1>
        <p className="text-lg text-gray-300 animate-fadeInDelay">
          Discover new tracks and expand your playlist.
        </p>
      </header>

      <main className="flex flex-col items-center gap-8 mt-10 w-full max-w-5xl">
        {loading ? (
          <p className="text-gray-400 text-center">Loading recommendations...</p>
        ) : (
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
        )}
      </main>
    </div>
  )
}