'use client'
import { useState } from 'react'
import { fetchFromAPI } from '@/lib/api'

export default function UserRecommendationPage() {
    const [userId, setUserId] = useState<string>('0');
    const [recommendations, setRecommendations] = useState<any[]>([]);
    const [loading, setLoading] = useState<boolean>(false);

    async function getRecommendations() {
        setLoading(true);
        try {
            const data = await fetchFromAPI('/recommend/user?model_name=graphsage', {
                method: 'POST',
                body: JSON.stringify({
                    userId: userId,
                    num_recommendations: 10,
                    exclude_listened: true,
                }),
            })
            setRecommendations(data.recommendations);
        } catch (err: any) {
            console.error("Error: " + err.message);
        } finally {
            setLoading(false);
        }
    }

    return (
        <main className="max-w-3xl mx-auto p-6">
            <h1 className="text-3xl font-bold mb-4">Recommend for User</h1>
            <input
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                className="border p-2 mb-4 w-full"
                placeholder="Enter user ID (e.g., 0)"
            />
            <button 
                onClick={getRecommendations}
                disabled={loading}
                className="bg-green-600 text-whit px-4 py-2 rounded"
            >
                {loading ? 'Loading...' : 'Get Recommendations'}
            </button>
            <div className="mt-6">
                {recommendations.length > 0 ? (
                    <ul className="space-y-3">
                        {recommendations.map((rec, i) => (
                            <li key={i} className="border rounded p-3">
                                <div className="font-semibold">{rec.song_name}</div>
                                <div className="text-sm text gray-600">
                                    {rec.artist_name}, {rec.album_name}, {rec.genre}
                                </div>
                            </li>
                        ))}
                    </ul>
                ) : null}
            </div>
        </main>
    )

}