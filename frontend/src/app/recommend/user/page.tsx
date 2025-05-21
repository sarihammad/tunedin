'use client';
import { useState } from 'react';
import { fetchFromAPI } from '@/lib/api';
import { MusicalNoteIcon } from '@heroicons/react/24/solid';

interface Recommendation {
    song_name: string;
    artist_name: string;
    album_name: string;
    genre: string;
}

export default function UserRecommendationPage() {
    const [userId, setUserId] = useState<string>('0');
    const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
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
            });
            setRecommendations(data.recommendations);
        } catch (err: unknown) {
            if (err instanceof Error) {
                alert(err.message);
            } else {
                alert('An unexpected error occurred');
            }
        } finally {
            setLoading(false);
        }
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-300 via-gray-700 to-black text-white flex flex-col items-center p-8 sm:p-20">
            <header className="flex flex-col items-center gap-4 animate-fadeIn">
                <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-gray-100 to-gray-400 animate-slideUp">
                    User Recommendations
                </h1>
                <p className="text-lg text-gray-300 animate-fadeInDelay">
                    Discover personalized music recommendations tailored just for you.
                </p>
            </header>

            <main className="flex flex-col items-center gap-8 mt-10 w-full max-w-3xl">
                <div className="w-full">
                    <input
                        value={userId}
                        onChange={(e) => setUserId(e.target.value)}
                        className="w-full p-4 rounded-lg bg-gray-800 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-red-500"
                        placeholder="Enter user ID (e.g., 0)"
                    />
                </div>
                <button
                    onClick={getRecommendations}
                    disabled={loading}
                    className={`bg-gradient-to-r from-red-500 to-red-700 text-white font-medium py-4 px-8 rounded-lg transition transform hover:scale-105 ${
                        loading ? 'opacity-50 cursor-not-allowed' : ''
                    }`}
                >
                    {loading ? 'Loading...' : 'Get Recommendations'}
                </button>

                <div className="mt-10 w-full">
                    {recommendations.length > 0 ? (
                        <ul className="grid grid-cols-1 gap-6">
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
                            {loading
                                ? 'Fetching recommendations...'
                                : 'No recommendations available. Try again!'}
                        </p>
                    )}
                </div>
            </main>
        </div>
    );
}