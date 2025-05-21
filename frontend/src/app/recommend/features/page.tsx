'use client';

import { useState } from "react";
import { fetchFromAPI } from '@/lib/api';
import { MusicalNoteIcon } from '@heroicons/react/24/solid';

interface Recommendation {
  song_name: string;
  artist_name: string;
  album_name: string;
  genre: string;
}

export default function FeatureRecommendationPage() {
  const [features, setFeatures] = useState({
    danceability: 0.5,
    energy: 0.5,
    speechiness: 0.1,
    acousticness: 0.3,
    instrumentalness: 0.0,
    liveness: 0.2,
    valence: 0.5,
    tempo: 120.0,
    loudness: -8.0,
  });

  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);

  const handleChange = (key: string, value: number) => {
    setFeatures((prev) => ({ ...prev, [key]: value }));
  };

  const fetchRecommendations = async () => {
    setLoading(true);
    try {
      const res = await fetchFromAPI("/recommend/features?model_name=graphsage", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features, num_recommendations: 10 }),
      });
      const data = await res.json();
      setRecommendations(data.recommendations || []);
    } catch (err) {
      console.error("Error fetching recommendations", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-300 via-gray-700 to-black text-white flex flex-col items-center p-8 sm:p-20">
      <header className="flex flex-col items-center gap-4 animate-fadeIn">
        <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-gray-100 to-gray-400 animate-slideUp">
          Feature-Based Recommendations
        </h1>
        <p className="text-lg text-gray-300 animate-fadeInDelay">
          Select audio feature values to get personalized song recommendations.
        </p>
      </header>

      <main className="flex flex-col items-center gap-8 mt-10 w-full max-w-3xl">
        <div className="w-full grid grid-cols-1 md:grid-cols-2 gap-6">
          {Object.entries(features).map(([key, value]) => (
            <div key={key} className="bg-gray-800 p-4 rounded-lg">
              <label htmlFor={key} className="block font-medium capitalize mb-1 text-white">
                {key.replace(/_/g, " ")}
              </label>
              <input
                type="range"
                id={key}
                min={key === "tempo" ? 60 : key === "loudness" ? -20 : 0}
                max={key === "tempo" ? 200 : key === "loudness" ? 0 : 1}
                step={key === "tempo" ? 1 : 0.01}
                value={value}
                onChange={(e) => handleChange(key, parseFloat(e.target.value))}
                className="w-full mt-2"
              />
              <span className="text-sm text-gray-500">{value.toFixed(2)}</span>
            </div>
          ))}
        </div>

        <button
          onClick={fetchRecommendations}
          className={`bg-gradient-to-r from-red-500 to-red-700 text-white font-medium py-4 px-8 rounded-lg transition transform hover:scale-105 ${
            loading ? 'opacity-50 cursor-not-allowed' : ''
          }`}
          disabled={loading}
        >
          {loading ? "Loading..." : "Get Recommendations"}
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