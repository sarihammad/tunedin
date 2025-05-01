"use client";

import { useState } from "react";
import { fetchFromAPI } from '@/lib/api'

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

  const [recommendations, setRecommendations] = useState<any[]>([]);
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
    <div className="max-w-3xl mx-auto py-10 px-4">
      <h1 className="text-2xl font-bold mb-6">Feature-Based Recommendations</h1>
      <p className="mb-4 text-gray-600">Select audio feature values to get personalized song recommendations.</p>

      {Object.entries(features).map(([key, value]) => (
        <div key={key} className="mb-4">
          <label htmlFor={key} className="block font-medium capitalize mb-1">
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
            className="w-full"
          />
          <span className="text-sm text-gray-500">{value.toFixed(2)}</span>
        </div>
      ))}

      <button
        onClick={fetchRecommendations}
        className="mt-6 px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
        disabled={loading}
      >
        {loading ? "Loading..." : "Get Recommendations"}
      </button>

      {recommendations.length > 0 && (
        <div className="mt-8">
          <h2 className="text-xl font-semibold mb-4">Recommendations</h2>
          <ul className="space-y-4">
            {recommendations.map((rec, i) => (
              <li key={i} className="border rounded p-4 shadow-sm">
                <p className="font-bold">{rec.song_name}</p>
                <p className="text-sm text-gray-600">{rec.artist_name} — {rec.album_name}</p>
                <p className="text-sm text-gray-500">Genre: {rec.genre}</p>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}