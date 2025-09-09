import { useState, useEffect } from 'react';
import { TrackCard } from '../components/TrackCard';
import { recommendationsApi, feedbackApi } from '../api/client';
import { usePlaylistStore } from '../stores/playlistStore';
import { Heart, RefreshCw, AlertCircle } from 'lucide-react';

export function Discover() {
  const [userId, setUserId] = useState('1');
  const [recommendations, setRecommendations] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cacheHit, setCacheHit] = useState(false);
  const { addTrack, isTrackSaved } = usePlaylistStore();

  const loadRecommendations = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await recommendationsApi.getUserRecommendations(userId, 20);
      setRecommendations(response.items);
      setCacheHit(response.cache_hit);
    } catch (err) {
      setError('Failed to load recommendations. Please try again.');
      console.error('Error loading recommendations:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleFeedback = async (trackId: string, event: 'like' | 'skip') => {
    try {
      await feedbackApi.submitFeedback({
        user_id: userId,
        track_id: trackId,
        event,
        ts: Math.floor(Date.now() / 1000),
      });
      
      if (event === 'like') {
        // Add to playlist
        const track = recommendations.find(r => r.track_id === trackId);
        if (track) {
          addTrack({
            track_id: trackId,
            title: `Track ${trackId}`,
            artist: `Artist ${trackId.split('_')[0]}`,
            score: track.score,
          });
        }
      }
    } catch (err) {
      console.error('Error submitting feedback:', err);
    }
  };

  useEffect(() => {
    loadRecommendations();
  }, [userId]);

  return (
    <div className="discover-page">
      <div className="discover-header">
        <h2>Discover Music</h2>
        <div className="user-controls">
          <label htmlFor="userId">User ID:</label>
          <input
            id="userId"
            type="text"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            className="user-input"
          />
          <button
            onClick={loadRecommendations}
            disabled={loading}
            className="refresh-button"
          >
            <RefreshCw className={loading ? 'spinning' : ''} size={16} />
            Refresh
          </button>
        </div>
      </div>

      {cacheHit && (
        <div className="cache-indicator">
          <span className="cache-badge">Cached</span>
        </div>
      )}

      {error && (
        <div className="error-message">
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      {loading && (
        <div className="loading-state">
          <RefreshCw className="spinning" size={24} />
          <p>Loading recommendations...</p>
        </div>
      )}

      {!loading && !error && (
        <div className="recommendations-grid">
          {recommendations.map((item, index) => (
            <TrackCard
              key={item.track_id}
              track={item}
              rank={index + 1}
              onLike={() => handleFeedback(item.track_id, 'like')}
              onSkip={() => handleFeedback(item.track_id, 'skip')}
              isSaved={isTrackSaved(item.track_id)}
            />
          ))}
        </div>
      )}

      {!loading && !error && recommendations.length === 0 && (
        <div className="empty-state">
          <Heart size={48} />
          <h3>No recommendations found</h3>
          <p>Try refreshing or changing the user ID.</p>
        </div>
      )}
    </div>
  );
}

