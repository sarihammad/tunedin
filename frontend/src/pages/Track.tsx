import { useParams } from 'react-router-dom';
import { useState, useEffect } from 'react';
import { ArrowLeft, Heart, SkipForward, Play } from 'lucide-react';
import { Link } from 'react-router-dom';
import { feedbackApi } from '../api/client';

export function Track() {
  const { trackId } = useParams<{ trackId: string }>();
  const [loading, setLoading] = useState(false);

  const handleFeedback = async (event: 'play' | 'like' | 'skip') => {
    if (!trackId) return;
    
    setLoading(true);
    try {
      await feedbackApi.submitFeedback({
        user_id: '1', // Default user for demo
        track_id: trackId,
        event,
        ts: Math.floor(Date.now() / 1000),
      });
      
      // Show feedback confirmation
      console.log(`Feedback submitted: ${event} for track ${trackId}`);
    } catch (err) {
      console.error('Error submitting feedback:', err);
    } finally {
      setLoading(false);
    }
  };

  if (!trackId) {
    return (
      <div className="track-page">
        <div className="error-state">
          <h2>Track not found</h2>
          <Link to="/" className="back-link">
            <ArrowLeft size={16} />
            Back to Discover
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="track-page">
      <div className="track-header">
        <Link to="/" className="back-link">
          <ArrowLeft size={16} />
          Back to Discover
        </Link>
      </div>

      <div className="track-content">
        <div className="track-artwork">
          <div className="artwork-placeholder">
            🎵
          </div>
        </div>

        <div className="track-info">
          <h1 className="track-title">Track {trackId}</h1>
          <p className="track-artist">Artist {trackId.split('_')[0]}</p>
          <p className="track-genre">Genre: Electronic</p>
          <p className="track-duration">Duration: 3:45</p>
          
          <div className="track-description">
            <p>
              This is a sample track description. In a real application, 
              this would contain detailed information about the track, 
              including lyrics, album information, and more.
            </p>
          </div>

          <div className="track-actions">
            <button
              className="action-button play-button"
              onClick={() => handleFeedback('play')}
              disabled={loading}
            >
              <Play size={20} />
              Play
            </button>
            
            <button
              className="action-button like-button"
              onClick={() => handleFeedback('like')}
              disabled={loading}
            >
              <Heart size={20} />
              Like
            </button>
            
            <button
              className="action-button skip-button"
              onClick={() => handleFeedback('skip')}
              disabled={loading}
            >
              <SkipForward size={20} />
              Skip
            </button>
          </div>
        </div>
      </div>

      <div className="track-details">
        <h3>Track Details</h3>
        <div className="details-grid">
          <div className="detail-item">
            <span className="detail-label">Track ID:</span>
            <span className="detail-value">{trackId}</span>
          </div>
          <div className="detail-item">
            <span className="detail-label">Artist:</span>
            <span className="detail-value">Artist {trackId.split('_')[0]}</span>
          </div>
          <div className="detail-item">
            <span className="detail-label">Genre:</span>
            <span className="detail-value">Electronic</span>
          </div>
          <div className="detail-item">
            <span className="detail-label">Year:</span>
            <span className="detail-value">2023</span>
          </div>
        </div>
      </div>
    </div>
  );
}

