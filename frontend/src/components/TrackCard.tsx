import { Heart, SkipForward, Play, Music } from 'lucide-react';
import { Link } from 'react-router-dom';

interface TrackCardProps {
  track: {
    track_id: string;
    score: number;
  };
  rank: number;
  onLike: () => void;
  onSkip: () => void;
  isSaved: boolean;
}

export function TrackCard({ track, rank, onLike, onSkip, isSaved }: TrackCardProps) {
  const artistId = track.track_id.split('_')[0];
  const trackTitle = `Track ${track.track_id}`;
  const artistName = `Artist ${artistId}`;

  return (
    <div className="track-card">
      <div className="track-card-header">
        <div className="track-rank">#{rank}</div>
        <div className="track-score">
          {Math.round(track.score * 100)}%
        </div>
      </div>

      <div className="track-card-content">
        <div className="track-artwork">
          <Music size={32} />
        </div>

        <div className="track-info">
          <h3 className="track-title">
            <Link to={`/track/${track.track_id}`}>
              {trackTitle}
            </Link>
          </h3>
          <p className="track-artist">{artistName}</p>
          <p className="track-id">ID: {track.track_id}</p>
        </div>
      </div>

      <div className="track-card-actions">
        <button
          className={`action-button like-button ${isSaved ? 'saved' : ''}`}
          onClick={onLike}
          title={isSaved ? 'Already saved' : 'Add to playlist'}
        >
          <Heart size={16} fill={isSaved ? 'currentColor' : 'none'} />
          {isSaved ? 'Saved' : 'Like'}
        </button>

        <button
          className="action-button skip-button"
          onClick={onSkip}
          title="Skip this track"
        >
          <SkipForward size={16} />
          Skip
        </button>

        <Link
          to={`/track/${track.track_id}`}
          className="action-button play-button"
          title="View track details"
        >
          <Play size={16} />
          Details
        </Link>
      </div>
    </div>
  );
}

