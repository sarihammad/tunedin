import { useState } from 'react';
import { usePlaylistStore, SavedTrack } from '../stores/playlistStore';
import { X, Play, Trash2, ChevronUp, ChevronDown } from 'lucide-react';

export function PlaylistBar() {
  const { savedTracks, removeTrack, clearPlaylist } = usePlaylistStore();
  const [isExpanded, setIsExpanded] = useState(false);

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp).toLocaleDateString();
  };

  const handlePlayAll = () => {
    // In a real app, this would start playing the playlist
    console.log('Playing all tracks:', savedTracks);
  };

  return (
    <div className={`playlist-bar ${isExpanded ? 'expanded' : ''}`}>
      <div className="playlist-header" onClick={() => setIsExpanded(!isExpanded)}>
        <div className="playlist-info">
          <span className="playlist-title">My Playlist</span>
          <span className="playlist-count">{savedTracks.length} tracks</span>
        </div>
        <div className="playlist-controls">
          <button
            className="playlist-button"
            onClick={(e) => {
              e.stopPropagation();
              handlePlayAll();
            }}
            title="Play all"
          >
            <Play size={16} />
          </button>
          <button
            className="playlist-button"
            onClick={(e) => {
              e.stopPropagation();
              clearPlaylist();
            }}
            title="Clear playlist"
          >
            <Trash2 size={16} />
          </button>
          {isExpanded ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
        </div>
      </div>

      {isExpanded && (
        <div className="playlist-content">
          <div className="playlist-tracks">
            {savedTracks.map((track, index) => (
              <div key={`${track.track_id}-${track.timestamp}`} className="playlist-track">
                <div className="track-number">{index + 1}</div>
                <div className="track-details">
                  <div className="track-title">{track.title}</div>
                  <div className="track-artist">{track.artist}</div>
                </div>
                <div className="track-meta">
                  <span className="track-score">{Math.round(track.score * 100)}%</span>
                  <span className="track-date">{formatTimestamp(track.timestamp)}</span>
                </div>
                <button
                  className="remove-button"
                  onClick={() => removeTrack(track.track_id)}
                  title="Remove from playlist"
                >
                  <X size={14} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

