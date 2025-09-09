import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface SavedTrack {
  track_id: string;
  title: string;
  artist: string;
  score: number;
  timestamp: number;
}

interface PlaylistState {
  savedTracks: SavedTrack[];
  addTrack: (track: Omit<SavedTrack, 'timestamp'>) => void;
  removeTrack: (trackId: string) => void;
  clearPlaylist: () => void;
  isTrackSaved: (trackId: string) => boolean;
}

export const usePlaylistStore = create<PlaylistState>()(
  persist(
    (set, get) => ({
      savedTracks: [],
      
      addTrack: (track) => {
        const { savedTracks } = get();
        const newTrack: SavedTrack = {
          ...track,
          timestamp: Date.now(),
        };
        
        // Check if track is already saved
        if (!savedTracks.some(t => t.track_id === track.track_id)) {
          set({ savedTracks: [...savedTracks, newTrack] });
        }
      },
      
      removeTrack: (trackId) => {
        const { savedTracks } = get();
        set({ 
          savedTracks: savedTracks.filter(track => track.track_id !== trackId) 
        });
      },
      
      clearPlaylist: () => {
        set({ savedTracks: [] });
      },
      
      isTrackSaved: (trackId) => {
        const { savedTracks } = get();
        return savedTracks.some(track => track.track_id === trackId);
      },
    }),
    {
      name: 'tunedin-playlist',
      version: 1,
    }
  )
);

