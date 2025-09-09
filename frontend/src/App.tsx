import { Routes, Route } from 'react-router-dom';
import { Discover } from './pages/Discover';
import { Track } from './pages/Track';
import { PlaylistBar } from './components/PlaylistBar';
import { usePlaylistStore } from './stores/playlistStore';

function App() {
  const { savedTracks } = usePlaylistStore();

  return (
    <div className="app">
      <header className="app-header">
        <div className="container">
          <h1 className="app-title">
            <span className="app-logo">🎵</span>
            TunedIn
          </h1>
          <p className="app-subtitle">AI-Powered Music Discovery</p>
        </div>
      </header>

      <main className="app-main">
        <div className="container">
          <Routes>
            <Route path="/" element={<Discover />} />
            <Route path="/track/:trackId" element={<Track />} />
          </Routes>
        </div>
      </main>

      {savedTracks.length > 0 && <PlaylistBar />}
    </div>
  );
}

export default App;

