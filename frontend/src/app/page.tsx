import Image from "next/image";
import { MusicalNoteIcon, ClockIcon, DevicePhoneMobileIcon } from '@heroicons/react/24/solid';

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-300 via-gray-700 to-black text-white flex flex-col items-center justify-between p-8 sm:p-20">
      <header className="flex flex-col items-center gap-4 animate-fadeIn">
        <Image
          src="/logo.svg"
          alt="App Logo"
          width={120}
          height={120}
          priority
        />
        <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-gray-100 to-gray-400 animate-slideUp">
          TunedIn
        </h1>
        <p className="text-lg text-gray-300 animate-fadeInDelay">
          Discover music, curated just for you.
        </p>
        <p className="text-sm text-gray-500 animate-fadeInDelay2">
          Unleash your personalized music experience with cutting-edge recommendations.
        </p>
      </header>

      <main className="flex flex-col items-center gap-8">
        <h2 className="text-3xl font-semibold text-gray-300">
          Your next favorite song is just a click away.
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 w-full max-w-2xl">
          <a
            href="/explore"
            className="bg-gradient-to-r from-red-500 to-red-700 hover:shadow-xl hover:shadow-red-700/50 text-white font-medium py-6 px-8 rounded-xl transition transform hover:scale-105 flex items-center justify-center"
          >
            Explore Music
          </a>
          <a
            href="/signup"
            className="bg-gradient-to-r from-red-500 to-red-700 hover:shadow-xl hover:shadow-red-700/50 text-white font-medium py-6 px-8 rounded-xl transition transform hover:scale-105 flex items-center justify-center"
          >
            Get Started
          </a>
        </div>
      </main>

      <section className="mt-20 text-center max-w-5xl">
        <h2 className="text-4xl font-bold text-gray-200 mb-4">
          Features You'll Love
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8 mt-8">
          <div className="bg-gray-800 p-6 rounded-lg hover:shadow-lg hover:scale-105 transition transform hover:-translate-y-1">
            <MusicalNoteIcon className="w-10 h-10 text-red-500 mb-4 hover:rotate-12 transition-transform" />
            <h3 className="text-2xl font-semibold text-white mb-2">Smart Playlists</h3>
            <p className="text-gray-400">
              Create playlists based on your unique taste and mood, powered by intelligent algorithms.
            </p>
          </div>
          <div className="bg-gray-800 p-6 rounded-lg hover:shadow-lg hover:scale-105 transition transform hover:-translate-y-1">
            <ClockIcon className="w-10 h-10 text-red-500 mb-4 hover:rotate-12 transition-transform" />
            <h3 className="text-2xl font-semibold text-white mb-2">Real-Time Updates</h3>
            <p className="text-gray-400">
              Stay updated with the latest music releases as soon as they drop.
            </p>
          </div>
          <div className="bg-gray-800 p-6 rounded-lg hover:shadow-lg hover:scale-105 transition transform hover:-translate-y-1">
            <DevicePhoneMobileIcon className="w-10 h-10 text-red-500 mb-4 hover:rotate-12 transition-transform" />
            <h3 className="text-2xl font-semibold text-white mb-2">Cross-Platform Sync</h3>
            <p className="text-gray-400">
              Access your playlists and favorites from any device, anytime.
            </p>
          </div>
        </div>
      </section>

      <section className="mt-20 text-center">
        <h2 className="text-4xl font-bold text-gray-200 mb-4">
          Ready to find your sound?
        </h2>
        <p className="text-lg text-gray-400 mb-8">
          Join thousands of music lovers discovering their next favorite tracks.
        </p>
        <a
          href="/signup"
          className="bg-red-500 hover:bg-red-600 text-white font-bold py-4 px-12 rounded-lg transition transform hover:scale-105 flex items-center justify-center"
        >
          Join Now
        </a>
      </section>
    </div>
  );
}
