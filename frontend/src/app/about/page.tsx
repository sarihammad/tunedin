
// About Page for TunedIn
'use client';

import { LightBulbIcon, MusicalNoteIcon, UserGroupIcon } from '@heroicons/react/24/solid';

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-300 via-gray-700 to-black text-white flex flex-col items-center p-8 sm:p-20">
      <header className="flex flex-col items-center gap-4 animate-fadeIn">
        <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-gray-100 to-gray-400 animate-slideUp">
          About TunedIn
        </h1>
        <p className="text-lg text-gray-300 animate-fadeInDelay">
          Discover the story behind the music, and the technology that brings it to you.
        </p>
      </header>

      <main className="flex flex-col items-center gap-12 mt-10 w-full max-w-4xl">
        <section className="grid grid-cols-1 md:grid-cols-2 gap-8 w-full">
          <div className="bg-gray-800 p-6 rounded-lg hover:shadow-lg hover:scale-105 transition transform hover:-translate-y-1">
            <LightBulbIcon className="w-10 h-10 text-red-500 mb-4 hover:rotate-12 transition-transform" />
            <h2 className="text-2xl font-semibold text-white mb-2">Our Vision</h2>
            <p className="text-gray-400">
              At TunedIn, we believe music discovery should be seamless, personal, and immersive. 
              Our platform is built to understand your unique taste and connect you with the music you love.
            </p>
          </div>

          <div className="bg-gray-800 p-6 rounded-lg hover:shadow-lg hover:scale-105 transition transform hover:-translate-y-1">
            <MusicalNoteIcon className="w-10 h-10 text-red-500 mb-4 hover:rotate-12 transition-transform" />
            <h2 className="text-2xl font-semibold text-white mb-2">The Technology</h2>
            <p className="text-gray-400">
              Powered by state-of-the-art machine learning models, TunedIn learns from your listening habits 
              and curates music recommendations that match your style.
            </p>
          </div>

          <div className="bg-gray-800 p-6 rounded-lg hover:shadow-lg hover:scale-105 transition transform hover:-translate-y-1">
            <UserGroupIcon className="w-10 h-10 text-red-500 mb-4 hover:rotate-12 transition-transform" />
            <h2 className="text-2xl font-semibold text-white mb-2">Community Driven</h2>
            <p className="text-gray-400">
              Join a community of music lovers who share your passion. Explore playlists, discover new artists, 
              and connect with people who appreciate good music.
            </p>
          </div>

          <div className="bg-gray-800 p-6 rounded-lg hover:shadow-lg hover:scale-105 transition transform hover:-translate-y-1">
            <LightBulbIcon className="w-10 h-10 text-red-500 mb-4 hover:rotate-12 transition-transform" />
            <h2 className="text-2xl font-semibold text-white mb-2">Continuous Learning</h2>
            <p className="text-gray-400">
              Our recommendation engine improves with every listen, adapting to your evolving taste and bringing you fresh music daily.
            </p>
          </div>
        </section>

        <section className="text-center mt-16">
          <h2 className="text-4xl font-bold text-gray-200 mb-4">
            Join Us in Discovering the Soundtrack of Your Life
          </h2>
          <p className="text-lg text-gray-400 mb-8">
            Whether you're into classics, trending hits, or undiscovered gems, TunedIn is your gateway to musical exploration.
          </p>
          <a
            href="/signup"
            className="bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white font-bold py-4 px-12 rounded-lg transition transform hover:scale-105"
          >
            Get Started
          </a>
        </section>
      </main>
    </div>
  );
}