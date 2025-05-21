
'use client';

import { EnvelopeIcon, PhoneIcon, ChatBubbleLeftRightIcon } from '@heroicons/react/24/solid';

export default function ContactPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-300 via-gray-700 to-black text-white flex flex-col items-center p-8 sm:p-20">
      <header className="flex flex-col items-center gap-4 animate-fadeIn">
        <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-gray-100 to-gray-400 animate-slideUp">
          Get in Touch
        </h1>
        <p className="text-lg text-gray-300 animate-fadeInDelay">
          We’d love to hear from you! Whether you have a question or need assistance, you can reach us easily.
        </p>
      </header>

      <main className="flex flex-col items-center gap-12 mt-10 w-full max-w-4xl">
        <section className="grid grid-cols-1 md:grid-cols-3 gap-8 w-full">
          <div className="bg-gray-800 p-6 rounded-lg hover:shadow-lg hover:scale-105 transition transform hover:-translate-y-1">
            <EnvelopeIcon className="w-10 h-10 text-red-500 mb-4 hover:rotate-12 transition-transform" />
            <h2 className="text-2xl font-semibold text-white mb-2">Email Us</h2>
            <p className="text-gray-400">
              Send us an email at:
              <a href="mailto:support@tunedin.com" className="block text-red-500 hover:underline">
                support@tunedin.com
              </a>
            </p>
          </div>

          <div className="bg-gray-800 p-6 rounded-lg hover:shadow-lg hover:scale-105 transition transform hover:-translate-y-1">
            <PhoneIcon className="w-10 h-10 text-red-500 mb-4 hover:rotate-12 transition-transform" />
            <h2 className="text-2xl font-semibold text-white mb-2">Call Us</h2>
            <p className="text-gray-400">
              Speak with a representative at:
              <span className="block text-red-500 hover:underline">
                +1 (800) 123-4567
              </span>
            </p>
          </div>

          <div className="bg-gray-800 p-6 rounded-lg hover:shadow-lg hover:scale-105 transition transform hover:-translate-y-1">
            <ChatBubbleLeftRightIcon className="w-10 h-10 text-red-500 mb-4 hover:rotate-12 transition-transform" />
            <h2 className="text-2xl font-semibold text-white mb-2">Live Chat</h2>
            <p className="text-gray-400">
              Chat with us in real-time for quick assistance.
              <a href="/livechat" className="block text-red-500 hover:underline">
                Start a Chat
              </a>
            </p>
          </div>
        </section>

        <section className="bg-gray-800 p-8 rounded-lg w-full max-w-2xl hover:shadow-lg hover:scale-105 transition transform hover:-translate-y-1">
          <h2 className="text-2xl font-semibold text-white mb-4">Send Us a Message</h2>
          <form className="flex flex-col gap-6">
            <input
              type="text"
              placeholder="Your Name"
              className="w-full p-4 rounded-lg bg-gray-700 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-red-500"
            />
            <input
              type="email"
              placeholder="Your Email"
              className="w-full p-4 rounded-lg bg-gray-700 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-red-500"
            />
            <textarea
              placeholder="Your Message"
              className="w-full p-4 rounded-lg bg-gray-700 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-red-500 h-32"
            />
            <button
              type="submit"
              className="bg-gradient-to-r from-red-500 to-red-700 text-white font-medium py-4 px-8 rounded-lg transition transform hover:scale-105"
            >
              Send Message
            </button>
          </form>
        </section>
      </main>
    </div>
  );
}