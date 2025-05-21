import React from 'react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-gray-900 text-gray-400 py-8">
      <div className="container mx-auto flex flex-col md:flex-row justify-between items-center gap-4">
        <div className="flex flex-col items-center md:items-start">
          <h3 className="text-lg font-bold text-gray-300 mb-2">TunedIn</h3>
          <p className="text-sm">Discover your sound. Connect with the music you love.</p>
          <p className="text-xs text-gray-500 mt-2">© 2025 TunedIn. All rights reserved.</p>
        </div>
        
        <div className="flex gap-6">
          <a
            href="https://twitter.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-400 hover:text-red-500 transition hover:underline"
          >
            Twitter
          </a>
          <a
            href="https://facebook.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-400 hover:text-red-500 transition hover:underline"
          >
            Facebook
          </a>
          <a
            href="https://instagram.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-400 hover:text-red-500 transition hover:underline"
          >
            Instagram
          </a>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
