'use client';

import { signIn } from 'next-auth/react';

export default function SocialAuth() {
  const handleGoogleSignIn = async () => {
    await signIn('google', {
      callbackUrl: '/dashboard',
    });
  };

  return (
    <div className="space-y-4">
      <button
        onClick={handleGoogleSignIn}
        type="button"
        className="w-full flex items-center justify-center py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
      >
        <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24" fill="currentColor">
          <path d="M21.35 11.1h-9.18v2.93h5.46c-.24 1.43-1.53 4.17-5.46 4.17-3.29 0-5.99-2.7-5.99-6s2.7-6 5.99-6c1.87 0 3.13.8 3.85 1.5l2.65-2.65C17.74 3.61 15.67 2.7 13 2.7 7.87 2.7 3.7 6.87 3.7 12s4.17 9.3 9.3 9.3c5.37 0 8.9-3.78 8.9-9.1 0-.63-.06-1.23-.15-1.8z" />
        </svg>
        Continue with Google
      </button>
    </div>
  );
}