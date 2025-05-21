'use client';

import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import SocialAuth from '@/components/SocialAuth';
import { EyeIcon, EyeSlashIcon } from '@heroicons/react/24/outline';
import { useSession, signIn } from 'next-auth/react';

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const { data: session } = useSession();

  useEffect(() => {
    if (session?.user) {
      router.replace('/recommend/user');
    }
  }, [session]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      const res = await signIn('credentials', {
        redirect: false,
        email,
        password,
        remember: rememberMe,
        callbackUrl: '/recommend/user',
      });

      if (res?.error) {
        throw new Error(res.error);
      }

      router.push('/recommend/user');
    } catch (err: any) {
      setError(err.message || 'Login failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-300 via-gray-700 to-black text-white flex flex-col items-center justify-center p-8 sm:p-20">
      <header className="flex flex-col items-center gap-4 animate-fadeIn">
        <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-gray-100 to-gray-400 animate-slideUp">
          Login to TunedIn
        </h1>
        <p className="text-lg text-gray-300 animate-fadeInDelay">
          Access your personalized music recommendations.
        </p>
      </header>

      <div className="bg-gray-900 rounded-lg shadow-lg p-10 w-full max-w-md mt-10">
        <form
          onSubmit={handleSubmit}
          className="space-y-6"
        >
          {error && (
            <div className="p-3 bg-red-500 text-sm text-white rounded-md">
              {error}
            </div>
          )}
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full p-4 rounded-lg bg-gray-800 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-red-500"
          />
          <div className="relative">
            <input
              type={showPassword ? 'text' : 'password'}
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full p-4 rounded-lg bg-gray-800 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-red-500"
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute inset-y-0 right-3 flex items-center"
            >
              {showPassword ? (
                <EyeSlashIcon className="h-5 w-5 text-gray-500" />
              ) : (
                <EyeIcon className="h-5 w-5 text-gray-500" />
              )}
            </button>
          </div>
          <div className="flex items-center justify-between text-sm text-gray-400">
            <label htmlFor="remember-me" className="flex items-center space-x-2">
              <input
                id="remember-me"
                name="remember-me"
                type="checkbox"
                checked={rememberMe}
                onChange={(e) => setRememberMe(e.target.checked)}
                className="h-4 w-4 text-red-600 focus:ring-red-500 border-gray-300 rounded"
              />
              <span>Remember me</span>
            </label>
            <a href="/forgot-password" className="text-red-500 hover:underline">
              Forgot Password?
            </a>
          </div>
          <button
            type="submit"
            disabled={isLoading}
            className={`w-full bg-red-600 hover:bg-red-700 text-white font-bold py-3 rounded-lg transition ${
              isLoading ? 'opacity-50 cursor-not-allowed' : ''
            }`}
          >
            {isLoading ? 'Logging in...' : 'Login'}
          </button>
        </form>

        <div className="mt-6">
          <SocialAuth />
        </div>

        <p className="text-gray-400 mt-6 text-center">
          Don't have an account?{' '}
          <a href="/signup" className="text-red-500 hover:underline">
            Sign up here
          </a>
        </p>
      </div>
    </div>
  )
}
