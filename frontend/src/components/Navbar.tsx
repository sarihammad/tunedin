'use client';

import Link from 'next/link';
import Image from 'next/image';
import { useSession, signOut } from 'next-auth/react';
import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function Navbar() {
  const { data: session, status } = useSession();
  const [mounted, setMounted] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [showNavbar, setShowNavbar] = useState(true);
  const [lastScrollY, setLastScrollY] = useState(0);

  useEffect(() => {
    setMounted(true);
    let ticking = false;

    const handleScroll = () => {
      const currentScrollY = window.scrollY;

      if (!ticking) {
        window.requestAnimationFrame(() => {
          if (currentScrollY > lastScrollY && currentScrollY > 64) {
            setShowNavbar(false);
          } else if (currentScrollY < lastScrollY || currentScrollY < 64) {
            setShowNavbar(true);
          }
          setLastScrollY(currentScrollY);
          ticking = false;
        });
        ticking = true;
      }
    };

    handleScroll();
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lastScrollY]);

  if (!mounted || status === 'loading') return null;

  return (
    <>
      <AnimatePresence>
        {showNavbar && (
          <motion.nav
            initial={{ y: -64 }}
            animate={{ y: 0 }}
            exit={{ y: -64 }}
            transition={{ duration: 0.2 }}
            className="fixed top-0 left-0 right-0 z-40 bg-white shadow-md"
          >
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex justify-between h-16 items-center">
                
                <Link href={session ? `/dashboard` : `/`} className="flex items-center">
                  <Image
                    src="/assets/images/logo.png"
                    alt="TunedIn Logo"
                    width={72}
                    height={72}
                    className="mr-2"
                    priority
                  />
                </Link>

                <div className="hidden md:flex items-center space-x-4">
                  <Link
                    href="/explore"
                    className="text-gray-700 px-3 py-2 rounded-md text-sm font-medium hover:text-red-600"
                  >
                    Explore
                  </Link>
                  <Link
                    href="/features"
                    className="text-gray-700 px-3 py-2 rounded-md text-sm font-medium hover:text-red-600"
                  >
                    Features
                  </Link>
                  <Link
                    href="/about"
                    className="text-gray-700 px-3 py-2 rounded-md text-sm font-medium hover:text-red-600"
                  >
                    About
                  </Link>
                  <Link
                    href="/contact"
                    className="text-gray-700 px-3 py-2 rounded-md text-sm font-medium hover:text-red-600"
                  >
                    Contact
                  </Link>
                </div>

                <div className="flex items-center space-x-4">
                  {!session ? (
                    <>
                      <Link
                        href="/login"
                        className="text-gray-700 px-3 py-2 rounded-md text-sm font-medium hover:text-red-600"
                      >
                        Login
                      </Link>
                      <Link
                        href="/signup"
                        className="bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition"
                      >
                        Sign Up
                      </Link>
                    </>
                  ) : (
                    <button
                      onClick={() => signOut({ callbackUrl: '/' })}
                      className="bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition"
                    >
                      Sign Out
                    </button>
                  )}
                </div>

              </div>
            </div>
          </motion.nav>
        )}
      </AnimatePresence>
    </>
  );
}
