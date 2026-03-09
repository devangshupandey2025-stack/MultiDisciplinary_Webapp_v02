import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Leaf, Menu, X, LogIn, LogOut, User, Globe, Download } from 'lucide-react';
import AuthModal from './AuthModal';
import { getUser, signOut, onAuthStateChange } from '../services/supabase';
import { useLanguage } from '../i18n/LanguageContext';

export default function Layout({ children }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [user, setUser] = useState(null);
  const [showAuth, setShowAuth] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [installPrompt, setInstallPrompt] = useState(null);
  const location = useLocation();
  const { t, lang, changeLang, LANGUAGES } = useLanguage();

  useEffect(() => {
    getUser().then(setUser);
    const { data } = onAuthStateChange((_event, session) => {
      setUser(session?.user || null);
    });
    return () => data.subscription.unsubscribe();
  }, []);

  useEffect(() => {
    setMenuOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  useEffect(() => {
    const handler = (e) => {
      e.preventDefault();
      setInstallPrompt(e);
    };
    window.addEventListener('beforeinstallprompt', handler);
    window.addEventListener('appinstalled', () => setInstallPrompt(null));
    return () => {
      window.removeEventListener('beforeinstallprompt', handler);
    };
  }, []);

  const handleInstall = async () => {
    if (!installPrompt) return;
    installPrompt.prompt();
    const { outcome } = await installPrompt.userChoice;
    if (outcome === 'accepted') setInstallPrompt(null);
  };

  const handleSignOut = async () => {
    await signOut();
    setUser(null);
  };

  const navItems = [
    { path: '/', label: t('nav_detect') },
    { path: '/history', label: t('nav_history') },
    { path: '/about', label: t('nav_about') },
  ];

  const isHome = location.pathname === '/';

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header
        className={`sticky top-0 z-40 transition-all duration-300 ${
          scrolled || !isHome
            ? 'nav-header shadow-sm'
            : 'bg-transparent'
        }`}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <Link to="/" className="flex items-center gap-2.5">
              <Leaf
                className="w-6 h-6"
                style={{ color: scrolled || !isHome ? 'var(--color-sage)' : '#A8C5AA' }}
              />
              <span
                className="font-bold text-lg tracking-tight"
                style={{
                  fontFamily: 'var(--font-serif)',
                  color: scrolled || !isHome ? 'var(--color-forest)' : 'white',
                }}
              >
                {t('nav_brand')}
              </span>
            </Link>

            {/* Desktop nav */}
            <nav className="hidden md:flex items-center gap-1">
              {navItems.map(({ path, label }) => (
                <Link
                  key={path}
                  to={path}
                  className="px-4 py-2 rounded-full transition-all duration-200 text-sm font-medium"
                  style={{
                    color: scrolled || !isHome
                      ? location.pathname === path ? 'var(--color-forest)' : 'var(--color-text-muted)'
                      : location.pathname === path ? 'white' : 'rgba(255,255,255,0.7)',
                    background: location.pathname === path
                      ? scrolled || !isHome ? 'rgba(107,143,113,0.08)' : 'rgba(255,255,255,0.12)'
                      : 'transparent',
                  }}
                >
                  {label}
                </Link>
              ))}

              {/* Language Selector */}
              <div className="flex items-center gap-1 ml-3">
                <Globe
                  className="w-3.5 h-3.5"
                  style={{ color: scrolled || !isHome ? 'var(--color-text-muted)' : 'rgba(255,255,255,0.6)' }}
                />
                <select
                  value={lang}
                  onChange={(e) => changeLang(e.target.value)}
                  className={scrolled || !isHome ? 'lang-dropdown' : 'lang-dropdown-hero'}
                  aria-label={t('lang_label')}
                >
                  {LANGUAGES.map((l) => (
                    <option key={l.code} value={l.code}>{l.nativeName}</option>
                  ))}
                </select>
              </div>

              {installPrompt && (
                <button
                  onClick={handleInstall}
                  className="flex items-center gap-1.5 text-sm px-3 py-1.5 rounded-full transition-all hover:scale-105 ml-2"
                  style={{
                    color: scrolled || !isHome ? '#fff' : '#2C3E2D',
                    background: scrolled || !isHome ? 'var(--color-forest)' : 'rgba(255,255,255,0.9)',
                    fontWeight: 600,
                  }}
                >
                  <Download className="w-3.5 h-3.5" />
                  {t('install_app')}
                </button>
              )}

              {user ? (
                <div className="flex items-center gap-3 ml-4">
                  <span
                    className="text-sm opacity-70"
                    style={{ color: scrolled || !isHome ? 'var(--color-text-muted)' : 'rgba(255,255,255,0.7)' }}
                  >
                    <User className="w-3.5 h-3.5 inline mr-1" />
                    {user.email?.split('@')[0]}
                  </span>
                  <button
                    onClick={handleSignOut}
                    className="text-sm px-3 py-1.5 rounded-full transition-colors"
                    style={{
                      color: scrolled || !isHome ? 'var(--color-text-muted)' : 'rgba(255,255,255,0.7)',
                      background: scrolled || !isHome ? 'rgba(107,143,113,0.06)' : 'rgba(255,255,255,0.1)',
                    }}
                  >
                    <LogOut className="w-3.5 h-3.5 inline mr-1" />
                    {t('nav_signout')}
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => setShowAuth(true)}
                  className="btn-primary ml-4 !py-2 !px-5 text-sm"
                >
                  {t('nav_signin')}
                </button>
              )}
            </nav>

            {/* Mobile controls */}
            <div className="md:hidden flex items-center gap-2">
              <select
                value={lang}
                onChange={(e) => changeLang(e.target.value)}
                className={`text-xs py-1 px-2 ${scrolled || !isHome ? 'lang-dropdown' : 'lang-dropdown-hero'}`}
                aria-label={t('lang_label')}
              >
                {LANGUAGES.map((l) => (
                  <option key={l.code} value={l.code}>{l.nativeName}</option>
                ))}
              </select>
              <button
                onClick={() => setMenuOpen(!menuOpen)}
                className="p-1.5 rounded-full"
                style={{ color: scrolled || !isHome ? 'var(--color-forest)' : 'white' }}
              >
                {menuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </button>
            </div>
          </div>

          {/* Mobile nav */}
          <div
            className={`md:hidden overflow-hidden transition-all duration-300 ${
              menuOpen ? 'max-h-64 pb-4' : 'max-h-0'
            }`}
          >
            <div className="space-y-1 pt-2">
              {navItems.map(({ path, label }) => (
                <Link
                  key={path}
                  to={path}
                  className="block px-4 py-2.5 rounded-xl transition-colors text-sm"
                  style={{
                    color: scrolled || !isHome ? 'var(--color-forest)' : 'white',
                    background: location.pathname === path
                      ? scrolled || !isHome ? 'rgba(107,143,113,0.08)' : 'rgba(255,255,255,0.1)'
                      : 'transparent',
                  }}
                >
                  {label}
                </Link>
              ))}
              {user ? (
                <button
                  onClick={handleSignOut}
                  className="flex items-center gap-2 w-full px-4 py-2.5 rounded-xl text-sm text-left"
                  style={{ color: scrolled || !isHome ? 'var(--color-text-muted)' : 'rgba(255,255,255,0.7)' }}
                >
                  <LogOut className="w-4 h-4" /> {t('nav_signout')}
                </button>
              ) : (
                <button
                  onClick={() => { setShowAuth(true); setMenuOpen(false); }}
                  className="flex items-center gap-2 w-full px-4 py-2.5 rounded-xl text-sm text-left"
                  style={{ color: scrolled || !isHome ? 'var(--color-forest)' : 'white' }}
                >
                  <LogIn className="w-4 h-4" /> {t('nav_signin')}
                </button>
              )}
              {installPrompt && (
                <button
                  onClick={() => { handleInstall(); setMenuOpen(false); }}
                  className="flex items-center gap-2 w-full px-4 py-2.5 rounded-xl text-sm text-left font-semibold"
                  style={{ color: scrolled || !isHome ? 'var(--color-forest)' : 'white' }}
                >
                  <Download className="w-4 h-4" /> {t('install_app')}
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className={`flex-1 page-enter ${isHome ? '-mt-16' : ''}`}>{children}</main>

      {/* Footer */}
      <footer className="footer-bg text-white/60 py-12">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <div className="flex items-center justify-center gap-2.5 mb-4">
            <Leaf className="w-5 h-5 text-white/40" />
            <span className="text-white/80 font-medium" style={{ fontFamily: 'var(--font-serif)', fontSize: '1.1rem' }}>
              {t('nav_brand')}
            </span>
          </div>
          <p className="text-sm text-white/50 max-w-md mx-auto leading-relaxed">
            {t('footer_text')}
          </p>
          <div className="section-divider my-6" style={{ opacity: 0.3 }} />
          <p className="text-xs text-white/30">
            {t('footer_disclaimer')}
          </p>
        </div>
      </footer>

      {showAuth && <AuthModal onClose={() => setShowAuth(false)} />}
    </div>
  );
}
