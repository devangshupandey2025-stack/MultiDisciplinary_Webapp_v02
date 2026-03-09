import { useState } from 'react';
import { X, Leaf } from 'lucide-react';
import { signIn, signUp } from '../services/supabase';
import { useLanguage } from '../i18n/LanguageContext';

export default function AuthModal({ onClose }) {
  const { t } = useLanguage();
  const [mode, setMode] = useState('signin');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setLoading(true);

    const fn = mode === 'signin' ? signIn : signUp;
    const { data, error: err } = await fn(email, password);

    setLoading(false);
    if (err) {
      setError(err.message);
    } else if (mode === 'signup') {
      setSuccess(t('auth_success'));
    } else {
      onClose();
    }
  };

  return (
    <div className="fixed inset-0 modal-overlay flex items-center justify-center z-50 p-4">
      <div className="rounded-2xl max-w-md w-full p-8 relative"
        style={{
          background: 'var(--color-ivory)',
          border: '1px solid rgba(107,143,113,0.12)',
          boxShadow: '0 24px 80px rgba(44,62,45,0.2)',
        }}>
        <button onClick={onClose} className="absolute top-4 right-4 hover:opacity-70 transition-opacity"
          style={{ color: 'var(--color-text-muted)' }}>
          <X className="w-5 h-5" />
        </button>

        <div className="text-center mb-6">
          <div className="feature-icon mx-auto mb-4" style={{ width: '3.5rem', height: '3.5rem' }}>
            <Leaf className="w-5 h-5" style={{ color: 'var(--color-sage)' }} />
          </div>
          <h2 className="text-2xl" style={{ fontFamily: 'var(--font-serif)', color: 'var(--color-forest)' }}>
            {mode === 'signin' ? t('auth_welcome') : t('auth_create')}
          </h2>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-medium uppercase tracking-wider mb-1.5" style={{ color: 'var(--color-text-muted)' }}>
              {t('auth_email')}
            </label>
            <input
              type="email" value={email} onChange={e => setEmail(e.target.value)}
              className="w-full px-4 py-3 rounded-xl focus:outline-none transition-all text-sm"
              style={{
                background: 'white',
                border: '1.5px solid rgba(107,143,113,0.2)',
                color: 'var(--color-forest)',
              }}
              onFocus={(e) => e.target.style.borderColor = 'var(--color-sage)'}
              onBlur={(e) => e.target.style.borderColor = 'rgba(107,143,113,0.2)'}
              required
            />
          </div>
          <div>
            <label className="block text-xs font-medium uppercase tracking-wider mb-1.5" style={{ color: 'var(--color-text-muted)' }}>
              {t('auth_password')}
            </label>
            <input
              type="password" value={password} onChange={e => setPassword(e.target.value)}
              className="w-full px-4 py-3 rounded-xl focus:outline-none transition-all text-sm"
              style={{
                background: 'white',
                border: '1.5px solid rgba(107,143,113,0.2)',
                color: 'var(--color-forest)',
              }}
              onFocus={(e) => e.target.style.borderColor = 'var(--color-sage)'}
              onBlur={(e) => e.target.style.borderColor = 'rgba(107,143,113,0.2)'}
              required minLength={6}
            />
          </div>

          {error && <p className="text-sm" style={{ color: '#C07050' }}>{error}</p>}
          {success && <p className="text-sm" style={{ color: 'var(--color-sage)' }}>{success}</p>}

          <button
            type="submit" disabled={loading}
            className="btn-primary w-full py-3 text-center"
          >
            {loading ? t('auth_loading') : mode === 'signin' ? t('auth_signin') : t('auth_signup')}
          </button>
        </form>

        <p className="mt-5 text-center text-sm" style={{ color: 'var(--color-text-muted)' }}>
          {mode === 'signin' ? (
            <>{t('auth_no_account')}{' '}
              <button onClick={() => setMode('signup')} className="font-semibold hover:underline" style={{ color: 'var(--color-sage)' }}>
                {t('auth_signup')}
              </button>
            </>
          ) : (
            <>{t('auth_has_account')}{' '}
              <button onClick={() => setMode('signin')} className="font-semibold hover:underline" style={{ color: 'var(--color-sage)' }}>
                {t('auth_signin')}
              </button>
            </>
          )}
        </p>
      </div>
    </div>
  );
}
