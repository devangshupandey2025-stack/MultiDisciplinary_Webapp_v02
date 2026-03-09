import { useState, useEffect } from 'react';
import { Clock, Leaf, AlertCircle } from 'lucide-react';
import { getHistory } from '../services/api';
import { getUser } from '../services/supabase';
import { useLanguage } from '../i18n/LanguageContext';

export default function HistoryPage() {
  const { t } = useLanguage();
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [user, setUser] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const load = async () => {
      try {
        const u = await getUser();
        setUser(u);
        if (u) {
          const data = await getHistory();
          setHistory(data.predictions || []);
        }
      } catch (err) {
        console.error('Failed to load history:', err);
        setError(err.message || 'Failed to load history');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  if (loading) {
    return (
      <div className="max-w-3xl mx-auto px-4 py-20 text-center">
        <div className="animate-spin w-8 h-8 border-3 rounded-full mx-auto"
          style={{ borderColor: 'var(--color-mint)', borderTopColor: 'transparent' }} />
        <p className="mt-4 text-sm" style={{ color: 'var(--color-text-muted)' }}>{t('history_loading')}</p>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="max-w-3xl mx-auto px-4 py-20 text-center">
        <div className="feature-icon mx-auto mb-6" style={{ width: '5rem', height: '5rem' }}>
          <AlertCircle className="w-7 h-7" style={{ color: 'var(--color-sage)' }} />
        </div>
        <h2 className="text-2xl mb-3" style={{ fontFamily: 'var(--font-serif)', color: 'var(--color-forest)' }}>
          {t('history_signin')}
        </h2>
        <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>{t('history_signin_desc')}</p>
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto px-4 py-12">
      <div className="mb-10">
        <h1 className="text-3xl md:text-4xl flex items-center gap-3" style={{ fontFamily: 'var(--font-serif)', color: 'var(--color-forest)' }}>
          <Clock className="w-7 h-7" style={{ color: 'var(--color-sage)' }} />
          {t('history_title')}
        </h1>
        <div className="section-divider mt-4" style={{ margin: '1rem 0 0 0' }} />
      </div>

      {error && (
        <div className="badge-danger rounded-xl p-4 mb-6 border" style={{ borderColor: 'rgba(192,112,80,0.15)' }}>
          <p className="text-sm">{error}</p>
        </div>
      )}

      {history.length === 0 ? (
        <div className="text-center py-20">
          <div className="feature-icon mx-auto mb-6" style={{ width: '5rem', height: '5rem' }}>
            <Leaf className="w-7 h-7" style={{ color: 'var(--color-sage)' }} />
          </div>
          <p style={{ color: 'var(--color-text-muted)' }}>{t('history_empty')}</p>
        </div>
      ) : (
        <div className="space-y-4">
          {history.map((item, i) => (
            <div key={item.id || i} className="glass-card p-5 flex items-center gap-4">
              {item.image_url && (
                <img src={item.image_url} alt="" className="w-14 h-14 sm:w-16 sm:h-16 rounded-xl object-cover" />
              )}
              <div className="flex-1 min-w-0">
                <p className="font-medium truncate" style={{ color: 'var(--color-forest)' }}>
                  {formatClassName(item.prediction_class)}
                </p>
                <p className="text-sm mt-0.5" style={{ color: 'var(--color-text-muted)' }}>
                  {t('history_confidence')}: {(item.probability * 100).toFixed(1)}%
                  {' · '}{t('history_uncertainty')}: {(item.uncertainty * 100).toFixed(1)}%
                </p>
              </div>
              <span className="text-xs whitespace-nowrap" style={{ color: 'var(--color-sage)' }}>
                {item.created_at ? new Date(item.created_at).toLocaleDateString() : ''}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function formatClassName(cls) {
  if (!cls) return '';
  return cls.replace(/___/g, ' — ').replace(/_/g, ' ');
}
