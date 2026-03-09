import { useState, useEffect } from 'react';
import { Brain, RefreshCw, BarChart3, CheckCircle, XCircle } from 'lucide-react';
import { useLanguage } from '../i18n/LanguageContext';
import { getFeedbackStats, triggerRetrain } from '../services/api';
import { toast } from 'sonner';

export default function ModelTraining() {
  const { t } = useLanguage();
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [retraining, setRetraining] = useState(false);

  useEffect(() => {
    loadStats();
  }, []);

  async function loadStats() {
    try {
      const data = await getFeedbackStats();
      setStats(data);
    } catch {
      setStats(null);
    } finally {
      setLoading(false);
    }
  }

  async function handleRetrain() {
    setRetraining(true);
    try {
      const result = await triggerRetrain();
      if (result.status === 'success') {
        toast.success(`${t('retrain_success')} (v${result.version}, ${result.final_accuracy})`);
        loadStats();
      } else {
        toast.error(result.message || t('retrain_error'));
      }
    } catch (e) {
      toast.error(e.response?.data?.detail || t('retrain_error'));
    } finally {
      setRetraining(false);
    }
  }

  if (loading) return null;
  if (!stats) return null;

  return (
    <div className="card rounded-2xl p-6 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Brain className="w-5 h-5" style={{ color: 'var(--color-sage)' }} />
          <h3 className="text-lg font-bold" style={{ color: 'var(--color-forest)' }}>
            {t('training_stats_title')}
          </h3>
        </div>
        <button
          onClick={loadStats}
          className="p-1.5 rounded-full hover:opacity-70 transition-opacity"
          title="Refresh"
        >
          <RefreshCw className="w-4 h-4" style={{ color: 'var(--color-text-muted)' }} />
        </button>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <div className="rounded-xl p-3 text-center" style={{ background: 'var(--color-warm)' }}>
          <div className="text-2xl font-bold" style={{ color: 'var(--color-forest)' }}>
            {stats.total}
          </div>
          <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
            {t('training_samples')}
          </div>
        </div>
        <div className="rounded-xl p-3 text-center" style={{ background: 'rgba(34,197,94,0.08)' }}>
          <div className="flex items-center justify-center gap-1">
            <CheckCircle className="w-4 h-4" style={{ color: '#22c55e' }} />
            <span className="text-2xl font-bold" style={{ color: '#22c55e' }}>{stats.correct}</span>
          </div>
          <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
            {t('training_correct')}
          </div>
        </div>
        <div className="rounded-xl p-3 text-center" style={{ background: 'rgba(239,68,68,0.08)' }}>
          <div className="flex items-center justify-center gap-1">
            <XCircle className="w-4 h-4" style={{ color: '#ef4444' }} />
            <span className="text-2xl font-bold" style={{ color: '#ef4444' }}>{stats.incorrect}</span>
          </div>
          <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
            {t('training_incorrect')}
          </div>
        </div>
      </div>

      {stats.last_training && (
        <div className="text-xs space-y-1" style={{ color: 'var(--color-text-muted)' }}>
          <p>{t('training_version')}: v{stats.last_training.version}</p>
          <p>{t('training_last')}: {stats.last_training.samples_used} samples, {(stats.last_training.final_accuracy * 100).toFixed(1)}% acc</p>
        </div>
      )}

      <button
        onClick={handleRetrain}
        disabled={retraining || !stats.can_retrain}
        className="w-full inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-sm font-semibold transition-all hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed"
        style={{ background: 'var(--color-forest)', color: '#fff' }}
      >
        <RefreshCw className={`w-4 h-4 ${retraining ? 'animate-spin' : ''}`} />
        {retraining ? t('retrain_training') : t('retrain_button')}
      </button>

      {!stats.can_retrain && stats.total < 5 && (
        <p className="text-xs text-center" style={{ color: 'var(--color-text-muted)' }}>
          {5 - stats.total} more feedback samples needed to enable retraining
        </p>
      )}
    </div>
  );
}
