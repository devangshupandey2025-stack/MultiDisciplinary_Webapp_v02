import { useState, useEffect, useRef } from 'react';
import { Clock, Leaf, AlertCircle, ChevronDown, ChevronUp, Download, Volume2, Square, Play, Pause, Bot, ThumbsUp, ThumbsDown, Pill, Lightbulb, Loader } from 'lucide-react';
import { getHistory, generateTTS } from '../services/api';
import { getUser } from '../services/supabase';
import { useLanguage } from '../i18n/LanguageContext';
import { toast } from 'sonner';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

export default function HistoryPage() {
  const { t, lang } = useLanguage();
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [user, setUser] = useState(null);
  const [error, setError] = useState(null);
  const [expandedId, setExpandedId] = useState(null);

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
          {history.map((item, i) => {
            const validation = parseValidation(item.gemini_validation);
            const isExpanded = expandedId === (item.id || i);

            return (
              <div key={item.id || i} className="glass-card overflow-hidden">
                {/* Compact row */}
                <div
                  className="p-5 flex items-center gap-4 cursor-pointer"
                  onClick={() => setExpandedId(isExpanded ? null : (item.id || i))}
                >
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
                  <div className="flex items-center gap-2">
                    <span className="text-xs whitespace-nowrap" style={{ color: 'var(--color-sage)' }}>
                      {item.created_at ? new Date(item.created_at).toLocaleDateString() : ''}
                    </span>
                    {isExpanded
                      ? <ChevronUp className="w-4 h-4" style={{ color: 'var(--color-text-muted)' }} />
                      : <ChevronDown className="w-4 h-4" style={{ color: 'var(--color-text-muted)' }} />
                    }
                  </div>
                </div>

                {/* Expanded details */}
                {isExpanded && (
                  <HistoryItemDetails
                    item={item}
                    validation={validation}
                    t={t}
                    lang={lang}
                  />
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function formatClassName(cls) {
  if (!cls) return '';
  return cls.replace(/___/g, ' — ').replace(/_/g, ' ');
}

function parseValidation(raw) {
  if (!raw) return null;
  if (typeof raw === 'object') return raw;
  try { return JSON.parse(raw); } catch { return null; }
}

function HistoryItemDetails({ item, validation, t, lang }) {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [speechRate, setSpeechRate] = useState(1);
  const [isDownloading, setIsDownloading] = useState(false);
  const [isLoadingTTS, setIsLoadingTTS] = useState(false);
  const detailsRef = useRef(null);
  const audioRef = useRef(null);
  const audioUrlRef = useRef(null);

  useEffect(() => {
    return () => {
      if (audioRef.current) { audioRef.current.pause(); audioRef.current = null; }
      if (audioUrlRef.current) { URL.revokeObjectURL(audioUrlRef.current); }
    };
  }, []);

  function getReportText() {
    const parts = [
      `${t('result_disease')}: ${formatClassName(item.prediction_class)}`,
      `${t('history_confidence')}: ${(item.probability * 100).toFixed(1)}%`,
    ];
    if (validation) {
      parts.push(
        t('result_validation_title'),
        validation.summary,
        `${t('result_confidence_assess')}: ${validation.confidence_assessment}`,
        `${t('result_reasoning')}: ${validation.reasoning}`,
      );
      if (validation.alternative_suggestions?.length > 0) {
        parts.push(`${t('result_alternatives')}: ${validation.alternative_suggestions.map(formatClassName).join(', ')}`);
      }
      if (validation.treatment_advice) {
        parts.push(`${t('result_treatment')}: ${validation.treatment_advice}`);
      }
    }
    return parts.join('. ');
  }

  async function handleDownload() {
    const el = detailsRef.current;
    if (!el || isDownloading) return;
    setIsDownloading(true);
    try {
      const canvas = await html2canvas(el, { scale: 2, useCORS: true, backgroundColor: '#FAFAF5', logging: false });
      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pw = pdf.internal.pageSize.getWidth();
      const ph = pdf.internal.pageSize.getHeight();
      const margin = 8;

      pdf.setFontSize(16);
      pdf.setTextColor(44, 62, 45);
      pdf.text('PlantGuard AI', pw / 2, 14, { align: 'center' });
      pdf.setFontSize(9);
      pdf.setTextColor(107, 143, 113);
      pdf.text(t('history_title'), pw / 2, 20, { align: 'center' });
      pdf.setDrawColor(168, 197, 170);
      pdf.line(margin, 23, pw - margin, 23);

      const imgW = pw - 2 * margin;
      const imgH = (canvas.height * imgW) / canvas.width;
      const maxH = ph - 42;
      if (imgH > maxH) {
        const s = maxH / imgH;
        pdf.addImage(imgData, 'PNG', (pw - imgW * s) / 2, 27, imgW * s, imgH * s);
      } else {
        pdf.addImage(imgData, 'PNG', margin, 27, imgW, imgH);
      }

      pdf.setFontSize(7);
      pdf.setTextColor(160, 160, 160);
      pdf.text(`Generated on ${new Date().toLocaleDateString()}`, pw / 2, ph - 6, { align: 'center' });

      const disease = formatClassName(item.prediction_class);
      pdf.save(`PlantGuard_History_${disease.replace(/[^a-zA-Z0-9]/g, '_')}.pdf`);
      toast.success(t('download_success'));
    } catch {
      toast.error(t('download_error'));
      window.print();
    } finally {
      setIsDownloading(false);
    }
  }

  async function handleSpeak() {
    if (isSpeaking) {
      if (audioRef.current) { audioRef.current.pause(); audioRef.current.currentTime = 0; }
      setIsSpeaking(false);
      setIsPaused(false);
      return;
    }
    const text = getReportText();
    if (!text) return;
    setIsLoadingTTS(true);
    try {
      const blob = await generateTTS(text, lang, speechRate);
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
      const url = URL.createObjectURL(blob);
      audioUrlRef.current = url;
      const audio = new Audio(url);
      audio.playbackRate = speechRate;
      audio.onended = () => { setIsSpeaking(false); setIsPaused(false); };
      audio.onerror = () => { setIsSpeaking(false); setIsPaused(false); };
      audioRef.current = audio;
      await audio.play();
      setIsSpeaking(true);
      setIsPaused(false);
    } catch (err) {
      console.error('TTS error:', err);
    } finally {
      setIsLoadingTTS(false);
    }
  }

  function handlePauseResume() {
    if (!audioRef.current) return;
    if (isPaused) { audioRef.current.play(); setIsPaused(false); }
    else { audioRef.current.pause(); setIsPaused(true); }
  }

  function handleSpeedChange(rate) {
    setSpeechRate(rate);
    if (audioRef.current && isSpeaking) {
      audioRef.current.playbackRate = rate;
    }
  }

  return (
    <div ref={detailsRef} className="px-5 pb-5 space-y-4 border-t" style={{ borderColor: 'rgba(168,197,170,0.15)' }}>
      {/* Action buttons */}
      <div className="flex items-center gap-2 flex-wrap pt-4">
        <button
          onClick={handleDownload}
          disabled={isDownloading}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-colors hover:opacity-80 disabled:opacity-50"
          style={{ background: 'var(--color-warm)', color: 'var(--color-forest)' }}
        >
          <Download className={`w-3.5 h-3.5 ${isDownloading ? 'animate-pulse' : ''}`} />
          {isDownloading ? '...' : t('result_download')}
        </button>

        {!isSpeaking ? (
          <button
            onClick={handleSpeak}
            disabled={isLoadingTTS}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-colors hover:opacity-80 disabled:opacity-50"
            style={{ background: 'var(--color-warm)', color: 'var(--color-forest)' }}
          >
            {isLoadingTTS ? <Loader className="w-3.5 h-3.5 animate-spin" /> : <Volume2 className="w-3.5 h-3.5" />}
            {isLoadingTTS ? '...' : t('result_listen')}
          </button>
        ) : (
          <div className="flex items-center gap-1.5 rounded-full px-2 py-1" style={{ background: 'rgba(107,143,113,0.08)', border: '1px solid rgba(168,197,170,0.25)' }}>
            <button
              onClick={handlePauseResume}
              className="inline-flex items-center justify-center w-7 h-7 rounded-full transition-colors hover:opacity-80"
              style={{ background: 'var(--color-warm)', color: 'var(--color-forest)' }}
            >
              {isPaused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
            </button>
            <button
              onClick={handleSpeak}
              className="inline-flex items-center justify-center w-7 h-7 rounded-full badge-danger transition-colors hover:opacity-80"
            >
              <Square className="w-3 h-3" />
            </button>
            <div className="flex items-center gap-0.5 ml-1">
              {[0.5, 1, 1.5, 2].map((rate) => (
                <button
                  key={rate}
                  onClick={() => handleSpeedChange(rate)}
                  className="px-1.5 py-0.5 rounded text-[10px] font-semibold transition-colors"
                  style={{
                    background: speechRate === rate ? 'var(--color-forest)' : 'transparent',
                    color: speechRate === rate ? '#fff' : 'var(--color-text-muted)',
                  }}
                >
                  {rate}x
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Validation details */}
      {validation ? (
        <div className="space-y-4">
          <div className="flex items-center gap-2 flex-wrap">
            <Bot className="w-4 h-4" style={{ color: 'var(--color-sage)' }} />
            <span className="text-sm font-semibold" style={{ color: 'var(--color-forest)' }}>
              {t('result_validation_title')}
            </span>
            {validation.agrees ? (
              <span className="badge-success inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold ml-auto">
                <ThumbsUp className="w-3 h-3" /> {t('result_agrees')}
              </span>
            ) : (
              <span className="badge-danger inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold ml-auto">
                <ThumbsDown className="w-3 h-3" /> {t('result_disagrees')}
              </span>
            )}
          </div>

          <p className="text-xs font-medium rounded-lg p-3" style={{ background: 'var(--color-warm)', color: 'var(--color-forest)' }}>
            {validation.summary}
          </p>

          <div>
            <span className="text-[10px] font-semibold uppercase tracking-[0.12em]" style={{ color: 'var(--color-text-muted)' }}>
              {t('result_confidence_assess')}
            </span>
            <p className="text-xs mt-0.5" style={{ color: 'var(--color-forest)' }}>{validation.confidence_assessment}</p>
          </div>

          <div>
            <span className="text-[10px] font-semibold uppercase tracking-[0.12em]" style={{ color: 'var(--color-text-muted)' }}>
              {t('result_reasoning')}
            </span>
            <p className="text-xs mt-0.5" style={{ color: 'var(--color-forest)' }}>{validation.reasoning}</p>
          </div>

          {validation.alternative_suggestions?.length > 0 && (
            <div>
              <div className="flex items-center gap-1 mb-1">
                <Lightbulb className="w-3 h-3" style={{ color: 'var(--color-gold)' }} />
                <span className="text-[10px] font-semibold uppercase tracking-[0.12em]" style={{ color: 'var(--color-text-muted)' }}>
                  {t('result_alternatives')}
                </span>
              </div>
              <div className="flex flex-wrap gap-1">
                {validation.alternative_suggestions.map((alt, j) => (
                  <span key={j} className="badge-warning px-2 py-0.5 rounded-full text-[10px] font-medium">
                    {formatClassName(alt)}
                  </span>
                ))}
              </div>
            </div>
          )}

          {validation.treatment_advice && (
            <div className="rounded-lg p-3 border" style={{ background: 'rgba(107,143,113,0.04)', borderColor: 'rgba(168,197,170,0.2)' }}>
              <div className="flex items-center gap-1 mb-1">
                <Pill className="w-3 h-3" style={{ color: 'var(--color-sage)' }} />
                <span className="text-[10px] font-semibold uppercase tracking-[0.12em]" style={{ color: 'var(--color-sage)' }}>
                  {t('result_treatment')}
                </span>
              </div>
              <p className="text-xs leading-relaxed" style={{ color: 'var(--color-forest)' }}>{validation.treatment_advice}</p>
            </div>
          )}
        </div>
      ) : (
        <p className="text-xs italic pt-1" style={{ color: 'var(--color-text-muted)' }}>
          {t('history_no_validation')}
        </p>
      )}
    </div>
  );
}
