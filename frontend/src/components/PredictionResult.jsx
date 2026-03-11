import { useState, useEffect, useRef, useCallback } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { AlertTriangle, CheckCircle, HelpCircle, Bot, ThumbsUp, ThumbsDown, Pill, Lightbulb, Download, Volume2, Square, Play, Pause, Loader, Check, X, ChevronDown } from 'lucide-react';
import { useLanguage } from '../i18n/LanguageContext';
import { generateTTS, submitFeedback, getClasses } from '../services/api';
import { toast } from 'sonner';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

const COLORS = ['#2C3E2D', '#6B8F71', '#A8C5AA', '#C9A96E', '#8B7355'];

function useWindowWidth() {
  const [width, setWidth] = useState(typeof window !== 'undefined' ? window.innerWidth : 1024);
  useEffect(() => {
    const onResize = () => setWidth(window.innerWidth);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);
  return width;
}

export default function PredictionResult({ result, imageUrl }) {
  const { t, lang } = useLanguage();
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [speechRate, setSpeechRate] = useState(1);
  const [isDownloading, setIsDownloading] = useState(false);
  const [isLoadingTTS, setIsLoadingTTS] = useState(false);
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [showClassPicker, setShowClassPicker] = useState(false);
  const [allClasses, setAllClasses] = useState([]);
  const audioRef = useRef(null);
  const audioUrlRef = useRef(null);
  const reportRef = useRef(null);
  const windowWidth = useWindowWidth();
  const isMobile = windowWidth < 640;

  useEffect(() => {
    return () => {
      if (audioRef.current) { audioRef.current.pause(); audioRef.current = null; }
      if (audioUrlRef.current) { URL.revokeObjectURL(audioUrlRef.current); }
    };
  }, []);

  if (!result) return null;

  const confidence = result.probability * 100;
  const uncertainty = result.uncertainty * 100;
  const isHighConfidence = confidence >= 85;
  const isUncertain = uncertainty > 30;
  const validation = result.gemini_validation;

  function getReportText() {
    if (!validation) return '';
    const parts = [
      t('result_validation_title'),
      `${validation.agrees ? t('result_agrees') : t('result_disagrees')}`,
      validation.summary,
      `${t('result_confidence_assess')}: ${validation.confidence_assessment}`,
      `${t('result_reasoning')}: ${validation.reasoning}`,
    ];
    if (validation.alternative_suggestions?.length > 0) {
      parts.push(`${t('result_alternatives')}: ${validation.alternative_suggestions.map(formatClassName).join(', ')}`);
    }
    if (validation.treatment_advice) {
      parts.push(`${t('result_treatment')}: ${validation.treatment_advice}`);
    }
    return parts.join('. ');
  }

  async function handleDownload() {
    const el = reportRef.current;
    if (!el || isDownloading) return;
    setIsDownloading(true);

    try {
      // Pre-convert images to base64 so html2canvas can capture them reliably
      const images = el.querySelectorAll('img');
      const originalSrcs = [];
      await Promise.all(Array.from(images).map(async (img, i) => {
        originalSrcs[i] = img.src;
        if (img.src.startsWith('data:')) return;
        try {
          const res = await fetch(img.src);
          const blob = await res.blob();
          const dataUrl = await new Promise((resolve) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result);
            reader.readAsDataURL(blob);
          });
          img.src = dataUrl;
        } catch { /* keep original src */ }
      }));

      const canvas = await html2canvas(el, {
        scale: 2,
        useCORS: true,
        backgroundColor: '#FAFAF5',
        logging: false,
      });

      // Restore original image sources
      images.forEach((img, i) => { if (originalSrcs[i]) img.src = originalSrcs[i]; });

      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const margin = 8;
      const availableWidth = pageWidth - 2 * margin;

      // Header
      pdf.setFontSize(16);
      pdf.setTextColor(44, 62, 45);
      pdf.text('PlantGuard AI', pageWidth / 2, 14, { align: 'center' });
      pdf.setFontSize(9);
      pdf.setTextColor(107, 143, 113);
      pdf.text('Validation Report', pageWidth / 2, 20, { align: 'center' });
      pdf.setDrawColor(168, 197, 170);
      pdf.line(margin, 23, pageWidth - margin, 23);

      // Report image
      const startY = 27;
      const imgWidth = availableWidth;
      const imgHeight = (canvas.height * imgWidth) / canvas.width;
      const maxHeight = pageHeight - startY - 15;

      if (imgHeight > maxHeight) {
        const scale = maxHeight / imgHeight;
        const scaledW = imgWidth * scale;
        const scaledH = imgHeight * scale;
        pdf.addImage(imgData, 'PNG', (pageWidth - scaledW) / 2, startY, scaledW, scaledH);
      } else {
        pdf.addImage(imgData, 'PNG', margin, startY, imgWidth, imgHeight);
      }

      // Footer
      pdf.setFontSize(7);
      pdf.setTextColor(160, 160, 160);
      pdf.text(`Generated on ${new Date().toLocaleDateString()}`, pageWidth / 2, pageHeight - 6, { align: 'center' });

      const disease = formatClassName(result.class);
      pdf.save(`PlantGuard_Report_${disease.replace(/[^a-zA-Z0-9]/g, '_')}.pdf`);
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
      // Stop
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
    if (isPaused) {
      audioRef.current.play();
      setIsPaused(false);
    } else {
      audioRef.current.pause();
      setIsPaused(true);
    }
  }

  function handleSpeedChange(rate) {
    setSpeechRate(rate);
    if (audioRef.current && isSpeaking) {
      audioRef.current.playbackRate = rate;
    }
  }

  async function handleFeedback(isCorrect, selectedClass = null) {
    const actualClass = isCorrect ? result.class : selectedClass;
    if (!actualClass) return;

    try {
      await submitFeedback({
        predicted_class: result.class,
        actual_class: actualClass,
        is_correct: isCorrect,
        image_url: result.image_url || imageUrl || '',
      });
      setFeedbackSent(true);
      setShowClassPicker(false);
      toast.success(t('feedback_thanks'));
    } catch (e) {
      console.error('Feedback error:', e.response?.data || e.message);
      toast.error(t('feedback_error'));
    }
  }

  async function loadClasses() {
    if (allClasses.length > 0) return;
    try {
      const data = await getClasses();
      setAllClasses(data.classes || []);
    } catch { /* ignore */ }
  }

  const chartData = (result.top_k || []).map((item, i) => ({
    name: formatClassName(item.class),
    probability: (item.probability * 100).toFixed(1),
    fill: COLORS[i % COLORS.length],
  }));

  return (
    <div ref={reportRef} className="space-y-6">
      {/* Main prediction card */}
      <div className="glass-card p-4 sm:p-6 md:p-8 space-y-6 fade-in-up">
        <div className="flex flex-col md:flex-row gap-6">
          {imageUrl && (
            <div className="flex-shrink-0">
              <img src={imageUrl} alt="Uploaded leaf" className="w-40 h-40 sm:w-48 sm:h-48 object-cover rounded-2xl" style={{ boxShadow: '0 4px 20px rgba(44,62,45,0.1)' }} />
            </div>
          )}

          <div className="flex-1 space-y-4">
            <div>
              <span className="text-xs uppercase tracking-[0.15em]" style={{ color: 'var(--color-text-muted)' }}>
                {t('result_disease')}
              </span>
              <h3 className="text-2xl mt-1" style={{ fontFamily: 'var(--font-serif)', color: 'var(--color-forest)' }}>
                {formatClassName(result.class)}
              </h3>
              <p className="text-sm mt-1" style={{ color: 'var(--color-text-muted)' }}>
                {t('result_plant')}: {result.class?.split('___')[0]?.replace(/_/g, ' ')}
              </p>
            </div>

            {/* Confidence bar */}
            <div>
              <div className="flex justify-between text-sm mb-1.5">
                <span className="font-medium" style={{ color: 'var(--color-forest)' }}>{t('result_confidence')}</span>
                <span className="font-semibold" style={{ color: isHighConfidence ? 'var(--color-sage)' : 'var(--color-gold)' }}>
                  {confidence.toFixed(1)}%
                </span>
              </div>
              <div className="w-full rounded-full h-2.5" style={{ background: 'rgba(168,197,170,0.15)' }}>
                <div
                  className={`h-2.5 rounded-full transition-all duration-500 ${isHighConfidence ? 'conf-high' : 'conf-moderate'}`}
                  style={{ width: `${confidence}%` }}
                />
              </div>
            </div>

            {/* Uncertainty */}
            <div>
              <div className="flex justify-between text-sm mb-1.5">
                <span className="font-medium" style={{ color: 'var(--color-forest)' }}>{t('result_uncertainty')}</span>
                <span className="font-semibold" style={{ color: isUncertain ? '#C07050' : 'var(--color-sage)' }}>
                  {uncertainty.toFixed(1)}%
                </span>
              </div>
              <div className="w-full rounded-full h-2.5" style={{ background: 'rgba(168,197,170,0.15)' }}>
                <div
                  className={`h-2.5 rounded-full transition-all duration-500 ${isUncertain ? 'conf-uncertain' : 'conf-high'}`}
                  style={{ width: `${uncertainty}%` }}
                />
              </div>
            </div>

            {/* Status badge */}
            <div className="flex items-center gap-2 flex-wrap">
              {isHighConfidence && !isUncertain ? (
                <span className="badge-success inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium">
                  <CheckCircle className="w-4 h-4" /> {t('result_high_conf')}
                </span>
              ) : isUncertain ? (
                <span className="badge-danger inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium">
                  <AlertTriangle className="w-4 h-4" /> {t('result_uncertain')}
                </span>
              ) : (
                <span className="badge-warning inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium">
                  <HelpCircle className="w-4 h-4" /> {t('result_moderate_conf')}
                </span>
              )}
              <span className="text-xs" style={{ color: 'var(--color-mint)' }}>MobileNetV3</span>
            </div>
          </div>
        </div>

        {/* Top-K chart */}
        {chartData.length > 1 && (
          <div className="fade-in-up" style={{ animationDelay: '0.3s' }}>
            <h4 className="text-sm font-semibold mb-3" style={{ color: 'var(--color-forest)' }}>
              {t('result_top_predictions')}
            </h4>
            {isMobile ? (
              <div className="space-y-3">
                {chartData.map((item, i) => (
                  <div key={i}>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="font-medium truncate mr-2" style={{ color: 'var(--color-forest)', maxWidth: '65%' }}>
                        {item.name}
                      </span>
                      <span className="font-semibold whitespace-nowrap" style={{ color: COLORS[i % COLORS.length] }}>
                        {item.probability}%
                      </span>
                    </div>
                    <div className="w-full rounded-full h-2.5" style={{ background: 'rgba(168,197,170,0.12)' }}>
                      <div
                        className="h-2.5 rounded-full transition-all duration-700"
                        style={{ width: `${item.probability}%`, background: COLORS[i % COLORS.length] }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={chartData} layout="vertical" margin={{ left: 80, right: 10 }}>
                  <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 12, fill: '#6B7B6D' }} />
                  <YAxis dataKey="name" type="category" tick={{ fontSize: 11, fill: '#2C3E2D' }} width={80} />
                  <Tooltip formatter={(v) => `${v}%`} />
                  <Bar dataKey="probability" radius={[0, 6, 6, 0]} animationDuration={800}>
                    {chartData.map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        )}
      </div>

      {/* Validation Report */}
      {validation && (
        <div className="glass-card p-4 sm:p-6 md:p-8 space-y-5 fade-in-up" style={{ animationDelay: '0.2s' }}>
          <div className="flex items-center gap-2 mb-2 flex-wrap">
            <Bot className="w-5 h-5" style={{ color: 'var(--color-sage)' }} />
            <h4 className="text-lg" style={{ fontFamily: 'var(--font-serif)', color: 'var(--color-forest)' }}>
              {t('result_validation_title')}
            </h4>
            {validation.agrees ? (
              <span className="badge-success inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold ml-auto">
                <ThumbsUp className="w-3.5 h-3.5" /> {t('result_agrees')}
              </span>
            ) : (
              <span className="badge-danger inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold ml-auto">
                <ThumbsDown className="w-3.5 h-3.5" /> {t('result_disagrees')}
              </span>
            )}
          </div>

          {/* Download & TTS action buttons */}
          <div className="flex items-center gap-2 flex-wrap sm:flex-nowrap">
            <button
              onClick={handleDownload}
              disabled={isDownloading}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-colors hover:opacity-80 disabled:opacity-50"
              style={{ background: 'var(--color-warm)', color: 'var(--color-forest)' }}
              title={t('result_download')}
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
                title={t('result_listen')}
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
                  title={isPaused ? t('result_resume') : t('result_pause')}
                >
                  {isPaused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
                </button>
                <button
                  onClick={handleSpeak}
                  className="inline-flex items-center justify-center w-7 h-7 rounded-full badge-danger transition-colors hover:opacity-80"
                  title={t('result_stop_listen')}
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

          {/* Summary */}
          <p className="text-sm font-medium rounded-xl p-4" style={{ background: 'var(--color-warm)', color: 'var(--color-forest)' }}>
            {validation.summary}
          </p>

          {/* Confidence Assessment */}
          <div>
            <span className="text-xs font-semibold uppercase tracking-[0.12em]" style={{ color: 'var(--color-text-muted)' }}>
              {t('result_confidence_assess')}
            </span>
            <p className="text-sm mt-1" style={{ color: 'var(--color-forest)' }}>{validation.confidence_assessment}</p>
          </div>

          {/* Reasoning */}
          <div>
            <span className="text-xs font-semibold uppercase tracking-[0.12em]" style={{ color: 'var(--color-text-muted)' }}>
              {t('result_reasoning')}
            </span>
            <p className="text-sm mt-1" style={{ color: 'var(--color-forest)' }}>{validation.reasoning}</p>
          </div>

          {/* Alternative Suggestions */}
          {validation.alternative_suggestions?.length > 0 && (
            <div>
              <div className="flex items-center gap-1.5 mb-2">
                <Lightbulb className="w-4 h-4" style={{ color: 'var(--color-gold)' }} />
                <span className="text-xs font-semibold uppercase tracking-[0.12em]" style={{ color: 'var(--color-text-muted)' }}>
                  {t('result_alternatives')}
                </span>
              </div>
              <div className="flex flex-wrap gap-2">
                {validation.alternative_suggestions.map((alt, i) => (
                  <span key={i} className="badge-warning px-2.5 py-1 rounded-full text-xs font-medium">
                    {formatClassName(alt)}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Treatment Advice */}
          {validation.treatment_advice && (
            <div className="rounded-xl p-4 border" style={{ background: 'rgba(107,143,113,0.04)', borderColor: 'rgba(168,197,170,0.2)' }}>
              <div className="flex items-center gap-1.5 mb-2">
                <Pill className="w-4 h-4" style={{ color: 'var(--color-sage)' }} />
                <span className="text-xs font-semibold uppercase tracking-[0.12em]" style={{ color: 'var(--color-sage)' }}>
                  {t('result_treatment')}
                </span>
              </div>
              <p className="text-sm leading-relaxed" style={{ color: 'var(--color-forest)' }}>{validation.treatment_advice}</p>
            </div>
          )}
        </div>
      )}

      {/* Feedback Section */}
      {!feedbackSent ? (
        <div className="rounded-2xl p-4 border mt-4 fade-in-up" style={{ animationDelay: '0.4s', background: 'rgba(107,143,113,0.04)', borderColor: 'rgba(168,197,170,0.2)' }}>
          <p className="text-sm font-medium mb-3" style={{ color: 'var(--color-forest)' }}>
            {t('feedback_question')}
          </p>
          <div className="flex items-center gap-2 flex-wrap">
            <button
              onClick={() => handleFeedback(true)}
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-full text-sm font-medium transition-all hover:scale-105"
              style={{ background: '#22c55e', color: '#fff' }}
            >
              <Check className="w-4 h-4" /> {t('feedback_correct')}
            </button>
            <button
              onClick={() => { setShowClassPicker(true); loadClasses(); }}
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-full text-sm font-medium transition-all hover:scale-105"
              style={{ background: '#ef4444', color: '#fff' }}
            >
              <X className="w-4 h-4" /> {t('feedback_wrong')}
            </button>
          </div>

          {showClassPicker && (
            <div className="mt-3">
              <p className="text-xs mb-2" style={{ color: 'var(--color-text-muted)' }}>
                {t('feedback_select_class')}
              </p>
              <select
                onChange={(e) => handleFeedback(false, e.target.value)}
                defaultValue=""
                className="w-full rounded-xl px-3 py-2 text-sm border"
                style={{
                  background: 'var(--color-warm)',
                  color: 'var(--color-forest)',
                  borderColor: 'rgba(168,197,170,0.3)',
                }}
              >
                <option value="" disabled>-- {t('feedback_select_class')} --</option>
                {allClasses.map((cls) => (
                  <option key={cls} value={cls}>{formatClassName(cls)}</option>
                ))}
              </select>
            </div>
          )}
        </div>
      ) : (
        <div className="rounded-2xl p-4 mt-4 text-center" style={{ background: 'rgba(34,197,94,0.08)' }}>
          <p className="text-sm font-medium" style={{ color: '#22c55e' }}>
            ✓ {t('feedback_submitted')}
          </p>
        </div>
      )}

    </div>
  );
}

function formatClassName(cls) {
  if (!cls) return '';
  return cls
    .replace(/___/g, ' — ')
    .replace(/_/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}
