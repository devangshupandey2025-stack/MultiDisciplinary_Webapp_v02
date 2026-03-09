import { useState, useEffect, useRef } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { AlertTriangle, CheckCircle, HelpCircle, Bot, ThumbsUp, ThumbsDown, Pill, Lightbulb, Download, Volume2, Square } from 'lucide-react';
import { useLanguage } from '../i18n/LanguageContext';

const COLORS = ['#2C3E2D', '#6B8F71', '#A8C5AA', '#C9A96E', '#8B7355'];

const LANG_TO_BCP47 = {
  en: 'en-US', hi: 'hi-IN', ta: 'ta-IN', te: 'te-IN',
  bn: 'bn-IN', mr: 'mr-IN', kn: 'kn-IN', gu: 'gu-IN',
};

export default function PredictionResult({ result, imageUrl }) {
  const { t, lang } = useLanguage();
  const [isSpeaking, setIsSpeaking] = useState(false);
  const utteranceRef = useRef(null);

  useEffect(() => {
    return () => window.speechSynthesis?.cancel();
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

  function handleDownload() {
    const disease = formatClassName(result.class);
    const plant = result.class?.split('___')[0]?.replace(/_/g, ' ') || '';
    const date = new Date().toLocaleDateString();

    let text = '';
    text += '═══════════════════════════════════════════\n';
    text += '       PlantGuard AI — Validation Report\n';
    text += '═══════════════════════════════════════════\n\n';
    text += `${t('result_disease')}: ${disease}\n`;
    text += `${t('result_plant')}: ${plant}\n`;
    text += `${t('result_confidence')}: ${(result.probability * 100).toFixed(1)}%\n`;
    text += `${t('result_uncertainty')}: ${(result.uncertainty * 100).toFixed(1)}%\n\n`;

    if (validation) {
      text += '───────────────────────────────────────────\n';
      text += `  ${t('result_validation_title')}\n`;
      text += '───────────────────────────────────────────\n\n';
      text += `${validation.agrees ? '✓ ' + t('result_agrees') : '✗ ' + t('result_disagrees')}\n\n`;
      text += `${validation.summary}\n\n`;
      text += `${t('result_confidence_assess')}:\n  ${validation.confidence_assessment}\n\n`;
      text += `${t('result_reasoning')}:\n  ${validation.reasoning}\n\n`;
      if (validation.alternative_suggestions?.length > 0) {
        text += `${t('result_alternatives')}:\n`;
        validation.alternative_suggestions.forEach(alt => {
          text += `  • ${formatClassName(alt)}\n`;
        });
        text += '\n';
      }
      if (validation.treatment_advice) {
        text += `${t('result_treatment')}:\n  ${validation.treatment_advice}\n\n`;
      }
    }

    text += '═══════════════════════════════════════════\n';
    text += `PlantGuard AI | ${date}\n`;
    text += '═══════════════════════════════════════════\n';

    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `PlantGuard_Report_${disease.replace(/[^a-zA-Z0-9]/g, '_')}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function handleSpeak() {
    const synth = window.speechSynthesis;
    if (!synth) return;

    if (isSpeaking) {
      synth.cancel();
      setIsSpeaking(false);
      return;
    }

    const text = getReportText();
    if (!text) return;

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = LANG_TO_BCP47[lang] || 'en-US';
    utterance.rate = 0.9;
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);
    utteranceRef.current = utterance;
    synth.cancel();
    synth.speak(utterance);
    setIsSpeaking(true);
  }

  const chartData = (result.top_k || []).map((item, i) => ({
    name: formatClassName(item.class),
    probability: (item.probability * 100).toFixed(1),
    fill: COLORS[i % COLORS.length],
  }));

  return (
    <div className="space-y-6">
      {/* Main prediction card */}
      <div className="glass-card p-6 md:p-8 space-y-6">
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
          <div>
            <h4 className="text-sm font-semibold mb-3" style={{ color: 'var(--color-forest)' }}>
              {t('result_top_predictions')}
            </h4>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={chartData} layout="vertical" margin={{ left: 100 }}>
                <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 12, fill: '#6B7B6D' }} />
                <YAxis dataKey="name" type="category" tick={{ fontSize: 11, fill: '#2C3E2D' }} width={100} />
                <Tooltip formatter={(v) => `${v}%`} />
                <Bar dataKey="probability" radius={[0, 6, 6, 0]}>
                  {chartData.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Validation Report */}
      {validation && (
        <div className="glass-card p-6 md:p-8 space-y-5">
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
          <div className="flex items-center gap-2 flex-wrap">
            <button
              onClick={handleDownload}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-colors hover:opacity-80"
              style={{ background: 'var(--color-warm)', color: 'var(--color-forest)' }}
              title={t('result_download')}
            >
              <Download className="w-3.5 h-3.5" />
              {t('result_download')}
            </button>
            <button
              onClick={handleSpeak}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-colors hover:opacity-80 ${isSpeaking ? 'badge-danger' : ''}`}
              style={isSpeaking ? {} : { background: 'var(--color-warm)', color: 'var(--color-forest)' }}
              title={isSpeaking ? t('result_stop_listen') : t('result_listen')}
            >
              {isSpeaking ? <Square className="w-3.5 h-3.5" /> : <Volume2 className="w-3.5 h-3.5" />}
              {isSpeaking ? t('result_stop_listen') : t('result_listen')}
            </button>
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
