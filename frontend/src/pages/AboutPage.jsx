import { useEffect, useRef } from 'react';
import { Leaf, Shield, Cpu, BarChart3, Globe, Bot } from 'lucide-react';
import { useLanguage } from '../i18n/LanguageContext';

function useScrollReveal() {
  const ref = useRef(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          el.classList.add('visible');
          observer.unobserve(el);
        }
      },
      { threshold: 0.1 }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return ref;
}

export default function AboutPage() {
  const { t } = useLanguage();
  const headerRef = useScrollReveal();
  const cardsRef = useRef([]);
  const modelCardRef = useScrollReveal();

  useEffect(() => {
    const observers = [];
    cardsRef.current.forEach((el) => {
      if (!el) return;
      const observer = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting) {
            el.classList.add('visible');
            observer.unobserve(el);
          }
        },
        { threshold: 0.1 }
      );
      observer.observe(el);
      observers.push(observer);
    });
    return () => observers.forEach((o) => o.disconnect());
  }, []);

  const features = [
    { icon: Cpu, title: t('about_mobilenet'), desc: t('about_mobilenet_desc'), delay: 'delay-100' },
    { icon: Bot, title: t('about_gemini'), desc: t('about_gemini_desc'), delay: 'delay-200' },
    { icon: Shield, title: t('about_uncertainty'), desc: t('about_uncertainty_desc'), delay: 'delay-100' },
    { icon: BarChart3, title: t('about_reports'), desc: t('about_reports_desc'), delay: 'delay-200' },
    { icon: Globe, title: t('about_classes'), desc: t('about_classes_desc'), delay: 'delay-100' },
    { icon: Leaf, title: t('about_transfer'), desc: t('about_transfer_desc'), delay: 'delay-200' },
  ];

  return (
    <div className="max-w-4xl mx-auto px-4 py-12">
      <div ref={headerRef} className="scroll-reveal mb-12">
        <p className="text-xs uppercase tracking-[0.2em] mb-3" style={{ color: 'var(--color-gold)' }}>
          About the platform
        </p>
        <h1 className="text-3xl md:text-4xl mb-4" style={{ fontFamily: 'var(--font-serif)', color: 'var(--color-forest)' }}>
          {t('about_title')}
        </h1>
        <div className="section-divider mt-4" style={{ margin: '1rem 0 0 0' }} />
        <p className="text-base mt-6 leading-relaxed" style={{ color: 'var(--color-text-muted)' }}>
          {t('about_subtitle')}
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-14">
        {features.map(({ icon: Icon, title, desc, delay }, i) => (
          <div
            key={title}
            ref={(el) => (cardsRef.current[i] = el)}
            className={`scroll-reveal ${delay} feature-card text-left`}
          >
            <div className="feature-icon mb-4" style={{ margin: '0 0 1rem 0' }}>
              <Icon className="w-6 h-6" style={{ color: 'var(--color-sage)', strokeWidth: 1.5 }} />
            </div>
            <h3 className="text-lg mb-2" style={{ fontFamily: 'var(--font-serif)', color: 'var(--color-forest)' }}>{title}</h3>
            <p className="text-sm leading-relaxed" style={{ color: 'var(--color-text-muted)' }}>{desc}</p>
          </div>
        ))}
      </div>

      {/* Model Card */}
      <div ref={modelCardRef} className="scroll-reveal glass-card p-8 md:p-10">
        <h2 className="text-2xl mb-8" style={{ fontFamily: 'var(--font-serif)', color: 'var(--color-forest)' }}>
          {t('about_model_card')}
        </h2>

        <div className="space-y-7 text-sm">
          {[
            {
              title: t('about_architecture'),
              content: (
                <ul className="list-disc list-inside space-y-1.5" style={{ color: 'var(--color-text-muted)' }}>
                  <li><strong style={{ color: 'var(--color-forest)' }}>Vision:</strong> MobileNetV3-Large (timm, pretrained) → frozen backbone → 2-layer classifier head (512→38)</li>
                  <li><strong style={{ color: 'var(--color-forest)' }}>Validation:</strong> Gemini AI (gemini-3-flash-preview) — receives prediction + actual leaf image → returns visual validation report</li>
                  <li><strong style={{ color: 'var(--color-forest)' }}>Graceful degradation:</strong> If Gemini API is unavailable, MobileNet predictions still work without AI validation</li>
                </ul>
              ),
            },
            {
              title: t('about_dataset'),
              content: (
                <p style={{ color: 'var(--color-text-muted)' }}>
                  PlantVillage — 54,000+ images across 38 classes (14 crop species, healthy and diseased).
                  Stratified splits grouped by plant species to prevent data leakage.
                </p>
              ),
            },
            {
              title: t('about_training'),
              content: (
                <ul className="list-disc list-inside space-y-1.5" style={{ color: 'var(--color-text-muted)' }}>
                  <li>Frozen MobileNetV3-Large backbone for CPU-efficient training</li>
                  <li>AdamW optimizer, OneCycleLR scheduler, label smoothing (0.05)</li>
                  <li>Data augmentation: random resize/crop, flips, color jitter</li>
                  <li>Class-weighted loss for handling imbalanced classes</li>
                </ul>
              ),
            },
            {
              title: t('about_validation_pipeline'),
              content: (
                <ul className="list-disc list-inside space-y-1.5" style={{ color: 'var(--color-text-muted)' }}>
                  <li>Image metadata extraction: brightness, contrast, dominant color, green ratio, RGB means</li>
                  <li>MobileNet prediction + top-k alternatives + actual leaf image sent to Gemini AI</li>
                  <li>Structured JSON response: agreement, confidence assessment, reasoning, alternatives, treatment</li>
                  <li>Low temperature (0.3) for consistent, factual responses</li>
                </ul>
              ),
            },
            {
              title: t('about_limitations'),
              content: (
                <ul className="list-disc list-inside space-y-1.5" style={{ color: 'var(--color-text-muted)' }}>
                  <li>Trained only on PlantVillage images — may not generalize to all field conditions</li>
                  <li>Limited to 14 crop species and 38 classes</li>
                  <li>Gemini AI performs multimodal analysis — visually inspects the leaf image for disease symptoms</li>
                  <li>Performance may degrade on images with poor lighting, occlusion, or multiple diseases</li>
                  <li><strong style={{ color: '#C07050' }}>Not a replacement for expert agronomist diagnosis</strong></li>
                </ul>
              ),
            },
            {
              title: t('about_intended_use'),
              content: (
                <p style={{ color: 'var(--color-text-muted)' }}>
                  Screening tool for farmers and agricultural workers to quickly identify potential plant diseases.
                  High-uncertainty predictions and Gemini AI disagreements should always be verified by a qualified agronomist.
                </p>
              ),
            },
          ].map(({ title, content }) => (
            <section key={title}>
              <h3 className="font-semibold text-base mb-2" style={{ color: 'var(--color-sage)' }}>
                {title}
              </h3>
              {content}
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}
