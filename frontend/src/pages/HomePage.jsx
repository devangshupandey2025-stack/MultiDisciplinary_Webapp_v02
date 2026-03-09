import { useCallback, useState, useEffect, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Camera, Loader2, Sparkles, Shield, AlertTriangle, ImagePlus } from 'lucide-react';
import PredictionResult from '../components/PredictionResult';
import { predictDisease } from '../services/api';
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
      { threshold: 0.15 }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return ref;
}

export default function HomePage() {
  const { t } = useLanguage();
  const [selectedImage, setSelectedImage] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const cameraInputRef = useRef(null);
  const galleryInputRef = useRef(null);

  const featuresRef = useScrollReveal();
  const feature1Ref = useScrollReveal();
  const feature2Ref = useScrollReveal();
  const feature3Ref = useScrollReveal();

  const handleFileSelect = useCallback((file) => {
    setSelectedImage(file);
    setImageUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  }, []);

  const onDrop = useCallback((files) => {
    if (files.length > 0) handleFileSelect(files[0]);
  }, [handleFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.webp', '.bmp'] },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024,
  });

  const handleCameraCapture = (e) => {
    const file = e.target.files?.[0];
    if (file) handleFileSelect(file);
  };

  const handlePredict = async () => {
    if (!selectedImage) return;
    setLoading(true);
    setError(null);

    try {
      const prediction = await predictDisease(selectedImage);
      setResult(prediction);
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Prediction failed';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setImageUrl(null);
    setResult(null);
    setError(null);
  };

  return (
    <div>
      {/* Hero Section */}
      <section className="hero-section px-4 pt-32 pb-24 md:pt-40 md:pb-32">
        <div className="hero-leaf-bg" />
        <div className="hero-overlay" />
        <div className="relative z-10 max-w-3xl mx-auto text-center text-white">
          <p className="text-sm uppercase tracking-[0.25em] text-white/50 mb-6 font-light">
            {t('hero_powered')}
          </p>
          <h1
            className="text-5xl sm:text-6xl md:text-7xl mb-6 leading-[1.1]"
            style={{ fontFamily: 'var(--font-serif)', fontWeight: 400 }}
          >
            {t('hero_title_1')}{' '}
            <span className="italic" style={{ color: '#A8C5AA' }}>{t('hero_title_2')}</span>
          </h1>
          <p className="text-lg sm:text-xl text-white/65 max-w-xl mx-auto mb-10 leading-relaxed font-light">
            {t('hero_subtitle')}
          </p>

          {/* Scroll indicator */}
          <div className="mt-8 animate-bounce opacity-40">
            <div className="w-5 h-9 border border-white/30 rounded-full mx-auto flex justify-center pt-2">
              <div className="w-1 h-2.5 bg-white/50 rounded-full" />
            </div>
          </div>
        </div>
      </section>

      {/* Upload Section */}
      <section className="max-w-3xl mx-auto px-4 py-16 relative z-20">
        {!imageUrl ? (
          <div>
            <div className="text-center mb-10">
              <h2
                className="text-3xl sm:text-4xl mb-3"
                style={{ fontFamily: 'var(--font-serif)', color: 'var(--color-forest)' }}
              >
                {t('predict_btn')}
              </h2>
              <div className="section-divider mt-4 mb-6" />
            </div>

            {/* Drag & Drop */}
            <div
              {...getRootProps()}
              className={`upload-zone ${isDragActive ? 'active' : ''}`}
            >
              <input {...getInputProps()} />
              <Upload className="w-12 h-12 mx-auto mb-4" style={{ color: 'var(--color-sage)', strokeWidth: 1.5 }} />
              <p className="text-lg font-medium mb-1" style={{ color: 'var(--color-forest)' }}>
                {isDragActive ? t('upload_drag') : t('upload_title')}
              </p>
              <p className="text-sm mb-5" style={{ color: 'var(--color-text-muted)' }}>
                {t('upload_browse')}
              </p>
              <div className="flex justify-center gap-2 flex-wrap">
                {['JPG', 'PNG', 'WebP'].map((f) => (
                  <span key={f} className="px-3 py-1 rounded-full text-xs"
                    style={{ background: 'var(--color-warm)', color: 'var(--color-text-muted)' }}>
                    {f}
                  </span>
                ))}
                <span className="px-3 py-1 rounded-full text-xs"
                  style={{ background: 'rgba(201,169,110,0.12)', color: 'var(--color-earth)' }}>
                  {t('upload_max')}
                </span>
              </div>
            </div>

            {/* Camera + Gallery */}
            <div className="flex flex-col sm:flex-row gap-3 mt-6">
              <button
                onClick={() => cameraInputRef.current?.click()}
                className="camera-btn flex items-center justify-center gap-2 flex-1"
              >
                <Camera className="w-4 h-4" />
                {t('upload_camera')}
              </button>
              <button
                onClick={() => galleryInputRef.current?.click()}
                className="btn-secondary flex items-center justify-center gap-2 flex-1"
              >
                <ImagePlus className="w-4 h-4" />
                {t('upload_gallery')}
              </button>
              <input ref={cameraInputRef} type="file" accept="image/*" capture="environment" onChange={handleCameraCapture} className="hidden" />
              <input ref={galleryInputRef} type="file" accept="image/*" onChange={handleCameraCapture} className="hidden" />
            </div>
          </div>
        ) : (
          <div className="space-y-8">
            {/* Image preview */}
            <div className="glass-card p-8">
              <div className="flex flex-col sm:flex-row items-center gap-8">
                <img
                  src={imageUrl}
                  alt="Selected leaf"
                  className="w-48 h-48 sm:w-56 sm:h-56 object-cover rounded-2xl shadow-sm"
                />
                <div className="flex-1 text-center sm:text-left">
                  <p className="text-base font-medium mb-1" style={{ color: 'var(--color-forest)' }}>
                    {selectedImage?.name}
                  </p>
                  <p className="text-sm mb-6" style={{ color: 'var(--color-text-muted)' }}>
                    {(selectedImage?.size / 1024).toFixed(0)} KB
                  </p>
                  <div className="flex flex-col sm:flex-row gap-3">
                    <button
                      onClick={handlePredict}
                      disabled={loading}
                      className="btn-primary flex items-center justify-center gap-2"
                    >
                      {loading ? (
                        <><Loader2 className="w-4 h-4 animate-spin" /> {t('predict_loading')}</>
                      ) : (
                        <><Sparkles className="w-4 h-4" /> {t('predict_btn')}</>
                      )}
                    </button>
                    <button onClick={handleReset} className="btn-secondary">
                      {t('predict_another')}
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Error */}
            {error && (
              <div className="rounded-xl p-5 border" style={{ background: 'rgba(192,112,80,0.05)', borderColor: 'rgba(192,112,80,0.15)' }}>
                <p className="font-medium text-sm" style={{ color: '#8B4C3B' }}>{t('predict_error')}</p>
                <p className="text-sm mt-1" style={{ color: '#A05A3E' }}>{error}</p>
              </div>
            )}

            {/* Result */}
            {result && <PredictionResult result={result} imageUrl={imageUrl} />}
          </div>
        )}
      </section>

      {/* Features Section */}
      <section style={{ background: 'var(--color-warm)' }} className="py-20">
        <div className="max-w-5xl mx-auto px-4">
          <div ref={featuresRef} className="scroll-reveal text-center mb-14">
            <p className="text-xs uppercase tracking-[0.2em] mb-3" style={{ color: 'var(--color-gold)' }}>
              How it works
            </p>
            <h2 className="text-3xl md:text-4xl" style={{ fontFamily: 'var(--font-serif)', color: 'var(--color-forest)' }}>
              {t('hero_powered')}
            </h2>
            <div className="section-divider mt-5" />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              { ref: feature1Ref, icon: Sparkles, title: t('feature_ai_title'), desc: t('feature_ai_desc'), delay: 'delay-100' },
              { ref: feature2Ref, icon: Shield, title: t('feature_validate_title'), desc: t('feature_validate_desc'), delay: 'delay-200' },
              { ref: feature3Ref, icon: AlertTriangle, title: t('feature_uncertainty_title'), desc: t('feature_uncertainty_desc'), delay: 'delay-300' },
            ].map(({ ref, icon: Icon, title, desc, delay }) => (
              <div key={title} ref={ref} className={`scroll-reveal ${delay} feature-card`}>
                <div className="feature-icon">
                  <Icon className="w-6 h-6" style={{ color: 'var(--color-sage)', strokeWidth: 1.5 }} />
                </div>
                <h3 className="text-lg mb-2" style={{ fontFamily: 'var(--font-serif)', color: 'var(--color-forest)' }}>
                  {title}
                </h3>
                <p className="text-sm leading-relaxed" style={{ color: 'var(--color-text-muted)' }}>{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
