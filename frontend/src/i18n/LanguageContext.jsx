import { createContext, useContext, useState, useCallback } from 'react';
import translations, { LANGUAGES } from './translations';

const LanguageContext = createContext();

export function LanguageProvider({ children }) {
  const [lang, setLang] = useState(() => {
    const saved = localStorage.getItem('plantguard_lang');
    return saved && translations[saved] ? saved : 'en';
  });

  const changeLang = useCallback((code) => {
    if (translations[code]) {
      setLang(code);
      localStorage.setItem('plantguard_lang', code);
    }
  }, []);

  const t = useCallback(
    (key) => translations[lang]?.[key] || translations.en?.[key] || key,
    [lang]
  );

  return (
    <LanguageContext.Provider value={{ lang, changeLang, t, LANGUAGES }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  const ctx = useContext(LanguageContext);
  if (!ctx) throw new Error('useLanguage must be used within LanguageProvider');
  return ctx;
}
