import axios from 'axios';
import { getSession } from './supabase';

const API_BASE = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
});

// Add auth token to requests — always fetch fresh token from Supabase session
api.interceptors.request.use(async (config) => {
  try {
    const session = await getSession();
    if (session?.access_token) {
      localStorage.setItem('access_token', session.access_token);
      config.headers.Authorization = `Bearer ${session.access_token}`;
      return config;
    }
  } catch {
    // Fall back to stored token
  }
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export const predictDisease = async (imageFile, topK = 5) => {
  const formData = new FormData();
  formData.append('file', imageFile);
  const response = await api.post(`/predict?top_k=${topK}`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const getHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

export const getClasses = async () => {
  const response = await api.get('/classes');
  return response.data;
};

export const getHistory = async (limit = 20) => {
  const response = await api.get(`/history?limit=${limit}`);
  return response.data;
};

export const getStats = async () => {
  const response = await api.get('/stats');
  return response.data;
};

export const generateTTS = async (text, language = 'en', pace = 1.0) => {
  const response = await api.post('/tts', { text, language, pace }, {
    responseType: 'blob',
    timeout: 30000,
  });
  return response.data;
};

export const submitFeedback = async (feedbackData) => {
  const response = await api.post('/feedback', feedbackData);
  return response.data;
};

export const getFeedbackStats = async () => {
  const response = await api.get('/feedback/stats');
  return response.data;
};

export const triggerRetrain = async () => {
  const response = await api.post('/admin/retrain', {}, { timeout: 300000 });
  return response.data;
};

export default api;
