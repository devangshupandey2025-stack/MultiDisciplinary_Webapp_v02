<div align="center">

# 🌿 PlantGuard AI

### Intelligent Plant Disease Detection — Powered by MobileNetV3 × Gemini AI

[![Live Demo](https://img.shields.io/badge/Live%20Demo-plantguard--ai--one.vercel.app-2C3E2D?style=for-the-badge&logo=vercel)](https://plantguard-ai-one.vercel.app)
[![API Status](https://img.shields.io/badge/API-Render%20(Free)-6B8F71?style=for-the-badge&logo=render)](https://plantguard-api.onrender.com/api/health)
[![License](https://img.shields.io/badge/License-MIT-C9A96E?style=for-the-badge)](LICENSE)

</div>

---

PlantGuard AI is a full-stack web application that detects plant leaf diseases from photos using a **MobileNetV3 deep-learning classifier** and cross-validates results with **Google Gemini's multimodal AI**. Farmers and agronomists can upload a leaf photo and get an instant diagnosis, confidence score, treatment recommendations, and audio readout in **8 Indian languages**.

<div align="center">

| 🔬 ML Model | 🤖 LLM Cross-validation | 🔊 TTS | 🌐 Languages | 📱 PWA |
|:-----------:|:----------------------:|:------:|:------------:|:------:|
| MobileNetV3 | Gemini 2.5 Flash | Sarvam AI | 8 Indian languages | Installable |

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Disease Detection** | Classifies 38 plant diseases across 14 crop species with 95.24% validation F1-score |
| **Gemini Cross-validation** | Multimodal AI visually inspects the leaf image and validates / challenges the model's prediction |
| **Treatment Advice** | Gemini generates practical, crop-specific treatment steps |
| **Confidence & Uncertainty** | Monte-Carlo dropout uncertainty estimation alongside softmax confidence |
| **Top-5 Predictions** | Ranked chart of the model's top-5 candidate diseases |
| **PDF Report Download** | Export full diagnosis report as a styled PDF |
| **Text-to-Speech** | Sarvam AI reads the report aloud in Hindi, Tamil, Telugu, Bengali, Marathi, Kannada, Gujarati, or English |
| **Prediction History** | Authenticated users get a searchable history with expandable Gemini reports |
| **User Feedback & Retraining** | Users mark predictions correct/wrong → feedback collected → admin can fine-tune model head |
| **PWA / Install App** | Installable as a mobile app on Android/desktop via `beforeinstallprompt` |
| **Drone/Batch Watcher** | CLI script polls a folder and sends new images automatically to the API |
| **8-Language UI** | Full interface localization for EN, HI, TA, TE, BN, MR, KN, GU |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Browser / PWA                             │
│   React 19 + Vite + Tailwind CSS                                 │
│   ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│   │  HomePage   │  │ HistoryPage  │  │      AboutPage         │ │
│   │(Upload+Diag)│  │(Past Results)│  │(ModelTraining Dashboard)│ │
│   └──────┬──────┘  └──────┬───────┘  └────────────────────────┘ │
└──────────┼────────────────┼─────────────────────────────────────┘
           │  /api/* proxy  │
           ▼                ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Vercel (vercel.json rewrite)                   │
│         /api/* → https://plantguard-api.onrender.com/api/*       │
└──────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│              FastAPI Backend (Render — Free Tier)                 │
│                                                                   │
│  POST /api/predict                                                │
│    │                                                              │
│    ├─► MobileNetV3 (timm, CPU) ──────────► class + top-5 + σ    │
│    │                                                              │
│    └─► Gemini 2.0 Flash (multimodal) ───► visual validation +    │
│                                           treatment advice        │
│                                                                   │
│  POST /api/tts ─────────────────────────► Sarvam AI → WAV audio  │
│  POST /api/feedback ────────────────────► Supabase DB            │
│  POST /api/admin/retrain ───────────────► Fine-tune model head   │
└──────────────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Supabase (PostgreSQL)                          │
│    predictions   │  training_feedback  │  users (Supabase Auth)  │
└──────────────────────────────────────────────────────────────────┘
```

### Model Architecture

```
Input (224×224 RGB)
       │
       ▼
MobileNetV3-Large backbone (timm, frozen during inference)
       │
       ▼
Classification Head (trainable, ~500K params):
  Dropout(0.3) → Linear(feat_dim, 512) → ReLU → BatchNorm1d → Dropout(0.2) → Linear(512, 38)
       │
       ▼
Softmax → class probabilities + uncertainty (MC-Dropout entropy)
```

---

## 🚀 Live Deployment

| Service | URL |
|---------|-----|
| 🌐 Frontend (Vercel) | [plantguard-ai-one.vercel.app](https://plantguard-ai-one.vercel.app) |
| ⚙️ Backend API (Render) | [plantguard-api.onrender.com/api/health](https://plantguard-api.onrender.com/api/health) |

> **Note:** The backend runs on Render's **free tier** — it sleeps after 15 minutes of inactivity. The first request after a sleep takes 30–60 seconds to wake up.

---

## 📋 API Reference

All endpoints are prefixed with `/api/`.

### 🏥 Health

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/health` | — | Server status, model load state, Gemini availability |

**Response:**
```json
{ "status": "healthy", "model_loaded": true, "gemini_available": true, "device": "cpu" }
```

---

### 🌿 Prediction

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/predict?top_k=5` | Optional | Classify a leaf image; Gemini cross-validates the result |
| `GET` | `/classes` | — | List all 38 supported disease class names |

**Request:** `multipart/form-data` with field `file` (JPEG/PNG leaf image)

**Response:**
```json
{
  "class": "Tomato___Early_blight",
  "probability": 0.924,
  "uncertainty": 0.076,
  "top_k": [
    { "class": "Tomato___Early_blight", "probability": 0.924 },
    { "class": "Tomato___Late_blight",  "probability": 0.041 }
  ],
  "image_metadata": { "width": 256, "height": 256, "brightness": 0.43, "green_ratio": 0.51 },
  "gemini_validation": {
    "agrees": true,
    "confidence_assessment": "High — visible concentric necrotic rings",
    "reasoning": "The brown lesions with concentric ring pattern are characteristic of Early Blight...",
    "alternative_suggestions": ["Tomato___Septoria_leaf_spot"],
    "treatment_advice": "Apply mancozeb or chlorothalonil fungicide. Remove infected leaves.",
    "summary": "Confirmed Early Blight with high visual confidence"
  },
  "latency_ms": 312
}
```

---

### 🔐 Authentication

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/auth/signup` | — | Register with email + password |
| `POST` | `/auth/signin` | — | Sign in, receive JWT access token |

---

### 📜 History & Stats

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/history?limit=20` | ✅ Required | Paginated prediction history with Gemini reports |
| `GET` | `/stats` | ✅ Required | Aggregate stats (total predictions, accuracy) |

---

### 💬 Feedback & Training

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/feedback` | Optional | Submit correct/incorrect label for a prediction |
| `GET` | `/feedback/stats` | — | Feedback count, training status, `can_retrain` flag |
| `POST` | `/admin/retrain` | ✅ Required | Trigger fine-tuning of classification head (min 5 samples) |

**Feedback Request Body:**
```json
{
  "prediction_id": "uuid",
  "image_url": "https://...",
  "predicted_class": "Tomato___Early_blight",
  "actual_class": "Tomato___Septoria_leaf_spot",
  "is_correct": false
}
```

---

### 🔊 Text-to-Speech

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/tts` | — | Convert text to speech via Sarvam AI (max 1500 chars) |

**Request Body:**
```json
{ "text": "टमाटर की पत्ती में अगेती झुलसा रोग है।", "language": "hi" }
```

**Response:** WAV audio stream (`audio/wav`)

**Supported language codes:** `en`, `hi`, `ta`, `te`, `bn`, `mr`, `kn`, `gu`

---

## 🛠️ Local Development

### Prerequisites

- Python 3.11+
- Node.js 18+
- A [Supabase](https://supabase.com) project
- A [Google Gemini](https://aistudio.google.com) API key
- A [Sarvam AI](https://sarvam.ai) API key

### 1. Clone & Setup

```bash
git clone https://github.com/devangshupandey2025-stack/MultiDisciplinary_Webapp_v02.git
cd MultiDisciplinary_Webapp_v02
```

### 2. Backend

```bash
cd backend

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your keys (see Environment Variables below)

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
echo "VITE_API_URL=http://localhost:8000" > .env.local

# Start dev server
npm run dev
# → http://localhost:5173
```

### 4. Supabase Setup

Run these SQL statements in your Supabase SQL editor:

```sql
-- Prediction history
CREATE TABLE predictions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id),
  class TEXT NOT NULL,
  probability FLOAT NOT NULL,
  image_url TEXT,
  gemini_validation JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- User feedback for model retraining
CREATE TABLE training_feedback (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID,
  prediction_id UUID,
  image_url TEXT,
  predicted_class TEXT NOT NULL,
  actual_class TEXT,
  is_correct BOOLEAN NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);
```

---

## 🔑 Environment Variables

### Backend (`backend/.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `SUPABASE_URL` | ✅ | Your Supabase project URL |
| `SUPABASE_KEY` | ✅ | Supabase anon/public key |
| `SUPABASE_SERVICE_KEY` | ✅ | Supabase service role key (for admin ops) |
| `GEMINI_API_KEY` | ✅ | Google Gemini API key |
| `GEMINI_MODEL` | ❌ | Model name (default: `gemini-2.0-flash`) |
| `SARVAM_API_KEY` | ✅ | Sarvam AI API key for TTS |
| `MODELS_DIR` | ❌ | Checkpoint directory (default: `checkpoints`) |
| `FRONTEND_URL` | ✅ | Frontend origin for CORS (e.g. `https://yourapp.vercel.app`) |

### Frontend (`frontend/.env.local`)

| Variable | Required | Description |
|----------|----------|-------------|
| `VITE_API_URL` | ❌ | Backend API URL (default: uses Vercel proxy `/api/*`) |
| `VITE_SUPABASE_URL` | ✅ | Supabase project URL |
| `VITE_SUPABASE_ANON_KEY` | ✅ | Supabase anon key |

---

## 📦 Project Structure

```
MultiDisciplinary_Webapp_v02/
│
├── frontend/                         # React 19 + Vite + Tailwind CSS 4
│   ├── public/                       # Static assets + PWA icons
│   │   ├── pwa-192x192.png
│   │   └── pwa-512x512.png
│   ├── src/
│   │   ├── components/
│   │   │   ├── Layout.jsx            # App shell (nav, footer, PWA install button)
│   │   │   ├── PredictionResult.jsx  # Diagnosis card, chart, Gemini report, TTS, download
│   │   │   ├── AuthModal.jsx         # Login / signup modal
│   │   │   └── ModelTraining.jsx     # Fine-tuning dashboard
│   │   ├── pages/
│   │   │   ├── HomePage.jsx          # Image upload + live prediction
│   │   │   ├── HistoryPage.jsx       # Authenticated prediction history
│   │   │   └── AboutPage.jsx         # Project info + training dashboard
│   │   ├── services/
│   │   │   ├── api.js                # Axios API client
│   │   │   └── supabase.js           # Supabase auth helpers
│   │   ├── i18n/
│   │   │   ├── LanguageContext.jsx   # React context for language switching
│   │   │   └── translations.js       # 8-language string tables
│   │   ├── App.jsx                   # Router + providers + Toaster
│   │   ├── main.jsx                  # React DOM entry point
│   │   └── index.css                 # Global styles + animations
│   ├── index.html                    # PWA meta tags, viewport
│   ├── vite.config.js                # Vite + VitePWA config
│   ├── vercel.json                   # Vercel API proxy rewrite
│   └── package.json
│
├── backend/                          # FastAPI + PyTorch
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py             # All API route handlers
│   │   ├── ml/
│   │   │   ├── predictor.py          # MobileNetV3 inference engine
│   │   │   ├── gemini_validator.py   # Gemini cross-validation (with retry logic)
│   │   │   └── finetuner.py          # Feedback-driven fine-tuning engine
│   │   ├── services/
│   │   │   └── supabase_service.py   # DB operations (predictions, feedback, auth)
│   │   └── main.py                   # FastAPI app, lifespan, CORS
│   ├── requirements.txt
│   └── Dockerfile
│
├── checkpoints/                      # Model weights (not committed)
│   └── mobilenet_v3/
│       └── best_model.pt             # Trained checkpoint (val F1 = 0.9524)
│
├── scripts/
│   └── drone_watcher.py              # Folder watcher for automated batch prediction
│
├── ml_pipeline/                      # Training pipeline (EfficientNet, ResNet, ensemble)
│   ├── configs/                      # YAML training configs
│   ├── scripts/                      # train.py, evaluate.py, export_models.py
│   ├── models/                       # Model architecture definitions
│   └── ensemble/                     # Stacking & calibration
│
├── train_plantvillage.py             # Standalone training script for PlantVillage dataset
├── render.yaml                       # Render deployment config
├── docker-compose.yml                # Local full-stack Docker setup
└── README.md
```

---

## 🚀 Deployment

### Frontend → Vercel

```bash
cd frontend
npx vercel --prod
```

The `vercel.json` proxies all `/api/*` requests to the Render backend automatically:

```json
{
  "rewrites": [{ "source": "/api/:path*", "destination": "https://plantguard-api.onrender.com/api/:path*" }]
}
```

### Backend → Render

The `render.yaml` in the repo root configures the Render service automatically on connection. Set these env vars in the Render dashboard (or via CLI):

```bash
# Required
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...
SUPABASE_SERVICE_KEY=eyJ...
GEMINI_API_KEY=AIza...
SARVAM_API_KEY=sk_...
FRONTEND_URL=https://your-app.vercel.app

# Optional (defaults shown)
GEMINI_MODEL=gemini-2.0-flash
MODELS_DIR=checkpoints
PYTHON_VERSION=3.11.0
```

### Docker (Full Stack)

```bash
docker-compose up --build
# Frontend: http://localhost:5173
# Backend:  http://localhost:8000
```

---

## 🤖 Automated Batch Prediction (Drone/Phone Watcher)

The `scripts/drone_watcher.py` script monitors a folder and sends new images to the API automatically — useful for drones, trail cameras, or phone sync folders.

```bash
# Anonymous mode (no history saved)
python scripts/drone_watcher.py --folder /path/to/images

# Authenticated mode (saves to history)
python scripts/drone_watcher.py --folder /path/to/images \
  --email user@example.com --password yourpassword

# Options
--api-url    API base URL (default: https://plantguard-api.onrender.com)
--interval   Polling interval in seconds (default: 5)
--folder     Folder to watch for new images
```

---

## 🧠 Model Training

### Fine-tuning from User Feedback

The backend collects user corrections and supports fine-tuning the classification head (backbone stays frozen):

1. Users submit feedback via the UI (Correct ✅ / Wrong ❌ + correct class)
2. Feedback accumulates in the `training_feedback` Supabase table
3. When ≥5 samples collected, an admin can trigger retraining:

```bash
curl -X POST https://plantguard-api.onrender.com/api/admin/retrain \
  -H "Authorization: Bearer <admin_jwt_token>"
```

The fine-tuner:
- Downloads feedback images from Supabase storage
- Trains only the classification head (`~500K params`) for 5 epochs
- Uses AdamW optimizer, class-weighted CrossEntropyLoss, LR=1e-4
- Hot-swaps the updated model without server restart
- Saves versioned checkpoints in `checkpoints/mobilenet_v3/finetuned/`

### Training from Scratch (PlantVillage)

```bash
# Download PlantVillage dataset first, place in data/plantvillage/
python train_plantvillage.py \
  --data_dir data/plantvillage \
  --epochs 30 \
  --batch_size 32 \
  --output_dir checkpoints/mobilenet_v3
```

---

## 🌐 Supported Languages

| Code | Language | TTS Voice |
|------|----------|-----------|
| `en` | English | Sarvam AI (anushka) |
| `hi` | हिन्दी (Hindi) | Sarvam AI (anushka) |
| `ta` | தமிழ் (Tamil) | Sarvam AI (anushka) |
| `te` | తెలుగు (Telugu) | Sarvam AI (anushka) |
| `bn` | বাংলা (Bengali) | Sarvam AI (anushka) |
| `mr` | मराठी (Marathi) | Sarvam AI (anushka) |
| `kn` | ಕನ್ನಡ (Kannada) | Sarvam AI (anushka) |
| `gu` | ગુજરાતી (Gujarati) | Sarvam AI (anushka) |

---

## 🌿 Supported Plant Diseases (38 Classes)

14 crop species across 38 disease/healthy categories from the [PlantVillage dataset](https://github.com/spMohanty/PlantVillage-Dataset):

| Crop | Conditions |
|------|-----------|
| Apple | Scab, Black Rot, Cedar Apple Rust, Healthy |
| Cherry | Powdery Mildew, Healthy |
| Corn (Maize) | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca, Leaf Blight, Healthy |
| Orange | Haunglongbing (Citrus Greening) |
| Peach | Bacterial Spot, Healthy |
| Pepper | Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery Mildew |
| Strawberry | Leaf Scorch, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## 🧩 Tech Stack

### Backend
| | Package | Version |
|--|---------|---------|
| 🔧 Framework | FastAPI | ≥0.104 |
| ⚡ Server | Uvicorn | ≥0.24 |
| 🧠 ML | PyTorch (CPU) + timm | 2.1.0 |
| 🤖 LLM | google-genai | ≥1.0 |
| 🗄️ Database | supabase-py | ≥2.0 |
| 🌐 HTTP | httpx | ≥0.25 |
| 🖼️ Images | Pillow | ≥10.0 |

### Frontend
| | Package | Version |
|--|---------|---------|
| ⚛️ UI | React | 19.2.0 |
| ⚡ Build | Vite | 7.3.1 |
| 🎨 CSS | Tailwind CSS | 4.2.1 |
| 📊 Charts | Recharts | 3.8.0 |
| 🔔 Toasts | Sonner | 2.0.7 |
| 📄 PDF | jsPDF + html2canvas | 4.2.0 / 1.4.1 |
| 📱 PWA | vite-plugin-pwa | 1.2.0 |

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Architecture | MobileNetV3-Large |
| Training Dataset | PlantVillage (38 classes) |
| Validation F1-Score | **0.9524** |
| Inference Device | CPU |
| Input Size | 224 × 224 RGB |
| Parameters (head) | ~500K trainable |
| Uncertainty Method | MC-Dropout entropy |

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit: `git commit -m "Add amazing feature"`
4. Push: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

Built with ❤️ for Indian farmers · MobileNetV3 × Gemini AI × Sarvam AI

</div>

