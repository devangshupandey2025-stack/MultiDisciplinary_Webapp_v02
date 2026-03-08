# Deployment Playbook

## Prerequisites
- Python 3.11+
- Node.js 20+
- Supabase account (free tier works)
- Render account (free tier works)

---

## 1. Supabase Setup

1. Create a new project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor** and create the predictions table:

```sql
CREATE TABLE predictions (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id),
  prediction_class TEXT NOT NULL,
  probability FLOAT NOT NULL,
  uncertainty FLOAT NOT NULL,
  top_k JSONB,
  image_url TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Enable RLS
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

-- Users can only see their own predictions
CREATE POLICY "Users view own predictions" ON predictions
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users insert own predictions" ON predictions
  FOR INSERT WITH CHECK (auth.uid() = user_id);
```

3. Create a storage bucket named `plant-images` (public read)
4. Copy your **Project URL** and **anon key** from Settings → API

---

## 2. Local Development

```bash
# Backend
cd backend
cp .env.example .env
# Edit .env with your Supabase credentials
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend (separate terminal)
cd frontend
cp .env.example .env.local
# Edit .env.local with your Supabase and API URLs
npm install
npm run dev
```

---

## 3. Train Models (Optional - for full pipeline)

```bash
# Generate synthetic data for testing
python ml_pipeline/scripts/generate_synthetic_data.py --output_dir data/synthetic --num_classes 5 --num_images 20

# Train single model
python ml_pipeline/scripts/train.py --config ml_pipeline/configs/mobilenet_v3.yaml --data_dir data/synthetic --override training.epochs=5

# Train all 5 models
python ml_pipeline/scripts/train_all.py --data_dir data/plantvillage

# Run stacking ensemble
python ml_pipeline/scripts/stacking.py --models_dir checkpoints --data_dir data/plantvillage

# Calibrate
python ml_pipeline/scripts/calibrate.py --ensemble_dir checkpoints/ensemble

# Export
python ml_pipeline/scripts/export_models.py --checkpoint checkpoints/efficientnet_v2/best_model.pt --format onnx torchscript
```

---

## 4. Deploy to Render

### Backend (Web Service)
1. Push code to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Settings:
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Python 3
5. Add environment variables:
   - `SUPABASE_URL`
   - `SUPABASE_KEY`
   - `FRONTEND_URL` (your frontend URL)

### Frontend (Static Site)
1. New → Static Site
2. Connect same repo
3. Settings:
   - **Build Command**: `cd frontend && npm ci && npm run build`
   - **Publish Directory**: `frontend/dist`
4. Add environment variables:
   - `VITE_API_URL` → your backend URL + `/api`
   - `VITE_SUPABASE_URL`
   - `VITE_SUPABASE_ANON_KEY`

Or use the `render.yaml` blueprint for automatic setup.

---

## 5. Docker Deployment

```bash
# Build and run
docker-compose up --build

# Or individually
docker build -f backend/Dockerfile -t plantguard-backend .
docker run -p 8000:8000 --env-file backend/.env plantguard-backend
```

---

## 6. Edge Deployment

```bash
# Distill student model
python ml_pipeline/scripts/distill.py --teacher_dir checkpoints --data_dir data/plantvillage --output_dir checkpoints/distilled

# Export to TFLite with INT8 quantization
python ml_pipeline/scripts/export_models.py --checkpoint checkpoints/distilled/distilled_student.pt --format tflite --quantize int8

# Export to ONNX for TensorRT
python ml_pipeline/scripts/export_models.py --checkpoint checkpoints/distilled/distilled_student.pt --format onnx
# Then: trtexec --onnx=model.onnx --saveEngine=model.trt --int8
```

---

## 7. Monitoring Checklist

- [ ] Prometheus scraping `/metrics` endpoint
- [ ] Grafana dashboard for prediction rates, confidence, uncertainty
- [ ] Sentry DSN configured for error tracking
- [ ] Log prediction distributions and check for drift
- [ ] Alert on high uncertainty rate (>20% of predictions)
- [ ] Alert on class distribution shift (KL divergence > 0.1)
- [ ] Monitor per-class error rates weekly
