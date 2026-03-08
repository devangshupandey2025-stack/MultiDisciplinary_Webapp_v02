# Plant Leaf Disease Detection — Ensemble ML System

A production-grade ensemble system for plant leaf disease detection using PlantVillage dataset (38 classes, 14 crops).

## Quick Start

```bash
# 1. Install dependencies
pip install -r ml_pipeline/requirements.txt
pip install -r backend/requirements.txt
cd frontend && npm install

# 2. Generate synthetic data for testing
python ml_pipeline/scripts/generate_synthetic_data.py --output_dir data/synthetic --num_classes 5 --num_images 50

# 3. Train a single base model
python ml_pipeline/scripts/train.py --config ml_pipeline/configs/efficientnet_v2.yaml --data_dir data/plantvillage

# 4. Train all base models
python ml_pipeline/scripts/train_all.py --data_dir data/plantvillage

# 5. Run stacking ensemble
python ml_pipeline/scripts/stacking.py --models_dir checkpoints/ --data_dir data/plantvillage

# 6. Calibrate models
python ml_pipeline/scripts/calibrate.py --ensemble_dir checkpoints/ensemble

# 7. Export models
python ml_pipeline/scripts/export_models.py --checkpoint checkpoints/ensemble/best.pt --format onnx torchscript tflite

# 8. Run evaluation
python ml_pipeline/scripts/evaluate.py --ensemble_dir checkpoints/ensemble --data_dir data/plantvillage

# 9. Start inference server
cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000

# 10. Start frontend
cd frontend && npm run dev
```

## Architecture

- **5 Base Models**: EfficientNetV2-S, ResNet50, ConvNeXt-Tiny, Swin-Tiny, MobileNetV3-Large
- **Ensemble**: Validation-weighted soft voting → Stacked meta-learner (LightGBM)
- **Calibration**: Temperature scaling with ECE < 5%
- **Edge**: Knowledge distillation → MobileNetV3 student → TFLite INT8 (≤30MB)

## Project Structure

```
├── ml_pipeline/          # ML training & inference
│   ├── configs/          # YAML configs per model
│   ├── scripts/          # Training, eval, export scripts
│   ├── models/           # Model architectures
│   ├── data/             # Data loading & augmentation
│   ├── ensemble/         # Stacking & calibration
│   ├── distillation/     # Knowledge distillation
│   ├── quantization/     # INT8/FP16 quantization
│   └── tests/            # ML unit tests
├── backend/              # FastAPI inference server
│   ├── app/
│   │   ├── api/          # API routes
│   │   ├── ml/           # ML inference engine
│   │   ├── models/       # Pydantic models
│   │   └── services/     # Supabase integration
│   └── Dockerfile
├── frontend/             # React + Tailwind CSS
│   ├── src/
│   └── Dockerfile
├── deployment/           # Docker, Helm, monitoring
├── .github/workflows/    # CI/CD
└── docs/                 # Documentation & model card
```
