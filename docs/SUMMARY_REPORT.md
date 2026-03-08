# Summary Report: PlantGuard AI

## Expected Performance

| Metric | Server/Cloud | Edge/Mobile |
|--------|-------------|-------------|
| Top-1 Accuracy | 92–95% | 88–92% |
| Macro F1 | ≥0.90 | ≥0.85 |
| Per-class F1 | ≥0.85 | ≥0.80 |
| ECE (calibration) | <3% | <5% |
| Model Size | ~500MB (ensemble) | ≤30MB (TFLite) |
| Latency (single image) | ~50ms (GPU) | <100ms (mobile CPU) |
| Throughput | ~200 img/s (batch GPU) | ~10 img/s |

## Architecture Decisions

### Why 5 Diverse Architectures?
- **EfficientNetV2-S**: Best accuracy-efficiency ratio for server; compound scaling captures multi-scale features
- **ResNet50**: Robust baseline with strong residual learning; different inductive bias from attention models
- **ConvNeXt-Tiny**: Modernized CNN competing with ViTs; adds depth-wise convolution diversity
- **Swin-Tiny**: Shifted window attention captures global context that CNNs miss
- **MobileNetV3**: Lightweight for edge deployment; also adds diversity through inverted residual blocks

Architecture diversity maximizes ensemble benefit — models make different errors on different samples.

### Why Two-Stage Ensembling?
1. **Weighted soft voting** provides a strong, interpretable baseline with minimal overhead
2. **Stacked LightGBM meta-learner** captures non-linear inter-model correlations that averaging misses
3. **K-fold OOF predictions** prevent information leakage in the stacking process

### Why Temperature Scaling?
- Single-parameter calibration that preserves accuracy while fixing confidence
- Modern neural networks are systematically overconfident
- Temperature scaling is simple, effective, and adds zero inference cost

### Uncertainty Estimation
- **Predictive entropy**: Captures aleatoric uncertainty (inherent data noise)
- **Model disagreement (std)**: Captures epistemic uncertainty (model knowledge gaps)
- Combined score flags predictions that need human review

## Recommended Configurations

### Cloud/Server Deployment
- Full 5-model ensemble with stacking meta-learner
- EfficientNetV2 at 384×384, others at 224×224
- GPU batching with ONNX Runtime or TorchScript
- No model size constraint

### Edge/Mobile Deployment
- Distilled MobileNetV3 student from ensemble teacher
- TFLite INT8 quantized (≤15MB typical)
- 224×224 input resolution
- Single model inference with temperature-scaled confidence

## Trade-offs
- **Accuracy vs Latency**: Ensemble adds ~5x latency but +3-5% accuracy over single model
- **Calibration vs Accuracy**: Label smoothing + temperature scaling may slightly reduce raw accuracy but dramatically improve calibration
- **Edge vs Server**: Knowledge distillation retains ~90% of ensemble accuracy at 1/50th the model size
