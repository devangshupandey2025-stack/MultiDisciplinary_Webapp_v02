"""
Export models to ONNX, TorchScript, and TFLite formats.

Usage:
  python ml_pipeline/scripts/export_models.py --checkpoint checkpoints/efficientnet_v2/best_model.pt --format onnx torchscript
  python ml_pipeline/scripts/export_models.py --checkpoint checkpoints/mobilenet_v3/best_model.pt --format tflite --quantize float16
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml_pipeline.models.architectures import create_model


def export_onnx(model, input_size, output_path, num_classes=38, opset=17):
    """Export model to ONNX format."""
    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size)

    torch.onnx.export(
        model, dummy, output_path,
        input_names=['image'],
        output_names=['logits'],
        dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}},
        opset_version=opset,
        do_constant_folding=True,
    )

    # Verify
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX exported: {output_path} ({size_mb:.1f} MB)")
    return output_path


def export_torchscript(model, input_size, output_path):
    """Export model to TorchScript format."""
    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size)
    traced = torch.jit.trace(model, dummy)
    traced.save(output_path)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"TorchScript exported: {output_path} ({size_mb:.1f} MB)")
    return output_path


def export_tflite(model, input_size, output_path, quantize="none"):
    """
    Export model to TFLite via ONNX → TF → TFLite conversion.
    Quantization options: none, float16, int8
    """
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
    except ImportError:
        print("TFLite export requires: pip install onnx-tf tensorflow")
        print("Skipping TFLite export.")
        return None

    # First export to ONNX
    onnx_path = output_path.replace('.tflite', '.onnx')
    export_onnx(model, input_size, onnx_path)

    # ONNX → TF SavedModel
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_path = output_path.replace('.tflite', '_saved_model')
    tf_rep.export_graph(tf_path)

    # TF SavedModel → TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)

    if quantize == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantize == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def representative_dataset():
            for _ in range(100):
                data = np.random.randn(1, input_size, input_size, 3).astype(np.float32)
                yield [data]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"TFLite exported: {output_path} ({size_mb:.1f} MB, quantize={quantize})")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export models")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--format", nargs="+", choices=["onnx", "torchscript", "tflite"],
                        default=["onnx"])
    parser.add_argument("--quantize", type=str, default="none",
                        choices=["none", "float16", "int8"])
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cpu')  # Export on CPU
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = create_model(
        ckpt['model_type'], num_classes=ckpt['num_classes'],
        pretrained=False, dropout=0.0
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    input_size = ckpt.get('config', {}).get('data', {}).get('img_size', 224)
    model_name = ckpt['model_type']

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.checkpoint)
    os.makedirs(args.output_dir, exist_ok=True)

    for fmt in args.format:
        output_path = os.path.join(args.output_dir, f"{model_name}.{fmt}")
        if fmt == "onnx":
            output_path = os.path.join(args.output_dir, f"{model_name}.onnx")
            export_onnx(model, input_size, output_path)
        elif fmt == "torchscript":
            output_path = os.path.join(args.output_dir, f"{model_name}.pt")
            export_torchscript(model, input_size, output_path)
        elif fmt == "tflite":
            output_path = os.path.join(args.output_dir, f"{model_name}.tflite")
            export_tflite(model, input_size, output_path, args.quantize)


if __name__ == "__main__":
    main()
