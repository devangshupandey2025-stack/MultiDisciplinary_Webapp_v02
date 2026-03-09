"""
Generate synthetic dataset for CI/CD testing.
Creates small random images organized in PlantVillage format.

Usage:
  python ml_pipeline/scripts/generate_synthetic_data.py --output_dir data/synthetic --num_classes 5 --num_images 50
"""
import os
import argparse
import random
from PIL import Image
import numpy as np


SAMPLE_CLASSES = [
    "Tomato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Apple___healthy",
    "Apple___Apple_scab",
    "Corn_(maize)___healthy",
    "Corn_(maize)___Common_rust_",
    "Grape___healthy",
    "Grape___Black_rot",
    "Potato___healthy",
]


def generate_synthetic_dataset(output_dir: str, num_classes: int = 5,
                               num_images_per_class: int = 10, img_size: int = 64):
    """Generate small synthetic images for testing."""
    classes = SAMPLE_CLASSES[:num_classes]

    for split in ["train", "val", "test"]:
        split_count = {
            "train": num_images_per_class,
            "val": max(3, num_images_per_class // 4),
            "test": max(3, num_images_per_class // 4),
        }[split]

        for cls in classes:
            cls_dir = os.path.join(output_dir, split, cls)
            os.makedirs(cls_dir, exist_ok=True)

            # Generate color bias based on class (so model can learn)
            random.seed(hash(cls) % 2**32)
            base_color = [random.randint(50, 200) for _ in range(3)]

            for i in range(split_count):
                # Create an image with class-correlated color + noise
                img_array = np.random.randint(0, 50, (img_size, img_size, 3), dtype=np.uint8)
                for c in range(3):
                    img_array[:, :, c] = np.clip(
                        img_array[:, :, c].astype(np.int16) + base_color[c] + random.randint(-20, 20),
                        0, 255
                    ).astype(np.uint8)

                # Add some shapes for structure
                cx, cy = random.randint(10, img_size - 10), random.randint(10, img_size - 10)
                r = random.randint(5, 15)
                y, x = np.ogrid[-cy:img_size - cy, -cx:img_size - cx]
                mask = x * x + y * y <= r * r
                img_array[mask] = [random.randint(100, 255) for _ in range(3)]

                img = Image.fromarray(img_array.astype(np.uint8))
                img.save(os.path.join(cls_dir, f"synth_{i:04d}.png"))

    total = sum(
        len(os.listdir(os.path.join(output_dir, split, cls)))
        for split in ["train", "val", "test"]
        for cls in classes
        if os.path.exists(os.path.join(output_dir, split, cls))
    )
    print(f"Generated {total} synthetic images in {output_dir}")
    print(f"Classes: {classes}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic test dataset")
    parser.add_argument("--output_dir", type=str, default="data/synthetic")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--num_images", type=int, default=10,
                        help="Images per class for training split")
    parser.add_argument("--img_size", type=int, default=64)
    args = parser.parse_args()

    generate_synthetic_dataset(args.output_dir, args.num_classes,
                               args.num_images, args.img_size)


if __name__ == "__main__":
    main()
