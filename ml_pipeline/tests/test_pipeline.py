"""Unit tests for the ML pipeline."""
import os
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml_pipeline.models.architectures import create_model, MODEL_REGISTRY
from ml_pipeline.data.dataset import (
    PlantDiseaseDataset, get_train_transforms, get_val_transforms,
    MixUpCutMix, PLANTVILLAGE_CLASSES
)
from ml_pipeline.scripts.calibrate import TemperatureScaling, compute_ece
from ml_pipeline.scripts.generate_synthetic_data import generate_synthetic_dataset


@pytest.fixture(scope="module")
def synthetic_data_dir():
    """Generate a tiny synthetic dataset for testing."""
    tmpdir = tempfile.mkdtemp(prefix="plantguard_test_")
    generate_synthetic_dataset(tmpdir, num_classes=3, num_images_per_class=5, img_size=32)
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestModels:
    """Test model creation and forward pass."""

    @pytest.mark.parametrize("model_type", list(MODEL_REGISTRY.keys()))
    def test_model_creation(self, model_type):
        model = create_model(model_type, num_classes=10, pretrained=False, dropout=0.1)
        assert model is not None
        params = sum(p.numel() for p in model.parameters())
        assert params > 0

    def test_model_forward(self):
        model = create_model("mobilenet_v3", num_classes=5, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 5)

    def test_model_features(self):
        model = create_model("resnet50", num_classes=5, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        features = model.get_features(x)
        assert features.dim() == 2
        assert features.shape[0] == 2


class TestDataPipeline:
    """Test data loading and augmentation."""

    def test_transforms(self):
        for level in ["light", "medium", "heavy"]:
            t = get_train_transforms(img_size=64, augment_level=level)
            assert t is not None

    def test_val_transforms(self):
        t = get_val_transforms(img_size=64)
        assert t is not None

    def test_dataset(self, synthetic_data_dir):
        t = get_val_transforms(32)
        ds = PlantDiseaseDataset(os.path.join(synthetic_data_dir, "train"), transform=t)
        assert len(ds) > 0
        img, label = ds[0]
        assert img.shape[0] == 3
        assert isinstance(label, int)

    def test_class_weights(self, synthetic_data_dir):
        t = get_val_transforms(32)
        ds = PlantDiseaseDataset(os.path.join(synthetic_data_dir, "train"), transform=t)
        weights = ds.get_class_weights()
        assert len(weights) == len(ds.classes)
        assert (weights > 0).all()

    def test_mixup(self):
        mixer = MixUpCutMix(mixup_alpha=0.2, cutmix_prob=0.0, num_classes=5)
        images = torch.randn(4, 3, 32, 32)
        targets = torch.tensor([0, 1, 2, 3])
        mixed_img, mixed_targets = mixer(images, targets)
        assert mixed_img.shape == images.shape
        assert mixed_targets.shape == (4, 5)

    def test_cutmix(self):
        mixer = MixUpCutMix(mixup_alpha=0.2, cutmix_prob=1.0, num_classes=5)
        images = torch.randn(4, 3, 32, 32)
        targets = torch.tensor([0, 1, 2, 3])
        mixed_img, mixed_targets = mixer(images, targets)
        assert mixed_img.shape == images.shape


class TestCalibration:
    """Test temperature scaling and ECE."""

    def test_temperature_scaling(self):
        ts = TemperatureScaling()
        logits = np.random.randn(100, 5).astype(np.float32)
        labels = np.random.randint(0, 5, 100)
        temp = ts.fit(logits, labels)
        assert temp > 0

    def test_calibrate(self):
        ts = TemperatureScaling()
        logits = np.random.randn(50, 5).astype(np.float32)
        labels = np.random.randint(0, 5, 50)
        ts.fit(logits, labels)
        probs = ts.calibrate(logits)
        assert probs.shape == logits.shape
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_ece(self):
        probs = np.random.dirichlet(np.ones(5), size=100)
        labels = np.random.randint(0, 5, 100)
        ece = compute_ece(probs, labels)
        assert 0 <= ece <= 1

    def test_class_names(self):
        assert len(PLANTVILLAGE_CLASSES) == 38


class TestSyntheticData:
    """Test synthetic data generation."""

    def test_generate(self, synthetic_data_dir):
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(synthetic_data_dir, split)
            assert os.path.exists(split_dir)
            classes = os.listdir(split_dir)
            assert len(classes) == 3
