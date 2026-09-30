"""Supplied PA2 data, baseline, visualization, and experiment helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset, TensorDataset


SEED = 16720
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class RunConfig:
    name: str
    train_per_class: int
    val_per_class: int
    test_per_class: int | None
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float


SMOKE_CONFIG = RunConfig("smoke", 32, 16, 16, 64, 1, 3e-3, 1e-4)
STANDARD_CONFIG = RunConfig("standard", 2500, 500, None, 128, 25, 3e-3, 1e-4)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def balanced_indices(
    targets: Sequence[int], per_class: int, seed: int = SEED, offset: int = 0
) -> np.ndarray:
    labels = np.asarray(targets, dtype=np.int64)
    if labels.ndim != 1 or per_class <= 0 or offset < 0:
        raise ValueError("invalid targets, per_class, or offset")
    chosen: list[np.ndarray] = []
    for class_id in sorted(np.unique(labels).tolist()):
        candidates = np.flatnonzero(labels == class_id)
        rng = np.random.default_rng(seed + int(class_id))
        candidates = rng.permutation(candidates)
        end = offset + per_class
        if end > len(candidates):
            raise ValueError(f"class {class_id} has too few examples")
        chosen.append(candidates[offset:end])
    return np.sort(np.concatenate(chosen)).astype(np.int64)


def _config(mode: str) -> RunConfig:
    if mode == "smoke":
        return SMOKE_CONFIG
    if mode == "standard":
        return STANDARD_CONFIG
    raise ValueError("mode must be 'smoke' or 'standard'")


def build_cifar10_loaders(
    root: str | Path,
    mode: str = "smoke",
    *,
    download: bool = True,
    workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build deterministic balanced CIFAR-10 loaders.

    The same official training examples receive training and validation transforms
    through separate dataset objects. The official test set is kept whole in
    standard mode and balanced-subsampled in smoke mode.
    """

    from torchvision import datasets, transforms

    cfg = _config(mode)
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])
    root = Path(root)
    train_base = datasets.CIFAR10(root, train=True, download=download, transform=train_transform)
    val_base = datasets.CIFAR10(root, train=True, download=download, transform=eval_transform)
    test_base = datasets.CIFAR10(root, train=False, download=download, transform=eval_transform)
    train_indices = balanced_indices(train_base.targets, cfg.train_per_class, SEED, 0)
    val_indices = balanced_indices(
        train_base.targets, cfg.val_per_class, SEED, cfg.train_per_class
    )
    if cfg.test_per_class is None:
        test_indices = np.arange(len(test_base), dtype=np.int64)
    else:
        test_indices = balanced_indices(test_base.targets, cfg.test_per_class, SEED, 0)
    generator = torch.Generator().manual_seed(SEED)
    common = dict(
        batch_size=cfg.batch_size,
        num_workers=workers,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=workers > 0,
    )
    train_loader = DataLoader(Subset(train_base, train_indices.tolist()), shuffle=True, **common)
    val_loader = DataLoader(Subset(val_base, val_indices.tolist()), shuffle=False, **common)
    test_loader = DataLoader(Subset(test_base, test_indices.tolist()), shuffle=False, **common)
    return train_loader, val_loader, test_loader, list(train_base.classes)


def make_offline_smoke_loaders(
    *, num_classes: int = 10, train_size: int = 64, test_size: int = 32
) -> tuple[DataLoader, DataLoader]:
    """Create deterministic structured tensors for environment and API smoke tests."""

    generator = torch.Generator().manual_seed(SEED)

    def dataset(size: int) -> TensorDataset:
        labels = torch.arange(size) % num_classes
        images = 0.08 * torch.randn(size, 3, 32, 32, generator=generator)
        for index, label in enumerate(labels.tolist()):
            channel = label % 3
            row = 2 + (label * 3) % 24
            images[index, channel, row : row + 6, 4:28] += 1.0
        return TensorDataset(images, labels)

    return (
        DataLoader(dataset(train_size), batch_size=16, shuffle=False),
        DataLoader(dataset(test_size), batch_size=16, shuffle=False),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def per_class_recall(confusion: Tensor) -> Tensor:
    matrix = torch.as_tensor(confusion, dtype=torch.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion must be square")
    totals = matrix.sum(dim=1)
    return torch.where(totals > 0, matrix.diag() / totals, torch.zeros_like(totals))


def raw_pixel_embeddings(images: Tensor) -> Tensor:
    return F.normalize(images.flatten(1).to(torch.float32), dim=1, eps=1e-12)


def hog_embeddings(images: np.ndarray) -> np.ndarray:
    """Compute the supplied HOG baseline on an NHWC float image batch."""

    from skimage.color import rgb2gray
    from skimage.feature import hog

    array = np.asarray(images)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError("images must have shape (N,H,W,3)")
    features = []
    for image in array:
        vector = hog(
            rgb2gray(image),
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        vector = vector.astype(np.float32)
        vector /= np.linalg.norm(vector) + 1e-12
        features.append(vector)
    return np.stack(features)


def cosine_prototypes(
    features: Tensor, labels: Tensor, num_classes: int
) -> tuple[Tensor, Tensor]:
    features = F.normalize(torch.as_tensor(features, dtype=torch.float32), dim=1)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    prototypes = []
    for class_id in range(num_classes):
        selected = features[labels == class_id]
        if selected.numel() == 0:
            raise ValueError(f"class {class_id} has no training features")
        prototypes.append(F.normalize(selected.mean(0), dim=0))
    return torch.stack(prototypes), torch.arange(num_classes)


def prototype_predict(features: Tensor, prototypes: Tensor) -> tuple[Tensor, Tensor]:
    features = F.normalize(torch.as_tensor(features, dtype=torch.float32), dim=1)
    prototypes = F.normalize(torch.as_tensor(prototypes, dtype=torch.float32), dim=1)
    similarities = features @ prototypes.T
    score, prediction = similarities.max(dim=1)
    return prediction, score


class PlainBlock(nn.Module):
    """Two-convolution supplied ablation with no skip connection."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TinyPlainNet(nn.Module):
    """Depth/width-matched no-skip comparison supplied for Part C3."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
        )
        self.body = nn.Sequential(
            PlainBlock(32, 32),
            PlainBlock(32, 32),
            PlainBlock(32, 64, 2),
            PlainBlock(64, 64),
            PlainBlock(64, 128, 2),
            PlainBlock(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward_features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        feature_map = self.body(self.stem(x))
        return feature_map, self.pool(feature_map).flatten(1)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.forward_features(x)[1])


def classifier_weight_cam(model: nn.Module, image: Tensor, class_index: int | None = None) -> Tensor:
    """Return normalized classifier-weight CAM as non-causal diagnostic evidence."""

    model.eval()
    with torch.no_grad():
        feature_map, embedding = model.forward_features(image)
        logits = model.classifier(embedding)
        if class_index is None:
            class_index = int(logits.argmax(dim=1)[0])
        weights = model.classifier.weight[class_index]
        cam = torch.einsum("c,bchw->bhw", weights, feature_map).relu()
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        return cam.cpu()


def apply_perturbation(
    image: np.ndarray, kind: str, severity: float, *, seed: int = SEED
) -> np.ndarray:
    """Apply a deterministic pre-normalization nuisance transform to an RGB float image."""

    import cv2

    value = np.asarray(image, dtype=np.float32)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("image must be HxWx3")
    if kind == "brightness":
        result = value * severity
    elif kind == "blur":
        result = cv2.GaussianBlur(value, (0, 0), sigmaX=severity) if severity > 0 else value.copy()
    elif kind == "noise":
        result = value + np.random.default_rng(seed).normal(0, severity, value.shape)
    elif kind == "occlusion":
        result = value.copy()
        side_fraction = float(np.sqrt(np.clip(severity, 0, 1)))
        height, width = value.shape[:2]
        half_h = round(height * side_fraction / 2)
        half_w = round(width * side_fraction / 2)
        cy, cx = height // 2, width // 2
        result[cy - half_h : cy + half_h, cx - half_w : cx + half_w] = 0.0
    elif kind == "translation":
        matrix = np.float32([[1, 0, severity], [0, 1, severity]])
        result = cv2.warpAffine(value, matrix, (value.shape[1], value.shape[0]))
    elif kind == "grayscale":
        gray = cv2.cvtColor(value, cv2.COLOR_RGB2GRAY)[..., None]
        result = (1 - severity) * value + severity * np.repeat(gray, 3, axis=2)
    else:
        raise ValueError(f"unsupported perturbation: {kind}")
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def build_pretrained_extractor(name: str = "resnet18") -> tuple[nn.Module, int]:
    """Return an approved frozen torchvision feature extractor and output size."""

    from torchvision import models

    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        dimension = model.fc.in_features
        model.fc = nn.Identity()
    elif name == "swin_t":
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        dimension = model.head.in_features
        model.head = nn.Identity()
    else:
        raise ValueError("approved names are 'resnet18' and 'swin_t'")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, dimension


def plot_training_curves(history: dict[str, Sequence[float]]):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(history["train_loss"], marker="o", label="train")
    axes[0].plot(history["val_loss"], marker="o", label="validation")
    axes[0].set(title="Loss", xlabel="epoch")
    axes[1].plot(history["train_accuracy"], marker="o", label="train")
    axes[1].plot(history["val_accuracy"], marker="o", label="validation")
    axes[1].set(title="Accuracy", xlabel="epoch", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    return figure, axes


def plot_confusion_matrix(matrix: Tensor, class_names: Iterable[str]):
    import matplotlib.pyplot as plt

    counts = torch.as_tensor(matrix, dtype=torch.float32)
    normalized = counts / counts.sum(1, keepdim=True).clamp_min(1)
    names = list(class_names)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(normalized.numpy(), vmin=0, vmax=1, cmap="Blues")
    axis.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names)
    axis.set(xlabel="predicted", ylabel="true", title="Row-normalized confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    return figure, axis
