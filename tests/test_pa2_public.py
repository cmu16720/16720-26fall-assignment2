from __future__ import annotations

import importlib
import os

import pytest
import torch
from torch import nn


@pytest.fixture(scope="session")
def impl():
    """Load student code by default; staff CI redirects the neutral hook."""

    module_name = os.environ.get("PA2_MODULE", "student.pa2")
    return importlib.import_module(module_name)


def test_conv_norm_act_shape_and_order(impl) -> None:
    layer = impl.ConvNormAct(3, 7, stride=2)
    assert list(layer.children()) == [layer.conv, layer.norm, layer.act]
    assert layer.conv.bias is None
    assert layer(torch.randn(2, 3, 15, 17)).shape == (2, 7, 8, 9)


def test_residual_skip_contract(impl) -> None:
    identity = impl.ResidualBlock(16, 16, 1)
    projection = impl.ResidualBlock(16, 32, 2)
    assert isinstance(identity.skip, nn.Identity)
    assert not isinstance(projection.skip, nn.Identity)
    assert identity(torch.randn(3, 16, 8, 8)).shape == (3, 16, 8, 8)
    assert projection(torch.randn(3, 16, 8, 8)).shape == (3, 32, 4, 4)


def test_tiny_resnet_shapes_and_parameter_count(impl) -> None:
    model = impl.TinyResNet(10)
    feature_map, embedding = model.forward_features(torch.randn(3, 3, 32, 32))
    assert feature_map.shape == (3, 128, 8, 8)
    assert embedding.shape == (3, 128)
    assert model(torch.randn(3, 3, 32, 32)).shape == (3, 10)
    assert sum(parameter.numel() for parameter in model.parameters()) == 696_618


def test_confusion_orientation_and_absent_class(impl) -> None:
    labels = torch.tensor([0, 0, 1, 1])
    predictions = torch.tensor([0, 1, 1, 2])
    matrix = impl.compute_confusion_matrix(labels, predictions, 4)
    expected = torch.tensor(
        [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    )
    assert matrix.dtype == torch.int64
    assert torch.equal(matrix, expected)


def test_nearest_neighbors_self_exclusion_and_ties(impl) -> None:
    gallery = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    indices, scores = impl.nearest_neighbors(
        gallery[0], gallery, 2, exclude_index=0
    )
    assert indices.tolist() == [1, 2]
    assert torch.allclose(scores, torch.tensor([1.0, 0.0]))
