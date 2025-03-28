from __future__ import annotations

import pytest
import torch
from unipercept.nn.losses.guided import DGPLoss, PGTLoss


@pytest.fixture
def pgt_loss():
    return PGTLoss()


def test_pgt_loss(pgt_loss):
    dep_feat = torch.randn(4, 16, 32, 16)
    seg_true = torch.randint(-1, 10, (4, 32, 16))

    loss = pgt_loss(dep_feat, seg_true)

    print("PGT loss: ", loss.item())

    assert loss >= 0, loss.item()
    assert loss.isfinite(), loss.item()


@pytest.fixture
def dg_loss():
    return DGPLoss()


def test_all_zero_inputs(dg_loss):
    semantic_features = torch.zeros([1, 3, 5, 5])
    depth_gt = torch.zeros([1, 1, 5, 5])
    loss = dg_loss(semantic_features, depth_gt)
    assert torch.isclose(loss, torch.tensor(0.0)), loss.item()


def test_depth_only_variation(dg_loss):
    semantic_features = torch.zeros([1, 3, 5, 5])
    depth_gt = torch.linspace(1, 25, 25).view(1, 1, 5, 5)
    loss = dg_loss(semantic_features, depth_gt)
    assert loss < 0 or torch.isclose(loss, torch.tensor(0.0)), loss.item()


def test_semantic_only_variation(dg_loss):
    semantic_features = torch.linspace(1, 75, 75).view(1, 3, 5, 5)
    depth_gt = torch.zeros([1, 1, 5, 5])
    loss = dg_loss(semantic_features, depth_gt)
    assert loss < 0 or torch.isclose(loss, torch.tensor(0.0)), loss.item()


def test_depth_and_semantic_variation(dg_loss):
    semantic_features = torch.linspace(1, 75, 75).view(1, 3, 5, 5)
    depth_gt = torch.linspace(1, 25, 25).view(1, 1, 5, 5)
    loss = dg_loss(semantic_features, depth_gt)
    assert loss < 0 or torch.isclose(loss, torch.tensor(0.0)), loss.item()


def test_batch_consistency(dg_loss):
    semantic_features = torch.zeros([2, 3, 5, 5])
    depth_gt = torch.zeros([2, 1, 5, 5])
    loss1 = dg_loss(semantic_features[0:1], depth_gt[0:1])
    loss2 = dg_loss(semantic_features[1:2], depth_gt[1:2])
    loss_batch = dg_loss(semantic_features, depth_gt)
    assert torch.isclose(loss_batch, (loss1 + loss2) / 2)
