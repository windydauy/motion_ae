"""iFSQ 模块测试。"""
import torch
import pytest

from motion_ae.models.ifsq import iFSQ


@pytest.fixture
def ifsq():
    return iFSQ(levels=[5, 7, 3, 9])


def test_output_shapes(ifsq):
    B, D = 16, 4
    z_c = torch.randn(B, D)
    z_dequant, z_d, info = ifsq(z_c)

    assert z_dequant.shape == (B, D)
    assert z_d.shape == (B, D)
    assert info["z_c"].shape == (B, D)


def test_quantized_values_in_range(ifsq):
    """z_d 应该是 [0, L-1] 范围内的整数。"""
    z_c = torch.randn(64, 4) * 5.0
    _z_dequant, z_d, _info = ifsq(z_c)

    for i, L in enumerate(ifsq.levels):
        col = z_d[:, i]
        assert (col >= 0).all(), f"dim {i}: z_d has values < 0"
        assert (col <= L - 1).all(), f"dim {i}: z_d has values > L-1"
        # 应该是整数
        assert torch.allclose(col, col.round()), f"dim {i}: z_d not integers"


def test_dequantized_in_range(ifsq):
    """z_dequant 应该在 [-1, 1] 范围内。"""
    z_c = torch.randn(64, 4) * 10.0
    z_dequant, _z_d, _info = ifsq(z_c)

    assert (z_dequant >= -1.0 - 1e-6).all()
    assert (z_dequant <= 1.0 + 1e-6).all()


def test_ste_gradient_flows(ifsq):
    """STE 不应让梯度断掉：z_c.grad 应该非零。"""
    z_c = torch.randn(8, 4, requires_grad=True)
    z_dequant, _z_d, _info = ifsq(z_c)

    loss = z_dequant.sum()
    loss.backward()

    assert z_c.grad is not None, "Gradient is None — STE broken"
    assert (z_c.grad != 0).any(), "Gradient is all zeros — STE not passing through"


def test_codebook_size(ifsq):
    assert ifsq.codebook_size == 5 * 7 * 3 * 9


def test_uniform_levels():
    """所有维度相同级别数的常见配置。"""
    q = iFSQ(levels=[8, 8, 8, 8, 8, 8, 8, 8])
    z_c = torch.randn(4, 8)
    z_dequant, z_d, _ = q(z_c)
    assert z_dequant.shape == (4, 8)
    assert q.codebook_size == 8 ** 8
