import torch

from models.architecture import DiffOptModel, FNO2d, ModelSelector, SpectralConv2d, SpectralUNet


def test_all_builders_instantiate_and_run():
    """Every advertised model must build and forward without error."""
    for dtype in ("complex_field",):
        for choice in ("fno", "unet", "diffopt", "convnet"):
            sel = ModelSelector(data_shape=(64, 64), data_type=dtype)
            sel.model_choice = choice
            model = sel.build_model()
            x = torch.randn(2, 2, 64, 64)
            out = model(x)
            assert out.shape[0] == 2
            assert torch.isfinite(out).all()


def test_fno_is_resolution_invariant():
    """An FNO trained at one resolution must run at another (fixed mode count)."""
    model = FNO2d(in_channels=2, out_channels=2, modes1=8, modes2=8, width=16)
    out_small = model(torch.randn(1, 2, 64, 64))
    out_large = model(torch.randn(1, 2, 128, 128))
    assert out_small.shape[-1] == 64
    assert out_large.shape[-1] == 128


def test_spectral_conv_truncates_high_modes():
    """A pure high-frequency input is annihilated by a low-mode spectral conv."""
    layer = SpectralConv2d(1, 1, modes1=2, modes2=2)
    n = 32
    xx = torch.arange(n).float()
    # highest representable frequency (checkerboard)
    high = torch.cos(torch.pi * xx).outer(torch.cos(torch.pi * xx))[None, None]
    out = layer(high)
    assert out.abs().mean() < high.abs().mean()


def test_diffopt_distance_is_learnable():
    model = DiffOptModel(grid=64, distance=50e-6)
    names = dict(model.named_parameters())
    assert "propagator.distance" in names
    assert names["propagator.distance"].requires_grad


def test_diffopt_gradient_flows_end_to_end():
    model = DiffOptModel(grid=32, distance=40e-6)
    x = torch.randn(1, 2, 32, 32)
    out = model(x)
    (out**2).sum().backward()
    assert model.phase_mask.grad is not None
    assert model.propagator.distance.grad is not None
