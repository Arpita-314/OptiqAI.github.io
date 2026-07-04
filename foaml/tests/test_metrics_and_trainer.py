import torch

from models.architecture import FNO2d
from training.trainer import OpticsTrainer
from utils.metrics import FourierOpticsMetrics


# --- metrics -------------------------------------------------------------- #
def test_strehl_of_flat_wavefront_is_one():
    calc = FourierOpticsMetrics(633e-9, 5e-6)
    flat = torch.ones(128, 128, dtype=torch.complex64)
    assert abs(calc.strehl_ratio(flat) - 1.0) < 1e-4


def test_strehl_drops_with_aberration():
    calc = FourierOpticsMetrics(633e-9, 5e-6)
    torch.manual_seed(0)
    aberrated = torch.exp(1j * 1.0 * torch.randn(128, 128))
    s = calc.strehl_ratio(aberrated)
    assert 0.0 <= s < 0.9


def test_marechal_matches_definition_for_small_aberration():
    calc = FourierOpticsMetrics(633e-9, 5e-6)
    torch.manual_seed(1)
    small = torch.exp(1j * 0.2 * torch.randn(256, 256))
    # for small sigma, exact Strehl and Maréchal estimate should be close
    assert abs(calc.strehl_ratio(small) - calc.strehl_marechal(small)) < 0.05


# --- trainer losses ------------------------------------------------------- #
def test_energy_loss_zero_when_power_matches():
    tr = OpticsTrainer(torch.nn.Identity())
    field = torch.randn(4, 2, 16, 16)
    assert tr.energy_conservation_loss(field, field).item() < 1e-8


def test_complex_loss_zero_when_equal_and_positive_when_phase_differs():
    tr = OpticsTrainer(torch.nn.Identity())
    a = torch.randn(2, 2, 16, 16)
    assert tr.complex_loss(a, a).item() < 1e-6
    # rotate phase by 90 deg: swap channels with sign -> same amplitude, diff phase
    b = torch.stack([-a[:, 1], a[:, 0]], dim=1)
    assert tr.complex_loss(a, b).item() > 0.1


def test_training_reduces_loss():
    """A real FNO should reduce loss on a fixed identity-mapping toy task."""
    torch.manual_seed(0)
    model = FNO2d(in_channels=2, out_channels=2, modes1=6, modes2=6, width=12)
    tr = OpticsTrainer(model)
    x = torch.randn(16, 2, 32, 32)
    loader = tr.create_dataloader(x, x.clone(), batch_size=4)  # learn the identity
    start, _ = tr.validate(loader)
    tr.manual_train(loader, loader, {"epochs": 8, "patience": 8, "lambda_physics": 0.0, "lr": 3e-3})
    end, _ = tr.validate(loader)
    assert end < start
