import torch

from lobora.scheduler import H3FlowMatch, mse_flow_loss


def test_add_noise_and_target_signs():
    sched = H3FlowMatch(shift=2.22)
    x0 = torch.ones(1, 2, 2, 4, 4)
    noise = torch.zeros_like(x0)
    t = sched.timesteps[0]  # high sigma
    xt = sched.add_noise(x0, noise, t)
    target = sched.training_target(x0, noise)
    # target = ε − x₀ = −1
    assert torch.allclose(target, -x0)
    # x_t = (1−σ)x₀ + σ ε  → closer to noise (0) at high σ
    assert xt.abs().mean() < 1.0


def test_dit_timestep_is_one_minus_sigma():
    sched = H3FlowMatch()
    t = torch.tensor(1000.0)
    assert torch.allclose(sched.dit_timestep(t), torch.tensor(0.0))
    t = torch.tensor(0.0)
    assert torch.allclose(sched.dit_timestep(t), torch.tensor(1.0))


def test_mse_finite():
    pred = torch.randn(2, 2)
    target = torch.randn(2, 2)
    loss = mse_flow_loss(pred, target)
    assert torch.isfinite(loss)
