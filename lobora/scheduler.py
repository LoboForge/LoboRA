"""H3 rectified-flow conventions — copied from DiffSynth FlowMatchScheduler.

Do NOT use LensTrainer's ``noise - latents`` blindly without checking the
scheduler: DiffSynth's target is also ``noise - sample`` (= ε − x₀) with
``x_t = (1−σ) x₀ + σ ε`` and DiT timestep ``t = 1 − σ``.
Training shift is 2.22 / 2.22 (inference uses 12 / 3 — intentional).
"""

from __future__ import annotations

import torch


class H3FlowMatch:
    def __init__(
        self,
        *,
        num_train_timesteps: int = 1000,
        shift: float = 2.22,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.set_timesteps(num_train_timesteps, training=True)

    def set_timesteps(self, steps: int, *, training: bool = False, shift: float | None = None) -> None:
        shift = self.shift if shift is None else shift
        base = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)[:-1]
        self.sigmas = shift * base / (1 + (shift - 1) * base)
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.sigmas * self.num_train_timesteps
            y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
            y_shifted = y - y.min()
            self.linear_timesteps_weights = y_shifted * (steps / y_shifted.sum())
        else:
            self.linear_timesteps_weights = torch.ones(steps)

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        sigma = self._sigma_for(timestep)
        while sigma.ndim < original_samples.ndim:
            sigma = sigma.unsqueeze(-1)
        return (1 - sigma) * original_samples + sigma * noise

    def training_target(self, sample: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """ε − x₀  (DiffSynth FlowMatchScheduler.training_target)."""
        return noise - sample

    def dit_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        """t = 1 − σ, with σ = timestep / 1000."""
        sigma = timestep.float() / float(self.num_train_timesteps)
        return 1.0 - sigma

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        idx = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        return self.linear_timesteps_weights[idx]

    def sample_timestep(self, generator: torch.Generator | None = None) -> torch.Tensor:
        idx = torch.randint(0, len(self.timesteps), (1,), generator=generator)
        return self.timesteps[idx]

    def _sigma_for(self, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.ndim == 0:
            idx = torch.argmin((self.timesteps - timestep.cpu()).abs())
            return self.sigmas[idx].to(dtype=torch.float32)
        sigmas = []
        for t in timestep.reshape(-1).cpu():
            idx = torch.argmin((self.timesteps - t).abs())
            sigmas.append(self.sigmas[idx])
        return torch.stack(sigmas).to(dtype=torch.float32)


def mse_flow_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(pred.float(), target.float())
