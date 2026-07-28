import torch
from torch import Tensor

class NoiseScheduler():
    def __init__(self, schedule: str):
        assert schedule in ['linear', 'cosine']
        self.schedule = schedule
        
    def alpha(self, t: Tensor) -> Tensor:
        """
        Calculates alpha(t)
        """
        if t.dim() == 0:
            assert t >= 0 and t <= 1
        else:
            assert torch.all(t >= 0) and torch.all(t <= 1)
        
        if self.schedule == 'linear':
            return 1 - t
        elif self.schedule == 'cosine':
            return 1 - torch.cos((torch.pi/2) * (1 - t))
        else:
            raise NotImplementedError
        
    def alpha_dash(self, t: Tensor) -> Tensor:
        """
        Calculates alpha'(t)
        """
        if t.dim() == 0:
            assert t >= 0 and t <= 1
        else:
            assert torch.all(t >= 0) and torch.all(t <= 1)

        if self.schedule == 'linear':
            return -1
        elif self.schedule == 'cosine':
            return -(torch.pi/2) * torch.sin((torch.pi/2) * (1 - t))
        else:
            raise NotImplementedError


class RemaskingScheduler():
    def __init__(self, schedule: str, **kwargs):
        assert schedule in ['max_capped', 'rescaled']
        self.schedule = schedule
        if schedule == 'max_capped':
            self.eta_cap = kwargs['eta_cap']
        elif schedule == 'rescaled':
            self.eta_rescale = kwargs['eta_rescale']
        self.t_on = kwargs.pop('t_on', 1.0)
        self.t_off = kwargs.pop('t_off', 0.0)
            
    def sigma_max(self, alpha_t: Tensor, alpha_s: Tensor) -> Tensor:
        """
        Calculates the maximum value of sigma
        """
        return torch.min(
            torch.ones_like(alpha_t),
            (1 - alpha_s) / alpha_t,
        )
        
    def sigma(self, t: Tensor, alpha_t: Tensor, alpha_s: Tensor) -> Tensor:
        """
        Calculates sigma(t)
        """
        if self.schedule == 'max_capped':
            sigma = torch.min(
                torch.zeros_like(alpha_t) + self.eta_cap,
                self.sigma_max(alpha_t, alpha_s)
            )
        elif self.schedule == 'rescaled':
            sigma = self.eta_rescale * self.sigma_max(alpha_t, alpha_s)
        else:
            raise NotImplementedError
        sigma = torch.where(
            (self.t_on >= t) & (t >= self.t_off),
            sigma,
            torch.zeros_like(sigma)
        )
        return sigma


class DiscreteTimeScheduler():
    def __init__(self, schedule: str, num_timesteps: int):
        assert schedule in ['linear', 'cosine']
        self.schedule = schedule
        self.num_timesteps = num_timesteps

    def discrete_time(self, i):
        assert i >= 0 and i <= self.num_timesteps
        if i == 0:
            return torch.tensor(0)
        if self.schedule == 'linear':
            return torch.tensor(i / self.num_timesteps) 
        elif self.schedule == 'cosine':
            return torch.cos(
                (torch.pi / 2) * (1 - torch.tensor(i / self.num_timesteps))
            )
        else:
            raise ValueError(
                f"Invalid discretization schedule: {self.schedule}"
            )
