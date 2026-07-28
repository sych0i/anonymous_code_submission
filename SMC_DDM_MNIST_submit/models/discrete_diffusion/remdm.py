import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .utils.schedulers import NoiseScheduler, DiscreteTimeScheduler, RemaskingScheduler
from .mdm import MaskedDiffusion


class ReMaskingDiffusion(MaskedDiffusion):
    
    def __init__(self, *args, **kwargs):
        self.remasking_scheduler = RemaskingScheduler(
            schedule=kwargs.pop("remasking_schedule"),
            **kwargs.pop("remasking_kwargs", {})
        )
        super(ReMaskingDiffusion, self).__init__(*args, **kwargs)
    
    def sample_step(self, z_t: Tensor, i: int, device=torch.device('cpu')) -> tuple[Tensor, Tensor]:
        B, L, N = z_t.shape
        
        t = self.discrete_time_scheduler.discrete_time(i).to(device)
        s = self.discrete_time_scheduler.discrete_time(i - 1).to(device)
        alpha_t = self.scheduler.alpha(t)
        alpha_s = self.scheduler.alpha(s)
        sigma_t = self.remasking_scheduler.sigma(t, alpha_t, alpha_s)

        # 1. Calculate x_theta
        x_theta = self.denoising_model(
            z_t.reshape(B, *self.input_shape, N), 
            torch.full((B,), t.item(), device=device)
        )
        x_theta = x_theta.reshape(B, L, N)

        # 2. Calculate p_theta
        m = F.one_hot(torch.tensor(self.mask_index, device=device), num_classes=self.num_categories)
        m_weight = torch.where(
            (z_t.argmax(dim=-1) != self.mask_index).unsqueeze(dim=-1).expand(B, L, N),
            sigma_t,
            (1 - alpha_s - sigma_t * alpha_t) / (1 - alpha_t)
        )
        p_theta = (1 - m_weight) * x_theta + m_weight * m

        tol = 1e-7
        # Sanity check - probabilities should be non-negative
        assert torch.all(p_theta >= 0 - tol), f"Min = {p_theta.min()}, {t}"
        p_theta = p_theta.clamp(min=0)
        assert torch.all(x_theta >= 0), f"Min = {x_theta.min()}, {t}"
        # Sanity check - probabilities should sum up to 1
        assert torch.allclose(p_theta.sum(dim=-1), torch.ones(B, L, device=device)), f"Max = {torch.abs(p_theta.sum(dim=-1) - torch.ones(B, L, device=device)).max()}"
        assert torch.allclose(x_theta.sum(dim=-1), torch.ones(B, L, device=device)), f"Max = {torch.abs(x_theta.sum(dim=-1) - torch.ones(B, L, device=device)).max()}"
        return p_theta, x_theta
