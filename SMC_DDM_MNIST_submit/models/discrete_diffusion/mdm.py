import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .utils.schedulers import NoiseScheduler, DiscreteTimeScheduler


class MaskedDiffusion(nn.Module):
    EPSILON = 1e-8
    
    def __init__(
        self, 
        denoising_model: nn.Module, 
        input_shape: tuple,
        num_categories: int,
        mask_index: int,
        masking_schedule:str = "linear",
        num_timesteps:int = 1000,
        discretization_schedule:str = "cosine"
    ):
        super(MaskedDiffusion, self).__init__()
        self.denoising_model = denoising_model
        self.input_shape = input_shape
        self.num_categories = num_categories
        self.mask_index = mask_index
        self.scheduler = NoiseScheduler(masking_schedule)
        self.num_timesteps = num_timesteps
        self.discrete_time_scheduler = DiscreteTimeScheduler(discretization_schedule, num_timesteps)
     
        
    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        N = self.num_categories
        x = x.reshape(B, -1)
        L = x.shape[-1]
        
        u = torch.rand(1, device=x.device)
        t = torch.remainder(u + torch.arange(B, device=x.device)/B, 1) # Shape: (B)
        
        alpha = self.scheduler.alpha(t) # Shape: (B)
        weight = self.scheduler.alpha_dash(t) / (1 - alpha).clamp(min=self.EPSILON) # Shape: (B)
        
        # 1. Sample z_t from q(z_t | x)
        x = F.one_hot(x, num_classes=N) # Shape: (B, L, N)
        m = F.one_hot(torch.full((B, L), self.mask_index, device=x.device), num_classes=N) # Shape: (B, L, N)
        q = torch.distributions.Categorical(
            probs=alpha.view(B, 1, 1) * x  + (1 - alpha).view(B, 1, 1) * m
        )
        z_t = q.sample() # Shape: (B, L)
        z_t = F.one_hot(z_t, num_classes=N).float() # Shape: (B, L, N)
        
        # 2. Calculate x_theta
        x_theta = self.denoising_model(z_t.reshape(B, *self.input_shape, N), t).reshape(B, L, N)
        
        # 3. Calculate loss terms for each token
        loss_terms = torch.log(
            (x_theta * x).sum(dim=-1)
        ) # Shape: (B, L)
        
        # 4. Calculate loss
        loss = weight * loss_terms.sum(dim=-1)
        return loss.mean(dim=0)
     
    def sample(self, num_samples: int = 1, device=torch.device('cpu')) -> Tensor:
        N = self.num_categories
        z_t = torch.full((num_samples, *self.input_shape), self.mask_index, device=device)
        B = z_t.shape[0]
        z_t = z_t.reshape(B, -1) # Shape: (B, L)
        
        for i in range(self.num_timesteps, 0, -1):
            p_theta, _ = self.sample_step(F.one_hot(z_t, num_classes=N).float(), i, device)
            
            # 3. Sample z_s from p_theta
            z_s = torch.distributions.Categorical(probs=p_theta).sample() # Shape: (B, L)
            
            z_t = z_s
        
        x = z_t.reshape(B, *self.input_shape)
        # Sanity check - the final sample should not have any masked tokens
        assert torch.all(x != self.mask_index)
        return x
    
    def sample_step(self, z_t: Tensor, i: int, device=torch.device('cpu')) -> tuple[Tensor, Tensor]:
        B, L, N = z_t.shape
        
        t = self.discrete_time_scheduler.discrete_time(i).to(device)
        s = self.discrete_time_scheduler.discrete_time(i - 1).to(device)
        alpha_t = self.scheduler.alpha(t)
        alpha_s = self.scheduler.alpha(s)

        # 1. Calculate x_theta
        x_theta = self.denoising_model(
            z_t.reshape(B, *self.input_shape, N), 
            torch.full((B,), t.item(), device=device)
        )
        x_theta = x_theta.reshape(B, L, N)

        # 2. Calculate p_theta
        p_theta = ((1 - alpha_s) * z_t + (alpha_s - alpha_t) * x_theta) /  (1 - alpha_t)

        # Sanity check - probabilities should sum up to 1
        assert torch.allclose(p_theta.sum(dim=-1), torch.ones(B, L, device=device)), f"Max = {torch.abs(p_theta.sum(dim=-1) - torch.ones(B, L, device=device)).max()}"
        assert torch.allclose(x_theta.sum(dim=-1), torch.ones(B, L, device=device)), f"Max = {torch.abs(x_theta.sum(dim=-1) - torch.ones(B, L, device=device)).max()}"
        # Sanity check - probabilities should be non-negative
        assert torch.all(p_theta >= 0), f"Min = {p_theta.min()}, {t}"
        assert torch.all(x_theta >= 0), f"Min = {x_theta.min()}, {t}"
        return p_theta, x_theta
