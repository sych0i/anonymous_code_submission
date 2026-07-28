import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .utils.schedulers import NoiseScheduler, DiscreteTimeScheduler

class UniformDiffusion(nn.Module):
    EPSILON = 1e-8
    
    def __init__(
        self, 
        denoising_model: nn.Module, 
        input_shape: tuple,
        num_categories: int,
        noise_schedule:str = "linear",
        num_timesteps:int = 1000,
        discretization_schedule:str = "cosine"
    ):
        super(UniformDiffusion, self).__init__()
        self.denoising_model = denoising_model
        self.input_shape = input_shape
        self.num_categories = num_categories
        self.scheduler = NoiseScheduler(noise_schedule)
        self.num_timesteps = num_timesteps
        self.discrete_time_scheduler = DiscreteTimeScheduler(discretization_schedule, num_timesteps)
    
    
    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = x.reshape(B, -1)
        L = x.shape[-1]
        N = self.num_categories
        
        u = torch.rand(1, device=x.device)
        t = torch.remainder(u + torch.arange(B, device=x.device)/B, 1) # Shape: (B)
        
        alpha = self.scheduler.alpha(t) # Shape: (B)
        weight = self.scheduler.alpha_dash(t) / alpha.clamp(min=self.EPSILON) # Shape: (B)
    
        # 1. Sample z_t from q(z_t | x)
        x = F.one_hot(x, num_classes=N) # Shape: (B, L, N)
        u = torch.ones_like(x) / N # Shape: (B, L, N)
        q = torch.distributions.Categorical(
            probs=alpha.view(B, 1, 1) * x  + (1 - alpha).view(B, 1, 1) * u
        )
        z_t = q.sample() # Shape: (B, L)
        
        # 2. Calculate x_theta
        x_theta: Tensor = self.denoising_model(
            F.one_hot(z_t.reshape(B, *self.input_shape), num_classes=N).float(), 
            t
        ).reshape(B, L, N)
        
        # 3. Calcualte x_bar and x_theta_bar
        x_bar = N * alpha.view(B, 1, 1) * x + (1 - alpha).view(B, 1, 1) * torch.ones_like(x) # Shape: (B, L, N)
        x_theta_bar = N * alpha.view(B, 1, 1) * x_theta + (1 - alpha).view(B, 1, 1) * torch.ones_like(x_theta) # Shape: (B, L, N)

        # 4. Define i and j indices
        i = F.one_hot(z_t, num_classes=N).bool() # Shape: (B, L, N)
        j = ~i # Shape: (B, L, N)
        
        # 5. Calculate x_bar_i, x_theta_bar_i, x_bar_j, x_theta_bar_j
        x_bar_i = x_bar[i].reshape(B, L, 1) # Shape: (B, L, 1)
        x_theta_bar_i = x_theta_bar[i].reshape(B, L, 1) # Shape: (B, L, 1)
        x_bar_j = x_bar[j].reshape(B, L, N-1) # Shape: (B, L, N-1)
        x_theta_bar_j = x_theta_bar[j].reshape(B, L, N-1) # Shape: (B, L, N-1)
        
        # 6. Calculate loss term 1
        loss_term_1 = N / x_bar_i - N / x_theta_bar_i # Shape: (B, L, 1)
        loss_term_1 = loss_term_1.sum(dim=-1).sum(dim=-1) # Shape: (B)
        
        # 7. Calculate loss term 2
        loss_term_2 = (x_bar_j / x_bar_i) * torch.log(
            (x_bar_j / x_bar_i).clamp(min=self.EPSILON) * (x_theta_bar_i / x_theta_bar_j)
        ) # Shape (B, L, N-1)
        loss_term_2 = - loss_term_2.sum(dim=-1).sum(dim=-1) # Shape: (B)
        
        # 8. Calcualte total loss
        loss = (weight / N) * (loss_term_1 + loss_term_2) # Shape: (B)
        return loss.mean(dim=0)
    
    def sample(self, num_samples: int = 1, device='cpu') -> Tensor:
        N = self.num_categories
        z_t = torch.randint(0, N, (num_samples, *self.input_shape), device=device)
        B = z_t.shape[0]
        z_t = z_t.reshape(B, -1) # Shape: (B, L)
        L = z_t.shape[-1]
        
        for i in range(self.num_timesteps, 0, -1):
            p_theta, _ = self.sample_step(F.one_hot(z_t, num_classes=N).float(), i, device)
            
            # 3. Sample z_s from p_theta
            z_s = torch.distributions.Categorical(probs=p_theta).sample() # Shape: (B, L)
            
            z_t = z_s
        
        x = z_t.reshape(B, *self.input_shape)
        return x
    
    def sample_step(self, z_t: Tensor, i: int, device='cpu') -> tuple[Tensor, Tensor]:
        B, L, N = z_t.shape
        
        t = self.discrete_time_scheduler.discrete_time(i).to(device)
        s = self.discrete_time_scheduler.discrete_time(i - 1).to(device)
        alpha_t = self.scheduler.alpha(t)
        alpha_s = self.scheduler.alpha(s)
        
        # 1. Calculate x_theta
        x_theta: Tensor = self.denoising_model(
            z_t.reshape(B, *self.input_shape, N), 
            torch.full((B,), t, device=device)
        )
        x_theta = x_theta.reshape(B, L, N)
        
        # 2. Calculate p_theta
        if i == 1:
            p_theta = x_theta
        else:
            p_theta = (
                N * alpha_t * z_t * x_theta 
                + (alpha_t / alpha_s - alpha_t) * z_t 
                + (alpha_s - alpha_t) * x_theta
                + ((alpha_s - alpha_t) * (1 - alpha_s) / (N * alpha_s)) * torch.ones_like(z_t)
            ) / (
                N * alpha_t * (z_t * x_theta).sum(dim=-1, keepdim=True)
                + 1 - alpha_t
            )
        
        # Sanity check - probabilities should sum up to 1
        assert torch.allclose(x_theta.sum(dim=-1), torch.ones(B, L, device=device)), f"Max = {torch.abs(x_theta.sum(dim=-1) - torch.ones(B, L, device=device)).max()}, {i}"
        assert torch.allclose(p_theta.sum(dim=-1), torch.ones(B, L, device=device), atol=1e-4), f"Max = {torch.abs(p_theta.sum(dim=-1) - torch.ones(B, L, device=device)).max()}, {i}"
        p_theta = p_theta / p_theta.sum(dim=-1, keepdim=True)
        # Sanity check - probabilities should be non-negative
        assert torch.all(x_theta >= 0), f"Min = {x_theta.min()}, {i}"
        assert torch.all(p_theta >= 0), f"Min = {p_theta.min()}, {i}"
        return p_theta, x_theta
