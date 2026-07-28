from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from typing import Callable, Optional
from .utils import (
    compute_ess_from_log_w,
    normalize_log_weights, 
    normalize_weights
)
from .sampling_algorithms import partial_resample

def sequential_monte_carlo_base_tstep(
    model, 
    num_categories,
    T,
    N: int, 
    ESS_min, 
    intialize_particles_fn,
    resample_fn, 
    proposal_fn,
    compute_reward_fn,
    lambdas,
    kl_weight,
    reward_estimate_sample_count: int,
    use_partial_resampling: bool = False,
    partial_resample_size: Optional[int] = None,
    perform_final_resample=True,
    eps=1e-9,
    device=torch.device('cpu'),
    verbose=False,
    tstep: set[int] = set()
):
    model.eval()

    X_t = intialize_particles_fn(N, device=device)
    log_W_t = torch.zeros(N, device=device, requires_grad=False)
    input_shape = X_t.shape[1:]
    
    log_prob_diffusion = torch.zeros(N, device=device, requires_grad=False)
    log_prob_proposal = torch.zeros(N, device=device, requires_grad=False)
    log_twist_func_prev = torch.zeros(N, device=device, requires_grad=False)
    
    particles_trace = [X_t.cpu().numpy()]
    log_weights_trace = []
    ess_trace = []
    rewards_trace = []
    mean_exp_rewards_trace = []
    resampling_trace = []
    log_prob_diffusion_trace = [log_prob_diffusion.cpu().numpy()]
    log_prob_proposal_trace = [log_prob_proposal.cpu().numpy()]
    parent_trace = []

    for t in tqdm(range(T, 0, -1)):
        # T is always corrected as the initial guided target. For t < T,
        # correction at x_t is controlled by whether t itself belongs to G.
        should_correct_current = t == T or t in tstep

        if should_correct_current:
            with torch.no_grad():
                z_curr = F.one_hot(X_t.reshape(N, -1), num_classes=num_categories).float()
                x_s_probs, x0_probs = model.sample_step(z_curr, t, device=device)

                x0_samples = F.gumbel_softmax(
                    logits=torch.log(x0_probs + eps).unsqueeze(1).expand(-1, reward_estimate_sample_count, -1, -1),
                    hard=True,
                )

                rewards = compute_reward_fn(
                    x0_samples.reshape(N * reward_estimate_sample_count, *input_shape, num_categories),
                    with_grad=False,
                ).reshape(N, reward_estimate_sample_count).mean(dim=1)

                rewards_grad = torch.zeros_like(z_curr)
                log_twist_func = (lambdas[t] / kl_weight) * rewards
        else:
            with torch.no_grad():
                z_t = F.one_hot(X_t.reshape(N, -1), num_classes=num_categories).float()
                x_s_probs, _ = model.sample_step(z_t, t, device=device)
                rewards = torch.full((N,), torch.nan, device=device)
                rewards_grad = torch.zeros_like(z_t)
                log_twist_func = log_twist_func_prev

        rewards_trace.append(rewards.cpu().numpy())
        log_W_t += (log_prob_diffusion - log_prob_proposal + log_twist_func - log_twist_func_prev)
        log_W_t = normalize_log_weights(log_W_t)

        if torch.isnan(log_W_t).any() or torch.isinf(log_W_t).any():
            raise ValueError("NaN or Inf encountered in log_W_t")
        log_weights_trace.append(log_W_t.cpu().numpy())
        mean_exp_rewards_trace.append(
            torch.sum(normalize_weights(log_W_t) * torch.exp(rewards / kl_weight)).item()
            if should_correct_current
            else np.nan
        )

        ESS = compute_ess_from_log_w(log_W_t)
        ess_trace.append(ESS.item())

        if ESS < ESS_min and should_correct_current:
            if use_partial_resampling:
                p_size = partial_resample_size if partial_resample_size is None else N // 2
                resampled_indices, log_W_t = partial_resample(log_W_t, resample_fn, p_size)
            else:
                resampled_indices = resample_fn(log_W_t)
                log_W_t = log_W_t.zero_()

            X_t = X_t[resampled_indices]
            x_s_probs = x_s_probs[resampled_indices]
            log_twist_func = log_twist_func[resampled_indices]
            rewards_grad = rewards_grad[resampled_indices] if rewards_grad is not None else None
            resampling_trace.append(t)
            parent_trace.append(resampled_indices.cpu().numpy())
        else:
            parent_trace.append(np.arange(N).astype(int))

        # SMC-base always uses the reference transition. The correction for
        # target x_{t-1}, when selected, is performed at the next iteration.
        X_t, log_prob_proposal = proposal_fn(X_t, x_s_probs, t, lambdas, kl_weight, rewards_grad, model, reward_estimate_sample_count)
        particles_trace.append(X_t.cpu().numpy())

        diffusion_distribution = torch.distributions.Categorical(probs=x_s_probs)
        log_prob_diffusion = diffusion_distribution.log_prob(X_t.reshape(N, -1)).sum(dim=1)

        log_prob_diffusion_trace.append(log_prob_diffusion.cpu().numpy())
        log_prob_proposal_trace.append(log_prob_proposal.cpu().numpy())
        log_twist_func_prev = log_twist_func

    # Correct x_0 only when 0 belongs to G. Regardless of that choice, the
    # final weights are converted to an unweighted particle sample below.
    should_correct_final = 0 in tstep
    if should_correct_final:
        final_rewards = compute_reward_fn(F.one_hot(X_t, num_classes=num_categories).float())
        log_twist_func = (lambdas[0] / kl_weight) * final_rewards
    else:
        final_rewards = torch.full((N,), torch.nan, device=device)
        log_twist_func = log_twist_func_prev
    rewards_trace.append(final_rewards.cpu().numpy())

    log_W_t += (log_prob_diffusion - log_prob_proposal + log_twist_func - log_twist_func_prev)
    log_W_t = normalize_log_weights(log_W_t)
    log_weights_trace.append(log_W_t.cpu().numpy())
    mean_exp_rewards_trace.append(
        torch.sum(normalize_weights(log_W_t) * torch.exp(final_rewards / kl_weight)).item()
        if should_correct_final
        else np.nan
    )
    ess_trace.append(compute_ess_from_log_w(log_W_t).item())

    if perform_final_resample:
        resampled_indices = resample_fn(log_W_t)
        X_t = X_t[resampled_indices]
        log_W_t = torch.zeros_like(log_W_t)

    return {
        "X_0": X_t,
        "W_0": normalize_weights(log_W_t),
        "ess_trace": ess_trace,
        "rewards_trace": rewards_trace,
        "mean_exp_rewards_trace": mean_exp_rewards_trace,
        "particles_trace": particles_trace,
        "log_weights_trace": log_weights_trace,
        "resampling_trace": resampling_trace,
        "log_prob_diffusion_trace": log_prob_diffusion_trace,
        "log_prob_proposal_trace": log_prob_proposal_trace,
        "parent_trace": parent_trace,
        "tstep": tstep,
    }


def sequential_monte_carlo_grad_tstep(
    model,
    num_categories,
    T,
    N: int,
    ESS_min,
    intialize_particles_fn,
    resample_fn,
    proposal_fn,
    compute_reward_fn,
    lambdas,
    kl_weight,
    reward_estimate_sample_count: int,
    use_partial_resampling: bool = False,
    partial_resample_size: Optional[int] = None,
    perform_final_resample=True,
    eps=1e-9,
    device=torch.device('cpu'),
    verbose=False,
    tstep: set[int] = set()
):
    model.eval()

    X_t = intialize_particles_fn(N, device=device)
    log_W_t = torch.zeros(N, device=device, requires_grad=False)
    input_shape = X_t.shape[1:]

    log_prob_diffusion = torch.zeros(N, device=device, requires_grad=False)
    log_prob_proposal = torch.zeros(N, device=device, requires_grad=False)
    log_twist_func_prev = torch.zeros(N, device=device, requires_grad=False)

    particles_trace = [X_t.cpu().numpy()]
    log_weights_trace = []
    ess_trace = []
    rewards_trace = []
    mean_exp_rewards_trace = []
    resampling_trace = []
    log_prob_diffusion_trace = [log_prob_diffusion.cpu().numpy()]
    log_prob_proposal_trace = [log_prob_proposal.cpu().numpy()]
    parent_trace = []

    for t in tqdm(range(T, 0, -1)):
        # A selected t corrects x_t. A selected t-1 additionally needs the
        # value gradient at source x_t to construct q_{t-1|t}.
        should_correct_current = t == T or t in tstep
        should_guide_transition = t - 1 in tstep

        if should_guide_transition:
            z_t = F.one_hot(X_t.reshape(N, -1), num_classes=num_categories).float()
            z_t.requires_grad_()
            x_s_probs, x0_probs = model.sample_step(z_t, t, device=device)

            x0_samples = F.gumbel_softmax(
                logits=torch.log(x0_probs + eps).unsqueeze(1).expand(-1, reward_estimate_sample_count, -1, -1),
                hard=True,
            )
            rewards = compute_reward_fn(
                x0_samples.reshape(N * reward_estimate_sample_count, *input_shape, num_categories),
                with_grad=True,
            ).reshape(N, reward_estimate_sample_count).mean(dim=1)
            rewards_grad = torch.autograd.grad(outputs=rewards, inputs=z_t, grad_outputs=torch.ones_like(rewards))[0]
            x_s_probs = x_s_probs.detach()
            rewards = rewards.detach()
        elif should_correct_current:
            with torch.no_grad():
                z_t = F.one_hot(X_t.reshape(N, -1), num_classes=num_categories).float()
                x_s_probs, x0_probs = model.sample_step(z_t, t, device=device)
                x0_samples = F.gumbel_softmax(
                    logits=torch.log(x0_probs + eps).unsqueeze(1).expand(-1, reward_estimate_sample_count, -1, -1),
                    hard=True,
                )
                rewards = compute_reward_fn(
                    x0_samples.reshape(N * reward_estimate_sample_count, *input_shape, num_categories),
                    with_grad=False,
                ).reshape(N, reward_estimate_sample_count).mean(dim=1)
                rewards_grad = torch.zeros_like(z_t)
        else:
            with torch.no_grad():
                z_t = F.one_hot(X_t.reshape(N, -1), num_classes=num_categories).float()
                x_s_probs, _ = model.sample_step(z_t, t, device=device)
                rewards = torch.full((N,), torch.nan, device=device)
                rewards_grad = torch.zeros_like(z_t)

        if should_correct_current:
            log_twist_func = (lambdas[t] / kl_weight) * rewards
        else:
            log_twist_func = log_twist_func_prev

        rewards_trace.append(rewards.cpu().numpy())

        log_W_t += (log_prob_diffusion - log_prob_proposal + log_twist_func - log_twist_func_prev)
        log_W_t = normalize_log_weights(log_W_t)

        if torch.isnan(log_W_t).any() or torch.isinf(log_W_t).any():
            raise ValueError("NaN or Inf encountered in log_W_t")
        log_weights_trace.append(log_W_t.cpu().numpy())
        mean_exp_rewards_trace.append(
            torch.sum(normalize_weights(log_W_t) * torch.exp(rewards / kl_weight)).item()
            if should_correct_current
            else np.nan
        )
        
        # 3. Adaptive resampling (should_compute_reward일 때만 수행하도록 설정 가능하지만,
        # 안전을 위해 ESS 기반으로 유지하되, Grad가 0이면 일반 Diffusion 리샘플링이 됨)
        ESS = compute_ess_from_log_w(log_W_t)
        ess_trace.append(ESS.item())

        if ESS < ESS_min and should_correct_current:
            if use_partial_resampling:
                p_size = partial_resample_size if partial_resample_size is None else N // 2
                resampled_indices, log_W_t = partial_resample(log_W_t, resample_fn, p_size)
            else:
                resampled_indices = resample_fn(log_W_t)
                log_W_t = log_W_t.zero_()

            X_t = X_t[resampled_indices]
            x_s_probs = x_s_probs[resampled_indices]
            log_twist_func = log_twist_func[resampled_indices]
            rewards_grad = rewards_grad[resampled_indices] if rewards_grad is not None else None

            resampling_trace.append(t)
            parent_trace.append(resampled_indices.cpu().numpy())
        else:
            parent_trace.append(np.arange(N).astype(int))

        # A non-zero gradient is supplied only when target t-1 belongs to G,
        # so skipped transitions reduce exactly to the reference proposal.
        X_t, log_prob_proposal = proposal_fn(X_t, x_s_probs, t, lambdas, kl_weight, rewards_grad, model, reward_estimate_sample_count)
        particles_trace.append(X_t.cpu().numpy())
        
        diffusion_distribution = torch.distributions.Categorical(probs=x_s_probs)
        log_prob_diffusion = diffusion_distribution.log_prob(X_t.reshape(N, -1)).sum(dim=1)
        
        log_prob_diffusion_trace.append(log_prob_diffusion.cpu().numpy())
        log_prob_proposal_trace.append(log_prob_proposal.cpu().numpy())
        log_twist_func_prev = log_twist_func

    should_correct_final = 0 in tstep
    if should_correct_final:
        final_rewards = compute_reward_fn(F.one_hot(X_t, num_classes=num_categories).float())
        log_twist_func = (lambdas[0] / kl_weight) * final_rewards
    else:
        final_rewards = torch.full((N,), torch.nan, device=device)
        log_twist_func = log_twist_func_prev
    rewards_trace.append(final_rewards.cpu().numpy())

    log_W_t += (log_prob_diffusion - log_prob_proposal + log_twist_func - log_twist_func_prev)
    log_W_t = normalize_log_weights(log_W_t)
    log_weights_trace.append(log_W_t.cpu().numpy())
    mean_exp_rewards_trace.append(
        torch.sum(normalize_weights(log_W_t) * torch.exp(final_rewards / kl_weight)).item()
        if should_correct_final
        else np.nan
    )
    ess_trace.append(compute_ess_from_log_w(log_W_t).item())

    if perform_final_resample:
        resampled_indices = resample_fn(log_W_t)
        X_t = X_t[resampled_indices]
        log_W_t = torch.zeros_like(log_W_t)

    return {
        "X_0": X_t,
        "W_0": normalize_weights(log_W_t),
        "ess_trace": ess_trace,
        "rewards_trace": rewards_trace,
        "mean_exp_rewards_trace": mean_exp_rewards_trace,
        "particles_trace": particles_trace,
        "log_weights_trace": log_weights_trace,
        "resampling_trace": resampling_trace,   
        "log_prob_diffusion_trace": log_prob_diffusion_trace,
        "log_prob_proposal_trace": log_prob_proposal_trace,
        "parent_trace": parent_trace,
        "tstep": tstep,
    }
