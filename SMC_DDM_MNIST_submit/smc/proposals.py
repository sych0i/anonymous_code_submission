import torch
from torch import Tensor
from functools import partial


def reverse_as_proposal(X_t, x_s_probs, t, lambdas, kl_weight, rewards_grad, model, reward_estimate_sample_count) -> tuple[Tensor, Tensor]:
    """
    Returns:
        tuple[Tensor, Tensor]: A tuple containing the proposed samples and their log probabilities.
    """
    # 1. Define proposal distribution
    proposal_distribution = torch.distributions.Categorical(probs=x_s_probs)
    
    # 2. Sample new particles from the proposal
    X_s = proposal_distribution.sample()
    
    # 3. Calcualte log probabibility of the new particles under the proposal distribution
    log_prob_proposal = proposal_distribution.log_prob(X_s) # Shape: (N, L)
    log_prob_proposal = log_prob_proposal.sum(dim=-1) # Shape: (N)
        
    return X_s.reshape(X_t.shape), log_prob_proposal


def first_order_approximation_optimal_proposal(X_t, x_s_probs, t, lambdas, kl_weight, rewards_grad, model, reward_estimate_sample_count, gradient_clip_value=None) -> tuple[Tensor, Tensor]:
    # Centering rewards_grad does not change the resulting probability distribution
    rewards_grad = rewards_grad - rewards_grad.mean(dim=-1, keepdim=True)
    
    if gradient_clip_value is not None:
        rewards_grad = torch.clip(rewards_grad, -gradient_clip_value, gradient_clip_value)
    
    logits_proposal = torch.log(x_s_probs) + (lambdas[t-1] / kl_weight) * rewards_grad # Shape: (N, L, C)
    
    # 1. Define proposal distribution
    proposal_distribution = torch.distributions.Categorical(logits=logits_proposal)
    
    # 2. Sample new particles from the proposal
    X_s = proposal_distribution.sample()
    
    # 3. Calcualte log probabibility of the new particles under the proposal distribution
    log_prob_proposal = proposal_distribution.log_prob(X_s) # Shape: (N, L)
    log_prob_proposal = log_prob_proposal.sum(dim=-1) # Shape: (N)
        
    return X_s.reshape(X_t.shape), log_prob_proposal

def first_order_approximation_optimal_proposal_with_gradient_clipping(gradient_clip_value):
    return partial(first_order_approximation_optimal_proposal, gradient_clip_value=gradient_clip_value)

def debiasing_guidance(X_t, x_s_probs, t, lambdas, kl_weight, rewards_grad, model, reward_estimate_sample_count, gradient_clip_value=None, beta=1.) -> tuple[Tensor, Tensor]:
    return first_order_approximation_optimal_proposal(X_t, x_s_probs, t, lambdas, 1/beta, rewards_grad, model, reward_estimate_sample_count, gradient_clip_value=None)
