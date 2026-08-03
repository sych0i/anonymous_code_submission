import math
import typing

try:
    import flash_attn
    import flash_attn.layers.rotary
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out


def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training)

  return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
  return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
  return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
  def __init__(self, dim, base=10_000):
    super().__init__()
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)
    self.seq_len_cached = None
    self.cos_cached = None
    self.sin_cached = None

  def forward(self, x, seq_dim=1):
    seq_len = x.shape[seq_dim]
    if seq_len != self.seq_len_cached:
      self.seq_len_cached = seq_len
      t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
      freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
      emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
      # dims are: batch, seq_len, qkv, head, dim
      self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
      self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
      # This makes the transformation on v an identity.
      self.cos_cached[:,:,2,:,:].fill_(1.)
      self.sin_cached[:,:,2,:,:].fill_(0.)

    return self.cos_cached, self.sin_cached


def rotate_half(x):
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin):
  cos = cos[0,:,0,0,:cos.shape[-1]//2]
  sin = sin[0,:,0,0,:sin.shape[-1]//2]
  if HAS_FLASH_ATTN:
    return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)
  # fallback: apply RoPE to q and k only; qkv shape: (b, s, 3, h, d)
  # cos/sin are (s, d//2); expand to full head dim via split-half convention
  cos_full = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(2)  # (1, s, 1, d)
  sin_full = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(2)
  def _rot(x):
    return x * cos_full + rotate_half(x) * sin_full
  q = _rot(qkv[:, :, 0])
  k = _rot(qkv[:, :, 1])
  return torch.stack([q, k, qkv[:, :, 2]], dim=2)


# function overload
def modulate(x, shift, scale):
  return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim
  def forward(self, x):
    with torch.amp.autocast("cuda", enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None,None,:]


def residual_linear(x, W, x_skip, residual_scale):
  """x_skip + residual_scale * W @ x"""
  dim_out, dim_in = W.shape[0], W.shape[1]
  return torch.addmm(
    x_skip.view(-1, dim_out),
    x.view(-1, dim_in),
    W.T,
    alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
  """
  Embeds scalar timesteps into vector representations.
  """
  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
      - math.log(max_period)
      * torch.arange(start=0, end=half, dtype=torch.float32)
      / half).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding,
         torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb


class LabelEmbedder(nn.Module):
  """Embeds class labels into vector representations.

  Also handles label dropout for classifier-free guidance.
  """
  def __init__(self, num_classes, cond_size):
    super().__init__()
    self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
    self.num_classes = num_classes

    # TODO think of initializing with 0.02 std deviation like in original DiT paper

  def forward(self, labels):
    embeddings = self.embedding_table(labels)
    return embeddings


#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlock(nn.Module):
  def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
    super().__init__()
    self.n_heads = n_heads

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

    self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()


  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference


  def forward(self, x, rotary_cos_sin, c, seqlens=None):
    batch_size, seq_len = x.shape[0], x.shape[1]

    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    (shift_msa, scale_msa, gate_msa, shift_mlp,
     scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

    # attention operation
    x_skip = x
    x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

    qkv = self.attn_qkv(x)
    qkv = rearrange(qkv,
                    'b s (three h d) -> b s three h d',
                    three=3,
                    h=self.n_heads)
    with torch.amp.autocast("cuda", enabled=False):
      cos, sin = rotary_cos_sin
      qkv = apply_rotary_pos_emb(
        qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
    qkv = rearrange(qkv, 'b s ... -> (b s) ...')
    if seqlens is None:
      cu_seqlens = torch.arange(
        0, (batch_size + 1) * seq_len, step=seq_len,
        dtype=torch.int32, device=qkv.device)
    else:
      cu_seqlens = seqlens.cumsum(-1)
    if HAS_FLASH_ATTN:
      x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
        qkv, cu_seqlens, seq_len, 0., causal=False)
    else:
      # fallback: qkv is (b*s, 3, h, d) → standard SDPA
      qkv_b = rearrange(qkv, '(b s) three h d -> b three h s d', b=batch_size)
      q_b, k_b, v_b = qkv_b.unbind(dim=1)
      x = F.scaled_dot_product_attention(q_b, k_b, v_b, dropout_p=0.)
      x = rearrange(x, 'b h s d -> (b s) h d')

    x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)

    x = bias_dropout_scale_fn(self.attn_out(x),
                              None,
                              gate_msa,
                              x_skip,
                              self.dropout)

    # mlp operation
    x = bias_dropout_scale_fn(
      self.mlp(modulate_fused(
        self.norm2(x), shift_mlp, scale_mlp)),
      None, gate_mlp, x, self.dropout)
    return x



class EmbeddingLayer(nn.Module):
  def __init__(self, dim, vocab_dim):
    super().__init__()
    self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
    torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

  def forward(self, x):
    return self.embedding[x]


class DDitFinalLayer(nn.Module):
  def __init__(self, hidden_size, out_channels, cond_dim):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()

    self.adaLN_modulation = nn.Linear(cond_dim,
                                      2 * hidden_size,
                                      bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()


  def forward(self, x, c):
    shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
    x = modulate_fused(self.norm_final(x), shift, scale)
    x = self.linear(x)
    return x


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, config, vocab_size: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    self.config = config
    self.vocab_size = vocab_size
    self.causal = False #temporary fix since we are not using causal models

    self.vocab_embed = EmbeddingLayer(config.model.hidden_size,
                                      vocab_size)
    self.sigma_map = TimestepEmbedder(config.model.cond_dim)
    self.rotary_emb = Rotary(
      config.model.hidden_size // config.model.n_heads)

    blocks = []
    for _ in range(config.model.n_blocks):
      blocks.append(DDiTBlock(config.model.hidden_size,
                              config.model.n_heads,
                              config.model.cond_dim,
                              dropout=config.model.dropout))
    self.blocks = nn.ModuleList(blocks)
    # self.layers = nn.ModuleDict({
    # 'output_layer': DDitFinalLayer(config.model.hidden_size, vocab_size, config.model.cond_dim)
    # })
    self.output_layer = DDitFinalLayer(
      config.model.hidden_size,
      vocab_size,
      config.model.cond_dim)
    self.scale_by_sigma = config.model.scale_by_sigma

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return  bias_dropout_add_scale_fused_inference

  # def forward(self, indices, sigma):
  #   x = self.vocab_embed(indices)
  #   c = F.silu(self.sigma_map(sigma))

  #   rotary_cos_sin = self.rotary_emb(x)

  #   with torch.cuda.amp.autocast(dtype=torch.bfloat16):
  #     for i in range(len(self.blocks)):
  #       x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
  #     x = self.output_layer(x, c)

  #   return x

  def forward(self, indices, sigma,
              cond=None,
              x_emb=None,
              return_hidden_states=False):
    if return_hidden_states:
      hidden_states = []

    if self.causal:
      c = None
    else:
      c = F.silu(self.sigma_map(sigma))
    if cond is not None:
      if self.cond_map is None:
        raise ValueError("Conditioning variable provided, "
                         "but Model was not initialized "
                         "with condition embedding layer.")
      else:
        if c is None:  # AR (self.causal is True)
          c = F.silu(self.cond_map(cond))
        else:
          c = c + F.silu(self.cond_map(cond))

    if x_emb is None:
      if indices.ndim == 2:
        # Ordinary generation path: integer token indices, (B, L).
        x = self.vocab_embed(indices)
      elif indices.ndim == 3:
        # Differentiable relaxation used by online SMC-grad: a one-hot or
        # probability vector is embedded by the same vocabulary table's
        # weighted sum.  At an exact one-hot this is algebraically identical
        # to the integer lookup above, while retaining gradients with respect
        # to the input probabilities.
        if indices.shape[-1] != self.vocab_size:
          raise ValueError(
            'Soft token input must have final dimension equal to '
            f'vocab_size={self.vocab_size}, got shape {tuple(indices.shape)}')
        x = F.linear(indices.to(dtype=self.vocab_embed.embedding.dtype),
                     self.vocab_embed.embedding.T)
      else:
        raise ValueError(
          'DIT input must be integer indices with shape (B, L) or soft '
          f'tokens with shape (B, L, V), got shape {tuple(indices.shape)}')
      if return_hidden_states:
        hidden_states.append(x)
      rotary_cos_sin = self.rotary_emb(x)

      with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i in range(len(self.blocks)):
          x = self.blocks[i](x, rotary_cos_sin, c,
                             seqlens=None)
          if return_hidden_states:
            hidden_states.append(x)
    else:
      x = x_emb

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
      #x = self.layers.output_layer(x, c)
      x = self.output_layer(x, c)

    if return_hidden_states:
      return x, hidden_states
    return x

class DITClassifier(nn.Module):
  """
  Classifier sharing the same architecture as the DIT model.
  """
  def __init__(self, model_config, vocab_size):
    super().__init__()
    if type(model_config) == dict:
      model_config = omegaconf.OmegaConf.create(model_config)

    self.config = model_config
    self.vocab_size = vocab_size
    self.causal = False #temporary hardcoded
    # self.causal = model_config.parameterization == 'ar'

    self.vocab_embed = EmbeddingLayer(
      model_config.classifier_model.hidden_size, self.vocab_size)

    if self.causal:
      self.sigma_map = None
    else:
      self.sigma_map = TimestepEmbedder(model_config.classifier_model.cond_dim)

    self.rotary_emb = Rotary(
      model_config.classifier_model.hidden_size // model_config.classifier_model.n_heads)

    blocks = []
    #use_adaLN = model_config.parameterization != 'ar'
    for _ in range(model_config.classifier_model.n_blocks):
      blocks.append(
        DDiTBlock(model_config.classifier_model.hidden_size,
                  model_config.classifier_model.n_heads,
                  model_config.classifier_model.cond_dim,
                  dropout=model_config.classifier_model.dropout,
                  # causal=self.causal,
                  #use_adaLN=use_adaLN
                  ))
    self.blocks = nn.ModuleList(blocks)

    self.scale_by_sigma = model_config.classifier_model.scale_by_sigma

    self.pooling = getattr(model_config.classifier_model, 'pooling', 'mean')
    # self.output_layer = nn.Linear(
    #   config.classifier_model.hidden_size,
    #   config.classifier_model.num_classes)
    self.output_layer = nn.Linear(model_config.classifier_model.hidden_size, 1) #to a scalar
    # self.output_layer = nn.Sequential(nn.Linear(model_config.classifier_model.hidden_size, model_config.classifier_model.hidden_size,),
    #                             nn.ReLU(),
    #                             nn.Linear(model_config.classifier_model.hidden_size, 1))

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return  bias_dropout_add_scale_fused_inference

  def forward(self, indices_or_one_hots, sigma, x_emb=None, attention_mask=None):
    if x_emb is None:
      if indices_or_one_hots.ndim == 2:  # indices (B, L)
        x = self.vocab_embed(indices_or_one_hots)
      else:  # one-hots (B, L, V)
        x = F.linear(indices_or_one_hots.to(torch.float),
                     self.vocab_embed.embedding.T)

      if self.causal:
        c = None
      else:
        c = F.silu(self.sigma_map(sigma))

      rotary_cos_sin = self.rotary_emb(x)

      with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i in range(len(self.blocks)):
          x = self.blocks[i](x, rotary_cos_sin, c,
                             seqlens=None)
    else:
      x = x_emb

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
      if self.pooling == 'mean':
        x = x.mean(dim=1)
      elif self.pooling == 'max':
        x = x.max(dim=1)
      elif self.pooling == 'cls':
        x = x[..., 0]
      elif self.pooling == 'last':
        x = x[..., -1]
      elif self.pooling == 'no_pooling':  # for ar_fudge
        pass
      elif self.pooling == 'attention_mean':  # for ar_pplm
        masked_x = x * attention_mask.unsqueeze(2)
        x = torch.sum(masked_x, dim=1) / (torch.sum(attention_mask, dim=1, keepdim=True) + 1e-15)
      else:
        raise NotImplementedError(
          f"`{self.pooling}` method not implemented.")
      x = self.output_layer(x)
    return x

  def load_pretrained_encoder(self, encoder: nn.Module):
    self.vocab_embed = encoder.vocab_embed
    self.sigma_map = encoder.sigma_map
    self.rotary_emb = encoder.rotary_emb
    self.blocks = encoder.blocks
