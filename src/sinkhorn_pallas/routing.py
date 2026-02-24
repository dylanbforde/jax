import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from kernels.pallas_sinkhorn import pallas_flash_sinkhorn


class RoutingLayer(nn.Module):
    num_heads: int = 8
    n_iters: int = 20
    use_pallas: bool = True

    @nn.compact
    def __call__(self, x):
        B, L, E = x.shape
        head_dim = 64

        q = nn.Dense(self.num_heads * head_dim)(x)
        k = nn.Dense(self.num_heads * head_dim)(x)

        q = q.reshape(B, L, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        v = nn.Dense(self.num_heads * head_dim)(x)
        v = v.reshape(B, L, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        if self.use_pallas:
            out = pallas_flash_sinkhorn(q, k, v, self.n_iters)
        else:
            logits = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)
            log_alpha = logits
            for _ in range(self.n_iters):
                log_alpha = log_alpha - jax.scipy.special.logsumexp(
                    log_alpha, axis=-1, keepdims=True
                )
                log_alpha = log_alpha - jax.scipy.special.logsumexp(
                    log_alpha, axis=-2, keepdims=True
                )
            attn = jnp.exp(log_alpha)
            out = jnp.matmul(attn, v)

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return nn.Dense(E)(out)
