import flax.linen as nn
import jax.numpy as jnp
from .routing import RoutingLayer


class AeroVitBlock(nn.Module):
    dim: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x):
        shortcut = x
        x = nn.LayerNorm()(x)
        x = RoutingLayer(num_heads=self.num_heads)(x)
        x = shortcut + x
        shortcut = x
        x = nn.LayerNorm()(x)
        mlp_dim = int(self.dim * self.mlp_ratio)
        x = nn.Dense(mlp_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim)(x)
        x = shortcut + x
        return x


class AeroVitModel(nn.Module):
    num_layers: int
    dim: int
    num_heads: int
    num_classes: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.dim)(x)

        for _ in range(self.num_layers):
            x = AeroVitBlock(dim=self.dim, num_heads=self.num_heads)(x)

        x = nn.LayerNorm()(x)
        x = jnp.mean(x, axis=1)
        x = nn.Dense(self.num_classes)(x)
        return x
