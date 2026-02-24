# A collection of kernels, jax implementations, and jax pallas implementations.
Ran using TPU v5e1 on colab.

Flash Sinkhorn in Pure JAx
- standard jax operations to simulate flash attenion like approach for sinkhorn.

Pallas Sinkhorn Kernel
- low level TPU kernel written in Jax Pallas, sinkhorn normalisation purely in SRAM, custom fused backward pass under compiler constraints of v5e1.

NN Models
- sinkhorn pallas
  - wrapped algorithms into a flax module, takes an impl flag allowing to swap between pure jax and pallas implementations.
  - integrated the routing layer into a vision transformer architecture, allowing the model to route information using transport rather than softmax attention.
