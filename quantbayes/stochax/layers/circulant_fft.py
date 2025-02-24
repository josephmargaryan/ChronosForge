import jax
import jax.numpy as jnp
import equinox as eqx
import jax.random
import numpy as np


class Circulant(eqx.Module):
    """
    A custom Equinox layer that implements a linear layer with a circulant matrix.

    The layer stores only the first row `c` (a vector of shape (n,)) and a bias vector
    (of shape (n,)). We define the circulant matrix C so that its first row is c and its i-th row is:
      C[i, :] = jnp.roll(c, i)

    The first column is computed by flipping `c` and then rolling by 1.
    The circulant multiplication is then computed via FFT:
      y = real( ifft( fft(x) * fft(first_col) ) ) + bias
    """
    first_row: jnp.ndarray  # shape (n,)
    bias: jnp.ndarray       # shape (n,)
    in_features: int = eqx.static_field()
    out_features: int = eqx.static_field()

    def __init__(self, in_features: int, *, key, init_scale: float = 1.0):
        self.in_features = in_features
        self.out_features = in_features
        # Split key for first_row and bias initialization.
        key1, key2 = jax.random.split(key)
        self.first_row = jax.random.normal(key1, (in_features,)) * init_scale
        self.bias = jax.random.normal(key2, (in_features,)) * init_scale

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Multiply input x (shape (..., n)) by the circulant matrix and add bias.
        """
        # Compute the "first column" of the circulant matrix:
        # first_row = [c0, c1, ..., c_{n-1}]
        # flip(first_row) = [c_{n-1}, c_{n-2}, ..., c0]
        # Rolling that by +1 gives: [c0, c_{n-1}, c_{n-2}, ..., c1]
        first_col = jnp.roll(jnp.flip(self.first_row), shift=1)
        fft_w = jnp.fft.fft(first_col)
        fft_x = jnp.fft.fft(x, axis=-1)
        y = jnp.fft.ifft(fft_x * fft_w, axis=-1)
        return jnp.real(y) + self.bias

