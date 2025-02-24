# test_spectral_model.py

import jax
import jax.numpy as jnp
import equinox as eqx

class SpectralDenseBlock(eqx.Module):
    """
    A custom spectral dense block that:
      1. Applies FFT on the input,
      2. Multiplies by a trainable complex mask (w_real and w_imag),
      3. Applies inverse FFT (taking the real part),
      4. Applies a pointwise MLP (Linear -> ReLU -> Linear),
      5. Adds a residual connection.
    
    This layer is defined for a single example with shape (in_features,).
    """
    in_features: int
    hidden_dim: int
    w_real: jnp.ndarray  # shape: (in_features,)
    w_imag: jnp.ndarray  # shape: (in_features,)
    linear1: eqx.nn.Linear  # maps (in_features,) -> (hidden_dim,)
    linear2: eqx.nn.Linear  # maps (hidden_dim,) -> (in_features,)

    def __init__(self, in_features: int, hidden_dim: int, *, key):
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.w_real = jax.random.normal(k1, (in_features,)) * 0.1
        self.w_imag = jax.random.normal(k2, (in_features,)) * 0.1
        self.linear1 = eqx.nn.Linear(in_features, hidden_dim, key=k3)
        self.linear2 = eqx.nn.Linear(hidden_dim, in_features, key=k4)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x is expected to be shape (in_features,)
        X_fft = jnp.fft.fft(x)  # shape (in_features,), complex-valued
        mask_complex = self.w_real + 1j * self.w_imag
        out_fft = X_fft * mask_complex
        x_time = jnp.fft.ifft(out_fft).real  # back to real signal
        h = self.linear1(x_time)
        h = jax.nn.relu(h)
        x_dense = self.linear2(h)
        return x_time + x_dense

class TestSpectral(eqx.Module):
    """
    A test model that uses one SpectralDenseBlock followed by a final linear layer.
    Processes a single sample of shape (in_features,).
    """
    layer1: SpectralDenseBlock
    out: eqx.nn.Linear  # maps (in_features,) -> (out_features,)

    def __init__(self, in_features: int, hidden_dim: int, out_features: int, *, key):
        k1, k2 = jax.random.split(key, 2)
        self.layer1 = SpectralDenseBlock(in_features, hidden_dim, key=k1)
        self.out = eqx.nn.Linear(in_features, out_features, key=k2)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (in_features,)
        h = self.layer1(x)
        return self.out(h)

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    in_features = 16
    hidden_dim = 32
    out_features = 4
    batch_size = 8

    # Initialize model (for a single sample)
    model = TestSpectral(in_features, hidden_dim, out_features, key=key)
    # Create a batch of inputs of shape (batch_size, in_features)
    X = jax.random.normal(key, (batch_size, in_features))
    # Use jax.vmap to apply the model to each sample in the batch.
    Y = jax.vmap(model)(X)
    print("TestSpectral output shape:", Y.shape)  # Expected: (batch_size, out_features)
