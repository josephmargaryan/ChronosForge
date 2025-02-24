# test_fourier_model.py

import jax
import jax.numpy as jnp
import equinox as eqx

def make_mask(n: int, n_modes: int):
    """Create a mask of length n with ones in the first and last n_modes positions."""
    mask = jnp.zeros((n,))
    mask = mask.at[:n_modes].set(1.0)
    mask = mask.at[-n_modes:].set(1.0)
    return mask

class FourierNeuralOperator1D(eqx.Module):
    """
    A single Fourier layer that:
      1. Computes the FFT of the input,
      2. Multiplies by a trainable spectral weight (with a mask),
      3. Computes the inverse FFT,
      4. Applies a pointwise MLP (Linear -> ReLU -> Linear),
      5. And adds a residual connection.
    
    This layer processes a single sample of shape (in_features,).
    """
    in_features: int
    hidden_dim: int
    n_modes: int
    spectral_weight: jnp.ndarray  # shape: (in_features,)
    linear1: eqx.nn.Linear      # maps (in_features,) -> (hidden_dim,)
    linear2: eqx.nn.Linear      # maps (hidden_dim,) -> (in_features,)

    def __init__(self, in_features: int, hidden_dim: int, n_modes: int, *, key):
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.n_modes = n_modes
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.spectral_weight = jax.random.normal(k1, (in_features,)) * 0.1
        self.linear1 = eqx.nn.Linear(in_features, hidden_dim, key=k2)
        self.linear2 = eqx.nn.Linear(hidden_dim, in_features, key=k3)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x is expected to be shape (in_features,)
        X_fft = jnp.fft.fft(x)
        mask = make_mask(self.in_features, self.n_modes)
        scale = 1.0 + mask * self.spectral_weight
        X_fft_mod = X_fft * scale
        x_ifft = jnp.fft.ifft(X_fft_mod).real
        h = self.linear1(x_ifft)
        h = jax.nn.relu(h)
        x_mlp = self.linear2(h)
        return x_ifft + x_mlp

class TestFourier(eqx.Module):
    """
    A test model that uses one FNO1DLayer followed by a final linear layer.
    Processes a single sample of shape (in_features,).
    """
    layer1: FourierNeuralOperator1D
    final_linear: eqx.nn.Linear  # maps (in_features,) -> (out_features,)

    def __init__(self, in_features: int, hidden_dim: int, n_modes: int, out_features: int, *, key):
        k1, k2 = jax.random.split(key, 2)
        self.layer1 = FourierNeuralOperator1D(in_features, hidden_dim, n_modes, key=k1)
        self.final_linear = eqx.nn.Linear(in_features, out_features, key=k2)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (in_features,)
        h = self.layer1(x)
        return self.final_linear(h)

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    in_features = 16
    hidden_dim = 32
    n_modes = 4
    out_features = 10
    batch_size = 8

    # Initialize model (for a single sample)
    model = TestFourier(in_features, hidden_dim, n_modes, out_features, key=key)
    # Create a batch of inputs with shape (batch_size, in_features)
    X = jax.random.normal(key, (batch_size, in_features))
    # Use jax.vmap to process the batch.
    Y = jax.vmap(model)(X)
    print("TestFourier output shape:", Y.shape)  # Expected: (batch_size, out_features)
