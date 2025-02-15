import jax
import jax.random as jr
import numpy as np
import equinox as eqx
import jax.numpy as jnp


@jax.custom_jvp
def circulant_matmul(weight_vector, x):
    # Forward pass using FFT
    w_fft = jnp.fft.fft(weight_vector)
    x_fft = jnp.fft.fft(x, axis=-1)
    y_fft = x_fft * w_fft[None, :].conjugate()
    return jnp.fft.ifft(y_fft, axis=-1).real


@circulant_matmul.defjvp
def circulant_matmul_jvp(primals, tangents):
    w, x = primals
    dw, dx = tangents
    # forward = circulant_matmul(w, x)   # shape (batch, b)

    # derivative wrt w => circ(dL/dy) x
    # derivative wrt x => circ(dL/dy) w
    # We can do it in the frequency domain directly

    # Just to illustrate the idea:
    w_fft = jnp.fft.fft(w)
    x_fft = jnp.fft.fft(x, axis=-1)
    dw_fft = jnp.fft.fft(dw)
    dx_fft = jnp.fft.fft(dx, axis=-1)

    # Forward pass again
    primal_y_fft = x_fft * w_fft.conjugate()
    primal_y = jnp.fft.ifft(primal_y_fft, axis=-1).real

    # JVP
    # d/dw => IRFFT( conj(x_fft) * d(w_fft) ) + IRFFT(...) ???
    # We'll keep it short in the example:
    tangent_y_fft = (x_fft * dw_fft.conjugate()) + (dx_fft * w_fft.conjugate())
    tangent_y = jnp.fft.ifft(tangent_y_fft, axis=-1).real

    return primal_y, tangent_y


class EfficientCirculantLinear(eqx.Module):
    """
    A custom Equinox layer that implements a linear layer with a circulant matrix.

    The layer stores only the first row `c` (a vector of shape (n,)).
    We define the circulant matrix C so that its first row is c and its i-th row is:

      C[i, :] = jnp.roll(c, i)

    Then, the first column is [c_0, c_{n-1}, c_{n-2}, ..., c_1].

    A well-known fact is that the eigenvalues of C are given by
      λ = fft(first_col)
    and one may compute C @ x via:

      y = real( ifft( fft(x) * fft(first_col) ) )

    Here we compute first_col by flipping c and then rolling by 1.

    Example use casae:
    class MyModel(eqx.Module):

        circulant_layer: CirculantLinear
        final_layer: eqx.nn.Linear

        # in_features and out_features are static (non-trainable) fields.
        in_features: int = eqx.static_field()
        out_features: int = eqx.static_field()

        def __init__(self, in_features: int, out_features: int, *, key):
            # We assume the hidden dimension equals the input dimension.
            self.in_features = in_features
            self.out_features = out_features
            key1, key2 = jax.random.split(key, 2)
            self.circulant_layer = CirculantLinear(in_features, key=key1, init_scale=1.0)
            self.final_layer = eqx.nn.Linear(in_features, out_features, key=key2)

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

            # Our custom layer supports batched inputs because FFT is applied along the last axis.
            h = self.circulant_layer(x)  # shape (batch, in_features)
            h = jax.nn.relu(h)
            # eqx.nn.Linear expects a single vector. We can use jax.vmap to apply it over the batch.
            y = jax.vmap(lambda z: self.final_layer(z))(h)
            return y
    """

    first_row: jnp.ndarray  # shape (n,)
    in_features: int = eqx.static_field()
    out_features: int = eqx.static_field()

    def __init__(self, in_features: int, *, key, init_scale: float = 1.0):
        # We assume a square weight matrix.
        self.in_features = in_features
        self.out_features = in_features
        self.first_row = jax.random.normal(key, (in_features,)) * init_scale

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Compute the "first column" from the first row.
        # first_row = [c0, c1, ..., c_{n-1}]
        # flip(first_row) = [c_{n-1}, c_{n-2}, ..., c0]
        # Rolling that by +1 gives: [c0, c_{n-1}, c_{n-2}, ..., c1]
        first_col = jnp.roll(jnp.flip(self.first_row), shift=1)
        # Use the custom FFT-based multiplication.
        return circulant_matmul(first_col, x)


def test_efficient_circulant_linear():
    key = jr.PRNGKey(42)
    n = 8
    layer = EfficientCirculantLinear(n, key=key, init_scale=1.0)

    key, subkey = jr.split(key)
    # Create a random input vector of shape (n,)
    x = jr.normal(subkey, (n,))

    # Compute output using our custom operator version.
    y_custom = layer(x)

    # For testing, explicitly build the circulant matrix.
    def circulant(first_row):
        return jnp.stack(
            [jnp.roll(first_row, i) for i in range(first_row.shape[0])], axis=0
        )

    first_col = jnp.roll(jnp.flip(layer.first_row), shift=1)
    C = circulant(first_col)
    y_direct = C @ x

    # Compare the two outputs, squeezing y_custom to remove the extra batch dimension.
    np.testing.assert_allclose(y_custom.squeeze(), y_direct, rtol=1e-5, atol=1e-5)
    print("Efficient CirculantLinear test passed!")
    print("Input x:\n", x)
    print("Efficient CirculantLinear output:\n", y_custom)
    print("Direct multiplication output:\n", y_direct)


if __name__ == "__main__":
    test_efficient_circulant_linear()
