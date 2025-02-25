import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import transforms

__all__ = [
    "JVPCirculant",
    "JVPBlockCirculant"
]

@jax.custom_jvp
def fft_matmul_custom(first_row: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    """
    Performs circulant matrix multiplication via FFT.
    Given the first row of a circulant matrix and an input X,
    computes:
        result = ifft( fft(first_row) * fft(X) ).real
    """
    # Compute FFT of the circulant's defining vector and of the input.
    first_row_fft = jnp.fft.fft(first_row, axis=-1)
    X_fft = jnp.fft.fft(X, axis=-1)
    # Multiply (with broadcasting) in Fourier domain.
    result_fft = first_row_fft[None, :] * X_fft
    result = jnp.fft.ifft(result_fft, axis=-1).real
    return result

@fft_matmul_custom.defjvp
def fft_matmul_custom_jvp(primals, tangents):
    first_row, X = primals
    d_first_row, dX = tangents

    # Recompute FFTs from the primal inputs (to avoid extra FFT calls in reverse mode).
    first_row_fft = jnp.fft.fft(first_row, axis=-1)
    X_fft = jnp.fft.fft(X, axis=-1)
    primal_out = jnp.fft.ifft(first_row_fft[None, :] * X_fft, axis=-1).real

    # Compute the directional derivatives.
    d_first_row_fft = jnp.fft.fft(d_first_row, axis=-1) if d_first_row is not None else 0.
    dX_fft = jnp.fft.fft(dX, axis=-1) if dX is not None else 0.
    tangent_out = jnp.fft.ifft(d_first_row_fft[None, :] * X_fft +
                               first_row_fft[None, :] * dX_fft,
                               axis=-1).real
    return primal_out, tangent_out


@jax.custom_jvp
def block_circulant_matmul_custom(W: jnp.ndarray, x: jnp.ndarray, d_bernoulli: jnp.ndarray = None) -> jnp.ndarray:
    """
    Performs block–circulant matrix multiplication via FFT.

    Parameters:
      W: shape (k_out, k_in, b) – each W[i,j,:] is the first row of a b×b circulant block.
      x: shape (batch, d_in) or (d_in,)
      d_bernoulli: optional diagonal (of ±1) to decorrelate projections.

    Returns:
      Output of shape (batch, k_out * b) (sliced to d_out if needed).

    The forward pass computes:
      1. Optionally scales x by the fixed Bernoulli diagonal.
      2. Pads x to length k_in*b and reshapes it to (batch, k_in, b).
      3. Computes W_fft = fft(W, axis=-1) and X_fft = fft(x_blocks, axis=-1).
      4. For each block row, sums over j:
            Y_fft[:, i, :] = sum_j X_fft[:, j, :] * conj(W_fft[i, j, :])
      5. Returns Y = ifft(Y_fft).real reshaped to (batch, k_out*b).
    """
    # Ensure x is 2D.
    if x.ndim == 1:
        x = x[None, :]
    batch_size, d_in = x.shape
    k_out, k_in, b = W.shape
    d_out = k_out * b

    # Optionally apply the Bernoulli diagonal.
    if d_bernoulli is not None:
        x = x * d_bernoulli[None, :]

    # Zero-pad x if needed.
    pad_len = k_in * b - d_in
    if pad_len > 0:
        x = jnp.pad(x, ((0, 0), (0, pad_len)))
    # Reshape x into blocks.
    x_blocks = x.reshape(batch_size, k_in, b)

    # Compute FFTs.
    W_fft = jnp.fft.fft(W, axis=-1)           # shape: (k_out, k_in, b)
    X_fft = jnp.fft.fft(x_blocks, axis=-1)      # shape: (batch, k_in, b)

    # Multiply in Fourier domain and sum over the input blocks.
    # For each output block row i:
    #   Y_fft[:, i, :] = sum_j X_fft[:, j, :] * conj(W_fft[i, j, :])
    Y_fft = jnp.sum(X_fft[:, None, :, :] * jnp.conjugate(W_fft)[None, :, :, :], axis=2)
    Y = jnp.fft.ifft(Y_fft, axis=-1).real
    out = Y.reshape(batch_size, k_out * b)
    return out

@block_circulant_matmul_custom.defjvp
def block_circulant_matmul_custom_jvp(primals, tangents):
    W, x, d_bernoulli = primals
    dW, dx, dd = tangents  # dd is the tangent for d_bernoulli (ignored here for simplicity)

    if x.ndim == 1:
        x = x[None, :]
    batch_size, d_in = x.shape
    k_out, k_in, b = W.shape

    # Forward pass (as above).
    if d_bernoulli is not None:
        x_eff = x * d_bernoulli[None, :]
    else:
        x_eff = x
    pad_len = k_in * b - d_in
    if pad_len > 0:
        x_eff = jnp.pad(x_eff, ((0, 0), (0, pad_len)))
    x_blocks = x_eff.reshape(batch_size, k_in, b)
    W_fft = jnp.fft.fft(W, axis=-1)
    X_fft = jnp.fft.fft(x_blocks, axis=-1)
    Y_fft = jnp.sum(X_fft[:, None, :, :] * jnp.conjugate(W_fft)[None, :, :, :], axis=2)
    primal_out = jnp.fft.ifft(Y_fft, axis=-1).real.reshape(batch_size, k_out * b)

    # Compute tangent for x.
    if dx is None:
        dx_eff = 0.
    else:
        if d_bernoulli is not None:
            dx_eff = dx * d_bernoulli[None, :]
        else:
            dx_eff = dx
    if dx_eff is not None:
        if pad_len > 0:
            dx_eff = jnp.pad(dx_eff, ((0, 0), (0, pad_len)))
        x_tangent_blocks = dx_eff.reshape(batch_size, k_in, b)
        X_tangent_fft = jnp.fft.fft(x_tangent_blocks, axis=-1)
    else:
        X_tangent_fft = 0.

    # Compute tangent for W.
    if dW is not None:
        W_tangent_fft = jnp.fft.fft(dW, axis=-1)
    else:
        W_tangent_fft = 0.

    # The directional derivative in the Fourier domain is:
    # dY_fft = sum_j [ X_tangent_fft[:, None, j, :] * conj(W_fft)[None, :, j, :] +
    #                  X_fft[:, None, j, :] * conj(W_tangent_fft)[None, :, j, :] ]
    dY_fft = jnp.sum(
        X_tangent_fft[:, None, :, :] * jnp.conjugate(W_fft)[None, :, :, :] +
        X_fft[:, None, :, :] * jnp.conjugate(W_tangent_fft)[None, :, :, :],
        axis=2)
    tangent_out = jnp.fft.ifft(dY_fft, axis=-1).real.reshape(batch_size, k_out * b)
    return primal_out, tangent_out


class JVPCirculant:
    """
    FFT–based circulant layer that uses a custom JVP rule for faster gradients.
    The forward pass computes:
        hidden = ifft( fft(first_row) * fft(X) ).real + bias
    and the JVP uses the saved FFT computations.
    """
    def __init__(
        self,
        in_features: int,
        name: str = "fft_layer",
        first_row_prior_fn=lambda shape: dist.Normal(0, 1).expand(shape).to_event(len(shape)),
        bias_prior_fn=lambda shape: dist.Normal(0, 1).expand(shape).to_event(1),
    ):
        self.in_features = in_features
        self.name = name
        self.first_row_prior_fn = first_row_prior_fn
        self.bias_prior_fn = bias_prior_fn

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        first_row = numpyro.sample(
            f"{self.name}_first_row", self.first_row_prior_fn([self.in_features])
        )
        bias_circulant = numpyro.sample(
            f"{self.name}_bias_circulant", self.bias_prior_fn([self.in_features])
        )
        hidden = fft_matmul_custom(first_row, X) + bias_circulant[None, :]
        return hidden

class JVPBlockCirculant:
    """
    Block–circulant layer with custom JVP rules.

    This layer:
      1. Samples W of shape (k_out, k_in, block_size) (each block is circulant).
      2. Optionally samples a Bernoulli diagonal for the input.
      3. Computes the forward pass via the custom FFT–based block–circulant matmul.
      4. Uses the custom JVP for faster gradient computation.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int,
        name: str = "block_circ_layer",
        W_prior_fn=lambda shape: dist.Normal(0, 1).expand(shape).to_event(len(shape)),
        use_diag: bool = True,
        bias_prior_fn=lambda shape: dist.Normal(0, 1).expand(shape).to_event(1),
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.name = name

        # Determine block counts along each dimension.
        self.k_in = (in_features + block_size - 1) // block_size
        self.k_out = (out_features + block_size - 1) // block_size

        self.W_prior_fn = W_prior_fn
        self.use_diag = use_diag
        self.bias_prior_fn = bias_prior_fn
        self.diag_prior = lambda shape: dist.TransformedDistribution(
            dist.Bernoulli(0.5).expand(shape).to_event(len(shape)),
            [transforms.AffineTransform(loc=-1.0, scale=2.0)]
        )

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        # Sample block–circulant weights.
        W = numpyro.sample(
            f"{self.name}_W",
            self.W_prior_fn([self.k_out, self.k_in, self.block_size]),
        )

        # Optionally sample and apply the Bernoulli diagonal.
        if self.use_diag:
            d_bernoulli = numpyro.sample(
                f"{self.name}_D",
                self.diag_prior([self.in_features]),
            )
        else:
            d_bernoulli = None

        # Compute the block–circulant multiplication via the custom JVP function.
        out = block_circulant_matmul_custom(W, X, d_bernoulli)

        # Sample and add bias.
        b = numpyro.sample(
            f"{self.name}_bias",
            self.bias_prior_fn([self.out_features]),
        )
        out = out + b[None, :]

        # If the padded output dimension is larger than out_features, slice the result.
        if self.k_out * self.block_size > self.out_features:
            out = out[:, : self.out_features]

        return out
