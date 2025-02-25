import jax
import jax.numpy as jnp
import equinox as eqx
import jax.random as jr

__all__ = [
    "Circulant",
    "BlockCirculant"
]

@jax.custom_jvp
def circulant_matmul(x: jnp.ndarray, first_row: jnp.ndarray) -> jnp.ndarray:
    """
    Compute y = C x, where C is a circulant matrix defined by its first row.
    Instead of forming C explicitly, we compute via FFT:
      - first, compute first_col = roll(flip(first_row), shift=1)
      - then, y = IFFT( FFT(x) * FFT(first_col) )
    """
    # Compute first_col from first_row
    first_col = jnp.roll(jnp.flip(first_row), shift=1)
    fft_first_col = jnp.fft.fft(first_col)
    fft_x = jnp.fft.fft(x, axis=-1)
    y = jnp.fft.ifft(fft_x * fft_first_col, axis=-1).real
    return y

@circulant_matmul.defjvp
def circulant_matmul_jvp(primals, tangents):
    x, first_row = primals
    dx, dfirst_row = tangents
    # Forward pass (same as in circulant_matmul)
    first_col = jnp.roll(jnp.flip(first_row), shift=1)
    fft_first_col = jnp.fft.fft(first_col)
    fft_x = jnp.fft.fft(x, axis=-1)
    y = jnp.fft.ifft(fft_x * fft_first_col, axis=-1).real
    # Tangent contribution from x:
    dfft_x = jnp.fft.fft(dx, axis=-1)
    dy_dx = jnp.fft.ifft(dfft_x * fft_first_col, axis=-1).real
    # Tangent contribution from first_row:
    dfirst_col = jnp.roll(jnp.flip(dfirst_row), shift=1)
    dfft_first_col = jnp.fft.fft(dfirst_col)
    dy_df = jnp.fft.ifft(fft_x * dfft_first_col, axis=-1).real
    return y, dy_dx + dy_df


@jax.custom_jvp
def block_circulant_matmul(W: jnp.ndarray, x: jnp.ndarray, d_bernoulli: jnp.ndarray) -> jnp.ndarray:
    """
    Compute block-circulant multiplication.

    W: shape (k_out, k_in, b) where each row is the first row of a circulant block.
    x: input array of shape (batch, d_in) or (d_in,). If necessary, x is zero-padded
       to length k_in * b.
    d_bernoulli: optional diagonal of shape (d_in,) (applied elementwise if given).

    Returns: y, shape (batch, k_out * b).
    """
    if x.ndim == 1:
        x = x[None, :]
    batch_size, d_in = x.shape
    k_out, k_in, b = W.shape
    # Apply d_bernoulli if provided.
    if d_bernoulli is not None:
        x = x * d_bernoulli[None, :]
    # Zero-pad if needed.
    pad_len = k_in * b - d_in
    if pad_len > 0:
        x = jnp.pad(x, ((0, 0), (0, pad_len)))
    # Reshape into blocks.
    x_blocks = x.reshape(batch_size, k_in, b)
    def one_block_mul(w_ij, x_j):
        c_fft = jnp.fft.fft(w_ij)  # (b,)
        X_fft = jnp.fft.fft(x_j, axis=-1)  # (batch, b)
        return jnp.fft.ifft(X_fft * jnp.conjugate(c_fft)[None, :], axis=-1).real
    def compute_blockrow(i):
        def sum_over_j(carry, j):
            w_ij = W[i, j, :]  # (b,)
            x_j = x_blocks[:, j, :]  # (batch, b)
            return carry + one_block_mul(w_ij, x_j), None
        init = jnp.zeros((batch_size, b))
        out_time, _ = jax.lax.scan(sum_over_j, init, jnp.arange(k_in))
        return out_time  # (batch, b)
    out_blocks = jax.vmap(compute_blockrow)(jnp.arange(k_out))  # (k_out, batch, b)
    out_reshaped = jnp.transpose(out_blocks, (1, 0, 2)).reshape(batch_size, k_out * b)
    return out_reshaped

@block_circulant_matmul.defjvp
def block_circulant_matmul_jvp(primals, tangents):
    W, x, d_bernoulli = primals
    dW, dx, dd = tangents
    y = block_circulant_matmul(W, x, d_bernoulli)
    # Derivative with respect to x:
    dy_dx = block_circulant_matmul(W, dx, d_bernoulli)
    # Derivative with respect to W:
    dy_dW = block_circulant_matmul(dW, x, d_bernoulli)
    # For simplicity, we ignore the derivative with respect to d_bernoulli.
    return y, dy_dx + dy_dW


class JVPCirculant(eqx.Module):
    """
    A circulant layer that uses a circulant weight matrix defined by its first row.
    The layer stores only the first row (a vector of shape (n,)) and a bias vector (shape (n,)).
    The forward pass computes:
        y = circulant_matmul(x, first_row) + bias
    where circulant_matmul is accelerated via a custom JVP rule.
    """
    first_row: jnp.ndarray  # shape (n,)
    bias: jnp.ndarray       # shape (n,)
    in_features: int = eqx.static_field()
    out_features: int = eqx.static_field()

    def __init__(self, in_features: int, *, key, init_scale: float = 1.0):
        self.in_features = in_features
        self.out_features = in_features  # circulant matrices are square
        key1, key2 = jr.split(key)
        self.first_row = jr.normal(key1, (in_features,)) * init_scale
        self.bias = jr.normal(key2, (in_features,)) * init_scale

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x can have any batch shape ending with in_features.
        y = circulant_matmul(x, self.first_row)
        return y + self.bias



class JVPBlockCirculant(eqx.Module):
    """
    A block-circulant layer that uses a block-circulant weight matrix and a bias vector.

    - W is of shape (k_out, k_in, b), where each slice W[i, j] is the first row of a circulant block.
    - D_bernoulli is an optional diagonal (shape (in_features,)) with ±1 entries.
    - A bias vector of shape (out_features,) is added to the output.
    - The forward pass calls block_circulant_matmul, which has a custom JVP rule.
    """
    W: jnp.ndarray             # shape: (k_out, k_in, b)
    D_bernoulli: jnp.ndarray   # shape: (in_features,)
    bias: jnp.ndarray          # shape: (out_features,)
    in_features: int = eqx.static_field()
    out_features: int = eqx.static_field()
    block_size: int = eqx.static_field()
    k_in: int = eqx.static_field()
    k_out: int = eqx.static_field()

    def __init__(self, in_features: int, out_features: int, block_size: int, *, key, init_scale: float = 0.1, use_bernoulli_diag: bool = True, use_bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        k_in = (in_features + block_size - 1) // block_size
        k_out = (out_features + block_size - 1) // block_size
        object.__setattr__(self, "k_in", k_in)
        object.__setattr__(self, "k_out", k_out)
        k1, k2, k3 = jr.split(key, 3)
        self.W = jr.normal(k1, (k_out, k_in, block_size)) * init_scale
        if use_bernoulli_diag:
            self.D_bernoulli = jnp.where(jr.bernoulli(k2, p=0.5, shape=(in_features,)), 1.0, -1.0)
        else:
            self.D_bernoulli = jnp.ones((in_features,))
        if use_bias:
            self.bias = jr.normal(k3, (out_features,)) * init_scale
        else:
            self.bias = jnp.zeros((out_features,))

    def __call__(self, x: jnp.ndarray, *, key=None, state=None, **kwargs) -> jnp.ndarray:
        single_example = (x.ndim == 1)
        if single_example:
            x = x[None, :]
        y = block_circulant_matmul(self.W, x, self.D_bernoulli)
        y = y + self.bias[None, :]
        if single_example:
            y = y[0]
        return y
