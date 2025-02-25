import jax
import jax.numpy as jnp
import equinox as eqx
import jax.random as jr

__all__ = ["Circulant", "BlockCirculant"]


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


class JVPCirculant(eqx.Module):
    """
    A circulant layer that uses a circulant weight matrix defined by its first row.
    The layer stores only the first row (a vector of shape (n,)) and a bias vector (shape (n,)).
    The forward pass computes:
        y = circulant_matmul(x, first_row) + bias
    where circulant_matmul is accelerated via a custom JVP rule.
    """

    first_row: jnp.ndarray  # shape (n,)
    bias: jnp.ndarray  # shape (n,)
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


@jax.custom_jvp
def block_circulant_matmul_custom(
    W: jnp.ndarray, x: jnp.ndarray, d_bernoulli: jnp.ndarray
):
    """
    Compute the block-circulant matrix multiplication:
       out = B x
    where B is defined via blocks of circulant matrices.

    - W has shape (k_out, k_in, b): each row W[i, j, :] is the first row of a b×b circulant block.
    - x is the input, of shape (batch, in_features) or (in_features,).
    - d_bernoulli is an optional diagonal (of shape (in_features,)) of ±1 entries.

    This function performs the following:
      1. (Optionally) multiplies x elementwise by d_bernoulli.
      2. Zero-pads x so that its length equals k_in * b.
      3. Reshapes x into (batch, k_in, b).
      4. For each output block i, sums over j the circulant multiplication via FFT:
           FFT(x_block) * conj(FFT(W[i,j]))  → summed over j, then inverse FFT.
    """
    # Ensure x has a batch dimension.
    single_example = x.ndim == 1
    if single_example:
        x = x[None, :]
    batch_size = x.shape[0]
    d_in = x.shape[-1]
    k_in = W.shape[1]
    b = W.shape[-1]

    # (1) Multiply by Bernoulli diagonal if provided.
    if d_bernoulli is not None:
        x_d = x * d_bernoulli[None, :]
    else:
        x_d = x

    # (2) Zero-pad x_d to length k_in * b.
    pad_len = k_in * b - d_in
    if pad_len > 0:
        x_d = jnp.pad(x_d, ((0, 0), (0, pad_len)), mode="constant", constant_values=0.0)

    # (3) Reshape into blocks: (batch, k_in, b)
    X_blocks = x_d.reshape(batch_size, k_in, b)
    # Compute FFT along the block dimension.
    X_fft = jnp.fft.fft(X_blocks, axis=-1)  # shape: (batch, k_in, b)

    # (4) Compute FFT for the circulant blocks in W.
    W_fft = jnp.fft.fft(W, axis=-1)  # shape: (k_out, k_in, b)

    # (5) For each output block row, sum over input blocks.
    def compute_block_row(i):
        # Multiply: for each input block j,
        #   multiply X_fft[:, j, :] with conj(W_fft[i, j, :])
        prod = X_fft * jnp.conjugate(W_fft[i, :, :])[None, :, :]  # (batch, k_in, b)
        sum_over_j = jnp.sum(prod, axis=1)  # (batch, b)
        # Inverse FFT to get the circulant product (real-valued).
        return jnp.fft.ifft(sum_over_j, axis=-1).real  # (batch, b)

    # Compute for all block rows (vmap over i=0,...,k_out-1).
    block_out = jax.vmap(compute_block_row)(jnp.arange(W.shape[0]))  # (k_out, batch, b)
    # Reshape: transpose to (batch, k_out, b) then flatten last two dims.
    out = jnp.transpose(block_out, (1, 0, 2)).reshape(batch_size, W.shape[0] * b)
    if single_example:
        out = out[0]
    return out


@block_circulant_matmul_custom.defjvp
def block_circulant_matmul_jvp(primals, tangents):
    W, x, d_bernoulli = primals
    dW, dx, dd = tangents  # dd corresponds to the tangent for d_bernoulli
    # ----- Forward Pass -----
    single_example = x.ndim == 1
    if single_example:
        x = x[None, :]
    batch_size = x.shape[0]
    d_in = x.shape[-1]
    k_in = W.shape[1]
    b = W.shape[-1]

    if d_bernoulli is not None:
        x_d = x * d_bernoulli[None, :]
    else:
        x_d = x

    pad_len = k_in * b - d_in
    if pad_len > 0:
        x_d = jnp.pad(x_d, ((0, 0), (0, pad_len)), mode="constant", constant_values=0.0)
    X_blocks = x_d.reshape(batch_size, k_in, b)
    X_fft = jnp.fft.fft(X_blocks, axis=-1)  # (batch, k_in, b)
    W_fft = jnp.fft.fft(W, axis=-1)  # (k_out, k_in, b)

    def compute_block_row(i):
        prod = X_fft * jnp.conjugate(W_fft[i, :, :])[None, :, :]
        sum_over_j = jnp.sum(prod, axis=1)
        return jnp.fft.ifft(sum_over_j, axis=-1).real  # (batch, b)

    block_out = jax.vmap(compute_block_row)(jnp.arange(W.shape[0]))
    out = jnp.transpose(block_out, (1, 0, 2)).reshape(batch_size, W.shape[0] * b)
    if single_example:
        out = out[0]

    # ----- Tangent (JVP) Computation -----
    # First, differentiate through the input multiplication by d_bernoulli.
    if d_bernoulli is not None:
        # d(x_d) = (dx * d_bernoulli) + (x * dd)
        dx_d = dx * d_bernoulli[None, :] + (x * dd[None, :] if dd is not None else 0.0)
    else:
        dx_d = dx

    if pad_len > 0:
        dx_d = jnp.pad(
            dx_d, ((0, 0), (0, pad_len)), mode="constant", constant_values=0.0
        )
    dX_blocks = dx_d.reshape(batch_size, k_in, b)
    dX_fft = jnp.fft.fft(dX_blocks, axis=-1)  # (batch, k_in, b)

    # For dW, if provided compute its FFT; otherwise, treat as zero.
    if dW is not None:
        dW_fft = jnp.fft.fft(dW, axis=-1)  # (k_out, k_in, b)
    else:
        dW_fft = 0.0

    # For each output block row, the tangent contribution is given by:
    #   ifft( sum_j ( dX_fft[:, j, :] * conj(W_fft[i, j, :])
    #                + X_fft[:, j, :] * conj(dW_fft[i, j, :]) ) )
    def compute_block_row_tangent(i):
        term1 = dX_fft * jnp.conjugate(W_fft[i, :, :])[None, :, :]
        term2 = X_fft * (
            jnp.conjugate(dW_fft[i, :, :])[None, :, :] if dW is not None else 0.0
        )
        sum_over_j = jnp.sum(term1 + term2, axis=1)
        return jnp.fft.ifft(sum_over_j, axis=-1).real  # (batch, b)

    dblock_out = jax.vmap(compute_block_row_tangent)(jnp.arange(W.shape[0]))
    d_out = jnp.transpose(dblock_out, (1, 0, 2)).reshape(batch_size, W.shape[0] * b)
    if single_example:
        d_out = d_out[0]
    return out, d_out


class JVPBlockCirculant(eqx.Module):
    """
    Equinox module implementing a block-circulant layer that uses a custom JVP rule.

    Parameters:
      - W has shape (k_out, k_in, b), where each W[i,j,:] is the first row of a circulant block.
      - D_bernoulli is an optional diagonal of ±1 entries (shape: (in_features,)).
      - bias is added after the block-circulant multiplication.
      - in_features and out_features are the overall dimensions (they may be padded up to a multiple of b).

    The forward pass computes:
         out = block_circulant_matmul_custom(W, x, D_bernoulli) + bias
    and the custom JVP rule reuses FFT computations to accelerate gradient evaluation.
    """

    W: jnp.ndarray  # shape: (k_out, k_in, b)
    D_bernoulli: jnp.ndarray  # shape: (in_features,)
    bias: jnp.ndarray  # shape: (out_features,)
    in_features: int = eqx.static_field()
    out_features: int = eqx.static_field()
    block_size: int = eqx.static_field()
    k_in: int = eqx.static_field()
    k_out: int = eqx.static_field()

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int,
        *,
        key,
        init_scale: float = 0.1,
        use_bernoulli_diag: bool = True,
        use_bias: bool = True,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        # Determine the number of blocks along the input and output dimensions.
        k_in = (in_features + block_size - 1) // block_size
        k_out = (out_features + block_size - 1) // block_size
        object.__setattr__(self, "k_in", k_in)
        object.__setattr__(self, "k_out", k_out)

        # Initialize W with shape (k_out, k_in, block_size)
        k1, k2, k3 = jr.split(key, 3)
        self.W = jr.normal(k1, (k_out, k_in, block_size)) * init_scale

        # Initialize the Bernoulli diagonal if enabled.
        if use_bernoulli_diag:
            diag_signs = jnp.where(
                jr.bernoulli(k2, p=0.5, shape=(in_features,)), 1.0, -1.0
            )
        else:
            diag_signs = jnp.ones((in_features,))
        object.__setattr__(self, "D_bernoulli", diag_signs)

        # Initialize bias if requested.
        if use_bias:
            self.bias = jr.normal(k3, (out_features,)) * init_scale
        else:
            self.bias = jnp.zeros((out_features,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Compute block-circulant multiplication using the custom JVP rule.
        out = block_circulant_matmul_custom(self.W, x, self.D_bernoulli)
        # (Optional) slice the output if the padded output dimension is larger than out_features.
        k_out_b = self.k_out * self.block_size
        if k_out_b > self.out_features:
            out = out[..., : self.out_features]
        # Add bias.
        return out + self.bias[None, :] if out.ndim > 1 else out + self.bias
