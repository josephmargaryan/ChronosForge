import jax
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

class BlockCirculant(eqx.Module):
    """
    Equinox module implementing a block-circulant weight matrix with bias.

    - W is of shape (k_out, k_in, b), where each slice W[i, j] is the "first row" 
      of a circulant block.
    - D_bernoulli is an optional diagonal (of shape (in_features,)) with ±1 entries.
    - A bias vector of shape (out_features,) is added to the final output.
    - in_features and out_features are the overall input and output dimensions,
      possibly padded up to multiples of b.
    """
    W: jnp.ndarray             # shape: (k_out, k_in, b)
    D_bernoulli: jnp.ndarray   # shape: (in_features,)
    bias: jnp.ndarray          # shape: (out_features,)
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
        """
        Initialize the BlockCirculant module.

        Parameters:
            in_features: Overall input dimension.
            out_features: Overall output dimension.
            block_size: The circulant block size, b.
            key: PRNG key for initialization.
            init_scale: Scaling factor for normal initialization.
            use_bernoulli_diag: If True, include the Bernoulli diagonal.
            use_bias: If True, include a bias parameter of shape (out_features,).
        """
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        # Determine k_in and k_out, the number of blocks along the input and output.
        k_in = (in_features + block_size - 1) // block_size
        k_out = (out_features + block_size - 1) // block_size
        object.__setattr__(self, "k_in", k_in)
        object.__setattr__(self, "k_out", k_out)

        # Initialize W with shape (k_out, k_in, block_size)
        k1, k2, k3 = jr.split(key, 3)
        W_init = jr.normal(k1, (k_out, k_in, block_size)) * init_scale

        # Bernoulli diagonal for input, shape (in_features,)
        if use_bernoulli_diag:
            diag_signs = jnp.where(
                jr.bernoulli(k2, p=0.5, shape=(in_features,)), 1.0, -1.0
            )
        else:
            diag_signs = jnp.ones((in_features,))
            
        # Initialize bias if requested, shape (out_features,)
        if use_bias:
            bias_init = jr.normal(k3, (out_features,)) * init_scale
        else:
            bias_init = jnp.zeros((out_features,))

        object.__setattr__(self, "W", W_init)
        object.__setattr__(self, "D_bernoulli", diag_signs)
        object.__setattr__(self, "bias", bias_init)

    def __call__(self, x: jnp.ndarray, *, key=None, state=None, **kwargs) -> jnp.ndarray:
        """
        Forward pass for block-circulant multiplication with bias.

        1. If x is a single sample, add a batch dimension.
        2. Multiply x elementwise by the Bernoulli diagonal.
        3. Zero-pad x (if needed) so that its length equals k_in * block_size.
        4. Reshape x into blocks of shape (batch, k_in, block_size).
        5. For each block-row i, sum over the circulant multiplications across the k_in blocks.
        6. Reshape the result to (batch, k_out * block_size) and slice to (batch, out_features).
        7. Add the bias.
        """
        single_example = (x.ndim == 1)
        if single_example:
            x = x[None, :]  # add batch dimension

        batch_size = x.shape[0]
        d_in = self.in_features
        d_out = self.out_features
        b = self.block_size
        k_in = self.k_in
        k_out = self.k_out

        # (2) Apply the Bernoulli diagonal.
        x_d = x * self.D_bernoulli[None, :]

        # (3) Zero-pad x_d if necessary.
        pad_len = k_in * b - d_in
        if pad_len > 0:
            x_d = jnp.pad(x_d, ((0, 0), (0, pad_len)), mode="constant", constant_values=0.0)

        # (4) Reshape into blocks.
        x_blocks = x_d.reshape(batch_size, k_in, b)

        # (5) Perform block-circulant multiplication via FFT.
        def one_block_mul(w_ij, x_j):
            # w_ij: (b,) - first row of a circulant block.
            # x_j: (batch, b) - corresponding block from x.
            c_fft = jnp.fft.fft(w_ij)  # shape (b,)
            X_fft = jnp.fft.fft(x_j, axis=-1)  # shape (batch, b)
            block_fft = X_fft * jnp.conjugate(c_fft)[None, :]
            return jnp.fft.ifft(block_fft, axis=-1).real  # shape (batch, b)

        def compute_blockrow(i):
            # For block-row i, sum over j=0,...,k_in-1.
            def sum_over_j(carry, j):
                w_ij = self.W[i, j, :]  # (b,)
                x_j = x_blocks[:, j, :]  # (batch, b)
                block_out = one_block_mul(w_ij, x_j)
                return carry + block_out, None
            init = jnp.zeros((batch_size, b))
            out_time, _ = jax.lax.scan(sum_over_j, init, jnp.arange(k_in))
            return out_time  # shape (batch, b)

        out_blocks = jax.vmap(compute_blockrow)(jnp.arange(k_out))  # shape (k_out, batch, b)
        out_reshaped = jnp.transpose(out_blocks, (1, 0, 2)).reshape(batch_size, k_out * b)

        # (6) Slice if needed.
        if k_out * b > d_out:
            out_reshaped = out_reshaped[:, :d_out]

        # (7) Add bias.
        out_final = out_reshaped + self.bias[None, :]

        if single_example:
            out_final = out_final[0]
        return out_final