import jax
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


class BlockCirculantLinear(eqx.Module):
    """
    Equinox module implementing a block-circulant weight matrix:
      - W is shape (k_out, k_in, b), each slice W[i,j] is a length-b "first row"
        of a circulant block.
      - in_features and out_features are the overall input and output dims,
        possibly padded up to multiples of b.
      - D_bernoulli is an optional diagonal +/- 1 matrix for all input dims
        (the paper calls it "Bernoulli diagonal" to reduce correlation).
    """

    W: jnp.ndarray  # shape (k_out, k_in, b)
    D_bernoulli: jnp.ndarray  # shape (d_in,), each +1 or -1
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
        init_scale=0.1,
        use_bernoulli_diag: bool = True,
    ):
        """
        :param in_features: int - dimensionality of input
        :param out_features: int - dimensionality of output
        :param block_size: int - b
        :param key: PRNG key for random init
        :param init_scale: scale factor for random normal init
        :param use_bernoulli_diag: whether to include the D_bernoulli
        """
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        # We'll define k_in, k_out as the number of blocks along each dimension.
        k_in = (in_features + block_size - 1) // block_size
        k_out = (out_features + block_size - 1) // block_size
        object.__setattr__(self, "k_in", k_in)
        object.__setattr__(self, "k_out", k_out)

        # Each block is parameterized by a vector of size b (the "first row").
        # So total shape is (k_out, k_in, b).
        k1, k2 = jr.split(key, 2)
        W_init = jr.normal(k1, (k_out, k_in, block_size)) * init_scale

        # D_bernoulli is shape (in_features,). You can sample ±1 for each dimension.
        if use_bernoulli_diag:
            diag_signs = jnp.where(
                jr.bernoulli(k2, p=0.5, shape=(in_features,)), 1.0, -1.0
            )
        else:
            diag_signs = jnp.ones((in_features,))
        object.__setattr__(self, "W", W_init)
        object.__setattr__(self, "D_bernoulli", diag_signs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass:
          - x shape can be (batch, d_in) or (d_in,) [we handle both].
          - multiply x by the diagonal Bernoulli sign matrix D if used
          - slice x into blocks of length b
          - for each block-row i, sum over block-cols j in freq domain
          - flatten or slice final result to length out_features
        """
        # (1) Possibly expand x to (batch, d_in).
        #     If x.ndim == 1, we treat as single example of shape (1, d_in).
        if x.ndim == 1:
            x = x[None, :]  # shape (1, d_in)

        batch_size = x.shape[0]
        d_in = self.in_features
        d_out = self.out_features
        b = self.block_size
        k_in = self.k_in
        k_out = self.k_out

        # (2) Apply the diagonal Bernoulli D to x
        #     x_d = D_bernoulli * x, broadcast over batch dimension
        # shape (batch, d_in)
        x_d = x * self.D_bernoulli[None, :]

        # (3) Zero-pad x_d if needed so that it has shape (batch, k_in*b).
        pad_len = k_in * b - d_in
        if pad_len > 0:
            pad_shape = ((0, 0), (0, pad_len))
            x_d = jnp.pad(x_d, pad_shape, mode="constant", constant_values=0.0)

        # (4) Reshape into blocks: (batch, k_in, b)
        x_blocks = x_d.reshape(batch_size, k_in, b)

        # We will produce an output array of shape (batch, k_out * b).
        # Then slice the first d_out columns if needed.
        # The block-circulant multiplication formula from the paper is:

        #  out[i] = sum_{j=1..k_in} circ(W[i,j]) x_blocks[j]
        #  each circ(W[i,j]) is an NxN circulant with N=b here.

        # (5) We'll do everything in the frequency domain. Summation is simpler:
        #   out_fft[i] = sum_j ( RFFT(W[i,j])^* * RFFT(x_blocks[j]) )
        #
        #   Then out[i] = IRFFT( out_fft[i] )

        def one_block_mul(w_ij, x_j):
            # w_ij: shape (b,) first row of circulant
            # x_j: shape (batch, b)
            # We do:
            #   c = fft(w_ij)  # shape (b,)
            #   X = fft(x_j, axis=-1)  # shape (batch, b)
            #   out_fft = c^* * X
            #   out_time = ifft(out_fft).real   # shape (batch, b)
            # Return out_time
            c_fft = jnp.fft.fft(w_ij)
            X_fft = jnp.fft.fft(x_j, axis=-1)
            block_fft = X_fft * jnp.conjugate(c_fft)[None, :]
            return jnp.fft.ifft(block_fft, axis=-1).real

        # We’ll accumulate out[i] for i in [0..k_out).
        # Actually we can do this with a double loop or vectorized approach.

        # shape (k_out, batch, b)
        def compute_blockrow(i):
            # sum_{j=1..k_in} circ(W[i,j]) x_blocks[:,j]
            # i.e. do the block-circulant multiplication for row i.
            # We'll build the sum in freq domain, then IRFFT once at the end.
            # Or we can do IRFFT block by block and sum in time domain.
            # Summation in time domain is also fine. We’ll do time domain for clarity.
            def sum_over_j(carry, j):
                w_ij = self.W[i, j]  # shape (b,)
                x_j = x_blocks[:, j, :]  # shape (batch, b)
                block_out = one_block_mul(w_ij, x_j)  # shape (batch, b)
                return carry + block_out, None

            init = jnp.zeros((batch_size, b))
            out_time, _ = jax.lax.scan(sum_over_j, init, jnp.arange(k_in))
            return out_time  # shape (batch, b)

        out_blocks = jax.vmap(compute_blockrow)(
            jnp.arange(k_out)
        )  # shape (k_out, batch, b)
        # We want to reorder to (batch, k_out*b).
        out_reshaped = jnp.transpose(out_blocks, (1, 0, 2)).reshape(
            batch_size, k_out * b
        )

        # (6) Finally slice to get the first d_out columns if k_out*b > d_out
        if k_out * b > d_out:
            out_reshaped = out_reshaped[:, :d_out]

        return out_reshaped  # shape (batch, d_out) if batch_size>1, else (1, d_out)


class MyBlockCirculantNet(eqx.Module):
    bc_layer: BlockCirculantLinear
    final_layer: eqx.nn.Linear

    def __init__(self, in_features, hidden_dim, *, key):
        key1, key2, key3 = jr.split(key, 3)
        self.bc_layer = BlockCirculantLinear(
            in_features=in_features,
            out_features=hidden_dim,
            block_size=16,  # choose b=16, for instance
            key=key1,
            init_scale=0.01,
            use_bernoulli_diag=True,
        )
        self.final_layer = eqx.nn.Linear(hidden_dim, 1, key=key2)

    def __call__(self, x):
        h = self.bc_layer(x)
        h = jax.nn.relu(h)
        return self.final_layer(h)


# Helper function: construct a circulant matrix from its first row.
def circulant(first_row: jnp.ndarray) -> jnp.ndarray:
    n = first_row.shape[0]
    return jnp.stack([jnp.roll(first_row, i) for i in range(n)], axis=0)


def construct_dense_block_circulant(layer: eqx.Module) -> jnp.ndarray:
    """
    Reconstruct the full dense block-circulant weight matrix from a BlockCirculantLinear instance.
    The resulting matrix will have shape (k_out * b, k_in * b).
    """
    b = layer.block_size
    k_in = layer.k_in
    k_out = layer.k_out
    # Build each circulant block and then form the full block matrix.
    block_rows = []
    for i in range(k_out):
        row_blocks = []
        for j in range(k_in):
            # Each block: circulant matrix with first row = layer.W[i,j]
            block = circulant(layer.W[i, j])
            row_blocks.append(block)
        # Concatenate blocks horizontally.
        row_concat = jnp.concatenate(row_blocks, axis=1)
        block_rows.append(row_concat)
    # Concatenate block rows vertically.
    dense_matrix = jnp.concatenate(block_rows, axis=0)
    return dense_matrix


def test_block_circulant_linear():
    key = jr.PRNGKey(42)
    # Define dimensions and block size.
    d_in = 16
    d_out = 16
    block_size = 4
    # Create instance of BlockCirculantLinear.
    layer = BlockCirculantLinear(
        in_features=d_in,
        out_features=d_out,
        block_size=block_size,
        key=key,
        init_scale=1.0,
        use_bernoulli_diag=True,
    )

    # Create a random input vector of shape (d_in,).
    key, subkey = jr.split(key)
    x = jr.normal(subkey, (d_in,))

    # Compute output using the block-circulant layer.
    y_layer = layer(x)

    # Construct the ground truth:
    # 1. Multiply x elementwise by the Bernoulli diagonal.
    x_d = x * layer.D_bernoulli
    # 2. Zero-pad x_d to length k_in * block_size.
    k_in = layer.k_in
    pad_len = k_in * block_size - d_in
    if pad_len > 0:
        x_d_padded = jnp.pad(x_d, (0, pad_len))
    else:
        x_d_padded = x_d
    # 3. Construct the full dense block-circulant matrix (shape: (k_out*b, k_in*b)).
    M = construct_dense_block_circulant(layer)
    # 4. Compute dense multiplication: y_dense = M @ x_d_padded.
    y_dense_full = M @ x_d_padded
    # 5. Slice the first d_out elements (if padded extra columns exist).
    y_direct = y_dense_full[:d_out]

    # Compare outputs.
    np.testing.assert_allclose(y_layer.squeeze(), y_direct, rtol=1e-5, atol=1e-5)
    print("BlockCirculantLinear test passed!")
    print("Input x:\n", x)
    print("Layer output:\n", y_layer)
    print("Direct multiplication output:\n", y_direct)


if __name__ == "__main__":
    test_block_circulant_linear()
