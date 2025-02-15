import jax
import numpy as np
import equinox as eqx
import jax.random as jr
import jax.numpy as jnp


@jax.custom_jvp
def block_circulant_matmul(W, X):
    """
    Computes block-circulant matrix multiplication in the frequency domain.

    Parameters:
      W: jnp.ndarray, shape (k_out, k_in, b)
         The "first rows" of the circulant blocks.
      X: jnp.ndarray, shape (batch, k_in, b)
         The input blocks.

    Returns:
      Y: jnp.ndarray, shape (batch, k_out, b)
         The result computed as:
         Y[i] = IRFFT( sum_{j=0}^{k_in-1} ( FFT(X[:,j,:]) * conj(FFT(W[i,j,:])) ) )
         for each block-row i.
    """
    # Get dimensions
    k_out, k_in, b = W.shape
    batch = X.shape[0]

    # Compute FFT over the last dimension for each block.
    X_fft = jnp.fft.fft(X, axis=-1)  # shape: (batch, k_in, b)
    W_fft = jnp.fft.fft(W, axis=-1)  # shape: (k_out, k_in, b)

    # Multiply each block: broadcast X_fft to (batch, 1, k_in, b)
    # and W_fft to (1, k_out, k_in, b), take complex conjugate of W_fft.
    result_fft = X_fft[:, None, :, :] * jnp.conjugate(W_fft)[None, :, :, :]
    # Sum over block columns (axis=2) to get shape: (batch, k_out, b)
    Y_fft = jnp.sum(result_fft, axis=2)

    # Inverse FFT to obtain the time domain result
    Y = jnp.fft.ifft(Y_fft, axis=-1).real
    return Y


@block_circulant_matmul.defjvp
def block_circulant_matmul_jvp(primals, tangents):
    W, X = primals
    dW, dX = tangents  # Same shapes as W and X

    # Compute the primal forward pass.
    Y = block_circulant_matmul(W, X)

    # Compute FFTs of the primal variables.
    X_fft = jnp.fft.fft(X, axis=-1)  # shape: (batch, k_in, b)
    W_fft = jnp.fft.fft(W, axis=-1)  # shape: (k_out, k_in, b)

    # Compute FFTs of the tangent variables.
    dX_fft = jnp.fft.fft(dX, axis=-1) if dX is not None else 0.0
    dW_fft = jnp.fft.fft(dW, axis=-1) if dW is not None else 0.0

    # Compute the tangent in the frequency domain.
    # For each block, the derivative is:
    #   tangent_fft = sum_{j=0}^{k_in-1} [ X_fft * conj(dW_fft) + dX_fft * conj(W_fft) ]
    term1 = (
        X_fft[:, None, :, :] * jnp.conjugate(dW_fft)[None, :, :, :]
    )  # shape: (batch, k_out, k_in, b)
    term2 = (
        dX_fft[:, None, :, :] * jnp.conjugate(W_fft)[None, :, :, :]
    )  # shape: (batch, k_out, k_in, b)
    tangent_fft = jnp.sum(term1 + term2, axis=2)  # shape: (batch, k_out, b)

    # Inverse FFT to get the tangent in time domain.
    tangent_Y = jnp.fft.ifft(tangent_fft, axis=-1).real

    return Y, tangent_Y


class EfficientBlockCirculantLinear(eqx.Module):
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
        Forward pass for BlockCirculantLinear using the custom block_circulant_matmul.
        """
        # (1) Ensure x has shape (batch, d_in)
        if x.ndim == 1:
            x = x[None, :]
        batch_size = x.shape[0]
        d_in = self.in_features
        d_out = self.out_features
        b = self.block_size
        k_in = self.k_in
        k_out = self.k_out

        # (2) Apply Bernoulli diagonal D
        x_d = x * self.D_bernoulli[None, :]

        # (3) Zero-pad x_d to length k_in * b if necessary.
        pad_len = k_in * b - d_in
        if pad_len > 0:
            x_d = jnp.pad(
                x_d, ((0, 0), (0, pad_len)), mode="constant", constant_values=0.0
            )

        # (4) Reshape into blocks: (batch, k_in, b)
        x_blocks = x_d.reshape(batch_size, k_in, b)

        # (5) Use our custom block_circulant_matmul to compute output in blocks.
        # The output has shape (batch, k_out, b)
        out_blocks = block_circulant_matmul(self.W, x_blocks)

        # (6) Flatten blocks to get shape (batch, k_out * b) and slice to d_out.
        out_flat = out_blocks.reshape(batch_size, k_out * b)
        if k_out * b > d_out:
            out_flat = out_flat[:, :d_out]

        return out_flat


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
    layer = EfficientBlockCirculantLinear(
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
