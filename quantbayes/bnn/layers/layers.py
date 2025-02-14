import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

__all__ = [
    "Linear",
    "FFTLinear",
    "FFTDirectPriorLinear",
    "BlockCirculantLayer",
    "BlockFFTDirectPriorLayer",
    "ParticleLinear",
    "FFTParticleLinear",
    "Conv1d",
    "Conv2d",
    "FFTConv1d",
    "FFTConv2d",
    "TransposedConv2d",
    "FFTTransposedConv2d",
    "MaxPool2d",
    "SelfAttention",
    "MultiHeadSelfAttention",
    "PositionalEncoding",
    "TransformerEncoder",
    "LayerNorm",
    "LSTM",
    "GaussianProcessLayer",
    "VariationalLayer",
]


class LayerNorm:
    def __init__(self, num_features, name="layer_norm"):
        self.num_features = num_features
        self.name = name

    def __call__(self, X):
        mean = jnp.mean(X, axis=-1, keepdims=True)
        variance = jnp.var(X, axis=-1, keepdims=True)
        epsilon = 1e-5  # Small constant for numerical stability
        scale = numpyro.sample(
            f"{self.name}_scale", dist.Normal(1.0, 0.1).expand([self.num_features])
        )
        shift = numpyro.sample(
            f"{self.name}_shift", dist.Normal(0.0, 0.1).expand([self.num_features])
        )
        normalized = (X - mean) / jnp.sqrt(variance + epsilon)
        return scale * normalized + shift


class MaxPool2d:
    """
    A 2D max-pooling layer.
    """

    def __init__(self, kernel_size=2, stride=2, name="maxpool2d"):
        """
        :param kernel_size: int
            Size of the pooling kernel.
        :param stride: int
            Stride for the pooling.
        :param name: str
            Name of the layer (not strictly used in your Bayesian parameter naming,
            but good for clarity).
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.name = name

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Perform max pooling on the input.

        :param X: jnp.ndarray
            Input tensor of shape `(batch_size, channels, height, width)`.

        :return: jnp.ndarray
            Pooled output tensor of shape `(batch_size, channels, pooled_height, pooled_width)`.
        """
        # Reduce window applies max pooling over the spatial dimensions (height, width).
        return jax.lax.reduce_window(
            X,
            init_value=-jnp.inf,
            computation=jax.lax.max,
            window_dimensions=(1, 1, self.kernel_size, self.kernel_size),
            window_strides=(1, 1, self.stride, self.stride),
            padding="VALID",  # For typical UNet, "VALID" pooling is common
        )


class Linear:
    """
    A fully connected layer with weights and biases sampled from specified distributions.

    Transforms inputs via a linear operation: `output = X @ weights + biases`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        name: str = "layer",
        weight_prior_fn=lambda shape: dist.Normal(0, 1).expand(shape),
        bias_prior_fn=lambda shape: dist.Normal(0, 1).expand(shape),
    ):
        """
        Initializes the Linear layer.

        :param in_features: int
            Number of input features.
        :param out_features: int
            Number of output features.
        :param name: str
            Name of the layer for parameter tracking (default: "layer").
        :param weight_prior_fn: function
            A function that takes a shape and returns a NumPyro distribution for the weights.
        :param bias_prior_fn: function
            A function that takes a shape and returns a NumPyro distribution for the biases.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.name = name
        self.weight_prior_fn = weight_prior_fn
        self.bias_prior_fn = bias_prior_fn

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Performs the linear transformation on the input.

        :param X: jnp.ndarray
            Input array of shape `(batch_size, in_features)`.
        :returns: jnp.ndarray
            Output array of shape `(batch_size, out_features)`.
        """
        w = numpyro.sample(
            f"{self.name}_w",
            self.weight_prior_fn([self.in_features, self.out_features]),
        )
        b = numpyro.sample(f"{self.name}_b", self.bias_prior_fn([self.out_features]))
        return jnp.dot(X, w) + b


def _fft_matmul(first_row: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    """
    Perform circulant matrix multiplication using FFT.

    This computes the multiplication of a circulant matrix (defined by its first row)
    with a matrix `X` using the Fast Fourier Transform (FFT) for efficiency.

    :param first_row: jnp.ndarray
        The first row of the circulant matrix, shape `(in_features,)`.
    :param X: jnp.ndarray
        The input matrix, shape `(batch_size, in_features)`.

    :returns: jnp.ndarray
        Result of the circulant matrix multiplication, shape `(batch_size, in_features)`.
    """
    first_row_fft = jnp.fft.fft(first_row, axis=-1)
    X_fft = jnp.fft.fft(X, axis=-1)
    result_fft = first_row_fft[None, :] * X_fft
    result = jnp.fft.ifft(result_fft, axis=-1).real
    return result


class FFTLinear:
    """
    FFT-based linear layer for efficient circulant matrix multiplication.

    This layer uses a circulant matrix (parameterized by its first row) and
    applies FFT-based matrix multiplication for computational efficiency.
    """

    def __init__(
        self,
        in_features: int,
        name: str = "fft_layer",
        first_row_prior_fn=lambda shape: dist.Normal(0, 1).expand(shape),
        bias_prior_fn=lambda shape: dist.Normal(0, 1).expand(shape),
    ):
        """
        Initialize the FFTLinear layer.

        :param in_features: int
            Number of input features.
        :param name: str
            Name of the layer, used for parameter naming (default: "fft_layer").
        :param first_row_prior_fn: function
            A function that takes a shape and returns a NumPyro distribution for the
            circulant matrix's first row.
        :param bias_prior_fn: function
            A function that takes a shape and returns a NumPyro distribution for the
            bias.
        """
        self.in_features = in_features
        self.name = name
        self.first_row_prior_fn = first_row_prior_fn
        self.bias_prior_fn = bias_prior_fn

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the forward pass of the FFTLinear layer.

        :param X: jnp.ndarray
            Input data, shape `(batch_size, in_features)`.
        :returns: jnp.ndarray
            Output of the FFT-based linear layer, shape `(batch_size, in_features)`.
        """
        first_row = numpyro.sample(
            f"{self.name}_first_row", self.first_row_prior_fn([self.in_features])
        )
        bias_circulant = numpyro.sample(
            f"{self.name}_bias_circulant", self.bias_prior_fn([self.in_features])
        )
        hidden = _fft_matmul(first_row, X) + bias_circulant[None, :]
        return hidden


class FFTDirectPriorLinear:
    """
    FFT-based linear layer that directly samples its Fourier coefficients.
    The idea is to place a prior in Fourier space, ensuring real outputs
    via Hermitian symmetry.

    - in_features: size of the input dimension.
    - name: prefix for NumPyro sample sites (e.g. "fft_direct_layer_real", "fft_direct_layer_imag").
    - prior_fn: a function taking a shape and returning a NumPyro distribution
      (default: Normal(0,1) on each real component).
    - We store the last computed "fft_full" for visualization after a forward pass.
    """

    def __init__(
        self,
        in_features: int,
        name: str = "fft_direct_layer",
        prior_fn=lambda shape: dist.Normal(0, 1).expand(shape).to_event(1),
    ):
        self.in_features = in_features
        self.name = name
        self.prior_fn = prior_fn
        # k = number of "independent" frequencies for real signals
        self.k = in_features // 2 + 1
        self.last_fourier_coeffs = None  # for optional later visualization

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass: samples real/imag parts in half-spectrum, reconstructs
        the full FFT (with Hermitian symmetry), multiplies input in freq domain,
        and returns the inverse transform (real part).
        """
        result, fft_full = self._forward_and_get_coeffs(X)
        # Keep a copy for visualization
        self.last_fourier_coeffs = jax.lax.stop_gradient(fft_full)
        return result

    def _forward_and_get_coeffs(self, X: jnp.ndarray):
        # 1) Sample real part for frequencies [0..k-1].
        real_coeff = numpyro.sample(f"{self.name}_real", self.prior_fn([self.k]))

        # 2) If k>1, sample imaginary part for [1..k-1].
        #    freq=0 is purely real. If in_features is even, freq=n/2 is real too.
        if self.k > 1:
            imag_coeff = numpyro.sample(
                f"{self.name}_imag", self.prior_fn([self.k - 1])
            )
            # Construct the half-spectrum as a complex vector
            # freq=0 -> real_coeff[0], freq=1..(k-1) -> real + i*imag
            independent_fft = jnp.concatenate(
                [real_coeff[:1], real_coeff[1:] + 1j * imag_coeff]  # frequency 0
            )
        else:
            # Edge case: in_features=1 => k=1
            independent_fft = real_coeff

        # 3) Enforce Hermitian symmetry to get the "full" spectrum of length `in_features`.
        #    If in_features is even, the last frequency (Nyquist) is real => handle that carefully.
        n = self.in_features
        if n % 2 == 0 and self.k > 1:
            # even n, so freq n/2 is real; that is independent_fft[-1].real
            nyquist = independent_fft[-1].real[None]  # forcibly real
            # everything else is mirrored
            fft_full = jnp.concatenate(
                [
                    independent_fft[:-1],
                    nyquist,  # freq = n/2
                    jnp.conj(independent_fft[1:-1])[::-1],  # mirror of freq=1..(k-2)
                ]
            )
        else:
            # odd n, or n=2 (k=2) edge case
            fft_full = jnp.concatenate(
                [independent_fft, jnp.conj(independent_fft[1:])[::-1]]
            )

        # 4) Multiply in freq domain: FFT(X) * fft_full
        X_fft = jnp.fft.fft(X, axis=-1)
        result_fft = X_fft * fft_full[None, ...]  # broadcast over batch
        result = jnp.fft.ifft(result_fft, axis=-1).real

        return result, fft_full

    def get_fourier_coeffs(self) -> jnp.ndarray:
        """
        Return the last computed *full* Fourier spectrum (complex, length `in_features`).
        Raises ValueError if no forward pass was done yet.
        """
        if self.last_fourier_coeffs is None:
            raise ValueError(
                f"No Fourier coefficients available for layer {self.name}. "
                "Call the layer once on some input to store them."
            )
        return self.last_fourier_coeffs

    def compute_fourier_coeffs(self, X: jnp.ndarray, rng_key):
        """
        Computes the Fourier coefficients in an eager (non-transformed) context.
        Accepts an RNG key so that numpyro.sample has one.
        """
        # Use a seed handler to supply the key.
        with numpyro.handlers.seed(rng_seed=rng_key):
            _, fft_full = self._forward_and_get_coeffs(X)
        return jax.lax.stop_gradient(fft_full)


def _block_circulant_matmul(W, x, d_bernoulli=None):
    """
    Perform block-circulant matmul using FFT.
    W: shape (k_out, k_in, b), each row W[i,j,:] is the "first row" of a b x b circulant block.
    x: shape (batch, d_in) or (d_in,)
    d_bernoulli: shape (d_in,) of ±1, if given
    Returns: shape (batch, d_out).
    """
    # If x is 1D, reshape to (1, d_in)
    if x.ndim == 1:
        x = x[None, :]
    batch_size, d_in = x.shape

    k_out, k_in, b = W.shape
    d_out = k_out * b # noqa

    # Possibly multiply x by the Bernoulli diagonal
    if d_bernoulli is not None:
        x = x * d_bernoulli[None, :]

    # Zero-pad x to length (k_in * b)
    pad_len = k_in * b - d_in
    if pad_len > 0:
        x = jnp.pad(x, ((0, 0), (0, pad_len)))

    # Reshape into blocks: shape (batch, k_in, b)
    x_blocks = x.reshape(batch_size, k_in, b)

    # We'll accumulate output for each block-row i
    def one_block_mul(w_ij, x_j):
        # w_ij: (b,)   first row of circulant
        # x_j:  (batch, b)
        c_fft = jnp.fft.fft(w_ij)  # (b,)
        X_fft = jnp.fft.fft(x_j, axis=-1)  # (batch, b)
        block_fft = X_fft * jnp.conjugate(c_fft)[None, :]
        return jnp.fft.ifft(block_fft, axis=-1).real  # (batch, b)

    def compute_blockrow(i):
        # Sum over j=0..k_in-1 of circ(W[i,j]) x_j
        def sum_over_j(carry, j):
            w_ij = W[i, j, :]  # (b,)
            x_j = x_blocks[:, j, :]  # (batch, b)
            block_out = one_block_mul(w_ij, x_j)
            return carry + block_out, None

        init = jnp.zeros((batch_size, b))
        out_time, _ = jax.lax.scan(sum_over_j, init, jnp.arange(k_in))
        return out_time  # shape (batch, b)

    out_blocks = jax.vmap(compute_blockrow)(jnp.arange(k_out))  # (k_out, batch, b)
    out_reshaped = jnp.transpose(out_blocks, (1, 0, 2)).reshape(batch_size, k_out * b)
    return out_reshaped


class BlockCirculantLayer:
    """
    NumPyro-style block-circulant layer:
      - Samples W of shape (k_out, k_in, b).
      - Optionally samples a Bernoulli diagonal for input dimension d_in.
      - Then does block-circulant matmul in the forward pass.
    """

    def __init__(
        self,
        in_features,
        out_features,
        block_size,
        name="block_circ_layer",
        W_prior_fn=lambda shape: dist.Normal(0, 1).expand(
            shape
        ),  # can add .to_event(...)
        diag_prior_fn=None,  # optional Bernoulli prior
        bias_prior_fn=None,  # optional bias
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.name = name

        # We'll define k_in, k_out
        self.k_in = (in_features + block_size - 1) // block_size
        self.k_out = (out_features + block_size - 1) // block_size

        self.W_prior_fn = W_prior_fn
        self.diag_prior_fn = diag_prior_fn
        self.bias_prior_fn = bias_prior_fn

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        # 1) Sample W
        W = numpyro.sample(
            f"{self.name}_W",
            self.W_prior_fn([self.k_out, self.k_in, self.block_size]),
        )

        # 2) (Optional) sample the Bernoulli diagonal
        d_bernoulli = None
        if self.diag_prior_fn is not None:
            d_bernoulli = numpyro.sample(
                f"{self.name}_D",
                self.diag_prior_fn([self.in_features]),
            )

        # 3) Do the block-circulant multiplication
        out = _block_circulant_matmul(W, X, d_bernoulli)

        # 4) If we want a bias, we can sample a vector of shape (out_features,)
        if self.bias_prior_fn is not None:
            b = numpyro.sample(
                f"{self.name}_bias",
                self.bias_prior_fn([self.out_features]),
            )
            out = out + b[None, :]

        # 5) Finally, slice if out_features < k_out * block_size
        #    (some block_size combos might exceed the exact out_features).
        k_out_b = self.k_out * self.block_size
        if k_out_b > self.out_features:
            out = out[:, : self.out_features]

        return out


def _ifft_block_multiply(block_fft: jnp.ndarray, x_block: jnp.ndarray) -> jnp.ndarray:
    """
    Given a single block's full FFT (length b) and the FFT of x_block,
    compute the time-domain result of circulant(block) @ x_block via:
       out_block = ifft( conj(block_fft) * X_fft ).real
    The 'conjugate' appears if we define the first-row convention for the block.
    Adjust if you prefer the direct multiply (depends on your convention).
    """
    X_fft = jnp.fft.fft(x_block, axis=-1)  # shape (batch, b)
    out_fft = X_fft * jnp.conjugate(block_fft)[None, :]
    return jnp.fft.ifft(out_fft, axis=-1).real


class BlockFFTDirectPriorLayer:
    """
    A Numpyro-compatible block-circulant layer that:
      - Divides the (in_features, out_features) space into (k_in, k_out) blocks, each of size (b x b).
      - Samples real/imag half-spectra for each block from prior distributions,
        then imposes Hermitian symmetry to get a length-b complex vector for each block.
      - Multiplies the input x by this block-circulant matrix in the frequency domain.
      - Optionally zero-pads the input if 'in_features' is not a multiple of 'b'.
      - Stores the reconstructed “full” block-level FFT arrays in `self.last_fourier_coeffs`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int,
        name: str = "block_fft_layer",
        real_prior_fn=lambda shape: dist.Normal(0.0, 1.0).expand(shape),
        imag_prior_fn=lambda shape: dist.Normal(0.0, 1.0).expand(shape),
    ):
        """
        :param in_features: total input dimension
        :param out_features: total output dimension
        :param block_size: size of each circulant block (b)
        :param name: prefix for NumPyro sample sites
        :param real_prior_fn: prior generator for real part, typically Normal(0,1).expand(...)
        :param imag_prior_fn: prior generator for imag part, typically Normal(0,1).expand(...)
        """
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.name = name

        # Number of blocks in each dimension
        self.k_in = (in_features + block_size - 1) // block_size
        self.k_out = (out_features + block_size - 1) // block_size

        # Half-spectrum size for real signals
        self.k_half = (block_size // 2) + 1

        # Store the user-specified prior functions
        self.real_prior_fn = real_prior_fn
        self.imag_prior_fn = imag_prior_fn

        # We'll store the final "full" FFT for each block (k_out, k_in, b) for plotting
        self.last_fourier_coeffs = None  # shape: (k_out, k_in, block_size) complex

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass. We:
          1) Sample real and imaginary half-spectrum for each block from the given priors,
          2) Enforce zero imaginary part at freq=0 and freq=b/2 (if b is even),
          3) Reconstruct the full length-b complex array with Hermitian symmetry,
          4) Multiply X by the resulting block-circulant matrix in the freq domain,
          5) Slice the result to size (batch, out_features) if needed,
          6) Store the block-level FFT arrays in self.last_fourier_coeffs for later retrieval.

        :param X: shape (batch, in_features) or (in_features,)
        :returns: shape (batch, out_features)
        """
        out, block_fft_full = self._forward_and_get_fft(X)

        # Keep a copy of the full block-level spectra for later visualization
        self.last_fourier_coeffs = jax.lax.stop_gradient(block_fft_full)

        return out

    def _forward_and_get_fft(self, X: jnp.ndarray):
        """Helper that does the actual logic and returns (output, block_fft_full)."""
        if X.ndim == 1:
            X = X[None, :]  # shape (1, in_features)
        batch_size, d_in = X.shape

        # (A) Sample the real and imaginary parts for shape (k_out, k_in, k_half).
        #     We'll name them "real" and "imag" sites, using self.name as a prefix.
        real_coeff = numpyro.sample(
            f"{self.name}_real",
            self.real_prior_fn([self.k_out, self.k_in, self.k_half]),
        )
        imag_coeff = numpyro.sample(
            f"{self.name}_imag",
            self.imag_prior_fn([self.k_out, self.k_in, self.k_half]),
        )
        # Enforce freq=0 is real-only => imag=0
        imag_coeff = imag_coeff.at[..., 0].set(0.0)
        # If block_size is even, freq=b/2 must be real => zero out the last imag index
        if (self.block_size % 2 == 0) and (self.k_half > 1):
            imag_coeff = imag_coeff.at[..., -1].set(0.0)

        # (B) Reconstruct the full b-point FFT for each block, imposing Hermitian symmetry
        def _reconstruct_block_spectrum(r_ij, i_ij):
            # r_ij, i_ij each shape (k_half,)
            # half_complex = r + 1j*i
            half_complex = r_ij + 1j * i_ij
            b = self.block_size
            if (b % 2 == 0) and (self.k_half > 1):
                # even length, last freq is real => place in the middle
                nyquist = half_complex[-1].real[None]
                block_fft = jnp.concatenate(
                    [
                        half_complex[:-1],
                        nyquist,
                        jnp.conjugate(half_complex[1:-1])[::-1],
                    ]
                )
            else:
                # odd length
                block_fft = jnp.concatenate(
                    [half_complex, jnp.conjugate(half_complex[1:])[::-1]]
                )
            return block_fft

        # Vectorize over (k_out, k_in).
        # The result has shape (k_out, k_in, block_size) (complex).
        block_fft_full = jax.vmap(
            lambda Rrow, Irow: jax.vmap(_reconstruct_block_spectrum)(Rrow, Irow),
            in_axes=(0, 0),
        )(real_coeff, imag_coeff)

        # (C) Zero-pad X if needed so X has length k_in * block_size
        pad_len = self.k_in * self.block_size - d_in
        if pad_len > 0:
            X = jnp.pad(X, ((0, 0), (0, pad_len)))

        # Reshape X into shape (batch_size, k_in, block_size)
        X_blocks = X.reshape(batch_size, self.k_in, self.block_size)

        # (D) Perform the block-circulant multiplication in time domain,
        #     using frequency-domain methods for each block row i.
        # out[i] = sum_j ifft(conjugate(block_fft[i,j]) * fft(X_blocks[j]))
        def compute_blockrow(i):
            def scan_over_j(carry, j):
                # block_fft_full[i, j] => shape (b,)
                # X_blocks[:, j, :]   => shape (batch_size, b)
                w_ij_fft = block_fft_full[i, j]
                x_j = X_blocks[:, j, :]
                block_out = _ifft_block_multiply(w_ij_fft, x_j)  # shape (batch_size, b)
                return carry + block_out, None

            init = jnp.zeros((batch_size, self.block_size))
            out_time, _ = jax.lax.scan(scan_over_j, init, jnp.arange(self.k_in))
            return out_time  # shape (batch_size, b)

        # shape (k_out, batch_size, b)
        out_blocks = jax.vmap(compute_blockrow)(jnp.arange(self.k_out))
        # reorder => (batch_size, k_out, b)
        out_reshaped = jnp.transpose(out_blocks, (1, 0, 2)).reshape(
            batch_size, self.k_out * self.block_size
        )

        # (E) Slice if out_features < k_out * block_size
        if self.k_out * self.block_size > self.out_features:
            out_reshaped = out_reshaped[:, : self.out_features]

        return out_reshaped, block_fft_full

    def get_fourier_coeffs(self) -> jnp.ndarray:
        """
        Return the last computed *full* block-level FFT array of shape
        (k_out, k_in, block_size) (complex). Raises ValueError if no forward pass was done yet.
        """
        if self.last_fourier_coeffs is None:
            raise ValueError(
                f"No Fourier coefficients available for layer '{self.name}'. "
                "Call the layer once on some input to store them."
            )
        return self.last_fourier_coeffs


class ParticleLinear:
    """
    A particle-aware fully connected layer.

    Applies linear transformations to inputs for each particle, with an aggregation step
    to ensure the output shape matches expectations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        name: str = "particle_layer",
        aggregation: str = "mean",
    ):
        """
        Initializes the ParticleLinear layer.

        :param in_features: int
            Number of input features.
        :param out_features: int
            Number of output features.
        :param name: str
            Layer name for parameter tracking.
        :param aggregation: str
            Method to aggregate across particles ('mean', 'sum', etc.).
        """
        self.in_features = in_features
        self.out_features = out_features
        self.name = name
        self.aggregation = aggregation

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Applies the particle-aware transformation.

        :param X: jnp.ndarray
            Input array of shape `(particles, batch_size, in_features)`.

        :returns: jnp.ndarray
            Output array of shape `(batch_size, out_features)`.
        """
        if X.ndim == 2:
            # Add particle dimension if missing
            X = X[jnp.newaxis, ...]  # Shape: (1, batch_size, in_features)

        particles = X.shape[0]
        w = numpyro.sample(
            f"{self.name}_w",
            dist.Normal(0, 1).expand([particles, self.in_features, self.out_features]),
        )
        b = numpyro.sample(
            f"{self.name}_b", dist.Normal(0, 1).expand([particles, self.out_features])
        )

        # Compute output for each particle
        particle_outputs = (
            jnp.einsum("pbi,pij->pbj", X, w) + b
        )  # (particles, batch_size, out_features)

        # Aggregate across particles
        if self.aggregation == "mean":
            aggregated_output = jnp.mean(particle_outputs, axis=0)
        elif self.aggregation == "sum":
            aggregated_output = jnp.sum(particle_outputs, axis=0)
        elif self.aggregation == "max":
            aggregated_output = jnp.max(particle_outputs, axis=0)
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation}")

        return aggregated_output


class FFTParticleLinear:
    """
    FFT-based particle-aware linear layer.

    Applies FFT-based linear transformations for each particle using circulant matrices
    and aggregates the outputs to ensure the final shape matches the expected dimensions.
    """

    def __init__(
        self,
        in_features: int,
        name: str = "fft_particle_layer",
        aggregation: str = "mean",
    ):
        """
        Initializes the FFTParticleLinear layer.

        :param in_features: int
            Number of input features (and output features due to circulant matrix property).
        :param name: str
            Name of the layer for parameter tracking.
        :param aggregation: str
            Method to aggregate across particles ('mean', 'sum', etc.).
        """
        self.in_features = in_features
        self.name = name
        self.aggregation = aggregation

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Applies the FFT-based particle-aware transformation.

        :param X: jnp.ndarray
            Input array of shape `(particles, batch_size, in_features)`.

        :returns: jnp.ndarray
            Output array of shape `(batch_size, in_features)`.
        """
        if X.ndim == 2:
            # Add particle dimension if missing
            X = X[jnp.newaxis, ...]  # Shape: (1, batch_size, in_features)

        particles = X.shape[0]

        # Sample first rows of circulant matrices and biases for each particle
        first_rows = numpyro.sample(
            f"{self.name}_first_rows",
            dist.Normal(0, 1).expand([particles, self.in_features]),
        )
        biases = numpyro.sample(
            f"{self.name}_biases",
            dist.Normal(0, 1).expand([particles, self.in_features]),
        )

        # Compute FFT-based transformation for each particle
        def fft_particle_transform(p_idx):
            first_row = first_rows[p_idx]  # Shape: (in_features,)
            bias = biases[p_idx]  # Shape: (in_features,)

            # FFT multiplication for each particle
            transformed = _fft_matmul(
                first_row, X[p_idx]
            )  # Shape: (batch_size, in_features)
            return transformed + bias  # Shape: (batch_size, in_features)

        # Apply transformation across particles
        particle_outputs = jax.vmap(fft_particle_transform)(
            jnp.arange(particles)
        )  # (particles, batch_size, in_features)

        # Aggregate across particles
        if self.aggregation == "mean":
            aggregated_output = jnp.mean(particle_outputs, axis=0)
        elif self.aggregation == "sum":
            aggregated_output = jnp.sum(particle_outputs, axis=0)
        elif self.aggregation == "max":
            aggregated_output = jnp.max(particle_outputs, axis=0)
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation}")

        return aggregated_output


class SelfAttention:
    """
    Implements a single-head self-attention mechanism with learnable weights.

    Self-attention computes attention scores for a sequence, enabling the model to
    focus on different parts of the sequence for each position.
    """

    def __init__(self, embed_dim: int, name: str = "self_attention"):
        """
        Initialize the SelfAttention layer.

        :param embed_dim: int
            Dimensionality of the embedding space.
        :param name: str, optional
            Name of the layer, used for parameter naming (default: "self_attention").
        """
        self.embed_dim = embed_dim
        self.name = name

    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Perform the forward pass of the self-attention layer.

        :param query: jnp.ndarray
            Query tensor of shape `(batch_size, seq_len, embed_dim)`.
        :param key: jnp.ndarray
            Key tensor of shape `(batch_size, seq_len, embed_dim)`.
        :param value: jnp.ndarray
            Value tensor of shape `(batch_size, seq_len, embed_dim)`.
        :param mask: jnp.ndarray, optional
            Attention mask of shape `(batch_size, seq_len, seq_len)`, where 1 indicates
            valid positions and 0 indicates masked positions (default: None).

        :returns: jnp.ndarray
            Output tensor of shape `(batch_size, seq_len, embed_dim)`.
        """
        d_k = self.embed_dim

        w_q = numpyro.sample(
            f"{self.name}_w_q",
            dist.Normal(0, 1).expand([self.embed_dim, self.embed_dim]),
        )
        w_k = numpyro.sample(
            f"{self.name}_w_k",
            dist.Normal(0, 1).expand([self.embed_dim, self.embed_dim]),
        )
        w_v = numpyro.sample(
            f"{self.name}_w_v",
            dist.Normal(0, 1).expand([self.embed_dim, self.embed_dim]),
        )

        q = jnp.dot(query, w_q)
        k = jnp.dot(key, w_k)
        v = jnp.dot(value, w_v)

        scores = jnp.matmul(q, k.transpose(0, 2, 1)) / jnp.sqrt(d_k)
        if mask is not None:
            scores = scores - 1e9 * (1 - mask)

        attention_weights = jax.nn.softmax(scores, axis=-1)
        output = jnp.matmul(attention_weights, v)

        return output


class MultiHeadSelfAttention:
    """
    Implements a multi-head self-attention mechanism with learnable weights.

    Multi-head self-attention splits the embedding space into multiple attention heads
    to capture diverse relationships within the input sequence.
    """

    def __init__(
        self, embed_dim: int, num_heads: int, name: str = "multihead_self_attention"
    ):
        """
        Initialize the MultiHeadSelfAttention layer.

        :param embed_dim: int
            Dimensionality of the embedding space.
        :param num_heads: int
            Number of attention heads. Must divide `embed_dim` evenly.
        :param name: str, optional
            Name of the layer, used for parameter naming (default: "multihead_self_attention").
        """
        assert (
            embed_dim % num_heads == 0
        ), "Embed dim must be divisible by the number of heads."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.name = name

    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Perform the forward pass of the multi-head self-attention layer.

        :param query: jnp.ndarray
            Query tensor of shape `(batch_size, seq_len, embed_dim)`.
        :param key: jnp.ndarray
            Key tensor of shape `(batch_size, seq_len, embed_dim)`.
        :param value: jnp.ndarray
            Value tensor of shape `(batch_size, seq_len, embed_dim)`.
        :param mask: jnp.ndarray, optional
            Attention mask of shape `(batch_size, seq_len, seq_len)`, where 1 indicates
            valid positions and 0 indicates masked positions (default: None).

        :returns: jnp.ndarray
            Output tensor of shape `(batch_size, seq_len, embed_dim)`.
        """
        batch_size, seq_len, _ = query.shape

        def project(input_tensor, suffix):
            w = numpyro.sample(
                f"{self.name}_w_{suffix}",
                dist.Normal(0, 1).expand([self.embed_dim, self.embed_dim]),
            )
            return jnp.dot(input_tensor, w)

        q = project(query, "q").reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        k = project(key, "k").reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        v = project(value, "v").reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )

        q, k, v = map(lambda x: x.transpose(0, 2, 1, 3), (q, k, v))

        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(self.head_dim)
        if mask is not None:
            scores = scores - 1e9 * (1 - mask[:, None, :, :])

        attention_weights = jax.nn.softmax(scores, axis=-1)
        attention_output = jnp.matmul(attention_weights, v)

        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.embed_dim
        )

        w_o = numpyro.sample(
            f"{self.name}_w_o",
            dist.Normal(0, 1).expand([self.embed_dim, self.embed_dim]),
        )
        return jnp.dot(attention_output, w_o)


class PositionalEncoding:
    """
    Implements learnable positional encodings for input sequences.

    Positional encodings are added to the input embeddings to inject information
    about the relative and absolute positions of tokens in a sequence.
    """

    def __init__(self, seq_len: int, embed_dim: int, name: str = "positional_encoding"):
        """
        Initialize the PositionalEncoding layer.

        :param seq_len: int
            The length of the input sequences.
        :param embed_dim: int
            The dimensionality of the embeddings.
        :param name: str, optional
            Name of the layer, used for parameter naming (default: "positional_encoding").
        """
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.name = name

        # Learnable positional embeddings
        self.positional_embeddings = numpyro.param(
            f"{self.name}_positional_embeddings",
            jax.random.normal(jax.random.PRNGKey(0), (seq_len, embed_dim)),
        )

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Add positional encodings to the input tensor.

        :param X: jnp.ndarray
            Input tensor of shape `(batch_size, seq_len, embed_dim)`.

        :returns: jnp.ndarray
            Output tensor of shape `(batch_size, seq_len, embed_dim)` with positional encodings added.

        :raises AssertionError:
            If the sequence length of the input tensor does not match the initialized `seq_len`.
        """
        assert (
            X.shape[1] == self.seq_len
        ), f"Input sequence length {X.shape[1]} does not match initialized sequence length {self.seq_len}."
        return X + self.positional_embeddings


class TransformerEncoder:
    """
    Implements a single transformer encoder block.

    A transformer encoder block consists of a multi-head self-attention mechanism,
    followed by a feedforward network with residual connections and layer normalization.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        name: str = "transformer_encoder",
    ):
        """
        Initialize the TransformerEncoder block.

        :param embed_dim: int
            Dimensionality of the input embeddings.
        :param num_heads: int
            Number of attention heads for the self-attention mechanism.
        :param hidden_dim: int
            Dimensionality of the hidden layer in the feedforward network.
        :param name: str, optional
            Name of the block, used for parameter naming (default: "transformer_encoder").
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.name = name

        self.self_attention = MultiHeadSelfAttention(
            embed_dim, num_heads, name=f"{name}_self_attention"
        )
        self.layer_norm1 = LayerNorm(embed_dim, name=f"{name}_layer_norm1")
        self.feedforward = Linear(embed_dim, hidden_dim, name=f"{name}_feedforward")
        self.out = Linear(hidden_dim, embed_dim, name=f"{name}_out")
        self.layer_norm2 = LayerNorm(embed_dim, name=f"{name}_layer_norm2")

    def __call__(self, X: jnp.ndarray, mask: jnp.ndarray = None) -> jnp.ndarray:
        """
        Perform the forward pass of the TransformerEncoder block.

        :param X: jnp.ndarray
            Input tensor of shape `(batch_size, seq_len, embed_dim)`.
        :param mask: jnp.ndarray, optional
            Attention mask of shape `(batch_size, seq_len, seq_len)`, where 1 indicates
            valid positions and 0 indicates masked positions (default: None).

        :returns: jnp.ndarray
            Output tensor of shape `(batch_size, seq_len, embed_dim)`.
        """
        attention_output = self.self_attention(X, X, X, mask)
        X = self.layer_norm1(attention_output + X)

        ff_output = jax.nn.relu(self.feedforward(X))
        ff_output = self.out(ff_output)

        X = self.layer_norm2(ff_output + X)
        return X


class Conv1d:
    """
    Implements a 1D convolutional layer with learnable weights and biases.

    This layer performs a convolution operation along the temporal (or spatial) dimension
    of the input, with support for custom stride, padding, and kernel size.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: str = "valid",
        name: str = "conv1d",
    ):
        """
        Initialize the Conv1d layer.

        :param in_channels: int
            Number of input channels.
        :param out_channels: int
            Number of output channels.
        :param kernel_size: int
            Size of the convolutional kernel.
        :param stride: int, optional
            Stride of the convolution (default: 1).
        :param padding: str, optional
            Padding type, either "valid" (no padding) or "same" (zero-padding, maintains size) (default: "valid").
        :param name: str, optional
            Name of the layer, used for parameter naming (default: "conv1d").
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding.upper()  # Convert to uppercase for JAX compatibility
        self.name = name

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the forward pass of the Conv1d layer.

        :param X: jnp.ndarray
            Input tensor of shape `(batch_size, in_channels, input_length)`.

        :returns: jnp.ndarray
            Output tensor of shape `(batch_size, out_channels, output_length)`, where `output_length`
            depends on the padding and stride configuration.
        """

        weight = numpyro.sample(
            f"{self.name}_weight",
            dist.Normal(0, 1).expand(
                [self.out_channels, self.in_channels, self.kernel_size]
            ),
        )
        bias = numpyro.sample(
            f"{self.name}_bias", dist.Normal(0, 1).expand([self.out_channels])
        )

        convolved = jax.lax.conv_general_dilated(
            X,
            weight,
            window_strides=(self.stride,),
            padding=self.padding,
            dimension_numbers=("NCH", "OIH", "NCH"),
        )

        convolved += bias[:, None]

        return convolved


class Conv2d:
    """
    Implements a 2D convolutional layer with learnable weights and biases.

    This layer performs a 2D convolution operation over spatial dimensions
    (height and width) of the input tensor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        stride: int | tuple = 1,
        padding: str = "valid",
        name: str = "conv2d",
    ):
        """
        Initialize the Conv2d layer.

        :param in_channels: int
            Number of input channels.
        :param out_channels: int
            Number of output channels.
        :param kernel_size: int or tuple
            Size of the convolutional kernel, either as an integer or a tuple (kernel_h, kernel_w).
        :param stride: int or tuple, optional
            Stride of the convolution, either as an integer or a tuple (stride_h, stride_w) (default: 1).
        :param padding: str, optional
            Padding mode, either "valid" (no padding) or "same" (maintains spatial dimensions) (default: "valid").
        :param name: str, optional
            Name of the layer, used for parameter naming (default: "conv2d").
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        self.name = name

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the forward pass of the Conv2d layer.

        :param X: jnp.ndarray
            Input tensor of shape `(batch_size, in_channels, height, width)`.

        :returns: jnp.ndarray
            Output tensor of shape `(batch_size, out_channels, new_height, new_width)`, where:
            - `new_height` and `new_width` depend on the padding, kernel size, and stride.

        :raises ValueError:
            If an unsupported padding mode is provided.
        """
        batch_size, in_channels, input_h, input_w = X.shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride

        # Determine padding
        if self.padding == "same":
            padding = "SAME"
        elif self.padding == "valid":
            padding = "VALID"
        else:
            raise ValueError(f"Unsupported padding mode: {self.padding}")

        weight = numpyro.sample(
            f"{self.name}_weight",
            dist.Normal(0, 1).expand(
                [self.out_channels, self.in_channels, kernel_h, kernel_w]
            ),
        )
        bias = numpyro.sample(
            f"{self.name}_bias",
            dist.Normal(0, 1).expand([self.out_channels]),
        )

        convolved = jax.lax.conv_general_dilated(
            X,
            weight,
            window_strides=(stride_h, stride_w),
            padding=padding,
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )

        bias = bias[:, None, None]
        convolved += bias
        return convolved


class FFTConv1d:
    """
    Implements a 1D convolutional layer using FFT-based computation.

    This layer performs a 1D convolution in the frequency domain for efficiency,
    particularly with large kernel sizes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        name: str = "fft_conv1d",
    ):
        """
        Initialize the FFTConv1d layer.

        :param in_channels: int
            Number of input channels.
        :param out_channels: int
            Number of output channels.
        :param kernel_size: int
            Size of the convolutional kernel.
        :param stride: int, optional
            Stride of the convolution (default: 1).
        :param name: str, optional
            Name of the layer, used for parameter naming (default: "fft_conv1d").
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.name = name

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the forward pass of the FFTConv1d layer.

        :param X: jnp.ndarray
            Input tensor of shape `(batch_size, in_channels, width)`.

        :returns: jnp.ndarray
            Output tensor of shape `(batch_size, out_channels, new_width)`, where:
            - `new_width = floor((width + kernel_size - 1) / stride)`.
        """
        batch_size, in_channels, width = X.shape

        weight = numpyro.sample(
            f"{self.name}_weight",
            dist.Normal(0, 1).expand(
                [self.out_channels, self.in_channels, self.kernel_size]
            ),
        )
        bias = numpyro.sample(
            f"{self.name}_bias",
            dist.Normal(0, 1).expand([self.out_channels]),
        )

        output = []
        for out_c in range(self.out_channels):
            channel_out = 0
            for in_c in range(self.in_channels):
                X_fft = jnp.fft.fft(X[:, in_c], n=width + self.kernel_size - 1, axis=-1)
                weight_fft = jnp.fft.fft(
                    weight[out_c, in_c], n=width + self.kernel_size - 1, axis=-1
                )

                conv_fft = X_fft * weight_fft

                channel_out += jnp.fft.ifft(conv_fft, axis=-1).real

            channel_out += bias[out_c]
            output.append(channel_out)

        output = jnp.stack(output, axis=1)

        if self.stride > 1:
            output = output[:, :, :: self.stride]

        return output


class FFTConv2d:
    """
    Implements a 2D convolutional layer using FFT-based computation.

    This layer performs a 2D convolution in the frequency domain, which is
    efficient for large kernels and input sizes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        padding: str = "same",  # Added padding parameter
        name: str = "fft_conv2d",
    ):
        """
        Initialize the FFTConv2d layer.

        :param in_channels: int
            Number of input channels.
        :param out_channels: int
            Number of output channels.
        :param kernel_size: int or tuple
            Size of the convolutional kernel, either as an integer or a tuple (kernel_h, kernel_w).
        :param padding: str, optional
            Padding mode, either "same" or "valid" (default: "same").
        :param name: str, optional
            Name of the layer, used for parameter naming (default: "fft_conv2d").
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        if padding not in ("same", "valid"):
            raise ValueError(f"Unsupported padding mode: {padding}")
        self.padding = padding
        self.name = name

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the forward pass of the FFTConv2d layer.

        :param X: jnp.ndarray
            Input tensor of shape `(batch_size, in_channels, height, width)`.

        :returns: jnp.ndarray
            Output tensor of shape `(batch_size, out_channels, new_height, new_width)`, where:
            - For "same" padding: `new_height = height`, `new_width = width`
            - For "valid" padding: `new_height = height - kernel_h + 1`, `new_width = width - kernel_w + 1`
        """
        batch_size, in_channels, height, width = X.shape
        kernel_h, kernel_w = self.kernel_size

        # Determine padding sizes
        if self.padding == "same":
            pad_h = kernel_h // 2
            pad_w = kernel_w // 2
            # Apply padding to height and width
            X_padded = jnp.pad(
                X,
                pad_width=((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                mode="constant",
                constant_values=0,
            )
        elif self.padding == "valid":
            X_padded = X
        else:
            raise ValueError(f"Unsupported padding mode: {self.padding}")

        # Update dimensions after padding
        _, _, H_padded, W_padded = X_padded.shape

        # Prepare weight and bias
        weight = numpyro.sample(
            f"{self.name}_weight",
            dist.Normal(0, 1).expand(
                [self.out_channels, self.in_channels, kernel_h, kernel_w]
            ),
        )
        bias = numpyro.sample(
            f"{self.name}_bias",
            dist.Normal(0, 1).expand([self.out_channels]),
        )

        # Determine FFT size
        out_h = H_padded + kernel_h - 1
        out_w = W_padded + kernel_w - 1

        output = []
        for out_c in range(self.out_channels):
            channel_out = 0
            for in_c in range(self.in_channels):
                # FFT of input
                X_fft = jnp.fft.fft2(X_padded[:, in_c], s=(out_h, out_w))
                # FFT of weight
                weight_fft = jnp.fft.fft2(weight[out_c, in_c], s=(out_h, out_w))

                # Element-wise multiplication in frequency domain
                conv_fft = X_fft * weight_fft

                # Inverse FFT to get the convolved output
                conv = jnp.fft.ifft2(conv_fft).real

                # Crop to desired size
                if self.padding == "same":
                    # To get the same size as input, center crop
                    start_h = (conv.shape[1] - height) // 2
                    start_w = (conv.shape[2] - width) // 2
                    conv_cropped = conv[
                        :, start_h : start_h + height, start_w : start_w + width
                    ]
                elif self.padding == "valid":
                    conv_cropped = conv[:, kernel_h - 1 : height, kernel_w - 1 : width]
                else:
                    raise ValueError(f"Unsupported padding mode: {self.padding}")

                channel_out += conv_cropped

            # Add bias
            channel_out += bias[out_c].reshape(1, 1, 1)

            output.append(channel_out)

        # Stack all output channels
        output = jnp.stack(output, axis=1)  # Shape: (batch_size, out_channels, H, W)

        return output


class TransposedConv2d:
    """
    Implements a 2D transposed convolutional layer with learnable weights and biases.
    This layer is commonly used for upsampling in neural networks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        stride: int | tuple = 1,
        padding: str = "valid",
        name: str = "transposed_conv2d",
    ):
        """
        Initialize the TransposedConv2d layer.

        :param in_channels: int
            Number of input channels.
        :param out_channels: int
            Number of output channels.
        :param kernel_size: int or tuple
            Size of the transposed convolutional kernel.
        :param stride: int or tuple, optional
            Stride of the transposed convolution (default: 1).
        :param padding: str, optional
            Padding type, either "valid" or "same" (default: "valid").
        :param name: str, optional
            Name of the layer (default: "transposed_conv2d").
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        self.name = name

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the forward pass of the TransposedConv2d layer.

        :param X: jnp.ndarray
            Input tensor of shape `(batch_size, in_channels, height, width)`.

        :returns: jnp.ndarray
            Output tensor of shape `(batch_size, out_channels, new_height, new_width)`.
        """
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride

        # Determine padding
        if self.padding == "same":
            padding = "SAME"
        elif self.padding == "valid":
            padding = "VALID"
        else:
            raise ValueError(f"Unsupported padding mode: {self.padding}")

        weight = numpyro.sample(
            f"{self.name}_weight",
            dist.Normal(0, 1).expand(
                [self.in_channels, self.out_channels, kernel_h, kernel_w]
            ),
        )
        bias = numpyro.sample(
            f"{self.name}_bias", dist.Normal(0, 1).expand([self.out_channels])
        )

        convolved = jax.lax.conv_transpose(
            lhs=X,
            rhs=weight,
            strides=(stride_h, stride_w),
            padding=padding,
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )

        convolved += bias[None, :, None, None]  # Add bias per channel
        return convolved


class FFTTransposedConv2d:
    """
    Implements a 2D transposed convolutional layer using FFT-based computation.

    This layer performs a 2D transposed convolution in the frequency domain
    for efficiency with large kernel sizes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        stride: int | tuple = 1,  # Add stride for compatibility
        padding: str = "same",  # Added padding parameter
        name: str = "fft_transposed_conv2d",
    ):
        """
        Initialize the FFTTransposedConv2d layer.

        :param in_channels: int
            Number of input channels.
        :param out_channels: int
            Number of output channels.
        :param kernel_size: int or tuple
            Size of the transposed convolutional kernel.
        :param stride: int or tuple, optional
            Stride of the transposed convolution (default: 1).
        :param padding: str, optional
            Padding mode, either "same" or "valid" (default: "same").
        :param name: str, optional
            Name of the layer (default: "fft_transposed_conv2d").
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        if padding not in ("same", "valid"):
            raise ValueError(f"Unsupported padding mode: {padding}")
        self.padding = padding
        self.name = name

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the forward pass of the FFTTransposedConv2d layer.

        :param X: jnp.ndarray
            Input tensor of shape `(batch_size, in_channels, height, width)`.

        :returns: jnp.ndarray
            Output tensor of shape `(batch_size, out_channels, new_height, new_width)`.
        """
        batch_size, in_channels, height, width = X.shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride

        # Prepare weight and bias
        weight = numpyro.sample(
            f"{self.name}_weight",
            dist.Normal(0, 1).expand(
                [self.in_channels, self.out_channels, kernel_h, kernel_w]
            ),
        )
        bias = numpyro.sample(
            f"{self.name}_bias", dist.Normal(0, 1).expand([self.out_channels])
        )

        # Calculate output spatial dimensions
        out_h = height * stride_h
        out_w = width * stride_w

        # Apply stride-based upsampling if stride > 1
        if self.stride != (1, 1):
            # Initialize a zero array with upsampled spatial dimensions
            upsampled = jnp.zeros((batch_size, in_channels, out_h, out_w))
            # Assign input values to the upsampled array with the given stride
            upsampled = upsampled.at[:, :, ::stride_h, ::stride_w].set(X)
        else:
            upsampled = X

        # Determine padding sizes for transposed convolution
        if self.padding == "same":
            pad_h = kernel_h // 2
            pad_w = kernel_w // 2
            # Apply padding to height and width
            upsampled_padded = jnp.pad(
                upsampled,
                pad_width=((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                mode="constant",
                constant_values=0,
            )
        elif self.padding == "valid":
            upsampled_padded = upsampled
        else:
            raise ValueError(f"Unsupported padding mode: {self.padding}")

        # Update dimensions after padding
        _, _, H_padded, W_padded = upsampled_padded.shape

        # Determine FFT size
        conv_h = H_padded + kernel_h - 1
        conv_w = W_padded + kernel_w - 1

        output = []
        for out_c in range(self.out_channels):
            channel_out = 0
            for in_c in range(self.in_channels):
                # FFT of upsampled input
                X_fft = jnp.fft.fft2(upsampled_padded[:, in_c], s=(conv_h, conv_w))
                # FFT of weight (note the transpose for transposed convolution)
                weight_fft = jnp.fft.fft2(weight[in_c, out_c], s=(conv_h, conv_w))

                # Element-wise multiplication in frequency domain
                conv_fft = X_fft * weight_fft

                # Inverse FFT to get the convolved output
                conv = jnp.fft.ifft2(conv_fft).real

                # Crop to desired size
                if self.padding == "same":
                    # To maintain the upsampled size
                    start_h = (conv.shape[1] - out_h) // 2
                    start_w = (conv.shape[2] - out_w) // 2
                    conv_cropped = conv[
                        :, start_h : start_h + out_h, start_w : start_w + out_w
                    ]
                elif self.padding == "valid":
                    # Calculate the amount to crop based on kernel size
                    conv_cropped = conv[:, kernel_h - 1 : out_h, kernel_w - 1 : out_w]
                else:
                    raise ValueError(f"Unsupported padding mode: {self.padding}")

                channel_out += conv_cropped

            # Add bias
            channel_out += bias[out_c].reshape(1, 1, 1)

            output.append(channel_out)

        # Stack all output channels
        output = jnp.stack(
            output, axis=1
        )  # Shape: (batch_size, out_channels, out_h, out_w)

        return output


class LSTM:
    """
    Implements a Long Short-Term Memory (LSTM) layer.

    The LSTM processes sequential data and maintains hidden and cell states
    across time steps for tasks like sequence modeling and time series forecasting.
    """

    def __init__(self, input_dim: int, hidden_dim: int, name: str = "lstm"):
        """
        Initialize the LSTM layer.

        :param input_dim: int
            Dimensionality of the input features.
        :param hidden_dim: int
            Dimensionality of the hidden state and cell state.
        :param name: str, optional
            Name of the LSTM layer, used for parameter naming (default: "lstm").
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.name = name

    def __call__(
        self, X: jnp.ndarray, init_state: tuple | None = None
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Perform the forward pass of the LSTM layer.

        :param X: jnp.ndarray
            Input tensor of shape `(batch_size, seq_len, input_dim)`.
        :param init_state: tuple or None, optional
            Initial hidden and cell states as `(h_0, c_0)` of shape `(batch_size, hidden_dim)`.
            If None, states are initialized to zeros (default: None).

        :returns: tuple
            - `outputs`: jnp.ndarray
                Output tensor of shape `(batch_size, seq_len, hidden_dim)` containing
                the hidden states across all time steps.
            - `final_state`: tuple[jnp.ndarray, jnp.ndarray]
                Final hidden state (`h_t`) and cell state (`c_t`), both of shape `(batch_size, hidden_dim)`.
        """
        batch_size, seq_len, _ = X.shape

        # Initialize weights
        Wf = numpyro.sample(
            f"{self.name}_Wf",
            dist.Normal(0, 1).expand(
                [self.input_dim + self.hidden_dim, self.hidden_dim]
            ),
        )
        Wi = numpyro.sample(
            f"{self.name}_Wi",
            dist.Normal(0, 1).expand(
                [self.input_dim + self.hidden_dim, self.hidden_dim]
            ),
        )
        Wc = numpyro.sample(
            f"{self.name}_Wc",
            dist.Normal(0, 1).expand(
                [self.input_dim + self.hidden_dim, self.hidden_dim]
            ),
        )
        Wo = numpyro.sample(
            f"{self.name}_Wo",
            dist.Normal(0, 1).expand(
                [self.input_dim + self.hidden_dim, self.hidden_dim]
            ),
        )
        bf = numpyro.sample(
            f"{self.name}_bf", dist.Normal(0, 1).expand([self.hidden_dim])
        )
        bi = numpyro.sample(
            f"{self.name}_bi", dist.Normal(0, 1).expand([self.hidden_dim])
        )
        bc = numpyro.sample(
            f"{self.name}_bc", dist.Normal(0, 1).expand([self.hidden_dim])
        )
        bo = numpyro.sample(
            f"{self.name}_bo", dist.Normal(0, 1).expand([self.hidden_dim])
        )

        # Initialize hidden and cell states
        if init_state is None:
            h_t = jnp.zeros((batch_size, self.hidden_dim))
            c_t = jnp.zeros((batch_size, self.hidden_dim))
        else:
            h_t, c_t = init_state

        outputs = []

        for t in range(seq_len):
            x_t = X[:, t, :]
            combined = jnp.concatenate([x_t, h_t], axis=-1)
            f_t = jax.nn.sigmoid(jnp.dot(combined, Wf) + bf)
            i_t = jax.nn.sigmoid(jnp.dot(combined, Wi) + bi)
            o_t = jax.nn.sigmoid(jnp.dot(combined, Wo) + bo)
            c_t_candidate = jnp.tanh(jnp.dot(combined, Wc) + bc)
            c_t = f_t * c_t + i_t * c_t_candidate
            h_t = o_t * jnp.tanh(c_t)
            outputs.append(h_t)

        return jnp.stack(outputs, axis=1), (h_t, c_t)


class GaussianProcessLayer:
    """
    Implements a Gaussian Process (GP) layer with a Radial Basis Function (RBF) kernel.

    This layer computes the covariance matrix (kernel) based on the input data and
    learnable parameters: length scale, variance, and noise.
    """

    def __init__(self, input_dim: int, name: str = "gp_layer"):
        """
        Initialize the Gaussian Process layer.

        :param input_dim: int
            Dimensionality of the input features.
        :param name: str, optional
            Name of the GP layer, used for parameter naming (default: "gp_layer").
        """
        self.input_dim = input_dim
        self.name = name

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the forward pass to compute the GP kernel matrix.

        :param X: jnp.ndarray
            Input data of shape `(num_points, input_dim)`, where `num_points` is
            the number of data points, and `input_dim` is the feature dimension.

        :returns: jnp.ndarray
            Covariance matrix (kernel) of shape `(num_points, num_points)`, representing
            pairwise relationships between input data points.
        """
        length_scale = numpyro.param(
            f"{self.name}_length_scale",
            jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        variance = numpyro.param(
            f"{self.name}_variance",
            jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        noise = numpyro.param(
            f"{self.name}_noise", jnp.array(1.0), constraint=dist.constraints.positive
        )

        pairwise_sq_dists = (
            jnp.sum(X**2, axis=-1, keepdims=True)
            - 2 * jnp.dot(X, X.T)
            + jnp.sum(X**2, axis=-1)
        )
        kernel = variance * jnp.exp(-0.5 * pairwise_sq_dists / length_scale**2)

        return kernel + noise * jnp.eye(X.shape[0])


class VariationalLayer:
    """
    Implements a variational layer for Bayesian neural networks, with learnable
    weight distributions (mean and variance).

    The layer uses variational inference to approximate the posterior distributions
    of weights during training.
    """

    def __init__(
        self, input_dim: int, output_dim: int, name: str = "variational_layer"
    ):
        """
        Initialize the variational layer.

        :param input_dim: int
            Dimensionality of the input features.
        :param output_dim: int
            Dimensionality of the output features.
        :param name: str, optional
            Name of the layer, used for parameter naming (default: "variational_layer").
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Perform the forward pass of the variational layer.

        :param X: jnp.ndarray
            Input tensor of shape `(batch_size, input_dim)`.

        :returns: jnp.ndarray
            Output tensor of shape `(batch_size, output_dim)`, computed as the
            dot product of the input and variationally sampled weights.
        """
        # Sample mean and variance for weights
        w_mu = numpyro.sample(
            f"{self.name}_w_mu",
            dist.Normal(0, 1).expand([self.input_dim, self.output_dim]),
        )
        w_sigma = numpyro.sample(
            f"{self.name}_w_sigma",
            dist.LogNormal(0.0, 0.1).expand([self.input_dim, self.output_dim]),
        )

        # Sample weights using the mean and variance
        weights = numpyro.sample(f"{self.name}_weights", dist.Normal(w_mu, w_sigma))

        # Compute the output
        return jnp.dot(X, weights)
