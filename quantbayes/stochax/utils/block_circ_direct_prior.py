import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt
import numpyro.handlers as handlers

__all__ = [
    "BlockFFTDirectPrior",
    "plot_block_fft_spectra",
    "visualize_block_circulant_kernels",
    "get_block_fft_full_for_given_params",
]


class BlockFFTDirectPrior(eqx.Module):
    """
    A deterministic Equinox layer that stores the half-spectrum (real, imag)
    for each block (k_out, k_in). The block-circulant multiplication is done in
    the frequency domain.
    """

    in_features: int = eqx.static_field()
    out_features: int = eqx.static_field()
    block_size: int = eqx.static_field()
    k_in: int = eqx.static_field()
    k_out: int = eqx.static_field()
    k_half: int = eqx.static_field()

    W_real: jnp.ndarray  # shape: (k_out, k_in, k_half)
    W_imag: jnp.ndarray  # shape: (k_out, k_in, k_half)

    def __init__(self, in_features, out_features, block_size, *, key, init_scale=0.1):
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.k_in = (in_features + block_size - 1) // block_size
        self.k_out = (out_features + block_size - 1) // block_size
        self.k_half = (block_size // 2) + 1

        key_r, key_i = jr.split(key, 2)
        shape = (self.k_out, self.k_in, self.k_half)
        real_init = jr.normal(key_r, shape) * init_scale
        imag_init = jr.normal(key_i, shape) * init_scale

        # Zero out the imaginary part at freq=0 (and freq=b/2 if block_size is even)
        imag_init = imag_init.at[..., 0].set(0.0)
        if (block_size % 2 == 0) and (self.k_half > 1):
            imag_init = imag_init.at[..., -1].set(0.0)

        object.__setattr__(self, "W_real", real_init)
        object.__setattr__(self, "W_imag", imag_init)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim == 1:
            x = x[None, :]
        batch_size, d_in = x.shape

        # Zero-pad x so that it fits into an integer number of blocks.
        pad_len = self.k_in * self.block_size - d_in
        if pad_len > 0:
            x = jnp.pad(x, ((0, 0), (0, pad_len)))
        x_blocks = x.reshape(batch_size, self.k_in, self.block_size)

        # This helper reconstructs the full Fourier vector for a given block
        def reconstruct_block_fft(r_ij, im_ij):
            b = self.block_size
            half_complex = r_ij + 1j * im_ij
            if (b % 2 == 0) and (self.k_half > 1):
                nyquist = half_complex[-1].real[None]
                block_fft = jnp.concatenate(
                    [
                        half_complex[:-1],
                        nyquist,
                        jnp.conjugate(half_complex[1:-1])[::-1],
                    ]
                )
            else:
                block_fft = jnp.concatenate(
                    [half_complex, jnp.conjugate(half_complex[1:])[::-1]]
                )
            return block_fft

        Wr = self.W_real
        Wi = self.W_imag

        def compute_blockrow(i):
            def fn(carry, j):
                x_j = x_blocks[:, j, :]  # shape: (batch_size, block_size)
                r_ij = Wr[i, j]
                im_ij = Wi[i, j]
                block_fft = reconstruct_block_fft(r_ij, im_ij)
                X_fft = jnp.fft.fft(x_j, axis=-1)
                out_fft = X_fft * jnp.conjugate(block_fft)[None, :]
                out_time = jnp.fft.ifft(out_fft, axis=-1).real
                return carry + out_time, None

            init = jnp.zeros((batch_size, self.block_size))
            out_time, _ = jax.lax.scan(fn, init, jnp.arange(self.k_in))
            return out_time  # shape: (batch_size, block_size)

        out_blocks = jax.vmap(compute_blockrow)(
            jnp.arange(self.k_out)
        )  # (k_out, batch_size, block_size)
        out_reshaped = jnp.transpose(out_blocks, (1, 0, 2)).reshape(
            batch_size, self.k_out * self.block_size
        )

        if self.k_out * self.block_size > self.out_features:
            out_reshaped = out_reshaped[:, : self.out_features]

        return out_reshaped

    def get_fourier_coeffs(self) -> jnp.ndarray:
        """
        Reconstruct and return the full Fourier coefficients for each block.
        Output shape: (k_out, k_in, block_size) (complex).
        """

        def reconstruct_full(r_ij, im_ij):
            b = self.block_size
            half_complex = r_ij + 1j * im_ij
            if (b % 2 == 0) and (self.k_half > 1):
                nyquist = half_complex[-1].real[None]
                return jnp.concatenate(
                    [
                        half_complex[:-1],
                        nyquist,
                        jnp.conjugate(half_complex[1:-1])[::-1],
                    ]
                )
            else:
                return jnp.concatenate(
                    [half_complex, jnp.conjugate(half_complex[1:])[::-1]]
                )

        # Vectorize first over the k_in dimension then over k_out.
        v_reconstruct = jax.vmap(
            lambda r_row, im_row: jax.vmap(reconstruct_full)(r_row, im_row),
            in_axes=(0, 0),
        )
        fft_full = v_reconstruct(self.W_real, self.W_imag)
        return fft_full


def plot_block_fft_spectra(fft_full_blocks: jnp.ndarray, show: bool = True):
    """
    Plot the FFT spectrum (magnitude) for each block weight matrix using stem plots.
    The subplots are arranged in a near-square grid.

    Expects fft_full_blocks with shape (k_out, k_in, block_size) (complex).
    """
    fft_blocks = np.asarray(fft_full_blocks)
    k_out, k_in, b = fft_blocks.shape
    total = k_out * k_in
    nrows = int(np.ceil(np.sqrt(total)))
    ncols = int(np.ceil(total / nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).flatten()

    for idx in range(total):
        i = idx // k_in  # block row (if original ordering is row-major)
        j = idx % k_in  # block column
        mag = np.abs(fft_blocks[i, j])
        ax = axes[idx]
        ax.stem(mag, linefmt="b-", markerfmt="bo", basefmt="r-")
        ax.set_title(f"Block ({i},{j}) FFT Mag")
        ax.set_xlabel("Freq index")
        ax.set_ylabel("Magnitude")

    # Hide any extra axes.
    for ax in axes[total:]:
        ax.set_visible(False)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def visualize_block_circulant_kernels(fft_full_blocks: jnp.ndarray, show: bool = True):
    """
    For each block, compute the time-domain kernel (via iFFT) and build the circulant matrix.
    The resulting circulant matrices are displayed in a near-square grid.

    Expects fft_full_blocks with shape (k_out, k_in, block_size) (complex).
    """
    fft_blocks = np.asarray(fft_full_blocks)
    k_out, k_in, b = fft_blocks.shape
    total = k_out * k_in
    nrows = int(np.ceil(np.sqrt(total)))
    ncols = int(np.ceil(total / nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).flatten()

    for idx in range(total):
        i = idx // k_in
        j = idx % k_in
        fft_block = fft_blocks[i, j]
        # Compute time-domain kernel via iFFT.
        time_kernel = jnp.fft.ifft(fft_block).real
        # Build the circulant matrix: each row is a roll of time_kernel.
        C = jnp.stack([jnp.roll(time_kernel, shift=k) for k in range(b)], axis=0)
        C_np = np.asarray(C)
        ax = axes[idx]
        im = ax.imshow(C_np, cmap="viridis")
        ax.set_title(f"Block ({i},{j}) Circulant")
        ax.set_xlabel("Index")
        ax.set_ylabel("Index")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[total:]:
        ax.set_visible(False)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def get_block_fft_full_for_given_params(model, X, param_dict, rng_key):
    """
    Given a dictionary of parameter draws (e.g. from MCMC) and an rng_key,
    run one forward pass so that the layer sees these values and saves
    its FFT arrays in `.last_fourier_coeffs`.
    """
    with handlers.seed(rng_seed=rng_key):
        with handlers.substitute(data=param_dict):
            _ = model(X)  # triggers the block_layer call
    # Now retrieve the block-layer’s stored FFT
    fft_full = model.block_layer.get_fourier_coeffs()
    return jax.device_get(fft_full)
