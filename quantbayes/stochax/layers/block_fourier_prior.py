import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np

from quantbayes.stochax.utils import visualize_block_circulant_layer

import matplotlib.pyplot as plt


def plot_block_fft_spectra_with_uncertainty(
    fft_samples_blocks: np.ndarray, show: bool = True
):
    """
    Plot the mean and 95% credible intervals for the magnitude and phase of each block's FFT.
    This version uses small markers and draws a line through the points.
    Subplots are arranged compactly with no individual subplot header, only an overall title.

    Parameters:
      fft_samples_blocks: np.ndarray of shape (num_samples, k_out, k_in, block_size)
      show: whether to call plt.show() at the end.

    Returns:
      Two matplotlib figures: one for magnitude and one for phase.
    """
    num_samples, k_out, k_in, b = fft_samples_blocks.shape
    total = k_out * k_in
    nrows = int(np.ceil(np.sqrt(total)))
    ncols = int(np.ceil(total / nrows))

    # Create figures for magnitude and phase.
    fig_mag, axes_mag = plt.subplots(nrows, ncols, figsize=(1.5 * ncols, 1.5 * nrows))
    fig_phase, axes_phase = plt.subplots(
        nrows, ncols, figsize=(1.5 * ncols, 1.5 * nrows)
    )

    # Flatten axes for easier indexing.
    axes_mag = np.array(axes_mag).flatten()
    axes_phase = np.array(axes_phase).flatten()

    for idx in range(total):
        i = idx // k_in
        j = idx % k_in

        # Extract complex FFT samples for this block.
        block_samples = fft_samples_blocks[:, i, j, :]  # shape (num_samples, b)
        mag_samples = np.abs(block_samples)
        phase_samples = np.angle(block_samples)

        # Compute statistics.
        mag_mean = mag_samples.mean(axis=0)
        mag_lower = np.percentile(mag_samples, 2.5, axis=0)
        mag_upper = np.percentile(mag_samples, 97.5, axis=0)

        phase_mean = phase_samples.mean(axis=0)
        phase_lower = np.percentile(phase_samples, 2.5, axis=0)
        phase_upper = np.percentile(phase_samples, 97.5, axis=0)

        freq_idx = np.arange(b)

        # Plot magnitude.
        ax_mag = axes_mag[idx]
        ax_mag.plot(freq_idx, mag_mean, "b-", lw=1)
        ax_mag.scatter(freq_idx, mag_mean, s=10, c="b", zorder=3)
        ax_mag.fill_between(freq_idx, mag_lower, mag_upper, color="blue", alpha=0.3)
        # Remove individual title and extra labels.
        ax_mag.set_xticks([])
        ax_mag.set_yticks([])
        ax_mag.set_title("")

        # Plot phase.
        ax_phase = axes_phase[idx]
        ax_phase.plot(freq_idx, phase_mean, "g-", lw=1)
        ax_phase.scatter(freq_idx, phase_mean, s=10, c="g", zorder=3)
        ax_phase.fill_between(
            freq_idx, phase_lower, phase_upper, color="green", alpha=0.3
        )
        ax_phase.set_xticks([])
        ax_phase.set_yticks([])
        ax_phase.set_title("")

    # Hide extra axes if any.
    for ax in axes_mag[total:]:
        ax.set_visible(False)
    for ax in axes_phase[total:]:
        ax.set_visible(False)

    fig_mag.suptitle("Block FFT Magnitude", fontsize=12)
    fig_phase.suptitle("Block FFT Phase", fontsize=12)
    fig_mag.tight_layout(rect=[0, 0, 1, 0.93])
    fig_phase.tight_layout(rect=[0, 0, 1, 0.93])

    if show:
        plt.show()
    return fig_mag, fig_phase


def visualize_block_circulant_kernels_with_uncertainty(
    fft_samples_blocks: np.ndarray, show: bool = True
):
    """
    Visualize the uncertainty in the time-domain circulant kernels for each block.
    Subplots are arranged compactly with small markers and a connecting line.

    Parameters:
      fft_samples_blocks: np.ndarray of shape (num_samples, k_out, k_in, block_size)
      show: whether to call plt.show() at the end.

    Returns:
      A matplotlib figure.
    """
    num_samples, k_out, k_in, b = fft_samples_blocks.shape
    total = k_out * k_in
    nrows = int(np.ceil(np.sqrt(total)))
    ncols = int(np.ceil(total / nrows))

    fig, axes = plt.subplots(nrows, ncols, figsize=(1.5 * ncols, 1.5 * nrows))
    axes = np.array(axes).flatten()

    for idx in range(total):
        i = idx // k_in
        j = idx % k_in

        block_fft_samples = fft_samples_blocks[:, i, j, :]  # shape (num_samples, b)
        time_kernels = np.array(
            [np.fft.ifft(sample).real for sample in block_fft_samples]
        )
        kernel_mean = time_kernels.mean(axis=0)
        kernel_lower = np.percentile(time_kernels, 2.5, axis=0)
        kernel_upper = np.percentile(time_kernels, 97.5, axis=0)

        t = np.arange(b)
        ax = axes[idx]
        ax.plot(t, kernel_mean, "k-", lw=1)
        ax.scatter(t, kernel_mean, s=10, c="k", zorder=3)
        ax.fill_between(t, kernel_lower, kernel_upper, color="gray", alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("")

    for ax in axes[total:]:
        ax.set_visible(False)

    fig.suptitle("Block Time-Domain Kernels", fontsize=12)
    fig.tight_layout(pad=0.5)
    if show:
        plt.show()
    return fig


def visualize_block_circulant_matrices_with_uncertainty(
    fft_samples_blocks: np.ndarray, show: bool = True
):
    """
    Visualize the uncertainty in the circulant matrices for each block.
    For each block, compute the mean time-domain kernel (via IFFT) and then roll it to
    form a circulant matrix. Plots are arranged compactly.

    Parameters:
        fft_samples_blocks: np.ndarray of shape (num_samples, k_out, k_in, block_size)
        show: whether to call plt.show() at the end.

    Returns:
        A matplotlib figure.
    """
    num_samples, k_out, k_in, b = fft_samples_blocks.shape
    total = k_out * k_in
    nrows = int(np.ceil(np.sqrt(total)))
    ncols = int(np.ceil(total / nrows))

    fig, axes = plt.subplots(nrows, ncols, figsize=(1.5 * ncols, 1.5 * nrows))
    axes = np.array(axes).flatten()

    for idx in range(total):
        i = idx // k_in
        j = idx % k_in

        block_fft_samples = fft_samples_blocks[:, i, j, :]
        time_kernels = np.array(
            [np.fft.ifft(sample).real for sample in block_fft_samples]
        )
        kernel_mean = time_kernels.mean(axis=0)
        # Construct circulant matrix from the mean kernel.
        C_mean = np.stack([np.roll(kernel_mean, shift=k) for k in range(b)], axis=0)

        ax = axes[idx]
        im = ax.imshow(C_mean, cmap="viridis", aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("")
        # Remove the colorbar call entirely:
        # plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[total:]:
        ax.set_visible(False)

    fig.suptitle("Block Circulant Matrices", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if show:
        plt.show()
    return fig


def visualize_block_circulant_layer(fft_samples_blocks: np.ndarray, show=True):
    """
    High-level function to visualize block circulant layers.
    Returns four figures: one for FFT magnitude, one for FFT phase, one for the time-domain kernels,
    and one for the circulant matrices.
    """
    fig_mag, fig_phase = plot_block_fft_spectra_with_uncertainty(
        fft_samples_blocks, show=False
    )
    fig_kernel = visualize_block_circulant_kernels_with_uncertainty(
        fft_samples_blocks, show=False
    )
    fig_circ = visualize_block_circulant_matrices_with_uncertainty(
        fft_samples_blocks, show=False
    )

    if show:
        plt.show()
    return fig_mag, fig_phase, fig_kernel, fig_circ


# A refactored block circulant layer.
class BlockFourierCirculant(eqx.Module):
    in_features: int = eqx.static_field()  # total input dimension
    out_features: int = eqx.static_field()  # total output dimension
    block_size: int = eqx.static_field()  # size of each block
    alpha: float = eqx.static_field()  # decay rate for frequency variance
    K: int = eqx.static_field()  # truncation index: keep frequencies < K
    k_in: int = eqx.static_field()  # number of input blocks
    k_out: int = eqx.static_field()  # number of output blocks
    k_half: int = eqx.static_field()  # half-spectrum size (block_size//2 + 1)

    # Fourier coefficients for each block, stored in a half-complex form.
    W_real: jnp.ndarray  # shape: (k_out, k_in, k_half)
    W_imag: jnp.ndarray  # shape: (k_out, k_in, k_half)

    def __init__(
        self,
        in_features,
        out_features,
        block_size,
        alpha=1.0,
        K=None,
        *,
        key,
        init_scale=0.1
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.alpha = alpha

        # Compute number of blocks along input and output.
        k_in = (in_features + block_size - 1) // block_size
        k_out = (out_features + block_size - 1) // block_size
        self.k_in = k_in
        self.k_out = k_out

        b = block_size
        k_half = b // 2 + 1
        if (K is None) or (K > k_half):
            K = k_half
        self.K = K
        self.k_half = k_half

        shape = (k_out, k_in, k_half)
        key_r, key_i = jr.split(key, 2)
        W_real = jr.normal(key_r, shape) * init_scale
        W_imag = jr.normal(key_i, shape) * init_scale
        # Force the DC component to be real:
        W_imag = W_imag.at[..., 0].set(0.0)
        if (b % 2 == 0) and (k_half > 1):
            W_imag = W_imag.at[..., -1].set(0.0)

        self.W_real = W_real
        self.W_imag = W_imag

    def get_full_fourier(self) -> jnp.ndarray:
        """
        For each block weight (for each output block i and input block j),
        reconstruct the full Fourier spectrum (of length block_size) from the half-spectrum.
        """
        freq_idx = jnp.arange(self.k_half)
        mask = (freq_idx < self.K).astype(jnp.float32)
        # Broadcast the mask to shape (k_out, k_in, k_half)
        mask = jnp.broadcast_to(mask, self.W_real.shape)

        # Apply the mask (zero out frequencies above the truncation index).
        r_masked = self.W_real * mask
        i_masked = self.W_imag * mask
        half_complex = r_masked + 1j * i_masked

        # Define a helper that reconstructs a full spectrum from a half-spectrum vector.
        def reconstruct_half(half):
            b = self.block_size
            if (b % 2 == 0) and (self.k_half > 1):
                # For even block sizes, treat the Nyquist frequency separately.
                nyquist = half[-1].real[None]
                full = jnp.concatenate(
                    [half[:-1], nyquist, jnp.conjugate(half[1:-1])[::-1]]
                )
            else:
                full = jnp.concatenate([half, jnp.conjugate(half[1:])[::-1]])
            return full

        # Apply the reconstruction over the last axis.
        # First vmap over the last axis for each (i,j) pair.
        reconstruct_vmap = jax.vmap(reconstruct_half, in_axes=0)
        # Then vmap over the first two dimensions.
        full_fft = jax.vmap(lambda mat: jax.vmap(reconstruct_half)(mat), in_axes=0)(
            half_complex
        )
        # full_fft has shape (k_out, k_in, block_size)
        return full_fft

    def get_fourier_coeffs(self) -> jnp.ndarray:
        # Wrapper for compatibility.
        return self.get_full_fourier()

    def get_r(self) -> jnp.ndarray:
        """
        Compute the time-domain block kernels via IFFT.
        Returns an array of shape (k_out, k_in, block_size).
        """
        full_fft = self.get_full_fourier()
        r_time = jnp.fft.ifft(full_fft, axis=-1).real
        return r_time

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Perform block circulant multiplication.

        Expects input x to have shape (batch_size, in_features).
        The input is zero-padded if necessary to match (k_in * block_size).
        Returns output with shape (batch_size, out_features).
        """
        bs = x.shape[0]
        # Pad x so its length is a multiple of block_size.
        pad_len = self.k_in * self.block_size - x.shape[1]
        if pad_len > 0:
            x = jnp.pad(x, ((0, 0), (0, pad_len)))
        # Reshape x into blocks: shape (batch_size, k_in, block_size)
        x_blocks = x.reshape(bs, self.k_in, self.block_size)

        # Get the time-domain kernels for each block: shape (k_out, k_in, block_size)
        kernels = self.get_r()

        # For each output block, sum the contributions from each input block.
        # Here we use a nested vmap to process the blocks over the batch.
        def block_multiply(output_idx, x_blocks):
            # For a fixed output block (output_idx), sum over all input blocks.
            def multiply_one_input(input_idx, x_block):
                # Multiply x_block (shape (block_size,)) with kernel for block (output_idx, input_idx)
                k = kernels[output_idx, input_idx]  # shape (block_size,)
                # Here we assume a circular convolution (can be done via FFT if desired)
                # For simplicity, we use elementwise multiplication and summation.
                # (In practice, you might replace this with a proper convolution.)
                return jnp.fft.ifft(jnp.fft.fft(x_block) * jnp.fft.fft(k)).real

            # Sum over input blocks.
            summed = jnp.sum(
                jax.vmap(multiply_one_input)(jnp.arange(self.k_in), x_blocks), axis=0
            )
            return summed  # shape (block_size,)

        # Apply block_multiply for each output block.
        output_blocks = jax.vmap(
            lambda idx: jax.vmap(lambda xb: block_multiply(idx, xb))(x_blocks)
        )(jnp.arange(self.k_out))
        # output_blocks shape: (k_out, batch_size, block_size)
        # Rearrange to (batch_size, k_out, block_size) and then flatten.
        out = jnp.transpose(output_blocks, (1, 0, 2)).reshape(
            bs, self.k_out * self.block_size
        )
        # Trim the output if it exceeds out_features.
        out = out[:, : self.out_features]
        return out


# --- Testing the refactored block circulant layer ---
if __name__ == "__main__":
    # Parameters for testing:
    in_features = 512
    out_features = 1024
    block_size = 62
    alpha = 1.0
    K = 7  # retain only first 7 frequencies per block
    key = jr.PRNGKey(42)

    # Instantiate the block circulant layer.
    block_layer = BlockFourierCirculant(
        in_features, out_features, block_size, alpha, K, key=key
    )

    # Create a dummy input: batch of 10 samples.
    x = jnp.linspace(-1, 1, in_features)[None, :].repeat(10, axis=0)

    # Forward pass.
    out = block_layer(x)
    print("Output shape:", out.shape)

    # Retrieve the full Fourier spectrum for visualization.
    fft_full = block_layer.get_full_fourier()
    fft_full_np = np.array(fft_full)
    print("Block FFT coefficients shape:", fft_full_np.shape)

    # (You can now use your visualization functions, e.g., visualize_block_circulant_layer,
    # to inspect fft_full_np.)
    # For example, if visualize_block_circulant_layer expects shape
    # (num_samples, k_out, k_in, block_size), you might wrap fft_full_np in an extra dimension:
    fft_samples = np.expand_dims(fft_full_np, axis=0)
    visualize_block_circulant_layer(fft_samples, show=True)
