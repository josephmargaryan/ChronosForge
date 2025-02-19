import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt


def plot_fft_spectrum_with_uncertainty(fft_samples, show=True):
    """
    fft_samples: shape (num_samples, n)
    Computes the mean and credible intervals for the magnitude and phase,
    and plots them.
    """
    # Compute statistics across samples
    mag_samples = np.abs(fft_samples)  # shape (num_samples, n)
    phase_samples = np.angle(fft_samples)  # shape (num_samples, n)

    mag_mean = mag_samples.mean(axis=0)
    phase_mean = phase_samples.mean(axis=0)

    # Compute, for example, 95% quantiles
    mag_lower = np.percentile(mag_samples, 2.5, axis=0)
    mag_upper = np.percentile(mag_samples, 97.5, axis=0)
    phase_lower = np.percentile(phase_samples, 2.5, axis=0)
    phase_upper = np.percentile(phase_samples, 97.5, axis=0)

    freq_idx = np.arange(fft_samples.shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(freq_idx, mag_mean, "b-", label="Mean Magnitude")
    axes[0].fill_between(
        freq_idx, mag_lower, mag_upper, color="blue", alpha=0.3, label="95% CI"
    )
    axes[0].set_title("FFT Magnitude")
    axes[0].set_xlabel("Frequency index")
    axes[0].set_ylabel("Magnitude")
    axes[0].legend()

    axes[1].plot(freq_idx, phase_mean, "g-", label="Mean Phase")
    axes[1].fill_between(
        freq_idx, phase_lower, phase_upper, color="green", alpha=0.3, label="95% CI"
    )
    axes[1].set_title("FFT Phase")
    axes[1].set_xlabel("Frequency index")
    axes[1].set_ylabel("Phase (radians)")
    axes[1].legend()

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def visualize_circulant_kernel_with_uncertainty(
    fft_samples: np.ndarray, show: bool = True
):
    """
    Visualize the uncertainty in the time-domain circulant kernel.

    Parameters:
        fft_samples: np.ndarray of shape (num_samples, n)
            Array of FFT coefficients from multiple posterior samples.
        show: bool, if True, calls plt.show().

    Returns:
        fig: the matplotlib figure.
    """
    num_samples, n = fft_samples.shape

    # Compute the time-domain kernel for each sample using the inverse FFT.
    time_kernels = np.array(
        [np.fft.ifft(fft_sample).real for fft_sample in fft_samples]
    )
    # time_kernels has shape (num_samples, n)

    # Compute summary statistics for the time-domain kernel at each time index.
    kernel_mean = time_kernels.mean(axis=0)
    kernel_lower = np.percentile(time_kernels, 2.5, axis=0)
    kernel_upper = np.percentile(time_kernels, 97.5, axis=0)

    # For the circulant matrix, compute the mean circulant matrix by rolling the mean kernel.
    C_mean = np.stack([np.roll(kernel_mean, i) for i in range(n)], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the mean kernel with error bars for uncertainty.
    # The fmt "o-" will draw both scatter points and a line connecting them.
    axes[0].errorbar(
        np.arange(n),
        kernel_mean,
        yerr=[kernel_mean - kernel_lower, kernel_upper - kernel_mean],
        fmt="o-",  # Added '-' to connect the points with a line.
        color="b",
        ecolor="lightgray",
        capsize=3,
    )
    axes[0].set_title("Circulant Kernel (Time Domain) K=None")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Amplitude")

    # Plot the mean circulant matrix.
    im = axes[1].imshow(C_mean, cmap="viridis")
    axes[1].set_title("Mean Circulant Matrix")
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Index")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def visualize_circulant_layer(fft_samples: np.ndarray, show=True):
    """
    Visualizes the FFT spectrum (magnitude and phase) and the time-domain circulant kernel.
    If fft_samples has multiple samples, uncertainty (e.g., 95% CI) is shown.

    Example Usage:
    from quantbayes.stochax.utils import get_fft_full_for_given_params, visualize_circulant_layer
    # Optionally Trigger the FFT layer's forward pass to update its stored coefficients.
    _ = net.fft_layer(x)
    posterior_samples = model.get_samples
    param_dict = {key: value[0] for key, value in posterior_samples.items()}

    # (2) Perform a forward pass to get the FFT coefficients for the circulant layer.
    fft_full = get_fft_full_for_given_params(model, X_test, param_dict, rng_key=jr.PRNGKey(0))

    # (3) To visualize uncertainty, loop over multiple posterior samples:
    fft_list = []
    n_samples = 50
    for i in range(n_samples):
        sample_param_dict = {key: value[i] for key, value in posterior_samples.items()}
        fft_sample = get_fft_full_for_given_params(model, X_test, sample_param_dict, rng_key=jr.PRNGKey(i))
        fft_list.append(fft_sample)

    # Convert the list to a NumPy array: shape (n_samples, n)
    fft_samples = np.stack(fft_list, axis=0)

    # (4) Call the high-level visualization function.
    fig_fft, fig_kernel = visualize_circulant_layer(fft_samples, show=True)
    """
    # Compute statistics (mean, lower, upper bounds) for FFT spectrum.
    fig1 = plot_fft_spectrum_with_uncertainty(fft_samples, show=False)

    # Compute time-domain kernels from fft_samples.
    fig2 = visualize_circulant_kernel_with_uncertainty(fft_samples, show=False)

    # Optionally combine or display them side by side.
    if show:
        plt.show()
    return fig1, fig2


# --- Your FourierPriorCirculant definition (as provided) ---
class FourierPriorCirculant(eqx.Module):
    R: int = eqx.static_field()  # Length of the real-valued vector r.
    K: int = eqx.static_field()  # Truncation: only frequencies < K are retained.
    alpha: float = eqx.static_field()  # Decay rate for frequency variance.
    k_half: int = eqx.static_field()  # R//2+1
    # The Fourier coefficients in the half-complex representation.
    fourier_coeffs_real: jnp.ndarray  # shape (k_half,)
    fourier_coeffs_imag: jnp.ndarray  # shape (k_half,)

    def __init__(self, R, K, alpha, *, key, init_scale=0.1):
        self.R = R
        self.alpha = alpha
        self.k_half = R // 2 + 1

        # Enforce that K does not exceed the available coefficients.
        if (K is None) or (K > self.k_half):
            K = self.k_half
        self.K = K

        key_r, key_i = jr.split(key, 2)
        # Initialize the Fourier coefficients.
        r_init = jr.normal(key_r, (self.k_half,)) * init_scale
        i_init = jr.normal(key_i, (self.k_half,)) * init_scale

        # Force the DC component to be real.
        i_init = i_init.at[0].set(0.0)
        # If R is even, the Nyquist frequency should also be real.
        if (R % 2 == 0) and (self.k_half > 1):
            i_init = i_init.at[-1].set(0.0)

        self.fourier_coeffs_real = r_init
        self.fourier_coeffs_imag = i_init

    def get_full_fourier(self):
        """
        Build the full Fourier spectrum (length R) from the half representation,
        applying the truncation mask.
        """
        freq_idx = jnp.arange(self.k_half)
        # (sigma is computed here if needed for scaling but is not applied)
        sigma = 1.0 / jnp.sqrt(1.0 + freq_idx**self.alpha)

        # Create a binary mask: only indices less than K are kept.
        mask = (freq_idx < self.K).astype(jnp.float32)

        # Apply the mask to the Fourier coefficients.
        r = self.fourier_coeffs_real * mask
        i = self.fourier_coeffs_imag * mask

        half_complex = r + 1j * i

        # Reconstruct the full Fourier spectrum using Hermitian symmetry.
        if (self.R % 2 == 0) and (self.k_half > 1):
            nyquist = half_complex[-1].real[None]  # ensure real
            full_fft = jnp.concatenate(
                [half_complex[:-1], nyquist, jnp.conjugate(half_complex[1:-1])[::-1]]
            )
        else:
            full_fft = jnp.concatenate(
                [half_complex, jnp.conjugate(half_complex[1:])[::-1]]
            )
        return full_fft

    def get_fourier_coeffs(self) -> jnp.ndarray:
        """Wrapper method for compatibility with previous code."""
        return self.get_full_fourier()

    def get_r(self):
        """
        Compute the time-domain vector r by taking the inverse DFT of the full spectrum.
        """
        full_fft = self.get_full_fourier()
        r_time = jnp.fft.ifft(full_fft).real
        return r_time

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Use the circulant vector r to perform a circulant multiplication.
        Assumes x has shape (..., R).
        """
        r = self.get_r()
        X_fft = jnp.fft.fft(x, axis=-1)
        r_fft = jnp.fft.fft(r)
        out_fft = X_fft * r_fft
        out = jnp.fft.ifft(out_fft, axis=-1).real
        return out


# --- End FourierPriorCirculant definition ---

if __name__ == "__main__":
    # For this test, we do not need the Bayesian machinery.
    # Set some parameters:
    R = 64  # length of the time-domain vector r.
    K = 7  # only retain first 7 frequencies (low-pass truncation)
    alpha = 1.0  # decay rate
    in_features = R  # For our test, the input length matches R.

    # Create a key and instantiate the layer.
    key = jr.PRNGKey(42)
    fourier_layer = FourierPriorCirculant(R, K, alpha, key=key)

    # Create a dummy input: a single sample with R elements.
    x = jnp.linspace(-1, 1, R)[None, :]  # Shape: (1, 64)

    # Do a forward pass (here, our layer applies circulant multiplication)
    pred = jax.vmap(fourier_layer)(x)
    print("Deterministic model prediction:", pred)

    # Retrieve the full Fourier spectrum and visualize it.
    fft_coeffs = fourier_layer.get_full_fourier()  # (complex vector of length R)
    fft_coeffs_np = np.array(fft_coeffs)
    print("FFT coefficients shape:", fft_coeffs_np.shape)

    # For visualization, wrap in an extra dimension (num_samples, R)
    fft_samples = np.expand_dims(fft_coeffs_np, axis=0)  # shape: (1, R)

    # Visualize the FFT spectrum (you should have a helper to show magnitude & phase)
    fig_fft, fig_kernel = visualize_circulant_layer(fft_samples, show=False)
    plt.show()
