import jax
import jax.random as jr
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from numpyro import handlers

__all__ = [
    "plot_block_fft_spectra",
    "visualize_block_circulant_kernels",
    "get_block_fft_full_for_given_params",
    "plot_fft_spectrum",
    "visualize_circulant_kernel",
    "get_fft_full_for_given_params",
    "compare_time_domain_truncation"
    "collect_block_r_i"
    "plot_prior_posterior_frequency",
]


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


def plot_fft_spectrum(fft_full: jnp.ndarray, show: bool = True):
    """
    Plot the FFT spectrum using stem plots for magnitude and phase.
    """
    # Compute magnitude and phase.
    mag = np.asarray(jnp.abs(fft_full))
    phase = np.asarray(jnp.arctan2(jnp.imag(fft_full), jnp.real(fft_full)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    markerline, stemlines, baseline = axes[0].stem(
        mag, linefmt="b-", markerfmt="bo", basefmt="r-"
    )
    axes[0].set_title("FFT Magnitude")
    axes[0].set_xlabel("Frequency index")
    axes[0].set_ylabel("Magnitude")

    markerline2, stemlines2, baseline2 = axes[1].stem(
        phase, linefmt="g-", markerfmt="go", basefmt="r-"
    )
    axes[1].set_title("FFT Phase")
    axes[1].set_xlabel("Frequency index")
    axes[1].set_ylabel("Phase (radians)")

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def visualize_circulant_kernel(fft_full: jnp.ndarray, show: bool = True):
    """
    Visualize the time-domain circulant kernel using a stem plot, and display the full circulant matrix.
    """
    n = fft_full.shape[0]
    time_kernel = jnp.fft.ifft(fft_full).real
    C = jnp.stack([jnp.roll(time_kernel, i) for i in range(n)], axis=0)
    C_np = np.asarray(C)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Stem plot for the time-domain kernel.
    markerline, stemlines, baseline = axes[0].stem(
        np.asarray(time_kernel), linefmt="b-", markerfmt="bo", basefmt="r-"
    )
    axes[0].set_title("Circulant Kernel (Time Domain)")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Amplitude")

    # Image for the circulant matrix.
    im = axes[1].imshow(C_np, cmap="viridis")
    axes[1].set_title("Circulant Matrix")
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Index")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def get_fft_full_for_given_params(model, X, param_dict, rng_key=jr.PRNGKey(0)):
    """
    Substitute a concrete parameter set into the model and run one forward pass
    with a provided rng_key so that the sample sites receive valid keys.
    This triggers model.fft_layer to store its Fourier coefficients.
    """
    with handlers.seed(rng_seed=rng_key):
        with handlers.substitute(data=param_dict):
            _ = model(X)  # this call now receives a proper PRNG key
    fft_full = model.fft_layer.get_fourier_coeffs()
    return jax.device_get(fft_full)


##################### Needs Testing ######################


def compare_time_domain_truncation(fft_full: jnp.ndarray, K: int, title: str = ""):
    """
    :param fft_full: shape (n,) complex
    :param K: cutoff in [0..n/2], for real signals half-spectrum is n//2+1
    """
    n = fft_full.shape[0]

    # time-domain kernel from the original
    kernel_orig = jnp.fft.ifft(fft_full).real

    # Let's get the half_spectrum length
    k_half = (n // 2) + 1 if (n % 2 == 0) else (n // 2 + 1)

    # Convert full->half. We'll do the standard approach:
    # half_complex[0] = fft_full[0]
    # half_complex[k] = fft_full[k] for k in [1..k_half-1], plus conj for the mirrored side.
    # There's a small difference if n is even, but let's do a small helper:
    def full_to_half(fft_full):
        # freq=0
        half = [fft_full[0]]
        # freq=1..k_half-1
        half.extend(fft_full[1:k_half])
        return jnp.array(half)

    # Reconstruct the half-spectrum
    half_orig = full_to_half(fft_full)
    # Force freq >= K to zero
    freq_idx = jnp.arange(k_half)
    keep_mask = freq_idx < K
    half_trunc = half_orig * keep_mask

    # Now rebuild the full from half_trunc
    if n % 2 == 0 and k_half > 1:
        nyquist = half_trunc[-1].real[None]  # forcibly real
        trunc_full = jnp.concatenate(
            [
                half_trunc[:-1],
                nyquist,
                jnp.conjugate(half_trunc[1:-1])[::-1],
            ]
        )
    else:
        trunc_full = jnp.concatenate(
            [
                half_trunc,
                jnp.conjugate(half_trunc[1:])[::-1],
            ]
        )

    kernel_trunc = jnp.fft.ifft(trunc_full).real

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].stem(np.asarray(kernel_orig), linefmt="b-", markerfmt="bo")
    axes[0].set_title("Original Kernel (Time Domain)")
    axes[0].set_xlabel("Index")

    axes[1].stem(np.asarray(kernel_trunc), linefmt="r-", markerfmt="ro")
    axes[1].set_title(f"Truncated Kernel (K={K})")
    axes[1].set_xlabel("Index")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()

    return fig


def collect_block_r_i(posterior, name_prefix, k_out, k_in, k_half):
    """
    Gather all block-specific real/imag sites into arrays of shape:
      real_samples: (num_samples, k_out, k_in, k_half)
      imag_samples: (num_samples, k_out, k_in, k_half)
    """
    num_samples = posterior[f"{name_prefix}_real_0_0"].shape[0]

    real_array = jnp.zeros((num_samples, k_out, k_in, k_half))
    imag_array = jnp.zeros((num_samples, k_out, k_in, k_half))

    for i in range(k_out):
        for j in range(k_in):
            r_key = f"{name_prefix}_real_{i}_{j}"
            i_key = f"{name_prefix}_imag_{i}_{j}"
            real_array = real_array.at[:, i, j, :].set(posterior[r_key])
            imag_array = imag_array.at[:, i, j, :].set(posterior[i_key])

    return real_array, imag_array


def plot_prior_posterior_frequency(
    posterior_samples_real: np.ndarray,
    posterior_samples_imag: np.ndarray,
    alpha: float,
    K: int,
    title: str = "Prior vs Posterior in Frequency Domain",
):
    """
    :param posterior_samples_real: shape (num_samples, k_half)
    :param posterior_samples_imag: shape (num_samples, k_half)
    :param alpha: prior hyperparam for 1 / sqrt(1 + freq^alpha)
    :param K: cutoff frequency
    """
    k_half = posterior_samples_real.shape[1]
    freqs = np.arange(k_half)

    # --- Prior std dev for each freq (before truncation).
    prior_std = 1.0 / np.sqrt(1.0 + freqs**alpha)
    # If freq >= K => truncated to 0
    prior_std_trunc = prior_std.copy()
    prior_std_trunc[K:] = 0.0

    # --- Posterior mean, std dev
    post_mean_r = posterior_samples_real.mean(axis=0)
    post_mean_i = posterior_samples_imag.mean(axis=0)
    post_std_r = posterior_samples_real.std(axis=0)
    post_std_i = posterior_samples_imag.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Plot Real part
    axes[0].plot(freqs, prior_std_trunc, "r--", label="Prior std (trunc)")
    axes[0].errorbar(
        freqs,
        post_mean_r,
        yerr=post_std_r,
        fmt="b.-",
        capsize=3,
        label="Posterior (real)",
    )
    axes[0].set_title("Real Part")
    axes[0].set_xlabel("Frequency index")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()

    # Plot Imag part
    axes[1].plot(freqs, prior_std_trunc, "r--", label="Prior std (trunc)")
    axes[1].errorbar(
        freqs,
        post_mean_i,
        yerr=post_std_i,
        fmt="g.-",
        capsize=3,
        label="Posterior (imag)",
    )
    axes[1].set_title("Imag Part")
    axes[1].set_xlabel("Frequency index")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
