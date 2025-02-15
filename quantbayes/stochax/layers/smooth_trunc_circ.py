import numpy as np
import matplotlib.pyplot as plt
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr


class SmoothTruncEquinoxCirculant(eqx.Module):
    in_features: int = eqx.static_field()
    alpha: float = eqx.static_field()
    K: int = eqx.static_field()
    k_half: int = eqx.static_field()

    fourier_coeffs_real: jnp.ndarray  # shape (k_half,)
    fourier_coeffs_imag: jnp.ndarray  # shape (k_half,)

    # We'll store the final full FFT in a mutable field if you like:
    _last_fft_full: jnp.ndarray = eqx.field(default=None, repr=False)

    def __init__(self, in_features, alpha=1.0, K=None, *, key, init_scale=0.1):
        self.in_features = in_features
        self.alpha = alpha
        k_half = in_features // 2 + 1
        if (K is None) or (K > k_half):
            K = k_half
        self.K = K
        self.k_half = k_half

        key_r, key_i = jr.split(key, 2)
        r_init = jr.normal(key_r, (k_half,)) * init_scale
        i_init = jr.normal(key_i, (k_half,)) * init_scale
        i_init = i_init.at[0].set(0.0)
        if (in_features % 2 == 0) and (k_half > 1):
            i_init = i_init.at[-1].set(0.0)

        self.fourier_coeffs_real = r_init
        self.fourier_coeffs_imag = i_init

        self._last_fft_full = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch = x.ndim == 2
        n = self.in_features

        freq_idx = jnp.arange(self.k_half)
        prior_std = 1.0 / jnp.sqrt(1.0 + freq_idx**self.alpha)
        mask = freq_idx < self.K
        # "Truncation" at forward time: zeroing out freq >= K
        r = self.fourier_coeffs_real * mask
        i = self.fourier_coeffs_imag * mask

        # Build half_complex
        half_complex = r + 1j * i
        if (n % 2 == 0) and (self.k_half > 1):
            nyquist = half_complex[-1].real[None]
            fft_full = jnp.concatenate(
                [half_complex[:-1], nyquist, jnp.conjugate(half_complex[1:-1])[::-1]]
            )
        else:
            fft_full = jnp.concatenate(
                [half_complex, jnp.conjugate(half_complex[1:])[::-1]]
            )

        object.__setattr__(self, "_last_fft_full", fft_full)

        X_fft = jnp.fft.fft(x, axis=-1) if batch else jnp.fft.fft(x)
        if batch:
            out_fft = X_fft * fft_full[None, :]
            out_time = jnp.fft.ifft(out_fft, axis=-1).real
        else:
            out_fft = X_fft * fft_full
            out_time = jnp.fft.ifft(out_fft).real
        return out_time

    def get_fourier_coeffs(self) -> jnp.ndarray:
        """Return the last computed full FFT array (complex, length in_features)."""
        if self._last_fft_full is None:
            raise ValueError(
                "No Fourier coefficients available for layer. "
                "Call the layer on some input first."
            )
        return self._last_fft_full


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

    # Plot them
    import matplotlib.pyplot as plt

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
