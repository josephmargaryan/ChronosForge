import equinox as eqx
import jax.numpy as jnp
import jax.random as jr


class SmoothTruncEquinoxCirculant(eqx.Module):
    in_features: int = eqx.static_field()
    alpha: float = eqx.static_field()
    K: int = eqx.static_field()
    k_half: int = eqx.static_field()

    # Instead of making prior_std a static field that changes,
    # we define it as a property computed from in_features, alpha, and K.
    # This keeps our design functional.
    fourier_coeffs_real: jnp.ndarray  # shape (k_half,)
    fourier_coeffs_imag: jnp.ndarray  # shape (k_half,)

    # We'll store the final full FFT in a mutable field for retrieval.
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
        # Ensure the DC component is purely real
        i_init = i_init.at[0].set(0.0)
        if (in_features % 2 == 0) and (k_half > 1):
            i_init = i_init.at[-1].set(0.0)

        self.fourier_coeffs_real = r_init
        self.fourier_coeffs_imag = i_init
        self._last_fft_full = None

    @property
    def prior_std(self) -> jnp.ndarray:
        """
        Compute and return the frequency-dependent standard deviations.
        This is based on the frequency indices and the parameter alpha.
        """
        freq_idx = jnp.arange(self.k_half)
        # Compute prior standard deviation for each frequency
        return 1.0 / jnp.sqrt(1.0 + freq_idx**self.alpha)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch = x.ndim == 2
        n = self.in_features

        freq_idx = jnp.arange(self.k_half)
        # Use the precomputed property for prior_std and also build a mask for truncation.
        # (Note: here we are not using prior_std to scale the parameters directly,
        # but it can be used in a bayesianize step to set the prior on each Fourier coeff.)
        mask = freq_idx < self.K

        # "Truncation" at forward time: zeroing out frequencies with index >= K.
        r = self.fourier_coeffs_real * mask
        i = self.fourier_coeffs_imag * mask

        # Build the half-complex array.
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
