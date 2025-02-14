import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

class SmoothTruncCirculantLayer:
    """
    NumPyro-based circulant layer that places a frequency-dependent Gaussian prior
    and truncates high frequencies (freq >= K).
    """

    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        K: int = None,
        name: str = "smooth_trunc_circ",
    ):
        self.in_features = in_features
        self.alpha = alpha
        self.name = name
        self.k_half = in_features // 2 + 1

        if K is None or K > self.k_half:
            K = self.k_half
        self.K = K

        # We'll store the final fft_full after forward pass.
        self._last_fft_full = None  # shape (in_features,) complex

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        # (1) freq-dependent scale for each index
        freq_indices = jnp.arange(self.k_half)
        prior_std = 1.0 / jnp.sqrt(1.0 + freq_indices**self.alpha)

        # (2) Zero out freq >= K
        mask = (freq_indices < self.K)
        effective_scale = prior_std * mask  # shape (k_half,)

        # (3) Sample
        real_part = numpyro.sample(
            f"{self.name}_real",
            dist.Normal(0.0, effective_scale).to_event(1),
        )
        imag_part = numpyro.sample(
            f"{self.name}_imag",
            dist.Normal(0.0, effective_scale).to_event(1),
        )

        # freq=0 => purely real
        imag_part = imag_part.at[0].set(0.0)
        # if even => freq=n/2 => real
        if (self.in_features % 2 == 0) and (self.k_half > 1):
            imag_part = imag_part.at[-1].set(0.0)

        half_complex = real_part + 1j * imag_part
        if (self.in_features % 2 == 0) and (self.k_half > 1):
            nyquist = half_complex[-1].real[None]
            fft_full = jnp.concatenate(
                [half_complex[:-1], nyquist, jnp.conjugate(half_complex[1:-1])[::-1]]
            )
        else:
            fft_full = jnp.concatenate(
                [half_complex, jnp.conjugate(half_complex[1:])[::-1]]
            )

        # Cache for get_fourier_coeffs()
        self._last_fft_full = jax.lax.stop_gradient(fft_full)

        # (5) Multiply
        X_fft = jnp.fft.fft(X, axis=-1) if (X.ndim == 2) else jnp.fft.fft(X)
        if X.ndim == 2:
            out_fft = X_fft * fft_full[None, :]
            out_time = jnp.fft.ifft(out_fft, axis=-1).real
        else:
            out_fft = X_fft * fft_full
            out_time = jnp.fft.ifft(out_fft).real
        return out_time

    def get_fourier_coeffs(self) -> jnp.ndarray:
        """Return the last-computed full FFT array (complex, length in_features)."""
        if self._last_fft_full is None:
            raise ValueError(
                "No Fourier coefficients available. "
                "Call the layer on some input first."
            )
        return self._last_fft_full



if __name__ == "__main__":
    import jax
    import jax.random as jr 

    from quantbayes import bnn
    from quantbayes.fake_data import generate_regression_data
    from quantbayes.bnn.utils import plot_hdi

    df = generate_regression_data()
    X, y = df.drop("target", axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)

    class MyNet(bnn.Module):
        def __init__(self):
            super().__init__(method="nuts", task_type="regression")
        def __call__(self, X, y=None):
            N, in_features = X.shape
            X = SmoothTruncCirculantLayer(in_features=in_features,
                                          alpha=1,
                                          K=3,
                                          name="tester")(X)
            X = jax.nn.tanh(X)
            X = bnn.Linear(in_features=in_features,
                           out_features=1,
                           name="out")(X)
            logits = X.squeeze()
            sigma = numpyro.sample("sigma", dist.Exponential(1.0))
            with numpyro.plate("data", N):
                numpyro.sample("likelihood", dist.Normal(logits, sigma), obs=y)

    train_key, val_key = jr.split(jr.key(34), 2)
    model = MyNet()
    model.compile(num_warmup=500, num_samples=1000)
    model.fit(X, y, train_key)
    model.visualize(X, y, posterior="likelihood")
    preds = model.predict(X, val_key, posterior="likelihood")
    plot_hdi(preds, X)

