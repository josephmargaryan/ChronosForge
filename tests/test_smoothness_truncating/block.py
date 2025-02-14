import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

class SmoothTruncBlockCirculantLayer:
    """
    NumPyro-based block-circulant layer. Each b x b block is parameterized
    by a half-spectrum with freq-dependent prior scale and optional truncation.
    Now vectorized for faster sampling, similar to BlockFFTDirectPriorLayer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int,
        alpha: float = 1.0,
        K: int = None,
        name: str = "smooth_trunc_block_circ",
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.alpha = alpha
        self.name = name

        self.k_in = (in_features + block_size - 1) // block_size
        self.k_out = (out_features + block_size - 1) // block_size
        self.b = block_size
        self.k_half = (block_size // 2) + 1
        if (K is None) or (K > self.k_half):
            K = self.k_half
        self.K = K

        # We'll store the final full block-level FFT in this field.
        self._last_block_fft = None  # shape (k_out, k_in, b) complex

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        if X.ndim == 1:
            X = X[None, :]
        bs, d_in = X.shape

        # 1) freq-dependent scale
        freq_idx = jnp.arange(self.k_half)
        prior_std = 1.0 / jnp.sqrt(1.0 + freq_idx**self.alpha)
        # zero out freq >= K
        mask = (freq_idx < self.K)
        eff_scale = prior_std * mask  # shape (k_half,)

        # 2) Sample real/imag of shape (k_out, k_in, k_half)
        real_coeff = numpyro.sample(
            f"{self.name}_real",
            dist.Normal(0.0, eff_scale).expand([self.k_out, self.k_in, self.k_half]),
        )
        imag_coeff = numpyro.sample(
            f"{self.name}_imag",
            dist.Normal(0.0, eff_scale).expand([self.k_out, self.k_in, self.k_half]),
        )

        # freq=0 => purely real
        imag_coeff = imag_coeff.at[..., 0].set(0.0)
        # if b even => freq=b/2 => real
        if (self.b % 2 == 0) and (self.k_half > 1):
            imag_coeff = imag_coeff.at[..., -1].set(0.0)

        # 3) Reconstruct the full b-length array for each (i,j).
        def reconstruct_fft(r_ij, i_ij):
            half_c = r_ij + 1j*i_ij
            if (self.b % 2 == 0) and (self.k_half > 1):
                nyquist = half_c[-1].real[None]
                block_fft = jnp.concatenate([
                    half_c[:-1],
                    nyquist,
                    jnp.conjugate(half_c[1:-1])[::-1]
                ])
            else:
                block_fft = jnp.concatenate([
                    half_c,
                    jnp.conjugate(half_c[1:])[::-1]
                ])
            return block_fft

        block_fft_full = jax.vmap(
            lambda Rrow, Irow: jax.vmap(reconstruct_fft)(Rrow, Irow),
            in_axes=(0, 0),
        )(real_coeff, imag_coeff)  # shape (k_out, k_in, b)

        # stop_gradient store for get_fourier_coeffs
        self._last_block_fft = jax.lax.stop_gradient(block_fft_full)

        # 4) Zero-pad X if needed, reshape
        pad_len = self.k_in*self.b - d_in
        if pad_len > 0:
            X = jnp.pad(X, ((0,0),(0,pad_len)))
        X_blocks = X.reshape(bs, self.k_in, self.b)

        # 5) Multiply in time domain
        def multiply_blockrow(i):
            # sum over j => ifft( conj(block_fft_full[i,j]) * fft(X_blocks[:,j,:]) )
            def scan_j(carry, j):
                w_ij = block_fft_full[i, j]  # shape (b,)
                x_j = X_blocks[:, j, :]      # shape (bs, b)
                X_fft = jnp.fft.fft(x_j, axis=-1)
                out_fft = X_fft * jnp.conjugate(w_ij)[None, :]
                out_time = jnp.fft.ifft(out_fft, axis=-1).real
                return carry + out_time, None

            init = jnp.zeros((bs, self.b))
            out_time, _ = jax.lax.scan(scan_j, init, jnp.arange(self.k_in))
            return out_time

        out_blocks = jax.vmap(multiply_blockrow)(jnp.arange(self.k_out))  # (k_out, bs, b)
        out_reshaped = jnp.transpose(out_blocks, (1,0,2)).reshape(bs, self.k_out*self.b)

        # slice if needed
        if self.k_out*self.b > self.out_features:
            out_reshaped = out_reshaped[:, :self.out_features]

        if X.shape[0] == 1 and bs == 1:
            out_reshaped = out_reshaped[0]
        return out_reshaped

    def get_fourier_coeffs(self) -> jnp.ndarray:
        """
        Return last-computed block-level FFT array (k_out, k_in, b).
        This is the full time-domain frequency representation for each block.
        """
        if self._last_block_fft is None:
            raise ValueError("No Fourier coefficients yet. Call the layer first.")
        return self._last_block_fft



if __name__ == "__main__":
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
            X = SmoothTruncBlockCirculantLayer(in_features=in_features,
                                                 out_features=16,
                                                 block_size=4,
                                                 alpha=1,
                                                 K=3,
                                                 name="tester")(X)
            X = jax.nn.tanh(X)
            X = bnn.Linear(in_features=16,
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