import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr


class SmoothTruncEquinoxBlockCirculant(eqx.Module):
    in_features: int = eqx.static_field()
    out_features: int = eqx.static_field()
    block_size: int = eqx.static_field()
    alpha: float = eqx.static_field()
    K: int = eqx.static_field()
    k_in: int = eqx.static_field()
    k_out: int = eqx.static_field()
    k_half: int = eqx.static_field()

    W_real: jnp.ndarray  # shape (k_out, k_in, k_half)
    W_imag: jnp.ndarray  # shape (k_out, k_in, k_half)

    _last_fourier_coeffs: jnp.ndarray = eqx.field(default=None, repr=False)
    # We'll store shape (k_out, k_in, block_size)

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

        k_in = (in_features + block_size - 1) // block_size
        k_out = (out_features + block_size - 1) // block_size
        self.k_in = k_in
        self.k_out = k_out
        b = block_size
        k_half = (b // 2) + 1
        if (K is None) or (K > k_half):
            K = k_half
        self.K = K
        self.k_half = k_half

        key_r, key_i = jr.split(key, 2)
        shape = (k_out, k_in, k_half)
        Wr = jr.normal(key_r, shape) * init_scale
        Wi = jr.normal(key_i, shape) * init_scale
        # freq=0 => imag=0
        Wi = Wi.at[..., 0].set(0.0)
        if (b % 2 == 0) and (k_half > 1):
            Wi = Wi.at[..., -1].set(0.0)

        self.W_real = Wr
        self.W_imag = Wi
        self._last_fourier_coeffs = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch = x.ndim == 2
        if not batch:
            x = x[None, :]
        bs, d_in = x.shape

        pad_len = self.k_in * self.block_size - d_in
        if pad_len > 0:
            x = jnp.pad(x, ((0, 0), (0, pad_len)))
        x_blocks = x.reshape(bs, self.k_in, self.block_size)

        # We'll reconstruct the full (k_out, k_in, block_size) arrays once.
        # Because we also do freq-dependent truncation, define freq scale and mask.
        freq_idx = jnp.arange(self.k_half)
        prior_std = 1.0 / jnp.sqrt(1.0 + freq_idx**self.alpha)
        freq_mask = (freq_idx < self.K).astype(jnp.float32)
        # We do the truncation in the forward pass by zeroing freq >= K.

        def reconstruct_block(r_ij, i_ij):
            # r_ij, i_ij shape (k_half,)
            # apply mask
            r_ij = r_ij * freq_mask
            i_ij = i_ij * freq_mask
            half_complex = r_ij + 1j * i_ij
            b = self.block_size
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

        # Vectorized reconstruction of shape (k_out, k_in, block_size)
        block_fft_full = jax.vmap(
            lambda Rrow, Irow: jax.vmap(reconstruct_block)(Rrow, Irow),
            in_axes=(0, 0),
        )(self.W_real, self.W_imag)

        # Store it for get_fourier_coeffs
        object.__setattr__(self, "_last_fourier_coeffs", block_fft_full)

        # Now multiply in time domain for each block row i
        def multiply_blockrow(i):
            # sum over j => circ(W[i,j]) x_j
            def sum_over_j(carry, j):
                fft_block = block_fft_full[i, j]  # shape (b,)
                x_j = x_blocks[:, j, :]
                X_fft = jnp.fft.fft(x_j, axis=-1)
                out_fft = X_fft * jnp.conjugate(fft_block)[None, :]
                out_time = jnp.fft.ifft(out_fft, axis=-1).real
                return carry + out_time, None

            init = jnp.zeros((bs, self.block_size))
            out_time, _ = jax.lax.scan(sum_over_j, init, jnp.arange(self.k_in))
            return out_time

        out_blocks = jax.vmap(multiply_blockrow)(
            jnp.arange(self.k_out)
        )  # (k_out, bs, b)
        out_reshaped = jnp.transpose(out_blocks, (1, 0, 2)).reshape(
            bs, self.k_out * self.block_size
        )

        if self.k_out * self.block_size > self.out_features:
            out_reshaped = out_reshaped[:, : self.out_features]

        if not batch:
            out_reshaped = out_reshaped[0]
        return out_reshaped

    def get_fourier_coeffs(self) -> jnp.ndarray:
        """
        Return the last-computed (k_out, k_in, block_size) complex array
        after the forward pass.
        """
        if self._last_fourier_coeffs is None:
            raise ValueError(
                "No Fourier coefficients available. "
                "Call the layer on some input first."
            )
        return self._last_fourier_coeffs
