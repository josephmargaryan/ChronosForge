import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr


class BlockCirculantProcess(eqx.Module):
    in_features: int = eqx.static_field()
    out_features: int = eqx.static_field()
    block_size: int = eqx.static_field()
    alpha: float = eqx.static_field()
    K: int = eqx.static_field()  # truncation index: frequencies >= K are zeroed.
    k_in: int = eqx.static_field()
    k_out: int = eqx.static_field()
    k_half: int = eqx.static_field()

    # Fourier coefficients for each block.
    W_real: jnp.ndarray  # shape (k_out, k_in, k_half)
    W_imag: jnp.ndarray  # shape (k_out, k_in, k_half)

    _last_fourier_coeffs: jnp.ndarray = eqx.field(default=None, repr=False)
    # This will store the reconstructed full Fourier coefficients,
    # with shape (k_out, k_in, block_size)

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

        # Compute the number of blocks for input and output.
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
        # Ensure that the DC frequency is purely real.
        Wi = Wi.at[..., 0].set(0.0)
        if (b % 2 == 0) and (k_half > 1):
            Wi = Wi.at[..., -1].set(0.0)

        self.W_real = Wr
        self.W_imag = Wi
        self._last_fourier_coeffs = None

    @property
    def prior_std(self) -> jnp.ndarray:
        """
        Compute the frequency-dependent standard deviations for the block Fourier coefficients.
        This property returns an array of shape (k_half,) based on the frequency indices.
        """
        freq_idx = jnp.arange(self.k_half)
        return 1.0 / jnp.sqrt(1.0 + freq_idx**self.alpha)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch = x.ndim == 2
        if not batch:
            x = x[None, :]
        bs, d_in = x.shape

        # Zero-pad x if needed so that its length fits an integer number of blocks.
        pad_len = self.k_in * self.block_size - d_in
        if pad_len > 0:
            x = jnp.pad(x, ((0, 0), (0, pad_len)))
        x_blocks = x.reshape(bs, self.k_in, self.block_size)

        # Use the property to get frequency-dependent std.
        freq_idx = jnp.arange(self.k_half)
        # In this deterministic forward pass, we don't multiply by prior_std,
        # but you can use it later when bayesianizing the parameters.
        freq_mask = (freq_idx < self.K).astype(jnp.float32)

        # Function to reconstruct a full Fourier block from the half-spectrum.
        def reconstruct_block(r_ij, i_ij):
            # Apply the frequency mask.
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

        # Vectorized reconstruction over (k_out, k_in).
        block_fft_full = jax.vmap(
            lambda Rrow, Irow: jax.vmap(reconstruct_block)(Rrow, Irow),
            in_axes=(0, 0),
        )(self.W_real, self.W_imag)

        # Store the reconstructed block FFT for later retrieval.
        object.__setattr__(self, "_last_fourier_coeffs", block_fft_full)

        # Multiply in the time domain.
        def multiply_blockrow(i):
            # Sum over the input blocks j.
            def sum_over_j(carry, j):
                fft_block = block_fft_full[i, j]  # shape (block_size,)
                x_j = x_blocks[:, j, :]  # shape (bs, block_size)
                X_fft = jnp.fft.fft(x_j, axis=-1)
                # Note the conjugation on the weight block.
                out_fft = X_fft * jnp.conjugate(fft_block)[None, :]
                out_time = jnp.fft.ifft(out_fft, axis=-1).real
                return carry + out_time, None

            init = jnp.zeros((bs, self.block_size))
            out_time, _ = jax.lax.scan(sum_over_j, init, jnp.arange(self.k_in))
            return out_time

        out_blocks = jax.vmap(multiply_blockrow)(
            jnp.arange(self.k_out)
        )  # shape (k_out, bs, block_size)
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
        Return the last-computed full Fourier coefficients array of shape (k_out, k_in, block_size).
        """
        if self._last_fourier_coeffs is None:
            raise ValueError(
                "No Fourier coefficients available. "
                "Call the layer on some input first."
            )
        return self._last_fourier_coeffs
