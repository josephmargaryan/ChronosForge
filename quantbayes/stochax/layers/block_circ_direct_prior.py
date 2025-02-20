import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


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

    def __call__(self, x: jnp.ndarray, *, key=None, state=None, **kwargs) -> jnp.ndarray:
        # If a single sample is passed, add a batch dimension.
        single_example = (x.ndim == 1)
        if single_example:
            x = x[None, :]
        
        batch_size, d_in = x.shape
        # Zero-pad x so that it fits into an integer number of blocks.
        pad_len = self.k_in * self.block_size - d_in
        if pad_len > 0:
            x = jnp.pad(x, ((0, 0), (0, pad_len)))
        x_blocks = x.reshape(batch_size, self.k_in, self.block_size)
        
        def reconstruct_block_fft(r_ij, im_ij):
            b = self.block_size
            half_complex = r_ij + 1j * im_ij
            if (b % 2 == 0) and (self.k_half > 1):
                nyquist = half_complex[-1].real[None]
                block_fft = jnp.concatenate(
                    [half_complex[:-1], nyquist, jnp.conjugate(half_complex[1:-1])[::-1]]
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
    
        out_blocks = jax.vmap(compute_blockrow)(jnp.arange(self.k_out))
        out_reshaped = jnp.transpose(out_blocks, (1, 0, 2)).reshape(
            batch_size, self.k_out * self.block_size
        )
    
        if self.k_out * self.block_size > self.out_features:
            out_reshaped = out_reshaped[:, : self.out_features]
    
        # If we originally had a single sample, remove the batch dimension.
        if single_example:
            out_reshaped = out_reshaped[0]
    
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