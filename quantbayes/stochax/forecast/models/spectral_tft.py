import jax
import jax.numpy as jnp
import equinox as eqx
import jax.random as jr
from jaxtyping import PRNGKeyArray

from quantbayes.stochax.forecast.models.temporal_fusion import LSTMEncoder, GatingModule


# --- Minimal Spectral Multihead Attention ---
class SpectralMultiheadAttention(eqx.Module):
    num_heads: int
    head_dim: int
    embed_dim: int
    seq_len: int = eqx.static_field()
    adaptive: bool
    dropout: eqx.nn.Dropout
    pre_norm: eqx.nn.LayerNorm
    base_filter: jnp.ndarray  # shape: (num_heads, freq_bins, 1)
    base_bias: jnp.ndarray  # shape: (num_heads, freq_bins, 1)
    adaptive_mlp: eqx.nn.MLP

    def __init__(
        self,
        embed_dim: int,
        seq_len: int,
        num_heads: int,
        dropout_rate: float,
        *,
        key: PRNGKeyArray
    ):
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.head_dim = embed_dim // num_heads
        freq_bins = seq_len // 2 + 1
        key, k1, k2, k3, k4 = jr.split(key, 5)
        self.base_filter = jr.normal(k1, (num_heads, freq_bins, 1))
        self.base_bias = jr.normal(k2, (num_heads, freq_bins, 1)) - 0.1
        self.adaptive_mlp = eqx.nn.MLP(
            in_size=embed_dim,
            out_size=num_heads * freq_bins * 2,
            width_size=embed_dim,
            depth=2,
            key=k3,
        )
        self.dropout = eqx.nn.Dropout(dropout_rate)
        self.pre_norm = eqx.nn.LayerNorm(embed_dim)
        self.adaptive = True

    def complex_activation(self, z: jnp.ndarray) -> jnp.ndarray:
        mag = jnp.abs(z)
        mag_act = jax.nn.gelu(mag)
        scale = mag_act / (mag + 1e-6)
        return z * scale

    def __call__(
        self, x: jnp.ndarray, training: bool, key: PRNGKeyArray
    ) -> jnp.ndarray:
        # x: (seq_len, embed_dim) or (B, seq_len, embed_dim)
        single_example = x.ndim == 2
        if single_example:
            x = x[None, ...]
        B, N, D = x.shape
        # Apply layer norm tokenwise.
        x_norm = jax.vmap(jax.vmap(self.pre_norm))(x)
        x_heads = x_norm.reshape(B, N, self.num_heads, self.head_dim)
        x_heads = jnp.transpose(
            x_heads, (0, 2, 1, 3)
        )  # (B, num_heads, seq_len, head_dim)
        F_fft = jnp.fft.rfft(x_heads, axis=2)
        # Compute global context from the entire sequence.
        context = x_norm.mean(axis=1)  # shape: (B, embed_dim)
        adapt_params = jax.vmap(self.adaptive_mlp)(context)
        freq_bins = F_fft.shape[2]  # should be (seq_len // 2 + 1)
        adapt_params = adapt_params.reshape(B, self.num_heads, freq_bins, 2)
        adaptive_scale = adapt_params[..., 0:1]
        adaptive_bias = adapt_params[..., 1:2]
        effective_filter = self.base_filter * (1 + adaptive_scale)
        effective_bias = self.base_bias + adaptive_bias
        F_fft_mod = F_fft * effective_filter + effective_bias
        F_fft_nl = self.complex_activation(F_fft_mod)
        x_filtered = jnp.fft.irfft(F_fft_nl, n=self.seq_len, axis=2)
        x_filtered = jnp.transpose(x_filtered, (0, 2, 1, 3))
        x_filtered = x_filtered.reshape(B, self.seq_len, self.embed_dim)
        key, subkey = jr.split(key)
        x_filtered = self.dropout(x_filtered, inference=not training, key=subkey)
        if single_example:
            return x_filtered[0]
        return x_filtered


# --- Spectral Temporal Fusion Transformer ---
class SpectralTemporalFusionTransformer(eqx.Module):
    lstm_encoder: eqx.Module  # assume your LSTMEncoder is defined elsewhere
    spectral_attn: SpectralMultiheadAttention
    gating: eqx.Module  # assume your GatingModule is defined elsewhere
    final_linear: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int,
        seq_len: int,
        *,
        key: PRNGKeyArray
    ):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.lstm_encoder = LSTMEncoder(input_size, hidden_size, key=k1)
        # Here we pass the true sequence length (e.g. 20) to spectral attention.
        self.spectral_attn = SpectralMultiheadAttention(
            embed_dim=hidden_size,
            seq_len=seq_len,
            num_heads=num_heads,
            dropout_rate=0.1,
            key=k2,
        )
        self.gating = GatingModule(2 * hidden_size, key=k3)
        self.final_linear = eqx.nn.Linear(hidden_size, 1, key=k4)
        self.hidden_size = hidden_size

    def __call__(
        self, x: jnp.ndarray, state: eqx.nn.State, *, key: PRNGKeyArray
    ) -> jnp.ndarray:
        """
        x: input sequence of shape (seq_len, input_size)
        """
        hs = self.lstm_encoder(x)  # shape: (seq_len, hidden_size)
        last_hidden = hs[-1]  # shape: (hidden_size,)
        # Apply spectral attention to the entire hidden sequence.
        attn_out = self.spectral_attn(
            hs, training=False, key=key
        )  # shape: (seq_len, hidden_size)
        # Summarize the attention output (e.g. via mean pooling).
        attn_summary = attn_out.mean(axis=0)  # shape: (hidden_size,)
        # Gated fusion between last LSTM output and the spectral summary.
        concat = jnp.concatenate([last_hidden, attn_summary], axis=-1)
        gate = self.gating(concat)
        fused = gate * attn_summary + (1 - gate) * last_hidden
        return self.final_linear(fused)


# --- Example usage ---
if __name__ == "__main__":
    # Example: each sequence has length 20 and input feature dimension 32.
    seq_len, D = 20, 32
    key = jr.PRNGKey(42)
    # Create a dummy sequence of shape (seq_len, D)
    x = jr.normal(key, (seq_len, D))
    model_key, run_key = jr.split(key)
    # Create the model with hidden_size = 32, num_heads = 4, and sequence length = 20.
    model = SpectralTemporalFusionTransformer(
        input_size=D, hidden_size=D, num_heads=4, seq_len=seq_len, key=model_key
    )
    pred = model(x, key=run_key)
    print("Spectral TFT prediction shape:", pred.shape)
