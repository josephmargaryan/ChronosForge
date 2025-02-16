import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from quantbayes.stochax.utils import (
    CirculantVisualizer,
    plot_fft_spectrum,
    visualize_circulant_kernel,
    get_fft_full_for_given_params,
)


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
            fft_layer = bnn.SmoothTruncCirculantLayer(
                in_features=in_features, alpha=1, K=3, name="fft_layer"
            )
            X = fft_layer(X)
            X = jax.nn.tanh(X)
            X = bnn.Linear(in_features=in_features, out_features=1, name="out")(X)
            logits = X.squeeze()
            sigma = numpyro.sample("sigma", dist.Exponential(1.0))
            with numpyro.plate("data", N):
                numpyro.sample("likelihood", dist.Normal(logits, sigma), obs=y)
            self.fft_layer = fft_layer

    train_key, val_key = jr.split(jr.key(34), 2)
    model = MyNet()
    model.compile(num_warmup=500, num_samples=1000)
    model.fit(X, y, train_key)
    model.visualize(X, y, posterior="likelihood")
    preds = model.predict(X, val_key, posterior="likelihood")
    plot_hdi(preds, X)

    posterior_samples = model.get_samples

    """    
    visualizer = CirculantVisualizer(
    model, 
    posterior_samples, 
    X,
    fft_layer_names=['fft_layer'],
    model_type='numpyro',
    overlay_samples=True,
    random_draws=30,
    ignore_keys=["logits", "likelihood", "sigma"]
    )

    # (A) Visualize one single posterior draw for the chosen Fourier layer.
    fig_single = visualizer.visualize_single('fft_layer', sample_index=0, rng_key=jr.PRNGKey(0))

    # (B) Visualize the full posterior uncertainty.
    figs = visualizer.visualize_posterior('fft_layer', n_draws=None)"""

    """    
    # (2) Perform a forward pass with a valid RNG key to get a concrete fft_full.
       
    param_dict = {key: value[0] for key, value in posterior_samples.items()}
    fft_full = get_fft_full_for_given_params(
        model, X, param_dict, rng_key=jr.PRNGKey(0)
    )

    # (3) Plot the Fourier spectrum and circulant kernel.
    fig1 = plot_fft_spectrum(fft_full, show=True)
    fig2 = visualize_circulant_kernel(fft_full, show=True)"""

    import numpy as np
    import matplotlib.pyplot as plt
    import jax

    class CirculantVisualizer:
        """
        Visualizer for a single circulant (FFT-based) layer.

        This class consolidates functions for plotting the FFT spectrum (magnitude
        and phase) as well as visualizing the time-domain circulant kernel.
        It gracefully handles conversion from JAX tracers/arrays to NumPy arrays.
        """

        def __init__(self, fft_full):
            # Convert to a numpy array if necessary.
            self.fft_full = self._to_numpy(fft_full)

        @staticmethod
        def _to_numpy(x):
            # This helper function converts jax arrays or tracers to np.array.
            if isinstance(x, (jax.Array,)):
                return np.asarray(x)
            return np.array(x)

        def plot_spectrum(self, show=True):
            """
            Plot the FFT spectrum (magnitude and phase) using stem plots.
            """
            fft_full_np = self._to_numpy(self.fft_full)
            mag = np.abs(fft_full_np)
            phase = np.angle(fft_full_np)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].stem(mag, linefmt="b-", markerfmt="bo", basefmt="r-")
            axes[0].set_title("FFT Magnitude")
            axes[0].set_xlabel("Frequency index")
            axes[0].set_ylabel("Magnitude")

            axes[1].stem(phase, linefmt="g-", markerfmt="go", basefmt="r-")
            axes[1].set_title("FFT Phase")
            axes[1].set_xlabel("Frequency index")
            axes[1].set_ylabel("Phase (radians)")

            plt.tight_layout()
            if show:
                plt.show()
            return fig

        def visualize_kernel(self, show=True):
            """
            Visualize the time-domain circulant kernel and its circulant matrix.
            """
            fft_full_np = self._to_numpy(self.fft_full)
            n = fft_full_np.shape[0]
            time_kernel = np.fft.ifft(fft_full_np).real
            # Build the circulant matrix.
            C = np.stack([np.roll(time_kernel, i) for i in range(n)], axis=0)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            # Stem plot of the time-domain kernel.
            axes[0].stem(time_kernel, linefmt="b-", markerfmt="bo", basefmt="r-")
            axes[0].set_title("Circulant Kernel (Time Domain)")
            axes[0].set_xlabel("Index")
            axes[0].set_ylabel("Amplitude")

            # Show the circulant matrix.
            im = axes[1].imshow(C, cmap="viridis")
            axes[1].set_title("Circulant Matrix")
            axes[1].set_xlabel("Index")
            axes[1].set_ylabel("Index")
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

            plt.tight_layout()
            if show:
                plt.show()
            return fig

        def plot_spectrum_with_uncertainty(self, fft_samples, show=True):
            """
            Plot the FFT spectrum with uncertainty estimates (e.g. 95% credible intervals).

            Parameters:
                fft_samples: array of shape (num_samples, n)
            """
            fft_samples_np = self._to_numpy(fft_samples)
            mag_samples = np.abs(fft_samples_np)
            phase_samples = np.angle(fft_samples_np)

            mag_mean = mag_samples.mean(axis=0)
            phase_mean = phase_samples.mean(axis=0)

            # 95% credible intervals.
            mag_lower = np.percentile(mag_samples, 2.5, axis=0)
            mag_upper = np.percentile(mag_samples, 97.5, axis=0)
            phase_lower = np.percentile(phase_samples, 2.5, axis=0)
            phase_upper = np.percentile(phase_samples, 97.5, axis=0)

            freq_idx = np.arange(fft_samples_np.shape[1])
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].plot(freq_idx, mag_mean, "b-", label="Mean Magnitude")
            axes[0].fill_between(
                freq_idx, mag_lower, mag_upper, color="blue", alpha=0.3, label="95% CI"
            )
            axes[0].set_title("FFT Magnitude with Uncertainty")
            axes[0].set_xlabel("Frequency index")
            axes[0].set_ylabel("Magnitude")
            axes[0].legend()

            axes[1].plot(freq_idx, phase_mean, "g-", label="Mean Phase")
            axes[1].fill_between(
                freq_idx,
                phase_lower,
                phase_upper,
                color="green",
                alpha=0.3,
                label="95% CI",
            )
            axes[1].set_title("FFT Phase with Uncertainty")
            axes[1].set_xlabel("Frequency index")
            axes[1].set_ylabel("Phase (radians)")
            axes[1].legend()

            plt.tight_layout()
            if show:
                plt.show()
            return fig

        def visualize_kernel_with_uncertainty(self, fft_samples, show=True):
            """
            Visualize the uncertainty in the time-domain circulant kernel.

            Parameters:
                fft_samples: array of shape (num_samples, n)
            """
            fft_samples_np = self._to_numpy(fft_samples)
            num_samples, n = fft_samples_np.shape

            # Compute time-domain kernels for all samples.
            time_kernels = np.array(
                [np.fft.ifft(fft_sample).real for fft_sample in fft_samples_np]
            )
            kernel_mean = time_kernels.mean(axis=0)
            kernel_lower = np.percentile(time_kernels, 2.5, axis=0)
            kernel_upper = np.percentile(time_kernels, 97.5, axis=0)

            # Mean circulant matrix.
            C_mean = np.stack([np.roll(kernel_mean, i) for i in range(n)], axis=0)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].errorbar(
                np.arange(n),
                kernel_mean,
                yerr=[kernel_mean - kernel_lower, kernel_upper - kernel_mean],
                fmt="o",
                color="b",
                ecolor="lightgray",
                capsize=3,
            )
            axes[0].set_title("Circulant Kernel (Time Domain) with Uncertainty")
            axes[0].set_xlabel("Index")
            axes[0].set_ylabel("Amplitude")

            im = axes[1].imshow(C_mean, cmap="viridis")
            axes[1].set_title("Mean Circulant Matrix")
            axes[1].set_xlabel("Index")
            axes[1].set_ylabel("Index")
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

            plt.tight_layout()
            if show:
                plt.show()
            return fig

    class BlockCirculantVisualizer:
        """
        Visualizer for block-circulant layers.

        This class consolidates functions for visualizing the FFT spectra of each
        block (using stem plots) and the time-domain circulant kernels for each block.
        It handles conversion from JAX tracers gracefully.
        """

        def __init__(self, fft_full_blocks):
            """
            Parameters:
                fft_full_blocks: array of shape (k_out, k_in, block_size)
                    The full Fourier coefficients for each block.
            """
            self.fft_full_blocks = self._to_numpy(fft_full_blocks)
            self.k_out, self.k_in, self.b = self.fft_full_blocks.shape

        @staticmethod
        def _to_numpy(x):
            if isinstance(x, (jax.Array,)):
                return np.asarray(x)
            return np.array(x)

        def plot_block_spectra(self, show=True):
            """
            Plot the FFT magnitude for each block weight matrix using stem plots.
            """
            fft_blocks = self.fft_full_blocks
            total = self.k_out * self.k_in
            nrows = int(np.ceil(np.sqrt(total)))
            ncols = int(np.ceil(total / nrows))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
            axes = np.array(axes).flatten()

            for idx in range(total):
                i = idx // self.k_in
                j = idx % self.k_in
                mag = np.abs(fft_blocks[i, j])
                ax = axes[idx]
                ax.stem(mag, linefmt="b-", markerfmt="bo", basefmt="r-")
                ax.set_title(f"Block ({i},{j}) FFT Mag")
                ax.set_xlabel("Freq index")
                ax.set_ylabel("Magnitude")

            # Hide any extra subplots.
            for ax in axes[total:]:
                ax.set_visible(False)
            plt.tight_layout()
            if show:
                plt.show()
            return fig

        def visualize_block_kernels(self, show=True):
            """
            For each block, compute the time-domain kernel (via iFFT) and
            display the corresponding circulant matrix.
            """
            fft_blocks = self.fft_full_blocks
            total = self.k_out * self.k_in
            nrows = int(np.ceil(np.sqrt(total)))
            ncols = int(np.ceil(total / nrows))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
            axes = np.array(axes).flatten()

            for idx in range(total):
                i = idx // self.k_in
                j = idx % self.k_in
                fft_block = fft_blocks[i, j]
                # Compute time-domain kernel.
                time_kernel = np.fft.ifft(fft_block).real
                # Build the circulant matrix.
                C = np.stack([np.roll(time_kernel, k) for k in range(self.b)], axis=0)
                ax = axes[idx]
                im = ax.imshow(C, cmap="viridis")
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

        def plot_block_spectra_with_uncertainty(self, fft_samples, show=True):
            """
            Given fft_samples with shape (num_samples, k_out, k_in, block_size),
            compute mean and 95% credible intervals for the magnitude and phase,
            and plot them blockwise.
            """
            fft_samples = self._to_numpy(fft_samples)
            num_samples, k_out, k_in, b = fft_samples.shape

            total = k_out * k_in
            nrows = int(np.ceil(np.sqrt(total)))
            ncols = int(np.ceil(total / nrows))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
            axes = np.array(axes).flatten()

            for idx in range(total):
                i = idx // k_in
                j = idx % k_in
                block_samples = fft_samples[
                    :, i, j, :
                ]  # shape (num_samples, block_size)
                mag_samples = np.abs(block_samples)
                phase_samples = np.angle(block_samples)
                mag_mean = mag_samples.mean(axis=0)
                phase_mean = phase_samples.mean(axis=0)
                mag_lower = np.percentile(mag_samples, 2.5, axis=0)
                mag_upper = np.percentile(mag_samples, 97.5, axis=0)
                phase_lower = np.percentile(phase_samples, 2.5, axis=0)
                phase_upper = np.percentile(phase_samples, 97.5, axis=0)
                freq_idx = np.arange(b)

                ax = axes[idx]
                ax.plot(freq_idx, mag_mean, "b-", label="Mean Mag")
                ax.fill_between(
                    freq_idx,
                    mag_lower,
                    mag_upper,
                    color="blue",
                    alpha=0.3,
                    label="95% CI",
                )
                ax.set_title(f"Block ({i},{j}) FFT Mag")
                ax.set_xlabel("Freq index")
                ax.set_ylabel("Magnitude")
                ax.legend()

            for ax in axes[total:]:
                ax.set_visible(False)
            plt.tight_layout()
            if show:
                plt.show()
            return fig

    # After triggering a forward pass:
    # _ = model.fft_layer(X[:1])
    fft_full = model.fft_layer.get_fourier_coeffs()
    fft_full = jax.device_get(fft_full)
    visualizer = CirculantVisualizer(fft_full)
    fig1 = visualizer.plot_spectrum()
    fig2 = visualizer.visualize_kernel()

    """_ = model.block_layer(X_test[:1])
    fft_full_blocks = model.block_layer.get_fourier_coeffs()
    block_visualizer = BlockCirculantVisualizer(fft_full_blocks)
    fig3 = block_visualizer.plot_block_spectra()
    fig4 = block_visualizer.visualize_block_kernels()
    """
