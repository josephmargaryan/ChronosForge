import jax
import jax.random as jr
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from numpyro import handlers

__all__ = [
    "CirculantVisualizer",
    "BlockCirculantVisualizer",
    "plot_block_fft_spectra",
    "visualize_block_circulant_kernels",
    "get_block_fft_full_for_given_params",
    "plot_fft_spectrum",
    "visualize_circulant_kernel",
    "get_fft_full_for_given_params",
    "compare_time_domain_truncation"
    "collect_block_r_i"
    "plot_prior_posterior_frequency",
    "visualize_circulant_layer",
    "visualize_block_circulant_layer"
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


def plot_fft_spectrum_with_uncertainty(fft_samples, show=True):
    """
    fft_samples: shape (num_samples, n)
    Computes the mean and credible intervals for the magnitude and phase,
    and plots them.
    """
    # Compute statistics across samples
    mag_samples = np.abs(fft_samples)  # shape (num_samples, n)
    phase_samples = np.angle(fft_samples)  # shape (num_samples, n)
    
    mag_mean = mag_samples.mean(axis=0)
    phase_mean = phase_samples.mean(axis=0)
    
    # Compute, for example, 95% quantiles
    mag_lower = np.percentile(mag_samples, 2.5, axis=0)
    mag_upper = np.percentile(mag_samples, 97.5, axis=0)
    phase_lower = np.percentile(phase_samples, 2.5, axis=0)
    phase_upper = np.percentile(phase_samples, 97.5, axis=0)
    
    freq_idx = np.arange(fft_samples.shape[1])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(freq_idx, mag_mean, 'b-', label='Mean Magnitude')
    axes[0].fill_between(freq_idx, mag_lower, mag_upper, color='blue', alpha=0.3, label='95% CI')
    axes[0].set_title("FFT Magnitude with Uncertainty")
    axes[0].set_xlabel("Frequency index")
    axes[0].set_ylabel("Magnitude")
    axes[0].legend()
    
    axes[1].plot(freq_idx, phase_mean, 'g-', label='Mean Phase')
    axes[1].fill_between(freq_idx, phase_lower, phase_upper, color='green', alpha=0.3, label='95% CI')
    axes[1].set_title("FFT Phase with Uncertainty")
    axes[1].set_xlabel("Frequency index")
    axes[1].set_ylabel("Phase (radians)")
    axes[1].legend()
    
    plt.tight_layout()
    if show:
        plt.show()
    return fig

def visualize_circulant_kernel_with_uncertainty(fft_samples: np.ndarray, show: bool = True):
    """
    Visualize the uncertainty in the time-domain circulant kernel.
    
    Parameters:
        fft_samples: np.ndarray of shape (num_samples, n)
            Array of FFT coefficients from multiple posterior samples.
        show: bool, if True, calls plt.show().
    
    Returns:
        fig: the matplotlib figure.
    """
    num_samples, n = fft_samples.shape

    # Compute the time-domain kernel for each sample using the inverse FFT.
    time_kernels = np.array([np.fft.ifft(fft_sample).real for fft_sample in fft_samples])
    # time_kernels has shape (num_samples, n)

    # Compute summary statistics for the time-domain kernel at each time index.
    kernel_mean = time_kernels.mean(axis=0)
    kernel_lower = np.percentile(time_kernels, 2.5, axis=0)
    kernel_upper = np.percentile(time_kernels, 97.5, axis=0)

    # For the circulant matrix, you could compute the mean circulant matrix
    # by taking the mean kernel and then rolling it.
    C_mean = np.stack([np.roll(kernel_mean, i) for i in range(n)], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot the mean kernel with error bars for uncertainty.
    axes[0].errorbar(np.arange(n), kernel_mean, 
                     yerr=[kernel_mean - kernel_lower, kernel_upper - kernel_mean],
                     fmt='o', color='b', ecolor='lightgray', capsize=3)
    axes[0].set_title("Circulant Kernel (Time Domain) with Uncertainty")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Amplitude")
    
    # For the circulant matrix, show the mean matrix.
    im = axes[1].imshow(C_mean, cmap="viridis")
    axes[1].set_title("Mean Circulant Matrix")
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Index")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_block_fft_spectra_with_phase(fft_full_blocks: jnp.ndarray, show: bool = True):
    """
    Plot both the magnitude and phase for each block weight matrix using subplots.
    Expects fft_full_blocks with shape (k_out, k_in, block_size) (complex).
    """
    fft_blocks = np.asarray(fft_full_blocks)
    k_out, k_in, b = fft_blocks.shape
    total = k_out * k_in
    
    # Create a grid for each block where we have two subplots (mag and phase)
    nrows = int(np.ceil(np.sqrt(total)))
    ncols = int(np.ceil(total / nrows))
    
    fig, axes = plt.subplots(nrows * 2, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(nrows, 2, ncols)  # shape (nrows, 2, ncols)
    axes = axes.reshape(-1, ncols)  # for easier iteration over rows
    
    idx = 0
    for i in range(nrows):
        for j in range(ncols):
            if idx < total:
                # determine block indices
                block_row = idx // k_in
                block_col = idx % k_in
                fft_block = fft_blocks[block_row, block_col]
                mag = np.abs(fft_block)
                phase = np.angle(fft_block)
                
                # Magnitude subplot (top row for this block)
                ax_mag = axes[i*2, j]
                ax_mag.stem(mag, linefmt="b-", markerfmt="bo", basefmt="r-")
                ax_mag.set_title(f"Block ({block_row},{block_col}) Mag")
                ax_mag.set_xlabel("Freq index")
                ax_mag.set_ylabel("Magnitude")
                
                # Phase subplot (bottom row for this block)
                ax_phase = axes[i*2 + 1, j]
                ax_phase.stem(phase, linefmt="g-", markerfmt="go", basefmt="r-")
                ax_phase.set_title(f"Block ({block_row},{block_col}) Phase")
                ax_phase.set_xlabel("Freq index")
                ax_phase.set_ylabel("Phase (rad)")
                
                idx += 1
            else:
                # Hide unused subplots
                for k in range(2):
                    axes[i*2 + k, j].set_visible(False)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_block_fft_spectra_with_uncertainty(fft_samples_blocks: np.ndarray, show: bool = True):
    """
    Plot the mean and 95% credible intervals for the magnitude and phase of each block's FFT.
    
    Parameters:
      fft_samples_blocks: np.ndarray of shape (num_samples, k_out, k_in, block_size)
      show: whether to call plt.show() at the end.
      
    Returns:
      A matplotlib figure.
    """
    num_samples, k_out, k_in, b = fft_samples_blocks.shape
    total = k_out * k_in

    # Prepare grids to store statistics for each block.
    # We'll create separate figures for magnitude and phase.
    fig_mag, axes_mag = plt.subplots(int(np.ceil(np.sqrt(total))), int(np.ceil(total / np.ceil(np.sqrt(total)))),
                                     figsize=(4 * int(np.ceil(np.sqrt(total))), 3 * int(np.ceil(np.sqrt(total)))))
    fig_phase, axes_phase = plt.subplots(int(np.ceil(np.sqrt(total))), int(np.ceil(total / np.ceil(np.sqrt(total)))),
                                         figsize=(4 * int(np.ceil(np.sqrt(total))), 3 * int(np.ceil(np.sqrt(total)))))

    axes_mag = np.array(axes_mag).flatten()
    axes_phase = np.array(axes_phase).flatten()
    
    for idx in range(total):
        # Determine block indices.
        i = idx // k_in
        j = idx % k_in
        
        # Extract all samples for block (i,j) => shape (num_samples, b)
        block_samples = fft_samples_blocks[:, i, j, :]  # complex values
        
        # Compute magnitude and phase samples: shape (num_samples, b)
        mag_samples = np.abs(block_samples)
        phase_samples = np.angle(block_samples)
        
        # Compute mean and 95% CI for magnitude.
        mag_mean = mag_samples.mean(axis=0)
        mag_lower = np.percentile(mag_samples, 2.5, axis=0)
        mag_upper = np.percentile(mag_samples, 97.5, axis=0)
        
        # Compute mean and 95% CI for phase.
        phase_mean = phase_samples.mean(axis=0)
        phase_lower = np.percentile(phase_samples, 2.5, axis=0)
        phase_upper = np.percentile(phase_samples, 97.5, axis=0)
        
        freq_idx = np.arange(b)
        
        # Plot magnitude uncertainty.
        ax_mag = axes_mag[idx]
        ax_mag.plot(freq_idx, mag_mean, 'b-', label='Mean')
        ax_mag.fill_between(freq_idx, mag_lower, mag_upper, color='blue', alpha=0.3, label='95% CI')
        ax_mag.set_title(f"Block ({i},{j}) Mag")
        ax_mag.set_xlabel("Freq index")
        ax_mag.set_ylabel("Magnitude")
        ax_mag.legend(fontsize=8)
        
        # Plot phase uncertainty.
        ax_phase = axes_phase[idx]
        ax_phase.plot(freq_idx, phase_mean, 'g-', label='Mean')
        ax_phase.fill_between(freq_idx, phase_lower, phase_upper, color='green', alpha=0.3, label='95% CI')
        ax_phase.set_title(f"Block ({i},{j}) Phase")
        ax_phase.set_xlabel("Freq index")
        ax_phase.set_ylabel("Phase (rad)")
        ax_phase.legend(fontsize=8)
    
    # Hide any extra subplots.
    for ax in axes_mag[total:]:
        ax.set_visible(False)
    for ax in axes_phase[total:]:
        ax.set_visible(False)
    
    fig_mag.tight_layout()
    fig_phase.tight_layout()
    if show:
        plt.show()
    return fig_mag, fig_phase

def visualize_block_circulant_kernels_with_uncertainty(fft_samples_blocks: np.ndarray, show: bool = True):
    """
    Visualize the uncertainty in the time-domain circulant kernels for each block.
    
    Parameters:
      fft_samples_blocks: np.ndarray of shape (num_samples, k_out, k_in, block_size)
      show: whether to call plt.show() at the end.
      
    Returns:
      A matplotlib figure.
    """
    num_samples, k_out, k_in, b = fft_samples_blocks.shape
    total = k_out * k_in
    
    # For each block, compute the time-domain kernels from each FFT sample.
    # We'll get an array of shape (num_samples, b) per block.
    # Then compute the mean and 95% CI across samples.
    fig, axes = plt.subplots(int(np.ceil(np.sqrt(total))), int(np.ceil(total / np.ceil(np.sqrt(total)))),
                             figsize=(4 * int(np.ceil(np.sqrt(total))), 3 * int(np.ceil(np.sqrt(total)))))
    axes = np.array(axes).flatten()
    
    for idx in range(total):
        i = idx // k_in
        j = idx % k_in
        
        # For block (i,j), get FFT samples.
        block_fft_samples = fft_samples_blocks[:, i, j, :]  # shape (num_samples, b)
        # Compute time-domain kernels via IFFT (per sample)
        time_kernels = np.array([np.fft.ifft(sample).real for sample in block_fft_samples])
        # time_kernels: shape (num_samples, b)
        
        # Compute mean and 95% CI for the kernel.
        # Compute mean and 95% CI for the kernel.
        kernel_mean = time_kernels.mean(axis=0)
        kernel_lower = np.percentile(time_kernels, 2.5, axis=0)
        kernel_upper = np.percentile(time_kernels, 97.5, axis=0)

        # Compute error bars and clip any negative values:
        lower_err = np.clip(kernel_mean - kernel_lower, a_min=0, a_max=None)
        upper_err = np.clip(kernel_upper - kernel_mean, a_min=0, a_max=None)

        # Plot error bars for the kernel using the clipped error values directly:
        ax = axes[idx]
        ax.errorbar(np.arange(b), kernel_mean,
                    yerr=[lower_err, upper_err],
                    fmt='o', color='b', ecolor='lightgray', capsize=3)

        ax.set_title(f"Block ({i},{j}) Kernel")
        ax.set_xlabel("Time index")
        ax.set_ylabel("Amplitude")
    
    # Hide any extra subplots.
    for ax in axes[total:]:
        ax.set_visible(False)
    
    plt.tight_layout()
    if show:
        plt.show()
    return fig

def visualize_block_circulant_matrices_with_uncertainty(fft_samples_blocks: np.ndarray, show: bool = True):
    """
    Visualize the uncertainty in the circulant matrices for each block.
    For each block, the time-domain kernel is computed via IFFT from the posterior
    samples, and the mean circulant matrix is obtained by rolling the mean kernel.
    
    Parameters:
        fft_samples_blocks: np.ndarray of shape (num_samples, k_out, k_in, block_size)
            Array of FFT coefficients from multiple posterior samples.
        show: bool, if True calls plt.show() at the end.
        
    Returns:
        A matplotlib figure showing the circulant matrices for each block.
    """
    num_samples, k_out, k_in, b = fft_samples_blocks.shape
    total = k_out * k_in
    nrows = int(np.ceil(np.sqrt(total)))
    ncols = int(np.ceil(total / nrows))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).flatten()
    
    for idx in range(total):
        i = idx // k_in
        j = idx % k_in
        
        # For block (i,j), compute the time-domain kernel for each sample.
        block_fft_samples = fft_samples_blocks[:, i, j, :]  # shape (num_samples, b)
        time_kernels = np.array([np.fft.ifft(sample).real for sample in block_fft_samples])
        # Compute the mean time-domain kernel.
        kernel_mean = time_kernels.mean(axis=0)
        # Reconstruct the circulant matrix by rolling the mean kernel.
        C_mean = np.stack([np.roll(kernel_mean, shift=k) for k in range(b)], axis=0)
        
        ax = axes[idx]
        im = ax.imshow(C_mean, cmap="viridis")
        ax.set_title(f"Block ({i},{j}) Circulant Matrix")
        ax.set_xlabel("Index")
        ax.set_ylabel("Index")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    for ax in axes[total:]:
        ax.set_visible(False)
        
    plt.tight_layout()
    if show:
        plt.show()
    return fig


##############################################
# Final versions
##############################################

def visualize_circulant_layer(fft_samples: np.ndarray, show=True):
    """
    Visualizes the FFT spectrum (magnitude and phase) and the time-domain circulant kernel.
    If fft_samples has multiple samples, uncertainty (e.g., 95% CI) is shown.
    """
    # Compute statistics (mean, lower, upper bounds) for FFT spectrum.
    fig1 = plot_fft_spectrum_with_uncertainty(fft_samples, show=False)
    
    # Compute time-domain kernels from fft_samples.
    fig2 = visualize_circulant_kernel_with_uncertainty(fft_samples, show=False)
    
    # Optionally combine or display them side by side.
    if show:
        plt.show()
    return fig1, fig2

def visualize_block_circulant_layer(fft_samples_blocks: np.ndarray, show=True):
    """
    Visualizes the FFT spectra, phase, time-domain kernels, and full circulant matrices for each block.
    """
    # FFT spectra with uncertainty.
    fig1, fig2 = plot_block_fft_spectra_with_uncertainty(fft_samples_blocks, show=False)
    # Time-domain kernels with uncertainty.
    fig3 = visualize_block_circulant_kernels_with_uncertainty(fft_samples_blocks, show=False)
    # Full circulant matrices.
    fig4 = visualize_block_circulant_matrices_with_uncertainty(fft_samples_blocks, show=False)
    
    if show:
        plt.show()
    return fig1, fig2, fig3, fig4



##############################################
# New consolidated classes 
##############################################

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

# For numpyro-based models.
import numpyro.handlers as handlers
import jax.random as jr

class CirculantVisualizer:
    """
    A class to visualize Fourier-domain parameters (for circulant layers) for both a single posterior sample 
    and across the entire posterior. It supports both Equinox-based layers (which require a forward pass to 
    trigger coefficient computation) and NumPyro-based layers (which use substitution/seed handlers).

    Parameters:
        model : object
            The model instance containing one or more Fourier layers.
        posterior_samples : dict
            Dictionary mapping parameter names to arrays of posterior samples.
            (All layers that are updated via the same posterior_samples will be assumed to be updated simultaneously.)
        X : array
            The input (e.g. X_test) needed to perform a forward pass.
        fft_layer_names : list of str, optional
            List of attribute names corresponding to Fourier layers in the model.
            Defaults to ['fft_layer'].
        model_type : str, either 'equinox' or 'numpyro'
            Determines how to trigger the Fourier coefficient computation.
        overlay_samples : bool, default False
            If True (when visualizing the posterior), overlay a few random individual FFT draws in the plots.
        random_draws : int, default 30
            Number of random posterior draws to overlay (if overlay_samples is True).
        ignore_keys : list of str, default []
            List of keys in posterior_samples to ignore (e.g. keys not used for the Fourier layers).
    """
    
    def __init__(self, model, posterior_samples, X, fft_layer_names=None,
                 model_type='equinox', overlay_samples=False, random_draws=30,
                 ignore_keys=None):
        self.model = model
        self.posterior_samples = posterior_samples
        self.X = X
        self.fft_layer_names = fft_layer_names if fft_layer_names is not None else ['fft_layer']
        self.model_type = model_type
        self.overlay_samples = overlay_samples
        self.random_draws = random_draws
        self.ignore_keys = ignore_keys if ignore_keys is not None else []
    
    def _filter_params(self, param_dict):
        """Filter out keys specified in self.ignore_keys."""
        return {key: value for key, value in param_dict.items() if key not in self.ignore_keys}

    def _get_fft_for_layer(self, layer_name, param_dict, rng_key):
        """
        Given a parameter dictionary and an RNG key, update the model and retrieve the Fourier coefficients 
        for the layer named `layer_name`.
        """
        # Get the Fourier layer.
        fourier_layer = getattr(self.model, layer_name)

        if self.model_type == 'equinox':
            # For Equinox layers, trigger a forward pass to update stored coefficients.
            _ = fourier_layer(self.X)
            fft_full = fourier_layer.get_fourier_coeffs()
        elif self.model_type == 'numpyro':
            # For NumPyro-based layers, use substitution and a seeded forward pass.
            with handlers.seed(rng_seed=rng_key):
                with handlers.substitute(data=param_dict):
                    _ = self.model(self.X)
            fft_full = fourier_layer.get_fourier_coeffs()
        else:
            raise ValueError("model_type must be either 'equinox' or 'numpyro'")
                
            # Bring fft_full to host (as a NumPy array)
        return jax.device_get(fft_full)
    
    def _plot_fft_spectrum(self, fft_full, show=True):
        """
        Plot a single realization of the Fourier spectrum (magnitude and phase).
        """
        fft_full = np.asarray(fft_full)
        mag = np.abs(fft_full)
        phase = np.arctan2(np.imag(fft_full), np.real(fft_full))
    
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

    def _plot_fft_spectrum_with_uncertainty(self, fft_samples, show=True):
        """
        Plot the mean FFT spectrum with 95% credible intervals.
        Optionally overlay a few random individual samples.
        
        fft_samples: np.ndarray of shape (num_samples, n)
        """
        # Compute magnitude and phase for all samples.
        fft_samples = np.asarray(fft_samples)
        mag_samples = np.abs(fft_samples)
        phase_samples = np.angle(fft_samples)
        
        # Compute statistics.
        mag_mean = mag_samples.mean(axis=0)
        phase_mean = phase_samples.mean(axis=0)
        mag_lower = np.percentile(mag_samples, 2.5, axis=0)
        mag_upper = np.percentile(mag_samples, 97.5, axis=0)
        phase_lower = np.percentile(phase_samples, 2.5, axis=0)
        phase_upper = np.percentile(phase_samples, 97.5, axis=0)
        
        freq_idx = np.arange(fft_samples.shape[1])
    
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # Plot magnitude.
        axes[0].plot(freq_idx, mag_mean, 'b-', label='Mean Magnitude')
        axes[0].fill_between(freq_idx, mag_lower, mag_upper, color='blue', alpha=0.3, label='95% CI')
        if self.overlay_samples:
            # Overlay a few random individual samples.
            idxs = np.random.choice(fft_samples.shape[0], self.random_draws, replace=False)
            for i in idxs:
                axes[0].plot(freq_idx, mag_samples[i, :], 'c-', alpha=0.3)
        axes[0].set_title("FFT Magnitude with Uncertainty")
        axes[0].set_xlabel("Frequency index")
        axes[0].set_ylabel("Magnitude")
        axes[0].legend()
    
        # Plot phase.
        axes[1].plot(freq_idx, phase_mean, 'g-', label='Mean Phase')
        axes[1].fill_between(freq_idx, phase_lower, phase_upper, color='green', alpha=0.3, label='95% CI')
        if self.overlay_samples:
            for i in idxs:
                axes[1].plot(freq_idx, phase_samples[i, :], 'y-', alpha=0.3)
        axes[1].set_title("FFT Phase with Uncertainty")
        axes[1].set_xlabel("Frequency index")
        axes[1].set_ylabel("Phase (radians)")
        axes[1].legend()
    
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def _plot_fft_complex_scatter(self, fft_samples, show=True):
        """
        (Additional visualization.)
        Plot the Fourier coefficients in the complex plane. The mean is highlighted and individual samples
        are shown as semi-transparent points.
        """
        fft_samples = np.asarray(fft_samples)
        all_real = fft_samples.real.flatten()
        all_imag = fft_samples.imag.flatten()
        fft_mean = fft_samples.mean(axis=0)
    
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(all_real, all_imag, color='gray', alpha=0.3, label="Samples")
        ax.scatter(fft_mean.real, fft_mean.imag, color='red', label="Mean", s=50)
        ax.set_xlabel("Real part")
        ax.set_ylabel("Imaginary part")
        ax.set_title("FFT Coefficients in Complex Plane")
        ax.legend()
        ax.grid(True)
    
        if show:
            plt.show()
        return fig

    def _plot_circulant_kernel_with_uncertainty(self, fft_samples, show=True):
        """
        Compute the time-domain circulant kernel (via IFFT) for each posterior sample and plot:
          - The mean kernel with 95% credible intervals (using error bars).
          - The mean circulant matrix (formed by rolling the mean kernel).
          
        fft_samples: np.ndarray of shape (num_samples, n)
        """
        fft_samples = np.asarray(fft_samples)
        num_samples, n = fft_samples.shape
        time_kernels = np.array([np.fft.ifft(fft_sample).real for fft_sample in fft_samples])
        kernel_mean = time_kernels.mean(axis=0)
        kernel_lower = np.percentile(time_kernels, 2.5, axis=0)
        kernel_upper = np.percentile(time_kernels, 97.5, axis=0)
    
        C_mean = np.stack([np.roll(kernel_mean, i) for i in range(n)], axis=0)
    
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
        axes[0].errorbar(np.arange(n), kernel_mean, 
                         yerr=[kernel_mean - kernel_lower, kernel_upper - kernel_mean],
                         fmt='o', color='b', ecolor='lightgray', capsize=3)
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

    def visualize_single(self, layer_name, sample_index=0, rng_key=None, show=True):
        """
        Visualize the Fourier spectrum (magnitude and phase) for a single posterior draw.
        
        Parameters:
            layer_name: str
                Name of the Fourier layer to visualize.
            sample_index: int, default 0
                Index of the sample to use.
            rng_key: jax.random.PRNGKey or None
                If None, uses a default key based on sample_index.
            show: bool
                Whether to call plt.show().
        """
        param_dict = {key: value[sample_index] for key, value in self.posterior_samples.items()}
        param_dict = self._filter_params(param_dict)
        if rng_key is None:
            rng_key = jr.PRNGKey(sample_index)
        fft_full = self._get_fft_for_layer(layer_name, param_dict, rng_key)
        fig = self._plot_fft_spectrum(fft_full, show=show)
        return fig

    def visualize_posterior(self, layer_name, n_draws=None, show=True):
        """
        Visualize the Fourier spectrum (with uncertainty) and the reconstructed circulant kernel 
        over many posterior draws for a given Fourier layer.
        
        Parameters:
            layer_name: str
                Name of the Fourier layer to visualize.
            n_draws: int or None
                Number of posterior samples to use. If None, uses the full number of samples.
            show: bool
                Whether to call plt.show() after plotting.
        
        Returns:
            A dictionary of figures.
        """
        sample_keys = list(self.posterior_samples.values())
        total_samples = sample_keys[0].shape[0]
        if n_draws is None:
            n_draws = total_samples
        else:
            n_draws = min(n_draws, total_samples)
    
        fft_list = []
        for i in range(n_draws):
            param_dict = {key: value[i] for key, value in self.posterior_samples.items()}
            param_dict = self._filter_params(param_dict)
            rng_key = jr.PRNGKey(i)
            fft_full = self._get_fft_for_layer(layer_name, param_dict, rng_key)
            fft_list.append(fft_full)
        fft_samples = np.stack(fft_list, axis=0)
    
        fig_fft = self._plot_fft_spectrum_with_uncertainty(fft_samples, show=show)
        fig_kernel = self._plot_circulant_kernel_with_uncertainty(fft_samples, show=show)
        fig_complex = self._plot_fft_complex_scatter(fft_samples, show=show)
    
        return {"fft_spectrum": fig_fft,
                "circulant_kernel": fig_kernel,
                "fft_complex": fig_complex}


"""
# Initialize the visualizer. For Equinox-based layers, set model_type='equinox';
# for NumPyro-based layers, use model_type='numpyro'.
visualizer = CirculantVisualizer(
    model, 
    posterior_samples, 
    X_test,
    fft_layer_names=['fft_layer'],
    model_type='equinox',
    overlay_samples=True,
    random_draws=30,
    ignore_keys=["logits"]
)

# (A) Visualize one single posterior draw for the chosen Fourier layer.
fig_single = visualizer.visualize_single('fft_layer', sample_index=0, rng_key=jr.PRNGKey(0))

# (B) Visualize the full posterior uncertainty.
figs = visualizer.visualize_posterior('fft_layer', n_draws=100)


"""
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.handlers as handlers

class BlockCirculantVisualizer:
    """
    A class to visualize block circulant layers’ Fourier-domain parameters.
    
    The block circulant layer is assumed to expose a method
        get_fourier_coeffs()
    that returns a complex FFT tensor of shape (k_out, k_in, block_size).
    
    This class supports:
      - Visualizing a single posterior draw:
          * FFT spectra (magnitude only or magnitude & phase) for each block in a near-square grid.
          * The corresponding time-domain circulant kernels (via IFFT).
      - Visualizing uncertainty across many posterior samples:
          * Mean and 95% credible intervals for the block FFT (both magnitude and phase).
          * Mean time-domain kernel (with error bars) for each block.
    
    Parameters:
        model : object
            The model instance that contains one or more block circulant layers.
        posterior_samples : dict
            Dictionary mapping parameter names to arrays of posterior samples.
        X : array
            Input data (e.g. X_test) required for a forward pass.
        block_layer_names : list of str, optional
            List of attribute names corresponding to block circulant layers.
            Defaults to ['block_layer'].
        model_type : str, either 'equinox' or 'numpyro'
            Determines how to trigger the layer’s forward pass.
        ignore_keys : list, optional
            List of keys to ignore from posterior_samples (e.g. ["logits"]). Defaults to ["logits"].
    """
    def __init__(self, model, posterior_samples, X, block_layer_names=None,
                 model_type='equinox', ignore_keys=None):
        self.model = model
        self.posterior_samples = posterior_samples
        self.X = X
        self.block_layer_names = block_layer_names if block_layer_names is not None else ['block_layer']
        self.model_type = model_type
        self.ignore_keys = ignore_keys if ignore_keys is not None else ["logits"]
    
    def _filter_params(self, param_dict):
        """Filter out keys specified in self.ignore_keys."""
        return {key: value for key, value in param_dict.items() if key not in self.ignore_keys}
    
    def _get_fft_for_block_layer(self, layer_name, param_dict, rng_key):
        """
        Given a parameter dictionary and an RNG key, trigger a forward pass so that the layer 
        sees these parameter values and returns its FFT tensor.
        """
        block_layer = getattr(self.model, layer_name)
        if self.model_type == 'equinox':
            # For Equinox-based layers, trigger a forward pass.
            _ = block_layer(self.X)
            fft_full = block_layer.get_fourier_coeffs()
        elif self.model_type == 'numpyro':
            with handlers.seed(rng_seed=rng_key):
                with handlers.substitute(data=param_dict):
                    _ = self.model(self.X)
            fft_full = block_layer.get_fourier_coeffs()
        else:
            raise ValueError("model_type must be either 'equinox' or 'numpyro'")
        return jax.device_get(fft_full)
    
    def _plot_block_fft_spectra(self, fft_full_blocks, show=True):
        """
        Plot the FFT magnitude for each block weight matrix in a near-square grid.
        
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
    
    def _plot_block_fft_spectra_with_phase(self, fft_full_blocks, show=True):
        """
        Plot both magnitude and phase for each block weight matrix in a near-square grid.
        
        Expects fft_full_blocks with shape (k_out, k_in, block_size) (complex).
        """
        fft_blocks = np.asarray(fft_full_blocks)
        k_out, k_in, b = fft_blocks.shape
        total = k_out * k_in
        
        nrows = int(np.ceil(np.sqrt(total)))
        ncols = int(np.ceil(total / nrows))
    
        # Each block gets two subplots: one for magnitude, one for phase.
        fig, axes = plt.subplots(nrows * 2, ncols, figsize=(4 * ncols, 3 * nrows))
        axes = np.array(axes).reshape(nrows, 2, ncols).reshape(-1, ncols)
    
        idx = 0
        for i in range(nrows):
            for j in range(ncols):
                if idx < total:
                    block_row = idx // k_in
                    block_col = idx % k_in
                    fft_block = fft_blocks[block_row, block_col]
                    mag = np.abs(fft_block)
                    phase = np.angle(fft_block)
    
                    ax_mag = axes[i*2, j]
                    ax_mag.stem(mag, linefmt="b-", markerfmt="bo", basefmt="r-")
                    ax_mag.set_title(f"Block ({block_row},{block_col}) Mag")
                    ax_mag.set_xlabel("Freq index")
                    ax_mag.set_ylabel("Magnitude")
    
                    ax_phase = axes[i*2 + 1, j]
                    ax_phase.stem(phase, linefmt="g-", markerfmt="go", basefmt="r-")
                    ax_phase.set_title(f"Block ({block_row},{block_col}) Phase")
                    ax_phase.set_xlabel("Freq index")
                    ax_phase.set_ylabel("Phase (rad)")
    
                    idx += 1
                else:
                    for k in range(2):
                        axes[i*2 + k, j].set_visible(False)
    
        plt.tight_layout()
        if show:
            plt.show()
        return fig
    
    def _plot_block_circulant_kernels(self, fft_full_blocks, show=True):
        """
        For each block, compute the time-domain circulant kernel (via IFFT) and show its full circulant matrix.
        
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
            time_kernel = jnp.fft.ifft(fft_block).real
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
    
    def _plot_block_fft_spectra_with_uncertainty(self, fft_samples_blocks, show=True):
        """
        Plot the mean and 95% credible intervals for the magnitude and phase of each block's FFT.
        
        Parameters:
            fft_samples_blocks: np.ndarray of shape (num_samples, k_out, k_in, block_size)
        Returns:
            Two figures: one for the FFT magnitude uncertainty and one for the FFT phase.
        """
        num_samples, k_out, k_in, b = fft_samples_blocks.shape
        total = k_out * k_in
    
        nrows = int(np.ceil(np.sqrt(total)))
        ncols = int(np.ceil(total / nrows))
    
        fig_mag, axes_mag = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        fig_phase, axes_phase = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes_mag = np.array(axes_mag).flatten()
        axes_phase = np.array(axes_phase).flatten()
    
        for idx in range(total):
            i = idx // k_in
            j = idx % k_in
            block_samples = fft_samples_blocks[:, i, j, :]  # shape (num_samples, b)
            mag_samples = np.abs(block_samples)
            phase_samples = np.angle(block_samples)
    
            mag_mean = mag_samples.mean(axis=0)
            mag_lower = np.percentile(mag_samples, 2.5, axis=0)
            mag_upper = np.percentile(mag_samples, 97.5, axis=0)
    
            phase_mean = phase_samples.mean(axis=0)
            phase_lower = np.percentile(phase_samples, 2.5, axis=0)
            phase_upper = np.percentile(phase_samples, 97.5, axis=0)
    
            freq_idx = np.arange(b)
    
            ax_mag = axes_mag[idx]
            ax_mag.plot(freq_idx, mag_mean, 'b-', label='Mean')
            ax_mag.fill_between(freq_idx, mag_lower, mag_upper, color='blue', alpha=0.3, label='95% CI')
            ax_mag.set_title(f"Block ({i},{j}) Mag")
            ax_mag.set_xlabel("Freq index")
            ax_mag.set_ylabel("Magnitude")
            ax_mag.legend(fontsize=8)
    
            ax_phase = axes_phase[idx]
            ax_phase.plot(freq_idx, phase_mean, 'g-', label='Mean')
            ax_phase.fill_between(freq_idx, phase_lower, phase_upper, color='green', alpha=0.3, label='95% CI')
            ax_phase.set_title(f"Block ({i},{j}) Phase")
            ax_phase.set_xlabel("Freq index")
            ax_phase.set_ylabel("Phase (rad)")
            ax_phase.legend(fontsize=8)
    
        for ax in axes_mag[total:]:
            ax.set_visible(False)
        for ax in axes_phase[total:]:
            ax.set_visible(False)
    
        fig_mag.tight_layout()
        fig_phase.tight_layout()
        if show:
            plt.show()
        return fig_mag, fig_phase
    
    def _plot_block_circulant_kernels_with_uncertainty(self, fft_samples_blocks, show=True):
        """
        Visualize uncertainty in the time-domain circulant kernels for each block.
        
        Parameters:
            fft_samples_blocks: np.ndarray of shape (num_samples, k_out, k_in, block_size)
        """
        num_samples, k_out, k_in, b = fft_samples_blocks.shape
        total = k_out * k_in
    
        nrows = int(np.ceil(np.sqrt(total)))
        ncols = int(np.ceil(total / nrows))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes = np.array(axes).flatten()
    
        for idx in range(total):
            i = idx // k_in
            j = idx % k_in
            block_fft_samples = fft_samples_blocks[:, i, j, :]  # shape (num_samples, b)
            # Compute time-domain kernels via IFFT.
            time_kernels = np.array([np.fft.ifft(sample).real for sample in block_fft_samples])
            kernel_mean = time_kernels.mean(axis=0)
            kernel_lower = np.percentile(time_kernels, 2.5, axis=0)
            kernel_upper = np.percentile(time_kernels, 97.5, axis=0)
    
            # Compute error bars (ensuring non-negative error lengths).
            lower_err = np.clip(kernel_mean - kernel_lower, a_min=0, a_max=None)
            upper_err = np.clip(kernel_upper - kernel_mean, a_min=0, a_max=None)
    
            ax = axes[idx]
            ax.errorbar(np.arange(b), kernel_mean, yerr=[lower_err, upper_err],
                        fmt='o', color='b', ecolor='lightgray', capsize=3)
            ax.set_title(f"Block ({i},{j}) Kernel")
            ax.set_xlabel("Time index")
            ax.set_ylabel("Amplitude")
    
        for ax in axes[total:]:
            ax.set_visible(False)
    
        plt.tight_layout()
        if show:
            plt.show()
        return fig
    
    def visualize_single(self, layer_name, sample_index=0, rng_key=None, with_phase=False, show=True):
        """
        Visualize a single posterior draw for a specified block circulant layer.
        
        Parameters:
            layer_name : str
                Name of the block circulant layer (attribute name in the model).
            sample_index : int, default 0
                Index of the posterior sample to use.
            rng_key : jax.random.PRNGKey or None
                RNG key to use; if None, defaults to jr.PRNGKey(sample_index).
            with_phase : bool, default False
                If True, also produce a plot of FFT spectra with phase.
            show : bool, default True
                Whether to call plt.show() after plotting.
        
        Returns:
            A dictionary of figures (keys include 'fft_spectra', optionally 'fft_spectra_with_phase',
            and 'circulant_kernels').
        """
        # Build parameter dictionary for this sample and filter out ignore_keys.
        param_dict = {key: value[sample_index] for key, value in self.posterior_samples.items()}
        param_dict = self._filter_params(param_dict)
        if rng_key is None:
            rng_key = jr.PRNGKey(sample_index)
        fft_full = self._get_fft_for_block_layer(layer_name, param_dict, rng_key)
    
        figs = {}
        figs['fft_spectra'] = self._plot_block_fft_spectra(fft_full, show=show)
        if with_phase:
            figs['fft_spectra_with_phase'] = self._plot_block_fft_spectra_with_phase(fft_full, show=show)
        figs['circulant_kernels'] = self._plot_block_circulant_kernels(fft_full, show=show)
        return figs
    
    def visualize_posterior(self, layer_name, n_draws=None, show=True):
        """
        Visualize uncertainty over the posterior for a given block circulant layer.
        
        This method loops over a specified number of posterior samples (or all samples if n_draws is None),
        retrieves the FFT blocks, and then produces uncertainty plots:
          - FFT spectra: mean and 95% credible intervals (separate figures for magnitude and phase).
          - Circulant kernels: mean and error bars for the time-domain kernel.
        
        Parameters:
            layer_name : str
                Name of the block circulant layer.
            n_draws : int or None
                Number of posterior samples to use. If None, uses all available samples.
            show : bool, default True
                Whether to call plt.show() after plotting.
        
        Returns:
            A dictionary with keys:
              'fft_spectra_uncertainty' : (fig_mag, fig_phase)
              'circulant_kernels_uncertainty' : fig for kernel uncertainty.
        """
        sample_keys = list(self.posterior_samples.values())
        total_samples = sample_keys[0].shape[0]
        if n_draws is None:
            n_draws = total_samples
        else:
            n_draws = min(n_draws, total_samples)
    
        fft_list = []
        for i in range(n_draws):
            param_dict = {key: value[i] for key, value in self.posterior_samples.items()}
            param_dict = self._filter_params(param_dict)
            rng_key = jr.PRNGKey(i)
            fft_full = self._get_fft_for_block_layer(layer_name, param_dict, rng_key)
            fft_list.append(fft_full)
        fft_samples = np.stack(fft_list, axis=0)  # shape: (n_draws, k_out, k_in, block_size)
    
        fig_mag, fig_phase = self._plot_block_fft_spectra_with_uncertainty(fft_samples, show=show)
        fig_kernel = self._plot_block_circulant_kernels_with_uncertainty(fft_samples, show=show)
    
        return {"fft_spectra_uncertainty": (fig_mag, fig_phase),
                "circulant_kernels_uncertainty": fig_kernel}

"""
# Initialize the visualizer.
# (For Equinox-based layers, set model_type='equinox'; for NumPyro-based layers, use 'numpyro'.)
visualizer = BlockCirculantVisualizer(model, posterior_samples, X_test,
                                      block_layer_names=['block_layer'],
                                      model_type='equinox',
                                      ignore_keys=["logits"])

# (A) Visualize a single posterior draw (sample index 0).
figs_single = visualizer.visualize_single('block_layer', sample_index=0, rng_key=jr.PRNGKey(123),
                                            with_phase=True, show=True)

# (B) Visualize uncertainty over the posterior (using, say, 100 draws).
figs_uncertainty = visualizer.visualize_posterior('block_layer', n_draws=100, show=True)

"""