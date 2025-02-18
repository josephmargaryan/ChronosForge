import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
import jax
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from quantbayes import bnn
from quantbayes.fake_data import generate_regression_data
import pandas as pd


def generate_periodic_regression_data(n_samples=1000, random_seed=42):
    """
    Generate synthetic regression data with a dominant low-frequency signal
    plus additional high-frequency noise.

    Parameters
    ----------
    n_samples : int
        Number of samples (rows).
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame containing a time variable and a target variable.
    """
    np.random.seed(random_seed)
    # Create a time variable that spans several periods.
    t = np.linspace(0, 10 * np.pi, n_samples)

    # Low-frequency (smooth) component.
    low_freq_component = np.sin(0.5 * t)

    # High-frequency (noisy) component; note the amplitude is small.
    high_freq_component = 0.3 * np.sin(10 * t)

    # Add some Gaussian noise.
    noise = 0.1 * np.random.randn(n_samples)

    # The target is the sum of these components.
    target = low_freq_component + high_freq_component + noise

    # For a univariate regression example, we use the time variable as the input.
    df = pd.DataFrame({"t": t, "target": target})
    return df


def visualize_gp_kernel(gp_layer, X):
    """
    Computes and visualizes the kernel matrix from the GP layer.

    Parameters:
      gp_layer: GaussianProcessLayer instance.
      X: jnp.ndarray of shape (num_points, input_dim)

    Returns:
      fig: Matplotlib figure.
    """
    # Compute the kernel matrix
    kernel_matrix = gp_layer(X)
    kernel_matrix_np = jax.device_get(kernel_matrix)  # Convert to NumPy array

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(kernel_matrix_np, cmap="viridis")
    ax.set_title("GP Kernel (Covariance Matrix)")
    ax.set_xlabel("Data Point Index")
    ax.set_ylabel("Data Point Index")
    fig.colorbar(cax)
    plt.tight_layout()
    plt.show()
    return fig


def sample_gp_prior(gp_layer, X, num_samples=5):
    """
    Draw samples from the GP prior and visualize them.

    Parameters:
      gp_layer: GaussianProcessLayer instance.
      X: jnp.ndarray of shape (num_points, input_dim)
      num_samples: int, number of GP samples to draw.

    Returns:
      fig: Matplotlib figure.
    """
    import numpy as np

    kernel_matrix = gp_layer(X)
    kernel_np = jax.device_get(kernel_matrix)
    # Ensure the kernel is symmetric positive-definite:
    L = np.linalg.cholesky(kernel_np + 1e-6 * np.eye(kernel_np.shape[0]))

    samples = []
    for i in range(num_samples):
        # Draw a sample from standard normal and scale it with the Cholesky factor
        sample = L @ np.random.randn(kernel_np.shape[0])
        samples.append(sample)

    samples = np.array(samples)  # Shape: (num_samples, num_points)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, sample in enumerate(samples):
        ax.plot(sample, label=f"Sample {i+1}")
    ax.set_title("Samples from the GP Prior")
    ax.set_xlabel("Data Point Index")
    ax.set_ylabel("Function Value")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig


def predict_gp(model, X_train, y_train, X_test):
    # Compute the training covariance matrix (with noise)
    K_train = jax.device_get(model.gp_layer(X_train))

    # Compute the cross-covariance between test and train data
    K_cross = jax.device_get(model.gp_layer(X_test, X_train))

    # Compute the test covariance (for predictive variance)
    K_test = jax.device_get(model.gp_layer(X_test))

    # Retrieve the noise parameter (as a concrete number) from the gp_layer attribute
    noise = jax.device_get(model.gp_layer.noise)

    # Add noise to training covariance matrix
    # (Note: Depending on your intended model, you might add noise once already in the kernel;
    # here we're following your original code which squares the noise value.)
    K_train_noise = K_train + (noise**2) * np.eye(K_train.shape[0])
    L = np.linalg.cholesky(K_train_noise + 1e-6 * np.eye(K_train.shape[0]))

    # Solve for alpha: K_train_noise⁻¹ y_train = L⁻T (L⁻¹ y_train)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, np.array(y_train)))

    # Predictive mean: K_cross @ alpha
    mean_pred = K_cross.dot(alpha)

    # Predictive variance: diag(K_test) - sum(v**2, axis=0) where v = L⁻¹ K_cross^T
    v = np.linalg.solve(L, K_cross.T)
    var_pred = np.diag(K_test) - np.sum(v**2, axis=0)

    return mean_pred, var_pred


from sklearn.decomposition import PCA


def visualize_predictions(X_test, mean_pred, var_pred):
    # Convert X_test to a NumPy array
    X_arr = np.array(X_test)

    # If multi-dimensional (more than 1 feature), reduce to 1D with PCA
    if X_arr.ndim > 1 and X_arr.shape[1] > 1:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=1)
        X_reduced = pca.fit_transform(X_arr).flatten()
    else:
        # If univariate, just flatten it.
        X_reduced = X_arr.flatten()

    # Sort for a nice plot
    order = np.argsort(X_reduced)
    X_plot = X_reduced[order]
    mean_pred = np.array(mean_pred)[order]
    std_pred = np.sqrt(np.array(var_pred))[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(X_plot, mean_pred, "b-", label="Predictive Mean")
    ax.fill_between(
        X_plot,
        mean_pred - 2 * std_pred,
        mean_pred + 2 * std_pred,
        color="blue",
        alpha=0.3,
        label="Uncertainty (±2 std)",
    )
    ax.set_title("GP Predictive Posterior")
    ax.set_xlabel(
        "Input" if X_arr.ndim == 1 or X_arr.shape[1] == 1 else "PCA Component 1"
    )
    ax.set_ylabel("Output")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig


class GaussianProcess(bnn.Module):
    def __init__(self):
        super().__init__(task_type="gp")

    def __call__(self, X, y=None):
        N, in_features = X.shape
        gp_layer = bnn.GaussianProcessLayer(input_dim=in_features, name="gp_layer")
        kernel_matrix = gp_layer(X)
        f = numpyro.sample(
            "f",
            dist.MultivariateNormal(
                loc=jnp.zeros(X.shape[0]), covariance_matrix=kernel_matrix
            ),
        )
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        with numpyro.plate("data", N):
            numpyro.sample("obs", dist.Normal(f, sigma), obs=y)
        # Immediately extract concrete (non-traced) values:
        self.gp_layer = gp_layer
        self.kernel_matrix = kernel_matrix


df = generate_periodic_regression_data()
X, y = df.drop("target", axis=1), df["target"]
X, y = jnp.array(X), jnp.array(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=24
)

tk, vk = jax.random.split(jax.random.key(123), 2)
model = GaussianProcess()
model.compile(num_warmup=500, num_samples=1000)
model.fit(X_train, y_train, tk)


preds = predict_gp(model, X_train, y_train, X_test)
mean_pred, var_pred = preds

# Compute RMSE (make sure y_test is a NumPy array)
y_test_np = np.array(y_test)
rmse = np.sqrt(mean_squared_error(y_test_np, mean_pred))
print("Root Mean Squared Error:", rmse)

# Visualize the predictions with uncertainty (±2 standard deviations)
fig = visualize_predictions(X_test, mean_pred, var_pred)


visualize_gp_kernel(model.gp_layer, X_test)
sample_gp_prior(model.gp_layer, X_train, num_samples=5)

################### Hybrid ########################


class Hybrid(bnn.Module):
    def __init__(self):
        super().__init__(task_type="gp")

    def __call__(self, X, y=None):
        N, in_features = X.shape
        fft_layer = bnn.SmoothTruncCirculantLayer(
            in_features=in_features, alpha=1, K=None, name="FFT Transformation"
        )
        X = fft_layer(X)
        gp_layer = bnn.GaussianProcessLayer(input_dim=in_features, name="gp_layer")
        kernel_matrix = gp_layer(X)
        f = numpyro.sample(
            "f",
            dist.MultivariateNormal(loc=jnp.zeros(N), covariance_matrix=kernel_matrix),
        )
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        with numpyro.plate("data", N):
            numpyro.sample("obs", dist.Normal(f, sigma), obs=y)
        self.gp_layer = gp_layer
        self.kernel_matrix = kernel_matrix
        self.fft_layer = fft_layer


tk, vk = jax.random.split(jax.random.key(223), 2)
model2 = Hybrid()
model2.compile(num_warmup=10, num_samples=10)
model2.fit(X_train, y_train, tk)

preds = predict_gp(model2, X_train, y_train, X_test)
mean_pred, var_pred = preds

# Compute RMSE (make sure y_test is a NumPy array)
y_test_np = np.array(y_test)
rmse = np.sqrt(mean_squared_error(y_test_np, mean_pred))
print("Root Mean Squared Error:", rmse)

# Visualize the predictions with uncertainty (±2 standard deviations)
fig = visualize_predictions(X_test, mean_pred, var_pred)

visualize_gp_kernel(model2.gp_layer, X_test)
sample_gp_prior(model2.gp_layer, X_train, num_samples=5)

from quantbayes.stochax.utils import (
    get_fft_full_for_given_params,
    visualize_circulant_layer,
)
import numpy as np

posterior_samples = model2.get_samples
for key, value in posterior_samples.items():
    print(key)
# (3) To visualize uncertainty, loop over multiple posterior samples:
fft_list = []
n_samples = 50
for i in range(n_samples):
    sample_param_dict = {
        key: value[i]
        for key, value in posterior_samples.items()
        if key in ["FFT Transformation_imag", "Transformation_real"]
    }
    fft_sample = get_fft_full_for_given_params(
        model2, X_test, sample_param_dict, rng_key=jax.random.PRNGKey(i)
    )
    fft_list.append(fft_sample)

# Convert the list to a NumPy array: shape (n_samples, n)
fft_samples = np.stack(fft_list, axis=0)

# (4) Call the high-level visualization function.
fig_fft, fig_kernel = visualize_circulant_layer(fft_samples, show=True)
