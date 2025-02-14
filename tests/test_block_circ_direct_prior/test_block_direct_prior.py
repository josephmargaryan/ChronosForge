import jax
import jax.random as jr
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from sklearn.model_selection import train_test_split

from quantbayes import bnn
from quantbayes import fake_data
from quantbayes.bnn.utils import BayesianAnalysis
from quantbayes.stochax.utils import (
    get_block_fft_full_for_given_params,
    plot_block_fft_spectra,
    visualize_block_circulant_kernels,
)

# Generate synthetic regression data.
df = fake_data.generate_regression_data(n_categorical=0, n_continuous=1)
X, y = df.drop("target", axis=1), df["target"]
X, y = jnp.array(X), jnp.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define your model.
class MyCircBlock(bnn.Module):
    def __init__(self):
        super().__init__(task_type="regression", method="nuts")

    def __call__(self, X, y=None):
        N, in_features = X.shape
        block_layer = bnn.BlockFFTDirectPriorLayer(
            in_features=in_features,
            out_features=16,
            block_size=4,
            name="block_fft_layer",
        )
        X = block_layer(X)
        X = jax.nn.tanh(X)
        X = bnn.Linear(in_features=16, out_features=1, name="out")(X)
        logits = X.squeeze()
        sigma = numpyro.sample("sigma", dist.Exponential(1.0))
        logits = numpyro.deterministic("logits", logits)
        with numpyro.plate("data", X.shape[0]):
            numpyro.sample("likelihood", dist.Normal(logits, sigma), obs=y)

        # Store the FFT layer for later visualization.
        self.block_layer = block_layer


# Create and compile your model.
train_key, val_key = jr.split(jr.PRNGKey(123), 2)
model = MyCircBlock()
model.compile(num_warmup=10, num_samples=10)
model.fit(X_train, y_train, jr.PRNGKey(34))
model.visualize(X_test, y_test, posterior="likelihood")

# Generate predictions.
posterior_preds = model.predict(X_test, val_key, posterior="likelihood")
posterior_samples = model.get_samples

# Perform Bayesian analysis.
bound = BayesianAnalysis(
    num_samples=len(X_train),
    delta=0.05,
    task_type="regression",
    inference_type="mcmc",
    posterior_samples=posterior_samples,
)

bound.compute_pac_bayesian_bound(
    predictions=posterior_preds, y_true=y_test, prior_mean=0.0, prior_std=1.0
)
print(posterior_samples.keys())

# --- Post-hoc FFT Visualization ---

# (1) Choose a concrete parameter set from the posterior.
# Here we take the first sample for each parameter.
param_dict = {key: value[0] for key, value in posterior_samples.items()}
param_dict = {
    key: value[0] for key, value in posterior_samples.items() if key != "logits"
}
# (2) Perform a forward pass with a valid RNG key to get a concrete fft_full.
fft_full = get_block_fft_full_for_given_params(
    model,
    X_test,
    param_dict,
    rng_key=jr.PRNGKey(123),
)

# (3) Plot the Fourier spectrum and circulant kernel.
fig1 = plot_block_fft_spectra(fft_full, show=True)
fig2 = visualize_block_circulant_kernels(fft_full, show=True)
