import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from quantbayes import bnn 
import numpyro
import numpyro.distributions as dist
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from quantbayes.fake_data import generate_regression_data

# Generate synthetic regression data
df = generate_regression_data(n_continuous=3)
X, y = df.drop("target", axis=1), df["target"]
X, y = jnp.array(X), jnp.array(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Define the model class.
class Test(bnn.Module):
    def __init__(self, prior: any, activation):
        super().__init__()
        self.prior = prior
        self.activation = activation

    def __call__(self, X, y=None):
        N, D = X.shape
        # Apply the circulant layer.
        X = bnn.CirculantProcess(
            in_features=D,
            prior_fn=self.prior
        )(X)
        # Use the provided activation function.
        X = self.activation(X)
        X = bnn.Linear(D, 1, name="out")(X)
        mu = X.squeeze()
        numpyro.sample("obs", dist.Normal(mu), obs=y)

# Wrap prior functions to accept extra keyword arguments.
def gaussian_prior(scale, **kwargs):
    return dist.Normal(0, scale)

def laplace_prior(scale, **kwargs):
    return dist.Laplace(0, scale)

def cauchy_prior(scale, **kwargs):
    return dist.Cauchy(0, scale)

priors = {
    "Gaussian": gaussian_prior,
    "Laplace": laplace_prior,
    "Cauchy": cauchy_prior
}

activations = {
    "tanh": jax.nn.tanh,
    "SiLU": jax.nn.silu,
    "GELU": jax.nn.gelu
}

def hyperparameter_tuning(X_train, y_train, X_val, y_val, priors, activations, seed=0):
    best_rmse = float('inf')
    best_config = None
    tuning_results = []

    for prior_name, prior_fn in priors.items():
        for act_name, act_fn in activations.items():
            print(f"Evaluating configuration: prior={prior_name}, activation={act_name}")
            # Use the same key for consistency.
            key = jr.PRNGKey(seed)
            # Instantiate the model with the current hyperparameters.
            model = Test(prior=prior_fn, activation=act_fn)
            # Compile the model (using 1 chain, 500 warmup, 1000 samples for tuning).
            model.compile(num_chains=1, num_warmup=10, num_samples=10)

            start_time = time.time()
            model.fit(X_train, y_train, key)
            end_time = time.time()
            run_time = end_time - start_time

            # Generate predictions on the validation set.
            k1, k2 = jr.split(key, 2)
            preds = model.predict(X_val, posterior="obs", rng_key=k2)
            mean_preds = np.array(preds).mean(axis=0)
            rmse = np.sqrt(mean_squared_error(np.array(y_val), mean_preds))

            config = {
                "prior": prior_name,
                "activation": act_name,
                "rmse": rmse,
                "time": run_time
            }
            tuning_results.append(config)
            print(f"Configuration results: RMSE = {rmse:.2f}, Time = {run_time:.2f}s\n")

            if rmse < best_rmse:
                best_rmse = rmse
                best_config = config

    return best_config, tuning_results

# Run hyperparameter tuning
best_config, tuning_results = hyperparameter_tuning(X_train, y_train, X_val, y_val, priors, activations, seed=0)

print("\nBest Hyperparameter Configuration:")
print(best_config)
