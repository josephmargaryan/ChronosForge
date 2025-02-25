import time
import jax
import jax.random as jr
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import numpyro
import numpyro.distributions as dist
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
import arviz as az
import warnings

from quantbayes import bnn
from quantbayes.bnn.utils import evaluate_mcmc
from quantbayes.fake_data import generate_regression_data

# Generate synthetic regression data
df = generate_regression_data(n_continuous=3)
X, y = df.drop("target", axis=1), df["target"]
X, y = jnp.array(X), jnp.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

class Test(bnn.Module):
    def __init__(self, alpha: float, K: int, prior: any):
        super().__init__()
        self.alpha = alpha
        self.K = K
        self.prior = prior
    def __call__(self, X, y=None):
        N, D = X.shape
        X = bnn.CirculantProcess(
            in_features=D,
            alpha=self.alpha,
            K=self.K,
            prior_fn=self.prior
            )(X)
        X = jax.nn.tanh(X)
        X = bnn.Linear(5, 1, name="out")(X)
        mu = X.squeeze()
        numpyro.sample("obs", dist.Normal(mu), obs=y)

def aggregate_diagnostics(diag_list):
    # Get all keys from the first diagnostic dictionary.
    keys = diag_list[0].keys()
    agg = {}
    for key in keys:
        values = []
        for diag in diag_list:
            try:
                # Convert each value to float (if it's a string, float() will work)
                val = float(diag[key])
                values.append(val)
            except Exception as e:
                # If conversion fails, skip this key
                continue
        if values:
            agg[key] = {"mean": np.mean(values), "std": np.std(values)}
        else:
            agg[key] = {"mean": None, "std": None}
    return agg

# List of seeds for multiple runs
seeds = [0, 1, 2, 3, 4]
rmse_list = []
time_list = []
diagnostics_list = []

for seed in seeds:
    key = jr.PRNGKey(seed)
    k1, k2 = jr.split(key, 2)
    model = Test()
    # For debugging/testing, we use 2 chains, 500 warmup, 1000 samples.
    model.compile(num_chains=2, num_warmup=500, num_samples=1000)
    start_time = time.time()
    model.fit(X_train, y_train, k1)
    end_time = time.time()
    run_time = end_time - start_time
    
    preds = model.predict(X_test, posterior="obs", rng_key=k2)
    # Compute mean prediction across chains/samples
    mean_preds = np.array(preds).mean(axis=0)
    rmse = np.sqrt(mean_squared_error(np.array(y_test), mean_preds))
    
    diagnostics = evaluate_mcmc(model)
    
    rmse_list.append(rmse)
    time_list.append(run_time)
    diagnostics_list.append(diagnostics)
    
    print(f"Seed {seed}: RMSE: {rmse:.2f}, Time: {run_time:.2f} s")
    print("Diagnostics:", diagnostics)

# Aggregate overall metrics from multiple runs
avg_rmse = np.mean(rmse_list)
std_rmse = np.std(rmse_list)
avg_time = np.mean(time_list)
std_time = np.std(time_list)
diagnostic_summary = aggregate_diagnostics(diagnostics_list)

print("\nOverall Results:")
print(f"RMSE: {avg_rmse:.2f} ± {std_rmse:.2f}")
print(f"Time: {avg_time:.2f} ± {std_time:.2f} s")

# Print full diagnostic summary table
print("\nDiagnostic Summary (averages ± std):")
for key, stat in diagnostic_summary.items():
    if stat["mean"] is not None:
        print(f"{key}: {stat['mean']:.2f} ± {stat['std']:.2f}")
    else:
        print(f"{key}: N/A")
