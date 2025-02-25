import jax 
import equinox as eqx
import jax.random as jr
import jax.numpy as jnp
import numpyro 
import numpyro.distributions as dist

from quantbayes import bnn
from quantbayes.fake_data import generate_regression_data
from quantbayes.stochax.utils import prior_fn, bayesianize
from quantbayes.stochax.layers import SpectralDenseBlock

df = generate_regression_data()
X, y = df.drop("target", axis=1), df["target"]
X, y = jnp.array(X), jnp.array(y)

class Det(eqx.Module):
    fc1: eqx.Module
    out: eqx.Module
    def __init__(self, key):
        k1, k2 = jr.split(key, 2)
        self.fc1 = SpectralDenseBlock(
            in_features=1,
            hidden_dim=32,
            out_features=16,
            key=k1
        )
        self.out = eqx.nn.Linear(16, 1, key=k2)
    def __call__(self, x):
        x = self.fc1(x)
        x = jax.nn.tanh(x)
        return self.out(x)
    

class Test(bnn.Module):
    def __init__(self):
        super().__init__()
    def __call__(self, X, y=None):
        N, D = X.shape 
        key = jr.key(2)
        net = Det(key)
        bayes = bayesianize(net, prior_fn)
        X = jax.vmap(bayes)(X)
        mu = X.squeeze()
        numpyro.sample("obs", dist.Normal(mu), obs=y)

k = jr.key(0)
model = Test()
model.compile()
model.fit(X, y, k)