import jax.numpy as jnp
from jax import tree_util
import equinox as eqx
import numpyro
import numpyro.distributions as dist

__all__ = [
    "bayesianize", 
    "prior_fn",
    "decaying_prior",
    "decaying_prior_block"
    ]
# ------------------------------------------------------------------------------
# Helper: Bayesianize an Equinox module
# ------------------------------------------------------------------------------
def bayesianize(module: eqx.Module, prior_fn):
    """
    Traverse the module's pytree and, for every array leaf, replace it with a
    NumPyro sample drawn from `prior_fn` (a function accepting a shape and
    returning a distribution). Each sample is tagged with a unique name.
    """
    leaves, treedef = tree_util.tree_flatten(module)
    new_leaves = []
    for i, leaf in enumerate(leaves):
        if isinstance(leaf, jnp.ndarray):
            # Draw a sample for this parameter using the provided prior function.
            new_leaf = numpyro.sample(f"param_{i}", prior_fn(leaf.shape))
            new_leaves.append(new_leaf)
        else:
            new_leaves.append(leaf)
    return tree_util.tree_unflatten(treedef, new_leaves)


# ------------------------------------------------------------------------------
# Helper: Prior function
# ------------------------------------------------------------------------------
def prior_fn(shape, mean=0.0, std=1.0, dist_cls=dist.Normal):
    """
    Returns a prior distribution for a given shape.

    By default, it returns a Normal(0, 1) prior. Users can override the mean,
    standard deviation, or even the distribution class if needed.

    Parameters:
        shape (tuple): The shape of the parameter array.
        mean (float, optional): Mean of the prior distribution. Default is 0.0.
        std (float, optional): Standard deviation of the prior distribution. Default is 1.0.
        dist_cls (callable, optional): The NumPyro distribution class to use. Default is Normal.

    Returns:
        A NumPyro distribution expanded to `shape` with the correct event dimensions.

    Example usage:
        custom_prior_fn = lambda shape: prior_fn(shape, mean=2.1, std=3.4, dist_cls=dist.Uniform)
    """
    return dist_cls(mean, std).expand(shape).to_event(len(shape))

def decaying_prior(shape, alpha=1.0):
    """
    Example uscase:
    fft_layer = FFTDirectPriorLinear(in_features=n, 
            key=key, 
            init_scale=1.0, 
            prior_fn=decaying_prior)
    """
    indices = jnp.arange(shape[0])
    # Standard deviation decays with frequency index
    stds = 1.0 / jnp.sqrt(1.0 + (indices ** alpha))
    return dist.Normal(0.0, stds).to_event(1)


def decaying_prior_block(shape, alpha=1.0):
    """
    Example usage:

    block_layer = BlockFFTDirectPriorLayer(
    in_features=..., 
    out_features=..., 
    block_size=..., 
    key=..., 
    real_prior_fn=lambda shape: decaying_prior_block(shape, alpha=1.0),
    imag_prior_fn=lambda shape: decaying_prior_block(shape, alpha=1.0)
    )

    """
    indices = jnp.arange(shape[0])
    stds = 1.0 / jnp.sqrt(1.0 + (indices ** alpha))
    return dist.Normal(0.0, stds).to_event(1)

