import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from quantbayes.stochax.utils import plot_fft_spectrum, visualize_circulant_kernel, get_fft_full_for_given_params  


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
                in_features=in_features, alpha=1, K=3, name="tester"
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
    param_dict = {key: value[0] for key, value in posterior_samples.items()}

    # (2) Perform a forward pass with a valid RNG key to get a concrete fft_full.
    fft_full = get_fft_full_for_given_params(
        model, X, param_dict, rng_key=jr.PRNGKey(0)
    )

    # (3) Plot the Fourier spectrum and circulant kernel.
    fig1 = plot_fft_spectrum(fft_full, show=True)
    fig2 = visualize_circulant_kernel(fft_full, show=True)