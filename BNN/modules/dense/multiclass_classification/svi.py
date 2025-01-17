from BNN.core.base_task import BaseTask
from BNN.core.base_inference import BaseInference
from typing import Callable
import jax
import jax.numpy as jnp
import numpyro
import numpy as np
import matplotlib.pyplot as plt
from numpyro.optim import Adam
from numpyro.infer import SVI, Trace_ELBO, Predictive, autoguide
from jax import random
import numpyro.distributions as dist


class DenseMultiClassSVI(BaseTask, BaseInference):
    """
    Dense multi-class classification model using SVI inference with flexible architecture.
    """

    def __init__(
        self,
        num_steps=500,
        model_type="shallow",  # New argument
        hidden_dim=10,
        num_classes=3,
        track_loss=False,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim
        self.model_type = model_type.lower()
        self.track_loss = track_loss
        self.num_classes = num_classes
        self.svi = None
        self.params = None
        self.loss = None

        # Validate the model_type
        if self.model_type not in ["shallow", "deep"]:
            raise ValueError("model_type must be 'shallow' or 'deep'.")

    def get_default_model(self) -> Callable:
        """
        Return the appropriate model based on the model_type.
        """
        if self.model_type == "shallow":
            return lambda X, y=None, **kwargs: self.multiclass_model(X, y)
        elif self.model_type == "deep":
            return lambda X, y=None, **kwargs: self.deep_multiclass_model(X, y)

    def multiclass_model(self, X, y=None):
        """
        Shallow Bayesian multi-class classification model with one dense layer.
        """
        input_dim = X.shape[1]
        hidden_dim = self.hidden_dim
        num_classes = self.num_classes

        w_hidden = numpyro.sample(
            "w_hidden", dist.Normal(0, 1).expand([input_dim, hidden_dim])
        )
        b_hidden = numpyro.sample("b_hidden", dist.Normal(0, 1).expand([hidden_dim]))

        w_out = numpyro.sample(
            "w_out", dist.Normal(0, 1).expand([hidden_dim, num_classes])
        )
        b_out = numpyro.sample("b_out", dist.Normal(0, 1).expand([num_classes]))

        hidden = jax.nn.silu(jnp.dot(X, w_hidden) + b_hidden)
        logits = jnp.dot(hidden, w_out) + b_out
        logits = jnp.clip(logits, a_min=-10, a_max=10)

        numpyro.deterministic("logits", logits)
        numpyro.sample("obs", dist.Categorical(logits=logits), obs=y)

    def deep_multiclass_model(self, X, y=None):
        """
        Deep Bayesian multi-class classification model with two dense layers.
        """
        input_dim = X.shape[1]
        hidden_dim = self.hidden_dim
        num_classes = self.num_classes

        # First dense layer
        w_hidden1 = numpyro.sample(
            "w_hidden1", dist.Normal(0, 1).expand([input_dim, hidden_dim])
        )
        b_hidden1 = numpyro.sample("b_hidden1", dist.Normal(0, 1).expand([hidden_dim]))
        hidden1 = jax.nn.silu(jnp.dot(X, w_hidden1) + b_hidden1)

        # Second dense layer
        w_hidden2 = numpyro.sample(
            "w_hidden2", dist.Normal(0, 1).expand([hidden_dim, hidden_dim])
        )
        b_hidden2 = numpyro.sample("b_hidden2", dist.Normal(0, 1).expand([hidden_dim]))
        hidden2 = jax.nn.silu(jnp.dot(hidden1, w_hidden2) + b_hidden2)

        # Output layer
        w_out = numpyro.sample(
            "w_out", dist.Normal(0, 1).expand([hidden_dim, num_classes])
        )
        b_out = numpyro.sample("b_out", dist.Normal(0, 1).expand([num_classes]))

        logits = jnp.dot(hidden2, w_out) + b_out
        logits = jnp.clip(logits, a_min=-10, a_max=10)

        numpyro.deterministic("logits", logits)
        numpyro.sample("obs", dist.Categorical(logits=logits), obs=y)

    def bayesian_inference(self, X_train, y_train, rng_key):
        """
        Perform SVI inference for the Bayesian regression model.
        """
        model = self.get_default_model()
        guide = autoguide.AutoNormal(model)
        optimizer = Adam(0.01)

        self.svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        svi_state = self.svi.init(
            rng_key,
            X_train,
            y_train,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
        )

        loss_progression = [] if self.track_loss else None

        for step in range(self.num_steps):
            svi_state, loss = self.svi.update(svi_state, X_train, y_train)
            if self.track_loss:
                loss_progression.append(loss)
            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss:.4f}")

        self.params = self.svi.get_params(svi_state)
        if self.track_loss:
            self.loss = loss_progression

    def retrieve_results(self) -> dict:
        """
        Retrieve inference results, optionally including losses.
        """
        if not self.params:
            raise RuntimeError("No inference results available. Fit the model first.")
        if self.track_loss:
            return {"svi": self.svi, "params": self.params, "loss": self.loss}
        return {"svi": self.svi, "params": self.params}

    def fit(self, X_train, y_train, rng_key):
        """
        Fit the model using MCMC.
        """
        self.bayesian_inference(X_train, y_train, rng_key)
        self.fitted = True

    def predict(self, X_test, rng_key, num_samples=100):
        """
        Predict regression values using the posterior.
        """
        if not self.fitted or self.svi is None:
            raise RuntimeError("Model must be fitted before making predictions.")
        predictive = Predictive(
            self.svi.model,
            guide=self.svi.guide,
            params=self.params,
            num_samples=num_samples,
        )
        rng_key = random.PRNGKey(1)
        pred_samples = predictive(
            rng_key, X=X_test, hidden_dim=self.hidden_dim, num_classes=self.num_classes
        )
        return pred_samples["logits"]

    def visualize(self, X, y, num_classes, features=(0, 1), resolution=100):
        """
        Visualizes multiclass decision boundaries with uncertainty.

        Args:
            X: Input data (shape: (N, D)).
            y: True class labels (shape: (N,)).
            num_classes: Number of classes.
            features: Tuple specifying the indices of the two features to visualize.
            resolution: Grid resolution for decision boundary visualization.
        """
        feature_1, feature_2 = features
        X_selected = X[:, [feature_1, feature_2]]
        x_min, x_max = X_selected[:, 0].min() - 1, X_selected[:, 0].max() + 1
        y_min, y_max = X_selected[:, 1].min() - 1, X_selected[:, 1].max() + 1
        xx, yy = jnp.meshgrid(
            jnp.linspace(x_min, x_max, resolution),
            jnp.linspace(y_min, y_max, resolution),
        )

        grid_points = jnp.c_[xx.ravel(), yy.ravel()]
        grid_points_full = jnp.zeros((grid_points.shape[0], X.shape[1]))
        grid_points_full = grid_points_full.at[:, feature_1].set(grid_points[:, 0])
        grid_points_full = grid_points_full.at[:, feature_2].set(grid_points[:, 1])

        # Predict on grid points
        pred_samples = self.predict(grid_points_full, jax.random.PRNGKey(0))
        grid_mean_predictions = jax.nn.softmax(pred_samples.mean(axis=0), axis=-1)

        # Compute entropy (uncertainty)
        grid_uncertainty = -jnp.sum(
            grid_mean_predictions * jnp.log(grid_mean_predictions + 1e-9), axis=1
        )

        grid_classes = jnp.argmax(grid_mean_predictions, axis=1).reshape(xx.shape)
        grid_uncertainty = grid_uncertainty.reshape(xx.shape)

        # Plot decision boundaries and uncertainty
        plt.figure(figsize=(12, 8))
        plt.contourf(
            xx, yy, grid_classes, levels=num_classes, cmap=plt.cm.RdYlBu, alpha=0.6
        )
        plt.colorbar(label="Predicted Class")
        plt.contourf(xx, yy, grid_uncertainty, levels=20, cmap="gray", alpha=0.4)
        plt.colorbar(label="Uncertainty (Entropy)")
        plt.scatter(
            X_selected[:, 0],
            X_selected[:, 1],
            c=y,
            edgecolor="k",
            cmap=plt.cm.RdYlBu,
            s=20,
        )
        plt.title(
            f"Multiclass Decision Boundaries with Uncertainty (Features {features[0]} and {features[1]})"
        )
        plt.xlabel(f"Feature {feature_1 + 1}")
        plt.ylabel(f"Feature {feature_2 + 1}")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":

    ############# Example Usage #############
    from BNN.modules.dense.multiclass_classification.svi import DenseMultiClassSVI
    from BNN.utils.entropy_analysis import EntropyAndMutualInformation
    from BNN.utils.generalization_bound import BayesianAnalysis
    from fake_data import generate_multiclass_classification_data
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss
    import jax.random as jr
    import jax.numpy as jnp

    df = generate_multiclass_classification_data(n_classes=3)
    X, y = df.drop("target", axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=24
    )
    key = jr.key(0)
    classifier = DenseMultiClassSVI(
        num_steps=1000, hidden_dim=10, num_classes=3, track_loss=True, model_type="deep"
    )
    classifier.fit(X_train, y_train, key)
    posteriors_preds = classifier.predict(X_test, key)
    print(posteriors_preds.shape)
    probabilities = jax.nn.softmax(posteriors_preds, axis=-1)
    results = classifier.retrieve_results()
    svi = results["svi"]
    params = results["params"]
    loss = results.get("loss")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss) + 1), loss, label="Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss over steps (Convergence)")
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    mean_preds = probabilities.mean(axis=0)
    logloss = log_loss(np.array(y_test), np.array(mean_preds))
    print(f"Log loss {logloss}")
    classifier.visualize(
        X=X_test, y=y_test, num_classes=3, features=(0, 1), resolution=100
    )
    analysis = BayesianAnalysis(len(X_train), delta=0.05, task_type="multiclass")
    layer_names = sorted(
        set(key.rsplit("_", 2)[0] for key in params.keys() if key.endswith("_auto_loc"))
    )
    # Compute PAC-Bayesian bound for MCMC
    bound = analysis.compute_pac_bayesian_bound(
        predictions=probabilities,
        y_true=y_test,
        posterior_samples=params,
        layer_names=layer_names,
        inference_type="svi",
        prior_mean=0,
        prior_std=1,
    )
    print("PAC-Bayesian Bound (MCMC):", bound)
    mi = analysis.compute_mutual_information_bound(
        posterior_samples=params,
        layer_names=layer_names,
        inference_type="svi",
    )
    print(f"Mutual Information bound: {mi}")
    uncertainty_quantification = EntropyAndMutualInformation("multiclass")
    mi, mi1 = uncertainty_quantification.compute_mutual_information(probabilities)
    uncertainty_quantification.visualize(mi, mi1)
