from numpyro.infer import NUTS, MCMC
import numpyro
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import random
from matplotlib.colors import ListedColormap
import jax


def run_inference(
    model, rng_key, X, y, num_samples=1000, num_warmup=500, init_strategy=None
):
    """
    Run MCMC using NUTS.
    """
    if init_strategy is not None:
        kernel = NUTS(model, init_strategy=init_strategy)
    else:
        kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key, X=X, y=y)
    mcmc.print_summary()
    return mcmc


def predict_binary(mcmc, X_test, model, sample_from="y"):
    """
    Generate predictions for binary classification using posterior samples from MCMC.

    Parameters:
    - mcmc: The MCMC object after sampling.
    - X_test: The test set input features (JAX array).
    - model: The probabilistic model used for inference.

    Returns:
    - predictions: A 2D array of shape (num_samples, len(X_test)), where each row corresponds
      to predictions for the test set from one posterior sample.
    """
    posterior_samples = mcmc.get_samples()

    predictive = numpyro.infer.Predictive(model, posterior_samples)

    preds = predictive(rng_key=random.PRNGKey(0), X=X_test)

    predictions = preds[sample_from]

    return predictions


def predict_regressor(mcmc, X_test, model):
    """
    Generate predictions for regression using posterior samples from MCMC.

    Parameters:
    - mcmc: The MCMC object after sampling.
    - X_test: The test set input features (JAX array).
    - model: The probabilistic model used for inference.

    Returns:
    - predictions: A 2D array of shape (num_samples, len(X_test)), where each row corresponds
      to predictions for the test set from one posterior sample.
    """
    posterior_samples = mcmc.get_samples()

    predictive = numpyro.infer.Predictive(model, posterior_samples)

    preds = predictive(rng_key=random.PRNGKey(0), X=X_test)

    predictions = preds["mean"]

    return predictions


def predict_multiclass(mcmc, X_test, model, sample_from="obs"):
    """
    Generate predictions for multiclass classification with explicit number of classes.

    Parameters:
    - mcmc: The MCMC object after sampling.
    - X_test: The test set input features (JAX array).
    - model: The probabilistic model used for inference.
    - n_classes: The number of classes to ensure consistency.

    Returns:
    - predictions: A 3D array of shape (num_samples, len(X_test), n_classes), where each slice
      corresponds to the predicted probabilities for each class from one posterior sample.
    """
    posterior_samples = mcmc.get_samples()
    predictive = numpyro.infer.Predictive(model, posterior_samples)
    preds = predictive(rng_key=random.PRNGKey(0), X=X_test)
    predictions = preds[sample_from]

    return predictions


def visualize_regression(
    X_test, y_test, mean_preds, lower_bound, upper_bound, feature_index=None
):
    """
    Visualize predictions with uncertainty bounds and true targets.

    Args:
        X_test (jnp.ndarray): Test features.
        y_test (jnp.ndarray): Test target values.
        mean_preds (jnp.ndarray): Mean predictions from the model.
        lower_bound (jnp.ndarray): Lower uncertainty bound.
        upper_bound (jnp.ndarray): Upper uncertainty bound.
        feature_index (int): Index of the feature to plot against y_test. If None or invalid, uses default.

    Returns:
        None. Displays the plot.
    """
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    mean_preds = np.array(mean_preds)
    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)

    if (
        X_test.shape[1] == 1
        or feature_index is None
        or not (0 <= feature_index < X_test.shape[1])
    ):
        feature_index = 0

    feature = X_test[:, feature_index]

    sorted_indices = np.argsort(feature)
    feature = feature[sorted_indices]
    y_test = y_test[sorted_indices]
    mean_preds = mean_preds[sorted_indices]
    lower_bound = lower_bound[sorted_indices]
    upper_bound = upper_bound[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.scatter(feature, y_test, color="blue", alpha=0.6, label="True Targets")
    plt.plot(
        feature,
        mean_preds,
        color="red",
        label="Mean Predictions",
        linestyle="-",
    )
    plt.fill_between(
        feature,
        lower_bound,
        upper_bound,
        color="gray",
        alpha=0.3,
        label="Uncertainty Bounds",
    )

    plt.xlabel(f"Feature {feature_index + 1}")
    plt.ylabel("Target (y_test)")
    plt.title("Model Predictions with Uncertainty and True Targets")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.show()


def visualize_binary(
    X, y, mcmc, predict, binary_model, feature_indices=(0, 1), grid_resolution=100
):
    """
    Visualize binary classification decision boundary with uncertainty.

    Args:
        X (jnp.ndarray): Input features.
        y (jnp.ndarray): Target labels (binary).
        mcmc: numpyro.infer.MCMC
            Trained MCMC object containing posterior samples.
        predict: Callable
            Prediction function for binary classification.
        binary_model: Callable
            The binary classification model to use for predictions.
        feature_indices (tuple): Indices of the two features to visualize (x and y axes).
        grid_resolution (int): Number of points for each grid axis (higher means finer grid).

    Returns:
        None. Displays the plot.
    """
    X = np.array(X)
    y = np.array(y)

    feature1_idx, feature2_idx = feature_indices
    feature1, feature2 = X[:, feature1_idx], X[:, feature2_idx]

    x_min, x_max = feature1.min() - 1, feature1.max() + 1
    y_min, y_max = feature2.min() - 1, feature2.max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    X_for_grid = np.zeros((grid.shape[0], X.shape[1]))
    X_for_grid[:, feature1_idx] = grid[:, 0]
    X_for_grid[:, feature2_idx] = grid[:, 1]
    for i in range(X.shape[1]):
        if i not in feature_indices:
            X_for_grid[:, i] = X[:, i].mean()

    grid_preds = predict(
        mcmc, jnp.array(X_for_grid), binary_model, sample_from="logits"
    )
    grid_preds = jax.nn.sigmoid(grid_preds)
    grid_mean = grid_preds.mean(axis=0).reshape(xx.shape)
    grid_uncertainty = grid_preds.var(axis=0).reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, grid_mean, levels=20, cmap="RdBu", alpha=0.8, vmin=0, vmax=1)
    plt.colorbar(label="Predicted Probability (Mean)")

    plt.imshow(
        grid_uncertainty,
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        cmap="binary",
        alpha=0.3,
        aspect="auto",
    )

    plt.contour(xx, yy, grid_mean, levels=[0.5], colors="black", linestyles="--")

    plt.scatter(
        feature1[y == 0],
        feature2[y == 0],
        color="blue",
        label="Class 0",
        edgecolor="k",
        alpha=0.6,
    )
    plt.scatter(
        feature1[y == 1],
        feature2[y == 1],
        color="red",
        label="Class 1",
        edgecolor="k",
        alpha=0.6,
    )

    plt.xlabel(f"Feature {feature1_idx + 1}")
    plt.ylabel(f"Feature {feature2_idx + 1}")
    plt.title("Binary Decision Boundary with Uncertainty")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()


def visualize_multiclass(
    X, y, mcmc, predict, multiclass_model, feature_indices=(0, 1), grid_resolution=100
):
    """
    Visualize multiclass classification decision boundary with uncertainty.

    Args:
        X (jnp.ndarray): Input features.
        y (jnp.ndarray): Target labels (integer class labels).
        mcmc: numpyro.infer.MCMC
            Trained MCMC object containing posterior samples.
        predict: Callable
            Prediction function for multiclass classification.
        multiclass_model: Callable
            The multiclass classification model to use for predictions.
        feature_indices (tuple): Indices of the two features to visualize (x and y axes).
        grid_resolution (int): Number of points for each grid axis (higher means finer grid).

    Returns:
        None. Displays the plot.
    """
    X = np.array(X)
    y = np.array(y)

    n_classes = len(np.unique(y))
    feature1_idx, feature2_idx = feature_indices
    feature1, feature2 = X[:, feature1_idx], X[:, feature2_idx]
    x_min, x_max = feature1.min() - 1, feature1.max() + 1
    y_min, y_max = feature2.min() - 1, feature2.max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    X_for_grid = np.zeros((grid.shape[0], X.shape[1]))
    X_for_grid[:, feature1_idx] = grid[:, 0]
    X_for_grid[:, feature2_idx] = grid[:, 1]

    for i in range(X.shape[1]):
        if i not in feature_indices:
            X_for_grid[:, i] = X[:, i].mean()
    grid_preds = predict(
        mcmc, jnp.array(X_for_grid), multiclass_model, sample_from="logits"
    )
    grid_preds = jax.nn.softmax(grid_preds, axis=-1)
    grid_mean = grid_preds.mean(axis=0).reshape(
        grid_resolution, grid_resolution, n_classes
    )
    grid_uncertainty = grid_preds.var(axis=0).reshape(
        grid_resolution, grid_resolution, n_classes
    )

    plt.figure(figsize=(10, 6))
    cmap = ListedColormap(plt.cm.tab10.colors[:n_classes])

    predicted_classes = grid_mean.argmax(axis=2)
    plt.contourf(
        xx,
        yy,
        predicted_classes,
        alpha=0.5,
        cmap=cmap,
        levels=np.arange(n_classes + 1) - 0.5,
    )
    plt.colorbar(ticks=np.arange(n_classes), label="Predicted Class")

    for class_idx in range(n_classes):
        plt.imshow(
            grid_uncertainty[:, :, class_idx],
            extent=(x_min, x_max, y_min, y_max),
            origin="lower",
            cmap="binary",
            alpha=0.3,
            aspect="auto",
        )

    for class_idx in range(n_classes):
        plt.scatter(
            feature1[y == class_idx],
            feature2[y == class_idx],
            label=f"Class {class_idx}",
            edgecolor="k",
            alpha=0.6,
        )

    plt.xlabel(f"Feature {feature1_idx + 1}")
    plt.ylabel(f"Feature {feature2_idx + 1}")
    plt.title("Multiclass Decision Boundary with Uncertainty")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()
