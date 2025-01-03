from BNN.DENSE.MCMC_METHOD.models import (
    regression_model,
    binary_model,
    multiclass_model,
    hierarchical_binary,
    hierarchical_multiclass,
    hierarchical_regressor,
)
from BNN.DENSE.MCMC_METHOD.utils import (
    run_inference,
    visualize_regression,
    predict_binary,
    predict_multiclass,
    predict_regressor,
    visualize_binary,
    visualize_multiclass,
)
from fake_data import (
    generate_simple_regression_data,
    generate_binary_classification_data,
    generate_multiclass_classification_data,
)
from sklearn.model_selection import train_test_split
import numpy as np
import jax.numpy as jnp
from jax import random
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    recall_score,
    precision_score,
    log_loss,
)
import jax


def test_regression():
    n_samples = 500
    n_features = 8
    random_seed = 42
    rng_key = random.key(0)

    simple_data = generate_simple_regression_data(
        n_samples, n_features, random_seed=random_seed
    )

    X, y = simple_data.drop(columns=["target"], axis=1), simple_data["target"]
    X, y = jnp.array(X), jnp.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=24, test_size=0.2
    )

    mcmc = run_inference(
        hierarchical_regressor,
        rng_key,
        X_train,
        y_train,
        num_samples=1000,
        num_warmup=500,
    )

    predictions = predict_regressor(mcmc, X_test, hierarchical_regressor)

    mean_preds = predictions.mean(axis=0)
    std_preds = predictions.std(axis=0)
    lower_bound = mean_preds - 1.96 * std_preds
    upper_bound = mean_preds + 1.96 * std_preds

    MSE = mean_squared_error(y_test, mean_preds)
    RMSE = root_mean_squared_error(y_test, mean_preds)
    MAE = mean_absolute_error(y_test, mean_preds)
    print(f"MSE: {MSE}\nRMSE: {RMSE}\nMAE: {MAE}")

    visualize_regression(X_test, y_test, mean_preds, lower_bound, upper_bound, 0)


def test_binary():
    rng_key = random.key(1)
    df = generate_binary_classification_data()
    X, y = df.drop(columns=["target"], axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=24
    )

    mcmc = run_inference(
        hierarchical_binary, rng_key, X_train, y_train, num_samples=1000, num_warmup=500
    )

    predictions = predict_binary(
        mcmc, X_test, hierarchical_binary, sample_from="logits"
    )
    mean_preds = predictions.mean(axis=0)
    probabilities = jax.nn.sigmoid(mean_preds)
    std_preds = predictions.std(axis=0)
    binary_preds = np.array((probabilities >= 0.5).astype(int))
    y_preds = np.array(y_test)
    accuracy = accuracy_score(y_preds, binary_preds)
    precision = precision_score(y_preds, binary_preds)
    recall = recall_score(y_preds, binary_preds)
    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}")

    visualize_binary(
        X_test,
        y_test,
        mcmc,
        predict_binary,
        binary_model,
        feature_indices=(0, 1),
        grid_resolution=200,
    )


def test_multiclass():
    rng_key = random.key(2)
    df = generate_multiclass_classification_data()
    X, y = df.drop(columns=["target"], axis=1), df["target"]
    X, y = jnp.array(X), jnp.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=45, test_size=0.2
    )

    mcmc = run_inference(hierarchical_multiclass, rng_key, X_train, y_train, 100, 50)
    predictions = predict_multiclass(
        mcmc, X_test, hierarchical_multiclass, sample_from="logits"
    )
    mean_preds = predictions.mean(axis=0)
    probabilities = jax.nn.softmax(mean_preds, axis=-1)
    std_preds = predictions.std(axis=0)

    loss = log_loss(np.array(y_test), np.array(probabilities))
    print(f"Loss: {loss}")

    visualize_multiclass(
        X_test,
        y_test,
        mcmc,
        predict_multiclass,
        multiclass_model,
        feature_indices=(0, 1),  # Features to visualize
        grid_resolution=200,  # Resolution for decision boundary
    )


if __name__ == "__main__":
    print("Testing Binary")
    test_binary()
    print("Testing Regressor")
    test_regression()
    print("Testing Multiclass")
    test_multiclass()
