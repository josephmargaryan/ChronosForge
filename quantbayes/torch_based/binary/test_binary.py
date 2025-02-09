# binary_script_torch.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Optional, Tuple


# ----------------------------------------------------------------
# 1. Model Definition
#    MLP for Binary Classification:
#    Output dimension = 1 (logits for BCEWithLogitsLoss)
# ----------------------------------------------------------------
class MLPBinaryClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns logits, shape: (batch_size, 1)
        return self.net(x)


# ----------------------------------------------------------------
# 2. Training Function
# ----------------------------------------------------------------
def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
) -> nn.Module:
    """
    Train a binary classification MLP using BCEWithLogitsLoss.
    """
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    train_losses = []
    val_losses = []

    def batch_generator(X, y, bs):
        n = len(X)
        indices = torch.randperm(n)
        for start in range(0, n, bs):
            end = start + bs
            batch_idx = indices[start:end]
            yield X[batch_idx], y[batch_idx]

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = []

        for batch_x, batch_y in batch_generator(X_train_t, y_train_t, batch_size):
            optimizer.zero_grad()
            logits = model(batch_x).squeeze(-1)  # (batch_size,)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))

        # Validation: run in evaluation mode
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).squeeze(-1)
            val_loss = loss_fn(val_logits, y_val_t).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % max(1, (num_epochs // 10)) == 0:
            print(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

    # Plot training and validation loss
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross Entropy")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model


# ----------------------------------------------------------------
# 3. Evaluation and Decision Boundary Visualization
# ----------------------------------------------------------------
def evaluate_model(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    features: Optional[Tuple[int, int]] = (0, 1),
) -> dict:
    """
    Perform deterministic evaluation for binary classification.
    Plots the decision boundary based on chosen features.

    Returns a dictionary with final predictions, accuracy, etc.
    """
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    # Set model to evaluation mode for deterministic predictions
    model.eval()

    with torch.no_grad():
        logits = model(X_val_t).squeeze(-1)
        probs = torch.sigmoid(logits)  # shape: (batch_size,)
        mean_preds = probs.cpu().numpy()

    # Accuracy (threshold at 0.5)
    y_pred_binary = (mean_preds > 0.5).astype(float)
    acc = np.mean(y_pred_binary == y_val)

    # Decision boundary for 2D data (features f1, f2)
    f1, f2 = features
    x_min, x_max = X_val[:, f1].min() - 0.5, X_val[:, f1].max() + 0.5
    y_min, y_max = X_val[:, f2].min() - 0.5, X_val[:, f2].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100),
    )
    grid_points = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    gp_t = torch.tensor(grid_points, dtype=torch.float32)

    # Use a single forward pass on the grid for predictions
    with torch.no_grad():
        grid_logits = model(gp_t).squeeze(-1)
        grid_probs = torch.sigmoid(grid_logits).cpu().numpy()

    # Reshape for contour plotting
    Z = grid_probs.reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(8, 6))
    # Contour where probability is 0.5
    plt.contourf(xx, yy, Z, levels=[0.0, 0.5, 1.0], alpha=0.3, colors=["red", "blue"])

    # Plot the validation points
    plt.scatter(
        X_val[y_val == 0, f1],
        X_val[y_val == 0, f2],
        c="red",
        edgecolor="k",
        label="Class 0",
    )
    plt.scatter(
        X_val[y_val == 1, f1],
        X_val[y_val == 1, f2],
        c="blue",
        edgecolor="k",
        label="Class 1",
    )
    plt.title(f"Binary Classification - Decision Boundary (Accuracy={acc:.2f})")
    plt.xlabel(f"Feature {f1}")
    plt.ylabel(f"Feature {f2}")
    plt.legend()
    plt.show()

    return {
        "mean_probs": mean_preds,
        "accuracy": acc,
    }


# ----------------------------------------------------------------
# 4. Example Test
# ----------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic binary data: two clusters
    np.random.seed(0)
    N = 200
    X0 = np.random.normal(loc=[-2, 0], scale=1.0, size=(N // 2, 2))
    X1 = np.random.normal(loc=[2, 0], scale=1.0, size=(N // 2, 2))
    X_all = np.vstack([X0, X1]).astype(np.float32)
    y_all = np.array([0] * (N // 2) + [1] * (N // 2), dtype=np.float32)

    # Shuffle data
    perm = np.random.permutation(N)
    X_all = X_all[perm]
    y_all = y_all[perm]

    # Split data
    train_size = int(0.8 * N)
    X_train, X_val = X_all[:train_size], X_all[train_size:]
    y_train, y_val = y_all[:train_size], y_all[train_size:]

    # Initialize model
    model = MLPBinaryClassifier(input_dim=2, hidden_dim=32)

    # Train model
    model = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        num_epochs=100,
        batch_size=16,
        learning_rate=1e-3,
    )

    # Evaluate model
    results = evaluate_model(model, X_val, y_val, features=(0, 1))
    print("Final Accuracy on Val:", results["accuracy"])
