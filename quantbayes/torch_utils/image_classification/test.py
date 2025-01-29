# image_classification_script_torch.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import List


# ----------------------------------------------------------------------------
# 1. Simple CNN Model (Replace with your actual model if desired)
# ----------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(
            64 * 8 * 8, 128
        )  # if input is 32x32 -> after 2 pools -> 8x8
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, 3, height, width)
        output: (batch_size, num_classes)
        """
        x = torch.relu(self.conv1(x))  # (batch_size, 32, h, w)
        x = self.pool(x)  # (batch_size, 32, h/2, w/2)
        x = torch.relu(self.conv2(x))  # (batch_size, 64, h/2, w/2)
        x = self.pool(x)  # (batch_size, 64, h/4, w/4)
        x = x.view(x.size(0), -1)  # flatten
        x = torch.relu(self.fc1(x))  # (batch_size, 128)
        x = self.dropout(
            x,
        )  # for MC if needed
        logits = self.fc2(x)  # (batch_size, num_classes)
        return logits


# ----------------------------------------------------------------------------
# 2. Training Function
# ----------------------------------------------------------------------------
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
    Train a classification model using cross-entropy loss.
    X_*: (N, 3, H, W)  y_*: (N,) integer class labels
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Convert data to torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Data generator
    def batch_generator(X, y, bs):
        n = len(X)
        indices = torch.randperm(n).to(device)
        for start in range(0, n, bs):
            end = start + bs
            batch_idx = indices[start:end]
            yield X[batch_idx], y[batch_idx]

    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = []

        # training loop
        for batch_x, batch_y in batch_generator(X_train_t, y_train_t, batch_size):
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))

        # validation
        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_t)
            val_loss = criterion(logits_val, y_val_t).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % max(1, (num_epochs // 5)) == 0:
            print(
                f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

    # Plot train/val losses
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model


# ----------------------------------------------------------------------------
# 3. Evaluate Model (Accuracy + Optional MC)
# ----------------------------------------------------------------------------
def evaluate_model(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_samples: int = 1,
    batch_size: int = 32,
) -> float:
    """
    Evaluate classification accuracy. If num_samples>1 and model has dropout,
    we can do multiple forward passes to gather stochastic predictions.
    """
    device = next(model.parameters()).device

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    def mini_batches(X, bs):
        n = len(X)
        for i in range(0, n, bs):
            yield X[i : i + bs]

    predictions = []

    for _ in range(num_samples):
        model.train()  # keep dropout "on" if we want MC sampling
        all_preds = []
        with torch.no_grad():
            for batch_x in mini_batches(X_val_t, batch_size):
                logits = model(batch_x)
                preds = torch.argmax(logits, dim=-1)
                all_preds.append(preds.cpu().numpy())
        all_preds = np.concatenate(all_preds, axis=0)  # shape (N_val,)
        predictions.append(all_preds)

    # (num_samples, N_val)
    preds_stacked = np.stack(predictions, axis=0)
    # simple approach: average and round or mode
    # We'll do a "mode" across samples
    final_preds = []
    for i in range(preds_stacked.shape[1]):
        vals, counts = np.unique(preds_stacked[:, i], return_counts=True)
        # pick the most frequent class across MC samples
        final_preds.append(vals[np.argmax(counts)])
    final_preds = np.array(final_preds)

    accuracy = (final_preds == y_val).mean()
    return float(accuracy)


# ----------------------------------------------------------------------------
# 4. Visualization
# ----------------------------------------------------------------------------
def visualize_predictions(
    model: nn.Module,
    X_samples: np.ndarray,
    y_samples: np.ndarray,
    class_names: List[str],
    num_plots: int = 5,
):
    """
    Display `num_plots` images with predicted vs ground-truth labels.
    X_samples: (N, 3, H, W)
    """
    device = next(model.parameters()).device

    idxs = np.random.choice(len(X_samples), num_plots, replace=False)
    plt.figure(figsize=(10, 2 * num_plots))

    for i, idx in enumerate(idxs):
        img_t = torch.tensor(X_samples[idx : idx + 1], dtype=torch.float32).to(device)
        logits = model(img_t)
        pred_label = int(torch.argmax(logits, dim=-1)[0].cpu().numpy())
        gt_label = int(y_samples[idx])

        # convert image to (H,W,3) for plotting
        img_np = X_samples[idx].transpose(1, 2, 0)  # from (3,H,W) to (H,W,3)

        plt.subplot(num_plots, 1, i + 1)
        plt.imshow(img_np, interpolation="nearest")
        plt.axis("off")
        plt.title(f"Predicted: {class_names[pred_label]}, GT: {class_names[gt_label]}")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# 5. Example Test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data: 32x32 images, 2 classes
    np.random.seed(42)
    N = 1000
    H, W = 32, 32
    C = 3
    X_all = np.random.rand(N, C, H, W).astype(np.float32)
    y_all = np.random.randint(low=0, high=2, size=(N,)).astype(np.int64)

    # split
    train_size = int(0.8 * N)
    X_train, X_val = X_all[:train_size], X_all[train_size:]
    y_train, y_val = y_all[:train_size], y_all[train_size:]

    # define model
    model = SimpleCNN(num_classes=2)

    # train
    model = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_epochs=10,
        batch_size=32,
        learning_rate=1e-3,
    )

    # evaluate
    acc = evaluate_model(model, X_val, y_val, num_samples=1, batch_size=32)
    print("Validation Accuracy:", acc)

    # visualize
    class_names = ["Class 0", "Class 1"]
    visualize_predictions(model, X_val, y_val, class_names, num_plots=5)
