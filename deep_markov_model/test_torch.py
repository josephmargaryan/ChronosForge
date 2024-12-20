import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Define Emitter Module
class Emitter(nn.Module):
    def __init__(self, input_dim, z_dim, emission_dim):
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_input = nn.Linear(emission_dim, input_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z_t):
        h1 = self.relu(self.lin_z_to_hidden(z_t))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        ps = self.sigmoid(self.lin_hidden_to_input(h2))
        return ps


# Define Gated Transition Module
class GatedTransition(nn.Module):
    def __init__(self, z_dim, transition_dim):
        super().__init__()
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1):
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate = self.sigmoid(self.lin_gate_hidden_to_z(_gate))
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)
        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean)))
        return loc, scale


# Define the Deep Markov Model
class DMM(nn.Module):
    def __init__(self, input_dim, z_dim, emission_dim, transition_dim):
        super().__init__()
        self.emitter = Emitter(input_dim, z_dim, emission_dim)
        self.trans = GatedTransition(z_dim, transition_dim)
        self.z_0 = nn.Parameter(torch.zeros(z_dim))

    def forward(self, mini_batch, mini_batch_mask):
        T_max = mini_batch.size(1)
        z_prev = self.z_0.expand(mini_batch.size(0), self.z_0.size(0))
        z_samples = []
        for t in range(1, T_max + 1):
            z_loc, z_scale = self.trans(z_prev)
            z_t = torch.distributions.Normal(z_loc, z_scale).rsample()
            z_samples.append(z_t)
            emission_probs_t = self.emitter(z_t)
            z_prev = z_t
        return torch.stack(z_samples, dim=1)


# Generate synthetic data
def generate_synthetic_data(N, D, T):
    X = torch.sin(torch.linspace(0, 10, T)).unsqueeze(0).unsqueeze(-1).repeat(N, 1, D)
    noise = 0.1 * torch.randn(N, T, D)
    X += noise
    y = X[:, :, 0].sum(dim=1, keepdim=True) + 0.1 * torch.randn(N, 1)
    return X, y


if __name__ == "__main__":
    # Generate data
    N, D, T = 100, 5, 20
    X, y = generate_synthetic_data(N, D, T)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize model and optimizer
    z_dim, emission_dim, transition_dim = 3, 10, 10
    model = DMM(D, z_dim, emission_dim, transition_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    num_epochs = 1000
    batch_size = 16
    losses = []

    print("Training...")
    for epoch in range(num_epochs):
        model.train()
        perm = torch.randperm(X_train.size(0))
        epoch_loss = 0.0
        for i in range(0, X_train.size(0), batch_size):
            indices = perm[i : i + batch_size]
            mini_batch = X_train[indices]
            mini_batch_mask = torch.ones(mini_batch.size(0), T)
            optimizer.zero_grad()
            z_samples = model(mini_batch, mini_batch_mask)
            loss = torch.mean(
                (z_samples[:, :, 0].sum(dim=1) - y_train[indices].squeeze()) ** 2
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / (X_train.size(0) // batch_size))
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {losses[-1]:.4f}")

    # Validation
    model.eval()
    with torch.no_grad():
        mini_batch_mask = torch.ones(X_test.size(0), T)
        z_samples = model(X_test, mini_batch_mask)
        y_pred = z_samples[:, :, 0].sum(dim=1, keepdim=True)
        mse = mean_squared_error(y_test.numpy(), y_pred.numpy())

    print(f"Validation MSE: {mse:.4f}")

    # Plot training loss
    plt.figure(figsize=(8, 6))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.show()

    # Plot predictions vs true values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test.numpy(), y_pred.numpy(), alpha=0.7)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True vs Predicted Values")
    plt.show()
