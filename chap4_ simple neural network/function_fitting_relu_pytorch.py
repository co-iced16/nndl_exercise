import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def target_function(x: np.ndarray) -> np.ndarray:
    # Nonlinear target with multiple frequencies.
    return np.sin(2.0 * x) + 0.3 * np.cos(5.0 * x) + 0.1 * x * x


def build_dataset(n_train: int = 512, n_test: int = 256):
    x_train = np.random.uniform(-3.0, 3.0, size=(n_train, 1)).astype(np.float32)
    x_test = np.linspace(-3.0, 3.0, n_test, dtype=np.float32).reshape(-1, 1)
    y_train = target_function(x_train).astype(np.float32)
    y_test = target_function(x_test).astype(np.float32)
    return x_train, y_train, x_test, y_test


class ReLURegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def main():
    set_seed(42)
    x_train, y_train, x_test, y_test = build_dataset()

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    x_test_t = torch.from_numpy(x_test)
    y_test_t = torch.from_numpy(y_test)

    model = ReLURegressor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 2000
    for epoch in range(epochs):
        model.train()
        pred = model(x_train_t)
        loss = criterion(pred, y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(x_test_t)
                test_loss = criterion(test_pred, y_test_t).item()
            print(
                f"epoch {epoch + 1:4d}/{epochs}, "
                f"train_mse={loss.item():.6f}, test_mse={test_loss:.6f}"
            )

    model.eval()
    with torch.no_grad():
        y_pred = model(x_test_t).numpy()
        final_test_mse = np.mean((y_pred - y_test) ** 2)

    print(f"final test mse: {final_test_mse:.6f}")

    plt.figure(figsize=(8, 5))
    plt.plot(x_test, y_test, label="target function", linewidth=2)
    plt.plot(x_test, y_pred, label="relu network fit", linewidth=2)
    plt.scatter(x_train, y_train, s=8, alpha=0.3, label="train samples")
    plt.legend()
    plt.title("Function Fitting with ReLU Network")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig("fitting_result.png", dpi=160)
    print("saved figure: fitting_result.png")


if __name__ == "__main__":
    main()
