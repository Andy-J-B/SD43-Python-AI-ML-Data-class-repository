"""
mnist_classifier.py

Fully‑implemented MNIST digit classifier using a tiny convolutional
neural network (PyTorch).  The script can be run directly or imported
by an automated test suite – all public functions have the signatures
defined in the template file.
"""

# -------------------------------------------------
# Imports
# -------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


# -------------------------------------------------
# 1. Device helper
# -------------------------------------------------
def get_device() -> torch.device:
    """
    Return the device to be used for training/inference.
    Uses CUDA if a GPU is available; otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------
# 2. Model definition – tiny ConvNet
# -------------------------------------------------
def build_model() -> nn.Module:
    """
    Tiny convolutional network for MNIST.
    Architecture:
        Conv2d(1, 16, 3, padding=1) → ReLU → MaxPool2d(2)
        Conv2d(16, 32, 3, padding=1) → ReLU → MaxPool2d(2)
        Flatten
        Linear(32 * 7 * 7, 128) → ReLU
        Linear(128, 10)
    """

    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            )
            # After two poolings: 28 → 14 → 7
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=32 * 7 * 7, out_features=128),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=128, out_features=10),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    return TinyCNN()


# -------------------------------------------------
# 3. Data loaders
# -------------------------------------------------
def get_data_loaders(batch_size: int = 64):
    """
    Returns (train_loader, test_loader) for MNIST with standard
    normalisation.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, test_loader


# -------------------------------------------------
# 4. One training epoch
# -------------------------------------------------
def train_one_epoch(
    model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device
) -> float:
    """
    Performs a full pass over ``loader`` and returns the average loss.
    """
    model.to(device)
    model.train()
    running_loss = 0.0
    total_batches = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_batches += 1

    avg_loss = running_loss / total_batches if total_batches > 0 else 0.0
    return avg_loss


# -------------------------------------------------
# 5. Evaluation (accuracy)
# -------------------------------------------------
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """
    Computes the classification accuracy of ``model`` on ``loader``.
    Returns a float in [0, 1].
    """
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)  # shape: (batch, 10)
            _, predicted = torch.max(outputs, dim=1)  # predicted class indices
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    return correct / total if total > 0 else 0.0


# -------------------------------------------------
# 6. Full training routine (entry point)
# -------------------------------------------------
def run_training(
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.001,
    checkpoint_path: str = "mnist_cnn.pt",
) -> None:
    """
    Orchestrates data loading, model creation, training and evaluation.
    Prints the loss and accuracy after each epoch.
    """
    device = get_device()
    print(f"Using device: {device}")

    # 1️⃣ Build model & move it to the right device
    model = build_model()
    model.to(device)

    # 2️⃣ Load data
    train_loader, test_loader = get_data_loaders(batch_size)

    # 3️⃣ Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4️⃣ Training loop
    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch}/{epochs} – "
            f"loss: {avg_loss:.4f} – "
            f"accuracy: {acc * 100:.2f}%"
        )

    # ----- SAVE THE TRAINED WEIGHTS -------------------------------------------------
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nModel checkpoint saved to '{checkpoint_path}'")

    # ----- FINAL EVALUATION ---------------------------------------------------------
    final_acc = evaluate(model, test_loader, device)
    print(f"Final test accuracy: {final_acc*100:.2f}%")


# -------------------------------------------------
# 7. Entry point
# -------------------------------------------------
if __name__ == "__main__":
    # Default hyper‑parameters: 5 epochs, batch size 64, LR 0.001
    run_training()
