"""
mnist_classifier_template.py

Starter skeleton for the “MNIST Hand‑Written Digits Classifier” project.
The student will fill in every block marked with TODO.

Public API (must stay unchanged – the test script will call these):
    get_device()                     → torch.device
    build_model()                    → torch.nn.Module
    get_data_loaders(batch_size) → (train_loader, test_loader)
    train_one_epoch(model, loader, criterion, optimizer, device) → float
    evaluate(model, loader, device) → float
    run_training(epochs, batch_size, lr) → None
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
    Return the torch device that should be used.
    If a CUDA‑enabled GPU is available, use it; otherwise fall back to CPU.
    **DO NOT** change the name or the return type – the tests rely on it.
    """
    # TODO: implement device selection (CUDA if available, else CPU)
    raise NotImplementedError


# -------------------------------------------------
# 2. Model definition – tiny ConvNet
# -------------------------------------------------
def build_model() -> nn.Module:
    """
    Build and return a tiny convolutional neural network for MNIST.
    Architecture (you may implement with nn.Sequential or a custom class):
        Conv2d(1 → 16, kernel=3, padding=1) → ReLU → MaxPool2d(2)
        Conv2d(16 → 32, kernel=3, padding=1) → ReLU → MaxPool2d(2)
        Flatten
        Linear(<computed‑in_features> → 128) → ReLU
        Linear(128 → 10)      # 10 digit classes
    The function should **return** an instantiated nn.Module.
    """
    # TODO: implement the model
    raise NotImplementedError


# -------------------------------------------------
# 3. Data loaders
# -------------------------------------------------
def get_data_loaders(batch_size: int = 64):
    """
    Download the MNIST dataset (if not present) and return two
    torch.utils.data.DataLoader objects:
        * train_loader  – shuffles the training set
        * test_loader   – does NOT shuffle the test set

    Apply the standard MNIST transform:
        transforms.ToTensor()
        transforms.Normalize(mean=0.1307, std=0.3081)

    Return: (train_loader, test_loader)
    """
    # TODO: create the two DataLoader objects and return them
    raise NotImplementedError


# -------------------------------------------------
# 4. One training epoch
# -------------------------------------------------
def train_one_epoch(
    model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device
) -> float:
    """
    Run a single training epoch.
    * Move the model to ``device`` (if it is not already there).
    * Iterate over ``loader``:
        – Move inputs & targets to ``device``.
        – Forward pass → compute loss.
        – Back‑propagation + optimizer step.
    * Accumulate the loss of every batch and return the **average** loss.
    """
    # TODO: implement the training loop for one epoch and return avg loss
    raise NotImplementedError


# -------------------------------------------------
# 5. Evaluation (accuracy)
# -------------------------------------------------
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """
    Compute classification accuracy of ``model`` on the data supplied by ``loader``.
    No gradients should be calculated.
    Return the accuracy as a float between 0 and 1 (correct / total).
    """
    # TODO: implement evaluation and return accuracy
    raise NotImplementedError


# -------------------------------------------------
# 6. Full training routine (entry point)
# -------------------------------------------------
def run_training(epochs: int = 5, batch_size: int = 64, lr: float = 0.001) -> None:
    """
    End‑to‑end training pipeline:
        1. device = get_device()
        2. model  = build_model()
        3. train_loader, test_loader = get_data_loaders(batch_size)
        4. criterion  = nn.CrossEntropyLoss()
        5. optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            avg_loss = train_one_epoch(model, train_loader,
                                      criterion, optimizer, device)
            acc = evaluate(model, test_loader, device)
            print(f"Epoch {epoch}/{epochs} – loss: {avg_loss:.4f} – "
                  f"accuracy: {acc*100:.2f}%")

    The function prints progress but returns nothing.
    """
    # TODO: glue everything together – see description above
    raise NotImplementedError


# -------------------------------------------------
# 7. Entry point for manual execution
# -------------------------------------------------
if __name__ == "__main__":
    # Running the file directly will start training with the default hyper‑parameters.
    run_training()
