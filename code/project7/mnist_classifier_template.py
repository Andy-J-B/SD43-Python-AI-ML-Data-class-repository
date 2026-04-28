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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------
# 2. Model definition – tiny ConvNet
# -------------------------------------------------
def build_model() -> nn.Module:
    """
    Build and return a tiny convolutional neural network for MNIST.
    Architecture (you may implement with nn.Sequential or a custom class):
        Conv2d(1 → 16, kernel=3, padding=1) → ReLU → MaxPool2d(2)
        ^ bro i read this as covid
        Conv2d(16 → 32, kernel=3, padding=1) → ReLU → MaxPool2d(2)
        Flatten
        Linear(<computed‑in_features> → 128) → ReLU
        Linear(128 → 10)      # 10 digit classes
    The function should **return** an instantiated nn.Module.
    """

    # TODO: implement the model
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
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=32 * 49, out_channels=128),
                nn.ReLU(in_plaace=True),
                nn.Linear(in_features=1, out_features=10),
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
        Download the MNIST dataset (if not present) and return two
        torch.utils.data.DataLoader objects:
            * train_loader  – shuffles the training set
            * test_loader   – does NOT shuffle the test set
    wait is it albert or caidon my bad
        Apply the standard MNIST transform:
            transforms.ToTensor()
            transforms.Normalize(mean=0.1307, std=0.3081)

        Return: (train_loader, test_loader)
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
    Run a single training epoch.
    * Move the model to ``device`` (if it is not already there).
    * Iterate over ``loader``:
        – Move inputs & targets to ``device``.
        – Forward pass → compute loss.
        – Back‑propagation + optimizer step.
    * Accumulate the loss of every batch and return the **average** loss.
    """
    # TODO: implement the training loop for one epoch and return avg loss
    model.to(device)
    model.train()  # set the model to training mode
    total_loss = 0.0
    total_batches = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()  # zero the parameter gradients
        outputs = model(images)  # forward pass
        loss = criterion(outputs, targets)  # compute loss
        loss.backward()  # back-propagation
        optimizer.step()  # update parameters

        total_loss += loss.item()  # accumulate loss
        total_batches += 1
    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    return avg_loss


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
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    return correct / total if total > 0 else 0


# -------------------------------------------------
# 6. Full training routine (entry point)
# -------------------------------------------------
def run_training(
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.001,
    checkpoint_path: str = "in_class_mnist.pt",
) -> None:
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
    device = get_device()
    model = build_model()
    model.to(device)
    train_loader, test_loader = get_data_loaders(batch_size)
    crit = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epoch + 1):
        average_loss = train_one_epoch(model, train_loader, crit, optimizer, device)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch: {epoch}/{epochs}")
        print(f"Loss: {average_loss}")
        print(f"Accuracy: {acc*100}")
    torch.save(model.state_dict(), checkpoint_path)
    final_acc = evaluate(model, test_loader, device)


# -------------------------------------------------
# 7. Entry point for manual execution
# -------------------------------------------------
if __name__ == "__main__":
    # Running the file directly will start training with the default hyper‑parameters.
    run_training()
