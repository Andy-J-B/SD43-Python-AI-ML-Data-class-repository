"""
main.py

Utility script that loads a previously trained MNIST model
(`mnist_cnn.pt` produced by mnist_classifier.py) and lets the
user classify an arbitrary image file.

Usage (from a terminal):
    python main.py path/to/image.png
"""

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T

# ------------------------------------------------------------------
# Import the *exact* model definition from the training file
# ------------------------------------------------------------------
# The file mnist_classifier.py must be in the same folder or on PYTHONPATH.
from mnist_classifier_final import build_model, get_device


# ------------------------------------------------------------------
# 1️⃣  Helper – preprocess a raw image exactly like the MNIST data
# ------------------------------------------------------------------
def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Load an image, convert to 28×28 greyscale, turn into a normalised tensor.
    Returns a tensor of shape (1, 1, 28, 28) ready to be fed to the model.
    """
    img = Image.open(image_path).convert("L")  # greyscale
    img = img.resize((28, 28), Image.BILINEAR)

    transform = T.Compose(
        [
            T.ToTensor(),  # → [0,1] float tensor, (1,28,28)
            T.Normalize((0.1307,), (0.3081,)),  # same as training
        ]
    )
    tensor = transform(img).unsqueeze(0)  # add batch dim → (1,1,28,28)
    return tensor


# ------------------------------------------------------------------
# 2️⃣  Core inference routine (the function you described)
# ------------------------------------------------------------------
def predict_image(
    image_path: str,
    model: nn.Module,
    device: torch.device,
) -> int:
    """
    Return the digit (0‑9) that the model predicts for *image_path*.
    """
    # 1. preprocess
    tensor = preprocess_image(image_path)

    # 2. move to device, forward pass
    model.eval()
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)  # shape (1,10)
        pred = logits.argmax(dim=1).item()

    return pred


# ------------------------------------------------------------------
# 3️⃣  CLI entry point
# ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a trained MNIST model and classify an image."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to a PNG/JPG/etc. image containing a handwritten digit.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="mnist_cnn.pt",
        help="Path to the model checkpoint produced by mnist_classifier.py",
    )
    args = parser.parse_args()

    # -------------------------------------------------
    # Verify the image exists
    # -------------------------------------------------
    img_path = Path(args.image_path)
    if not img_path.is_file():
        sys.exit(f"❌  Image not found: {img_path}")

    # -------------------------------------------------
    # Load the model architecture & weights
    # -------------------------------------------------
    device = get_device()
    model = build_model().to(device)

    try:
        state_dict = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✅  Loaded checkpoint '{args.ckpt}'")
    except FileNotFoundError:
        sys.exit(
            f"❌  Checkpoint not found: '{args.ckpt}'. "
            "Run `python mnist_classifier.py` first to create it."
        )
    except Exception as exc:
        sys.exit(f"❌  Failed to load checkpoint: {exc}")

    # -------------------------------------------------
    # Predict & show result
    # -------------------------------------------------
    digit = predict_image(str(img_path), model, device)
    print(f"🔢  Predicted digit: {digit}")


if __name__ == "__main__":
    main()
