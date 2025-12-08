"""
main_template.py

Utility script that loads a checkpoint created by
mnist_classifier_template.py and lets the user classify an
arbitrary image file from the command line.

How to use after you finish the TODOs:

    python main_template.py path/to/your_image.png

The script will print the predicted digit (0‑9) or an error
message if something is missing.
"""

# ------------------------------------------------------------------
# 0️⃣  Imports – you do not need to modify these
# ------------------------------------------------------------------
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T

# ------------------------------------------------------------------
# 1️⃣  Import the *exact* model‑building function and the device helper
# ------------------------------------------------------------------
# The training file you completed earlier should be named
# `mnist_classifier_template.py`.  Import the two public helpers from it.
# (If you renamed the file, change the import line accordingly.)
from mnist_classifier_template import build_model, get_device


# ------------------------------------------------------------------
# 2️⃣  Helper – preprocess a raw image exactly like the MNIST data
# ------------------------------------------------------------------
def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Load an image from *image_path*, convert it to a 28×28 greyscale
    tensor, normalise it with the same mean/std used for training,
    and add a batch dimension.

    Steps you must implement (in the order shown):
        1. Open the file with Pillow (`Image.open`) and force greyscale
           conversion (`convert("L")`).
        2. Resize the image to 28 × 28 pixels (use `Image.BILINEAR` as the
           resampling filter).
        3. Build a torchvision `Compose` transform that:
           – turns the image into a tensor (`ToTensor`), which rescales the
             pixel values to the range [0, 1] and adds a channel dimension,
           – normalises the tensor with the MNIST statistics
             `mean=0.1307`, `std=0.3081`.
        4. Apply the transform to the Pillow image.
        5. `unsqueeze(0)` the result so the final shape is
           `(1, 1, 28, 28)` (batch‑size = 1, channel = 1).
        6. Return the resulting tensor.

    The function should raise an exception only if the file cannot be
    opened; otherwise it returns the tensor.
    """
    # ---------- TODO: implement the preprocessing steps ----------
    raise NotImplementedError
    # -------------------------------------------------------------


# ------------------------------------------------------------------
# 3️⃣  Core inference routine – returns the predicted digit
# ------------------------------------------------------------------
def predict_image(image_path: str, model: nn.Module, device: torch.device) -> int:
    """
    Given a file path, a ready‑to‑use model and the device it runs on,
    return the digit (0‑9) that the model predicts.

    Required steps:
        1. Call `preprocess_image(image_path)` to obtain a tensor.
        2. Move that tensor to the same device as the model
           (`tensor.to(device)`).
        3. Put the model into evaluation mode (`model.eval()`).
        4. Run a forward pass inside a `torch.no_grad()` block
           (no gradients are needed for inference).
        5. The output of the model has shape `(1, 10)`.  The index of the
           largest logit is the predicted class – obtain it with
           `logits.argmax(dim=1).item()`.
        6. Return that integer.

    Do **not** modify the model’s parameters.
    """
    # ---------- TODO: implement the inference pipeline -------------
    raise NotImplementedError
    # -------------------------------------------------------------


# ------------------------------------------------------------------
# 4️⃣  Command‑line interface (CLI) – glue everything together
# ------------------------------------------------------------------
def main() -> None:
    """
    Parse command‑line arguments, load the checkpoint, build the model,
    and call `predict_image`.  Finally print the result.

    You must:
        * Define an `ArgumentParser` with a positional argument
          `image_path` (the path to the picture you want to classify) and
          an optional `--ckpt` argument (default: `mnist_cnn.pt`) that
          points to the checkpoint file produced by the training script.
        * Verify that `image_path` points to an existing file – if not,
          exit with an error message.
        * Obtain the device with `get_device()`.
        * Build the model by calling `build_model()` and move it to the
          device.
        * Load the checkpoint (`torch.load`) with `map_location=device`
          and restore the state dict onto the model.  Handle two possible
          errors:
            – `FileNotFoundError` → explain that the checkpoint is
              missing and that the user should run the training script
              first.
            – any other exception → print a generic load‑failure message.
        * Call `predict_image` with the parsed `image_path`, the model, and
          the device.
        * Print the predicted digit in a friendly format, e.g.
          “🔢  Predicted digit: 7”.

    The script should **exit cleanly** (using `sys.exit` only for error
    conditions) and otherwise finish with a single line showing the
    prediction.
    """
    # ---------- TODO: implement the CLI logic --------------------
    raise NotImplementedError
    # -------------------------------------------------------------


# ------------------------------------------------------------------
# 5️⃣  Entry‑point guard – runs only when the file is executed
# ------------------------------------------------------------------
if __name__ == "__main__":
    # When the student finishes the TODOs, this will start the program.
    main()
