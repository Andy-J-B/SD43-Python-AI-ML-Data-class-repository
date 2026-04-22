#!/usr/bin/env python3
# binary_nn_predict_template.py
# --------------------------------------------------------------
# Skeleton for loading a trained NN (saved in model.npz) and
# making a single‑image prediction (cat vs. dog).
# --------------------------------------------------------------
# DO NOT CHANGE ANY FUNCTION NAMES or argument signatures.
# --------------------------------------------------------------

import numpy as np
from PIL import Image
import sys


# ------------------------------------------------------------------
# 1️⃣  LOAD THE SAVED MODEL
# ------------------------------------------------------------------
def load_model():
    """
    Load the NPZ file that contains the trained weights, biases
    and the hyper‑parameters (image size & hidden layer size).

    Returns
        parameters : dict  with keys "W1","b1","W2","b2"
        image_size : int   (the size the training images were resized to)
        hidden_size : int  (the size of the hidden layer that was used)
    """
    # --------------------------------------------------------------
    # STEP 1 – load the .npz file (named exactly "model.npz")
    # --------------------------------------------------------------
    data = np.load("model.npz")
    # --------------------------------------------------------------
    # STEP 2 – create a dict called ``parameters`` and copy the four
    #         arrays from the NPZ into it
    #         (W1, b1, W2, b2)
    # --------------------------------------------------------------
    parameters = {}

    # --------------------------------------------------------------
    # STEP 3 – read the stored hyper‑parameters and cast to int
    # --------------------------------------------------------------
    parameters["W1"] = data["W1"]
    parameters["b1"] = data["b1"]
    parameters["W2"] = data["W2"]
    parameters["b2"] = data["b2"]
    image_size = int(data["image_size"])
    hidden_size = int(data["hidden_size"])
    return parameters, image_size, hidden_size

    # --------------------------------------------------------------
    # STEP 4 – return the dict and the two ints
    # --------------------------------------------------------------


# ------------------------------------------------------------------
# 2️⃣  RELU ACTIVATION (same as in the training script)
# ------------------------------------------------------------------
def relu(Z):
    """
    ReLU (Rectified Linear Unit) activation.
    Return max(0, Z) element‑wise.
    """
    # TODO: implement ``np.maximum(0, Z)`` and return the result
    return np.maximum(0, 2)


# ------------------------------------------------------------------
# 3️⃣  SIGMOID ACTIVATION (same as in the training script)
# ------------------------------------------------------------------
def sigmoid(Z):
    """
    Sigmoid activation.
    Return 1 / (1 + exp(-Z)).  No need to clip here – the values
    coming from the network are already safe.
    """
    # TODO: compute the sigmoid and return it
    return 1 / (1 + np.exp(-Z))


# ------------------------------------------------------------------
# 4️⃣  FORWARD PROPAGATION (inference only – no cache needed)
# ------------------------------------------------------------------
def forward_propagation(X, parameters):
    """
    Perform a single forward pass through the two‑layer network.
    Return the output probability (shape (1,1) for a single sample).

    Args
    ----
    X : np.ndarray, shape (1, n_features)
        The pre‑processed image.
    parameters : dict
        Contains "W1","b1","W2","b2" as loaded by ``load_model``.
    """
    # --------------------------------------------------------------
    # STEP 1 – unpack the weight matrices and bias vectors
    # --------------------------------------------------------------
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    # --------------------------------------------------------------
    # STEP 2 – compute the hidden layer (Z1 then ReLU)
    # --------------------------------------------------------------
    Z1 = np.dot(X, W1) + b1
    # --------------------------------------------------------------
    # STEP 3 – compute the output layer (Z2 then sigmoid)
    # --------------------------------------------------------------
    a1 = relu(Z1)
    Z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(Z2)
    # --------------------------------------------------------------
    # STEP 4 – return A2 (the probability that the image is a dog)
    # --------------------------------------------------------------
    return a2


def preprocess_image(image_path, image_size):

    image = Image.open(image_path)
    image = image.resize((image_size, image_size))

    image_array = np.array(image)
    image_array = image_array / 255.0

    image_flat = image_array.flatten()
    image_flat = image_flat.reshape(1, -1)

    return image_flat


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python binary_nn_predict.py image.jpg")
        sys.exit()

    image_path = sys.argv[1]

    parameters, image_size, hidden_size = load_model()

    X = preprocess_image(image_path, image_size)

    probability = forward_propagation(X, parameters)

    if probability[0][0] >= 0.5:
        print("Prediction: DOG")
        print("Confidence:", probability[0][0])
    else:
        print("Prediction: CAT")
        print("Confidence:", 1 - probability[0][0])
