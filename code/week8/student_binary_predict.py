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
    # data = np.load("model.npz")

    # --------------------------------------------------------------
    # STEP 2 – create a dict called ``parameters`` and copy the four
    #         arrays from the NPZ into it
    #         (W1, b1, W2, b2)
    # --------------------------------------------------------------
    # parameters = {}
    # parameters["W1"] = data["W1"]
    # parameters["b1"] = data["b1"]
    # parameters["W2"] = data["W2"]
    # parameters["b2"] = data["b2"]

    # --------------------------------------------------------------
    # STEP 3 – read the stored hyper‑parameters and cast to int
    # --------------------------------------------------------------
    # image_size = int(data["image_size"])
    # hidden_size = int(data["hidden_size"])

    # --------------------------------------------------------------
    # STEP 4 – return the dict and the two ints
    # --------------------------------------------------------------
    # return parameters, image_size, hidden_size

    pass


# ------------------------------------------------------------------
# 2️⃣  RELU ACTIVATION (same as in the training script)
# ------------------------------------------------------------------
def relu(Z):
    """
    ReLU (Rectified Linear Unit) activation.
    Return max(0, Z) element‑wise.
    """
    # TODO: implement ``np.maximum(0, Z)`` and return the result
    pass


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
    pass


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
    # W1 = parameters["W1"]
    # b1 = parameters["b1"]
    # W2 = parameters["W2"]
    # b2 = parameters["b2"]

    # --------------------------------------------------------------
    # STEP 2 – compute the hidden layer (Z1 then ReLU)
    # --------------------------------------------------------------
    # Z1 = np.dot(X, W1) + b1
    # A1 = relu(Z1)

    # --------------------------------------------------------------
    # STEP 3 – compute the output layer (Z2 then sigmoid)
    # --------------------------------------------------------------
    # Z2 = np.dot(A1, W2) + b2
    # A2 = sigmoid(Z2)

    # --------------------------------------------------------------
    # STEP 4 – return A2 (the probability that the image is a dog)
    # --------------------------------------------------------------
    # return A2

    pass


# ------------------------------------------------------------------
# 5️⃣  PRE‑PROCESS ONE IMAGE (must match the preprocessing used in training)
# ------------------------------------------------------------------
def preprocess_image(image_path, image_size):
    """
    Load an image from disk, resize it, normalise pixel values,
    flatten it, and reshape to a (1, n_features) row vector.

    Args
    ----
    image_path : str
        Path to the JPEG/PNG file supplied on the command line.
    image_size : int
        The size the model expects (e.g. 32 → resize to 32×32).

    Returns
    -------
    X : np.ndarray, shape (1, n_features)
        Ready to be fed into ``forward_propagation``.
    """
    # --------------------------------------------------------------
    # STEP 1 – open the image with Pillow and resize to (image_size, image_size)
    # --------------------------------------------------------------
    # image = Image.open(image_path)
    # image = image.resize((image_size, image_size))

    # --------------------------------------------------------------
    # STEP 2 – turn the image into a NumPy array of floats and normalise
    # --------------------------------------------------------------
    # image_array = np.array(image, dtype=np.float32) / 255.0

    # --------------------------------------------------------------
    # STEP 3 – flatten the 3‑D (h, w, 3) array into a 1‑D vector
    # --------------------------------------------------------------
    # image_flat = image_array.flatten()

    # --------------------------------------------------------------
    # STEP 4 – reshape into a row vector so its shape is (1, n_features)
    # --------------------------------------------------------------
    # X = image_flat.reshape(1, -1)

    # --------------------------------------------------------------
    # STEP 5 – return X
    # --------------------------------------------------------------
    # return X

    pass


# ------------------------------------------------------------------
# 🏁  MAIN – command‑line interface
# ------------------------------------------------------------------
if __name__ == "__main__":
    # --------------------------------------------------------------
    # STEP 1 – make sure the user supplied exactly one argument
    # --------------------------------------------------------------
    # if len(sys.argv) != 2:
    #     print("Usage: python binary_nn_predict.py image.jpg")
    #     sys.exit(1)

    # --------------------------------------------------------------
    # STEP 2 – grab the path to the image file from the command line
    # --------------------------------------------------------------
    # image_path = sys.argv[1]

    # --------------------------------------------------------------
    # STEP 3 – load the saved model (weights + hyper‑params)
    # --------------------------------------------------------------
    # parameters, image_size, hidden_size = load_model()

    # --------------------------------------------------------------
    # STEP 4 – preprocess the supplied image so it matches the training data
    # --------------------------------------------------------------
    # X = preprocess_image(image_path, image_size)

    # --------------------------------------------------------------
    # STEP 5 – run a forward pass to obtain the probability of “dog”
    # --------------------------------------------------------------
    # probability = forward_propagation(X, parameters)   # shape (1,1)

    # --------------------------------------------------------------
    # STEP 6 – turn the probability into a class label and print it
    # --------------------------------------------------------------
    # if probability[0][0] >= 0.5:
    #     print("Prediction: DOG")
    #     print("Confidence:", probability[0][0])
    # else:
    #     print("Prediction: CAT")
    #     # confidence for cat is 1 - prob(dog)
    #     print("Confidence:", 1 - probability[0][0])

    # --------------------------------------------------------------
    # END OF SCRIPT – replace all the ``pass`` statements (and delete
    # the comment blocks) with the real code shown above.
    # --------------------------------------------------------------
    pass
