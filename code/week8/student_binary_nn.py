#!/usr/bin/env python3
# cat_dog_classifier_template.py
# --------------------------------------------------------------
# Student skeleton for a tiny cat‑vs‑dog image classifier.
# --------------------------------------------------------------
# KEEP ALL FUNCTION NAMES EXACTLY as they appear – the auto‑grader
# will import this file and call the functions directly.
# --------------------------------------------------------------

# --------------------------------------------------------------
# IMPORTS (DO NOT MODIFY)
# --------------------------------------------------------------
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------------
# 1️⃣  DATA LOADING
# ------------------------------------------------------------------
def load_images(image_size: int):
    """
    Load the images, resize them, normalise the pixel values,
    flatten each picture to a 1‑D vector and create the label list.
    Finally split the data into train / test sets.

    INPUT
        image_size – the target width/height (the pictures are square).

    OUTPUT
        X_train, X_test, y_train, y_test – NumPy arrays.
        Labels: 0 for cat, 1 for dog.
    """
    # --------------------------------------------------------------
    # STEP 1 – create empty containers X (features) and y (labels)
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 2 – PROCESS THE CATS FOLDER
    # --------------------------------------------------------------
    # for every file in the folder "cats":
    #   * build the full file path
    #   * open the image, force 3 colour channels (RGB)
    #   * resize to (image_size, image_size)
    #   * convert to a NumPy array of type float32
    #   * divide by 255.0 to bring values into [0,1]
    #   * flatten the 3‑D array to 1‑D and append to X
    #   * append label 0 to y (cat)

    # --------------------------------------------------------------
    # STEP 3 – PROCESS THE DOGS FOLDER (same as cats, but label = 1)
    # --------------------------------------------------------------
    # for every file in the folder "dogs":  (repeat the steps above)

    # --------------------------------------------------------------
    # STEP 4 – TURN LISTS INTO NUMPY ARRAYS
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 5 – SPLIT INTO TRAIN / TEST (80 % train, 20 % test)
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # RETURN THE FOUR ARRAYS
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # END OF FUNCTION – the above lines are all comments.
    # Replace them with the real code and *remove* the "pass".
    # --------------------------------------------------------------
    pass


# ------------------------------------------------------------------
# 2️⃣  INITIALISE WEIGHTS
# ------------------------------------------------------------------
def initialize_parameters(input_size: int, hidden_size: int):
    """
    Create the weight matrices and bias vectors for a two‑layer NN:
        * W1 : (input_size, hidden_size) – small random numbers
        * b1 : (1, hidden_size)          – zeros
        * W2 : (hidden_size, 1)          – small random numbers
        * b2 : (1, 1)                    – zero

    Return a dictionary with those four entries.
    """
    # --------------------------------------------------------------
    # STEP 1 – set a reproducible random generator (seed optional)
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 2 – initialise W1 with Gaussian(mean=0, std=0.01)
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 3 – initialise b1 as zeros
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 4 – initialise W2 (Gaussian) and b2 (zeros)
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 5 – pack everything into a dict and return it
    # --------------------------------------------------------------

    pass


# ------------------------------------------------------------------
# 3️⃣  ACTIVATIONS
# ------------------------------------------------------------------
def relu(Z):
    """
    ReLU (Rectified Linear Unit) activation.
    Return max(0, Z) element‑wise.
    """
    # --------------------------------------------------------------
    # TODO: implement np.maximum(0, Z) and return the result
    # --------------------------------------------------------------
    pass


def sigmoid(Z):
    """
    Sigmoid activation.
    Return 1 / (1 + exp(-Z)).
    Clip Z to a safe range (e.g. -500…500) before exponentiation.
    """
    # --------------------------------------------------------------
    # TODO: clip Z, compute sigmoid, and return it
    # --------------------------------------------------------------
    pass


# ------------------------------------------------------------------
# 4️⃣  FORWARD PROPAGATION
# ------------------------------------------------------------------
def forward_propagation(X, parameters):
    """
    Perform one forward pass through the network:
        Z1 = X·W1 + b1
        A1 = ReLU(Z1)
        Z2 = A1·W2 + b2
        A2 = sigmoid(Z2)

    Return:
        A2 – network output (probabilities)
        cache – dict with Z1, A1, Z2, A2 (needed for back‑prop)
    """
    # --------------------------------------------------------------
    # STEP 1 – unpack weights/biases from the parameters dict
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 2 – compute Z1 = X @ W1 + b1
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 3 – apply ReLU → A1
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 4 – compute Z2 = A1 @ W2 + b2
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 5 – apply sigmoid → A2
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 6 – store intermediate values in cache and return them
    # --------------------------------------------------------------

    pass


# ------------------------------------------------------------------
# 5️⃣  LOSS (binary cross‑entropy)
# ------------------------------------------------------------------
def compute_loss(y_true, y_pred):
    """
    Binary cross‑entropy (log‑loss).

    y_true: shape (m,) with values 0 or 1
    y_pred: shape (m,1) with probability predictions

    Return a scalar loss.
    """
    # --------------------------------------------------------------
    # STEP 1 – reshape y_true to a column vector (m,1)
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 2 – compute the loss formula
    #   loss = -(1/m) * Σ[ y*log(p) + (1‑y)*log(1‑p) ]
    #   add a tiny epsilon (e.g. 1e‑9) inside the logs for safety
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 3 – return the scalar loss
    # --------------------------------------------------------------

    pass


# ------------------------------------------------------------------
# 6️⃣  BACKWARD PROPAGATION
# ------------------------------------------------------------------
def backward_propagation(X, y_true, parameters, cache):
    """
    Compute gradients of the loss w.r.t. every parameter.

    Returns a dict with keys:
        dW1, db1, dW2, db2
    """
    # --------------------------------------------------------------
    # STEP 0 – reshape y_true to column vector (m,1)
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 1 – unpack needed values
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 2 – compute output‑layer gradients
    #   dZ2 = A2 - y_true
    #   dW2 = (A1.T @ dZ2) / m
    #   db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 3 – back‑propagate through hidden layer (ReLU)
    #   dA1 = dZ2 @ W2.T
    #   dZ1 = dA1 * (Z1 > 0)          # derivative of ReLU
    #   dW1 = (X.T @ dZ1) / m
    #   db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    # --------------------------------------------------------------
    # ...

    # --------------------------------------------------------------
    # STEP 4 – pack everything into a dict and return
    # --------------------------------------------------------------
    # grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    # return grads

    pass


# ------------------------------------------------------------------
# 7️⃣  UPDATE PARAMETERS (gradient descent)
# ------------------------------------------------------------------
def update_parameters(parameters, grads, learning_rate):
    """
    Apply a single gradient‑descent step:
        param = param - lr * grad
    """
    # --------------------------------------------------------------
    # TODO: update each of W1, b1, W2, b2 using the corresponding gradient
    # --------------------------------------------------------------
    # parameters["W1"] -= learning_rate * grads["dW1"]
    # parameters["b1"] -= learning_rate * grads["db1"]
    # parameters["W2"] -= learning_rate * grads["dW2"]
    # parameters["b2"] -= learning_rate * grads["dbb2"]
    # return parameters

    pass


# ------------------------------------------------------------------
# 8️⃣  TRAINING LOOP
# ------------------------------------------------------------------
def train_model(
    X_train,
    y_train,
    input_size,
    hidden_size,
    epochs=100,
    learning_rate=0.1,
    verbose=True,
):
    """
    Train the neural‑network for a given number of epochs.
    Print the loss every 10 epochs (if verbose=True).
    Return the final parameters dictionary.
    """
    # --------------------------------------------------------------
    # STEP 1 – initialise the parameters dict
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 2 – start the epoch loop
    # --------------------------------------------------------------
    # for epoch in range(epochs):
    #     # ---- forward pass -------------------------------------------------

    #     # ---- compute loss -------------------------------------------------

    #     # ---- backward pass ------------------------------------------------

    #     # ---- update parameters --------------------------------------------

    #     # ---- optional printing ---------------------------------------------

    # --------------------------------------------------------------
    # STEP 3 – after the loop, return the learned parameters
    # --------------------------------------------------------------

    pass


# ------------------------------------------------------------------
# 9️⃣  PREDICTION
# ------------------------------------------------------------------
def predict(X, parameters, threshold=0.5):
    """
    Run a forward pass on X, then turn probabilities into class labels:
        prob >= threshold  → 1 (dog)
        prob <  threshold  → 0 (cat)

    Return a 1‑D NumPy array of 0/1 predictions.
    """
    # --------------------------------------------------------------
    # STEP 1 – obtain probabilities from forward_propagation
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 2 – threshold the probabilities
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 3 – return the prediction vector
    # --------------------------------------------------------------

    pass


# ------------------------------------------------------------------
# 🔟  SAVE MODEL  (already fully implemented – do NOT edit)
# ------------------------------------------------------------------
def save_model(parameters, image_size, hidden_size, filename="model.npz"):
    """
    Store the trained weights, biases and the hyper‑parameters in one .npz file.
    """
    np.savez(
        filename,
        W1=parameters["W1"],
        b1=parameters["b1"],
        W2=parameters["W2"],
        b2=parameters["b2"],
        image_size=image_size,
        hidden_size=hidden_size,
    )
    print(f"Model saved to {filename}")


# ------------------------------------------------------------------
# 🏁  MAIN – orchestrate the whole pipeline
# ------------------------------------------------------------------
if __name__ == "__main__":
    # --------------------------------------------------------------
    # STEP 1 – define the image size you want to work with
    #           (example: 32 → 32×32 pixels)
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 2 – load the data and obtain the train / test split
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 3 – determine the size of the input layer (number of features)
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 4 – pick a size for the hidden layer (example: 64 neurons)
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 5 – train the model

    # --------------------------------------------------------------
    # parameters = train_model(
    #     X_train,
    #     y_train,
    #     input_size=INPUT_SIZE,
    #     hidden_size=HIDDEN_SIZE,
    #     epochs=100,
    #     learning_rate=0.1,
    #     verbose=True,
    # )

    # --------------------------------------------------------------
    # STEP 6 – obtain predictions on the held‑out test set
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 7 – compute accuracy *manually* (no sklearn.metrics)
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 8 – print the final accuracy
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 9 – save the trained model for later reuse
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # All of the above lines are comments.  Replace them with real
    # code, **remove** the final `pass` and run the script to see
    # the network learn to distinguish cats from dogs!
    # --------------------------------------------------------------
    pass
