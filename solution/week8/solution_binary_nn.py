#!/usr/bin/env python3
# cat_dog_classifier_complete.py
# --------------------------------------------------------------
# A minimal neural‑network for binary image classification (cats vs. dogs)
# Built from scratch with NumPy – no deep‑learning libraries required.
# --------------------------------------------------------------

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------------
# 1. DATA LOADING -------------------------------------------------
# ------------------------------------------------------------------
def load_images(image_size: int):
    """
    Load, resize, normalise and flatten all images in the folders
    ``cats`` and ``dogs``.  Returns a classic train/test split.

    Parameters
    ----------
    image_size : int
        Width and height the images will be resized to (square).

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Training and testing feature matrices and label vectors.
        Labels: 0 = cat, 1 = dog.
    """
    X, y = [], []

    # --- cats -----------------------------------------------------
    for filename in os.listdir("cats"):
        path = os.path.join("cats", filename)
        img = Image.open(path).convert("RGB")  # ensure 3 channels
        img = img.resize((image_size, image_size))
        arr = np.array(img, dtype=np.float32) / 255.0  # normalise
        X.append(arr.flatten())
        y.append(0)  # cat = 0

    # --- dogs -----------------------------------------------------
    for filename in os.listdir("dogs"):
        path = os.path.join("dogs", filename)
        img = Image.open(path).convert("RGB")
        img = img.resize((image_size, image_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        X.append(arr.flatten())
        y.append(1)  # dog = 1

    X = np.array(X)  # shape: (n_samples, image_size*image_size*3)
    y = np.array(y)  # shape: (n_samples,)

    # Classic 80/20 train‑test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------------
# 2. INITIALISE WEIGHTS ---------------------------------------------
# ------------------------------------------------------------------
def initialize_parameters(input_size: int, hidden_size: int):
    """
    Initialise a two‑layer network (input → hidden → output)
    with small random values for the weights and zeros for the biases.

    Returns
    -------
    parameters : dict
        Dictionary with keys "W1", "b1", "W2", "b2".
    """
    rng = np.random.default_rng(seed=42)  # reproducible initialisation

    W1 = rng.normal(0, 0.01, size=(input_size, hidden_size))
    b1 = np.zeros((1, hidden_size))

    W2 = rng.normal(0, 0.01, size=(hidden_size, 1))
    b2 = np.zeros((1, 1))

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


# ------------------------------------------------------------------
# 3. ACTIVATIONS ---------------------------------------------------
# ------------------------------------------------------------------
def relu(Z):
    """ReLU activation: max(0, Z) applied element‑wise."""
    return np.maximum(0, Z)


def sigmoid(Z):
    """Sigmoid activation, safe for large negative/positive inputs."""
    # Clip to avoid overflow in np.exp
    Z = np.clip(Z, -500, 500)
    return 1.0 / (1.0 + np.exp(-Z))


# ------------------------------------------------------------------
# 4. FORWARD PASS --------------------------------------------------
# ------------------------------------------------------------------
def forward_propagation(X, parameters):
    """
    Compute a forward pass through the network.

    Returns
    -------
    A2 : np.ndarray
        Output of the network (probabilities, shape (m, 1)).
    cache : dict
        Intermediate values needed for back‑propagation.
    """
    W1, b1, W2, b2 = (
        parameters["W1"],
        parameters["b1"],
        parameters["W2"],
        parameters["b2"],
    )

    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


# ------------------------------------------------------------------
# 5. LOSS ---------------------------------------------------------
# ------------------------------------------------------------------
def compute_loss(y_true, y_pred):
    """
    Binary cross‑entropy (log‑loss).

    Parameters
    ----------
    y_true : np.ndarray, shape (m,)
        Ground‑truth labels (0 or 1).
    y_pred : np.ndarray, shape (m, 1)
        Predicted probabilities.

    Returns
    -------
    loss : float
    """
    m = y_true.shape[0]
    y_true = y_true.reshape(-1, 1)  # column vector

    # Add epsilon to avoid log(0)
    eps = 1e-9
    loss = -np.mean(
        y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)
    )
    return loss


# ------------------------------------------------------------------
# 6. BACKWARD PASS -------------------------------------------------
# ------------------------------------------------------------------
def backward_propagation(X, y_true, parameters, cache):
    """
    Compute gradients of the loss w.r.t. all parameters.

    Returns
    -------
    grads : dict
        Keys: "dW1", "db1", "dW2", "db2".
    """
    m = X.shape[0]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    y_true = y_true.reshape(-1, 1)

    # Output layer
    dZ2 = A2 - y_true  # (m, 1)
    dW2 = (A1.T @ dZ2) / m  # (hidden, 1)
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m  # (1, 1)

    # Hidden layer
    dA1 = dZ2 @ W2.T  # (m, hidden)
    dZ1 = dA1 * (cache["Z1"] > 0)  # ReLU derivative
    dW1 = (X.T @ dZ1) / m  # (input, hidden)
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m  # (1, hidden)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


# ------------------------------------------------------------------
# 7. PARAMETER UPDATE ----------------------------------------------
# ------------------------------------------------------------------
def update_parameters(parameters, grads, learning_rate):
    """
    Gradient‑descent step.
    """
    parameters["W1"] -= learning_rate * grads["dW1"]
    parameters["b1"] -= learning_rate * grads["db1"]
    parameters["W2"] -= learning_rate * grads["dW2"]
    parameters["b2"] -= learning_rate * grads["db2"]
    return parameters


# ------------------------------------------------------------------
# 8. TRAINING LOOP -------------------------------------------------
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
    Train the two‑layer NN on the training set.
    Prints loss every 10 epochs (if verbose=True).

    Returns
    -------
    parameters : dict
        Trained weights and biases.
    """
    parameters = initialize_parameters(input_size, hidden_size)

    for epoch in range(epochs):
        # forward
        y_pred, cache = forward_propagation(X_train, parameters)

        # loss
        loss = compute_loss(y_train, y_pred)

        # backward
        grads = backward_propagation(X_train, y_train, parameters, cache)

        # update
        parameters = update_parameters(parameters, grads, learning_rate)

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch:3d} / {epochs-1:3d}  -  loss: {loss:.5f}")

    return parameters


# ------------------------------------------------------------------
# 9. PREDICTION ---------------------------------------------------
# ------------------------------------------------------------------
def predict(X, parameters, threshold=0.5):
    """
    Run a forward pass and convert probabilities to class labels
    (0 = cat, 1 = dog) using the given threshold.

    Returns
    -------
    preds : np.ndarray, shape (m,)
        Integer predictions (0 or 1).
    """
    probs, _ = forward_propagation(X, parameters)
    preds = (probs >= threshold).astype(int).flatten()
    return preds


# ------------------------------------------------------------------
# 10. SAVE MODEL ---------------------------------------------------
# ------------------------------------------------------------------
def save_model(parameters, image_size, hidden_size, filename="model.npz"):
    """
    Store all learned weights, biases and hyper‑parameters in an NPZ file.
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
# 11. MAIN ---------------------------------------------------------
# ------------------------------------------------------------------
if __name__ == "__main__":
    # --------------------------------------------------------------
    # STEP 1 – hyper‑parameters
    # --------------------------------------------------------------
    IMAGE_SIZE = 32  # 32×32 pixels (feel free to change)
    HIDDEN_SIZE = 64  # number of hidden units
    EPOCHS = 100
    LR = 0.1  # learning rate

    # --------------------------------------------------------------
    # STEP 2 – load and split the data
    # --------------------------------------------------------------
    X_train, X_test, y_train, y_test = load_images(IMAGE_SIZE)

    # --------------------------------------------------------------
    # STEP 3 – train
    # --------------------------------------------------------------
    INPUT_SIZE = X_train.shape[1]  # = IMAGE_SIZE * IMAGE_SIZE * 3
    trained_params = train_model(
        X_train,
        y_train,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        epochs=EPOCHS,
        learning_rate=LR,
        verbose=True,
    )

    # --------------------------------------------------------------
    # STEP 4 – evaluate on the hold‑out set
    # --------------------------------------------------------------
    y_pred = predict(X_test, trained_params)
    accuracy = np.mean(y_pred == y_test)
    print(f"\nTest accuracy: {accuracy:.2%}")

    # --------------------------------------------------------------
    # STEP 5 – persist the model for later use
    # --------------------------------------------------------------
    save_model(trained_params, IMAGE_SIZE, HIDDEN_SIZE)
