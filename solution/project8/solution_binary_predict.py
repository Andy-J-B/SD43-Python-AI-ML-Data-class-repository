import numpy as np
from PIL import Image
import sys


def load_model():

    data = np.load("model.npz")

    parameters = {}
    parameters["W1"] = data["W1"]
    parameters["b1"] = data["b1"]
    parameters["W2"] = data["W2"]
    parameters["b2"] = data["b2"]

    image_size = int(data["image_size"])
    hidden_size = int(data["hidden_size"])

    return parameters, image_size, hidden_size


def relu(Z):

    result = np.maximum(0, Z)
    return result


def sigmoid(Z):

    result = 1 / (1 + np.exp(-Z))
    return result


def forward_propagation(X, parameters):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    return A2


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
