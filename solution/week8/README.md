## Setup

1. Install required packages:

```bash
python -m venv venv
source .venv/bin/activate
pip install kaggle Pillow scikit-learn
```

2. Place your Kaggle API key:

```bash
mkdir ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## Prepare Dataset

Run:

```bash
python auto_prepare_cats_dogs.py
```

This will:

- Download dataset
- Extract it
- Create `cats/` and `dogs/` folders
- Limit number of images
- Clean up extra files

---

## Train the Model

Run:

```bash
python solution_binary_nn.py
```

You should see training output like:

```
Epoch: 0 Loss: ...
Epoch: 10 Loss: ...
...
Final Accuracy: ...
Model saved to model.npz
```

This will create:

```
model.npz
```

---

## Test the Model on a New Image

Run:

```bash
python solution_binary_predict.py path_to_image.jpg
```

Example:

```bash
python solution_binary_predict.py dogs/example.jpg
```

Output:

```
Prediction: DOG
Confidence: 0.99
```
