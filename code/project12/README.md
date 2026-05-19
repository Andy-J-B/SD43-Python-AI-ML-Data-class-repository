# 🌳 Zoo Animal Decision Tree Classifier

A lightweight, from-scratch Python project that builds a Decision Tree to classify animals into biological categories based on their physical traits. This project automatically fetches the classic UCI Zoo Dataset, trains a decision tree using Information Gain, visualizes the resulting "Wisdom Tree," and classifies a mystery test animal.

## 📌 Project Overview

This tool demonstrates the core concepts of machine learning decision trees without relying on heavy external libraries like `scikit-learn` or `pandas`. By removing the animal names from the dataset before training, the model is forced to learn real biological rules (e.g., "Does it have feathers?" -> "It's a bird") rather than memorizing individual data points.

### ✨ Key Features

- **Live Data Fetching:** Directly pulls the latest clean version of the UCI Zoo Dataset from GitHub using `urllib`.
- **Anti-Cheat Cleaning:** Automatically strips the `animal_name` column from the dataset to ensure the AI learns from physical features alone.
- **Custom Decision Tree Implementation:** Uses a custom-built decision tree algorithm based on Information Gain.
- **Visual Tree Output:** Recursively prints a highly readable, nested structure of the trained "Wisdom Tree" to the console.
- **Human-Readable Classes:** Maps abstract target IDs (1-7) to real-world biological classes like _Mammal_, _Bird_, _Reptile_, and _Amphibian_.

---

## 📂 Project Structure

To run this project, ensure your workspace is structured with the following files:

- **`main.py`**: The main execution script. It orchestrates fetching the data, growing the tree, visualizing the results, and classifying the mystery test animal.
- **`dataset.py`**: Handles the data pipeline. It downloads the `zoo.data` CSV, parses the raw text, applies headers, and cleans the dataset by removing the initial name column.
- **`decision_tree.py`**: _(Required dependency)_ This file contains the logic for the tree itself, including the `Leaf` node classes, `build_tree` logic, and Information Gain calculations.

---

## 🚀 How to Run

1. **Prerequisites:** This project relies entirely on Python's standard library for data fetching and parsing. No `pip install` is required for the provided code.
2. **Ensure all files are present:** Make sure `main.py`, `dataset.py`, and your custom `decision_tree.py` are in the same directory.
3. **Execute the main script:**
   Open your terminal or command prompt and run:

```bash
python main.py

```

---

## 🧠 How it Works

1. **Fetching Data:** `dataset.py` reaches out to the raw GitHub URL and downloads the CSV text into memory.
2. **Preprocessing:** The dataset contains 18 columns. The first column (`animal_name`) is dropped, leaving 16 boolean/numeric feature columns (like `hair`, `feathers`, `aquatic`) and 1 target class column (`class_type`).
3. **Building the Tree:** `main.py` passes the cleaned data to `decision_tree.build_tree()`. The algorithm finds the best questions to ask (e.g., "Is feathers == 1?") by measuring which questions provide the most Information Gain.
4. **Classification:** A mystery array of features is fed into the tree. The script traverses the branches based on the array's boolean values until it hits a `Leaf` node, returning the predicted animal class.

### 🐧 The Mystery Animal

By default, the script tests a mystery animal with the following traits:

- No hair, produces no milk.
- Has feathers, lays eggs.
- Is aquatic, is a predator, but not airborne.
- **Result:** The tree will successfully classify this specialized marine bird (a Penguin) as a **Bird**.

```

```
