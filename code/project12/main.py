import dataset
import decision_tree

# Define a dictionary mapping the dataset's class numerical IDs (1-7) to actual string labels.
CLASS_MAPPING = {
    "1":"mammal",
    "2":"bird",
    "3":"reptile",
    "4":"fish",
    "5":"amphibian",
    "6":"bug",
    "7":"invertebrate"
}

def print_tree(node, spacing="", feature_names=None):
    # Check if the current node is a leaf structure.
    # If it is, look up the animal group names for the predictions and display them with spacing.
    # Retrieve the readable feature question name using the column index and feature_names list.
    # Print the current question branch format.
    # Recursively traverse down the true branch with increased indentation.
    # Recursively traverse down the false branch with increased indentation.
    pass

def classify(row, node):
    # If the node is an instance of a Leaf, return its stored predictions dictionary.
    # Evaluate the current row against the decision node's question match logic.
    # If the feature matches, recursively slide down the true branch.
    # If it fails, recursively slide down the false branch.
    if isinstance(node, decision_tree.Leaf):
        return node.predictions
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def main():
    # Call the dataset module function to pull the fresh data and headers from the web.
    # Build the recursive decision tree using the online dataset matrix.
    # Print a welcome banner and output the entire generated tree blueprint to the terminal.
    # Construct a custom mockup feature list representing a mystery test animal.
    # Run the test animal through the classification function and output the mapped result.
    print("Fetching live zoo data set from the web")
    feature_header, training_data = dataset.load_online_data()
    print("Growing the tree")
    tree = decision_tree.build_tree(training_data)
    print_tree(tree, feature_names=feature_header)
    mystery_animal=["0", "1", "1", "0", "0", "1", "1", "0", "1", "1", "0", "0", "0", "1", "0", "0", "1"]
    raw_prediction=classify(mystery_animal, tree)
    prediction=[CLASS_MAPPING.get(key,key)for key in raw_prediction.keys()]
    print(f"ai_classification: {raw_prediction}")

    


if __name__ == '__main__':
    main()