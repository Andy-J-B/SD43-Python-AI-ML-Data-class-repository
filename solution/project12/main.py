import dataset
import decision_tree

# Mapping target IDs to real category labels for presentation clarity
CLASS_MAPPING = {
    "1": "Mammal", "2": "Bird", "3": "Reptile", "4": "Fish", 
    "5": "Amphibian", "6": "Bug", "7": "Invertebrate"
}

def print_tree(node, spacing="", feature_names=None):
    if isinstance(node, decision_tree.Leaf):
        readable_preds = {CLASS_MAPPING.get(k, k): v for k, v in node.predictions.items()}
        print(spacing + "Predict -->", readable_preds)
        return

    feature_label = feature_names[node.question.column] if feature_names else node.question.column
    print(spacing + f"Is {feature_label} == {node.question.value}?")
    
    print(spacing + '  [Yes]:')
    print_tree(node.true_branch, spacing + "    ", feature_names)

    print(spacing + '  [No]:')
    print_tree(node.false_branch, spacing + "    ", feature_names)

def classify(row, node):
    if isinstance(node, decision_tree.Leaf):
        return node.predictions

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def main():
    print("Fetching live UCI Zoo Dataset from the web...")
    feature_headers, training_data = dataset.load_online_data()
    
    print("Growing the Wisdom Tree via Information Gain...")
    tree = decision_tree.build_tree(training_data)
    
    print("\n================ THE WISDOM TREE VISUALIZATION ================\n")
    print_tree(tree, feature_names=feature_headers)
    print("\n================================================================\n")
    
    # Mystery animal properties checklist matching features:
    # hair=0, feathers=1, eggs=1, milk=0, airborne=0, aquatic=1, predator=1, toothed=0 ...
    # This represents a specialized marine bird like a Penguin!
    mystery_animal = ["0", "1", "1", "0", "0", "1", "1", "0", "1", "1", "0", "0", "0", "1", "0", "0", "1"]
    
    print(f"Classifying a mystery test animal with features: {mystery_animal}")
    raw_prediction = classify(mystery_animal, tree)
    
    named_prediction = [CLASS_MAPPING.get(k, k) for k in raw_prediction.keys()]
    print(f"AI Classification Verdict: {named_prediction}\n")

if __name__ == '__main__':
    main()