# imports

# --- Setup Helper (Run this once to create our dataset) ---
def create_mini_dataset():
    """Creates a small local CSV file so we can practice loading data."""
    if not os.path.exists("movie_reviews.csv"):
        data = {
            "review": [
                "Amazing movie!", "I loved every second of it", "Brilliant acting and great plot", 
                "A masterpiece of modern cinema", "Highly recommended", "So much fun!", 
                "Best movie of the year", "Incredible visuals", "Heartwarming story", "A triumphant success",
                "Terrible movie.", "Waste of time", "I hated it", "Boring and dull", 
                "Worst acting I have ever seen", "Do not watch this", "Awful plot", 
                "Completely unoriginal", "A massive disappointment", "I fell asleep"
            ],
            "sentiment": [
                "Positive", "Positive", "Positive", "Positive", "Positive", "Positive", "Positive", "Positive", "Positive", "Positive",
                "Negative", "Negative", "Negative", "Negative", "Negative", "Negative", "Negative", "Negative", "Negative", "Negative"
            ]
        }
        pd.DataFrame(data).to_csv("movie_reviews.csv", index=False)
        print("Created movie_reviews.csv locally!")

# --- Step 1: Load the Data ---
    # TODO: Read the CSV file located at the filepath using pandas.
    # TODO: Return the loaded dataframe.
    pass

# --- Step 2 & 3: Split and Vectorize (Bag of Words) ---
    # TODO: Extract the 'review' column to serve as our input features.
    # TODO: Extract the 'sentiment' column to serve as our target labels.
    
    # TODO: Split the features and labels into training and testing sets using an 80/20 split.
    
    # TODO: Create a CountVectorizer object to translate words into numbers.
    
    # TODO: Fit the vectorizer on the training features and transform them into a matrix of numbers.
    # TODO: Transform the testing features into numbers using the already-fitted vectorizer.
    
    # TODO: Return the vectorized training features, the vectorized testing features, the training labels, the testing labels, and the vectorizer object.
    pass

# --- Step 4: Train the AI ---
    # TODO: Create a Multinomial Naive Bayes classifier object.
    # TODO: Train the classifier using the vectorized training features and the training labels.
    # TODO: Return the trained model.
    pass

# --- Step 5: Test the AI ---
    # TODO: Use the trained model to predict the sentiments of the vectorized testing features.
    # TODO: Calculate the accuracy score by comparing the predictions to the actual testing labels.
    # TODO: Return the accuracy score.
    pass

# --- Step 6: Make a Prediction ---
    # TODO: Place the input text string into a list.
    # TODO: Transform the text list into numbers using the provided vectorizer.
    # TODO: Use the trained model to predict the sentiment of the transformed text.
    # TODO: Return the first item from the prediction array.
    pass


# --- Main Application Execution ---
if __name__ == "__main__":
    # Create the CSV file
    create_mini_dataset()
    
    print("Loading data...")
    # TODO: Call the load_data function and store the result in a variable.
    
    print("Preparing and Vectorizing Data...")
    # TODO: Call the split_and_vectorize function and store the five returned values.
    
    print("Training Model...")
    # TODO: Call the train_ai function using the training data and store the model.
    
    print("Evaluating Model...")
    # TODO: Call the calculate_accuracy function using the testing data and print the result as a percentage.
    
    print("Setting up User Interface...")
    # TODO: Create a wrapper function that takes a single text input and returns the result of predict_review.
    # TODO: Create a Gradio Interface using the wrapper function, setting the input to a textbox and output to text.
    # TODO: Launch the Gradio Interface to open the web browser app.