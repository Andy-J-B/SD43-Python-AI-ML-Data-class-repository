import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import gradio as gr
import os

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
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# --- Step 2 & 3: Split and Vectorize (Bag of Words) ---
def split_and_vectorize(df):
    X = df['review']
    y = df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    vectorizer = CountVectorizer()
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    return X_train_vec, X_test_vec, y_train, y_test, vectorizer

# --- Step 4: Train the AI ---
def train_ai(X_train_vectorized, y_train):
    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)
    return model

# --- Step 5: Test the AI ---
def calculate_accuracy(model, X_test_vectorized, y_test):
    predictions = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# --- Step 6: Make a Prediction ---
def predict_review(text, model, vectorizer):
    text_list = [text]
    text_vectorized = vectorizer.transform(text_list)
    prediction = model.predict(text_vectorized)
    return prediction[0]


# --- Main Application Execution ---
if __name__ == "__main__":
    create_mini_dataset()
    
    print("Loading data...")
    df = load_data("movie_reviews.csv")
    
    print("Preparing and Vectorizing Data...")
    X_train_vec, X_test_vec, y_train, y_test, vectorizer = split_and_vectorize(df)
    
    print("Training Model...")
    trained_model = train_ai(X_train_vec, y_train)
    
    print("Evaluating Model...")
    accuracy = calculate_accuracy(trained_model, X_test_vec, y_test)
    print(f"Model Accuracy: {accuracy:.2%}")
    
    print("Setting up User Interface...")
    
    # Wrapper function for our UI to use
    def gradio_predict(user_text):
        return predict_review(user_text, trained_model, vectorizer)
        
    # Build the visual web interface
    interface = gr.Interface(
        fn=gradio_predict, 
        inputs=gr.Textbox(lines=3, placeholder="Type a movie review here..."), 
        outputs=gr.Textbox(label="AI Sentiment Prediction"),
        title="Live Sentiment Meter"
    )
    
    # Launch it!
    interface.launch()