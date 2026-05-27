# Import the module needed to fetch data from a web URL.
# Import the built-in CSV module to parse rows smoothly.

# Define a string variable holding the direct URL to the raw online CSV file.
import csv
import urllib.request

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data"

header = [
    "animal_name", "hair", "feathers", "eggs", "milk", "airborne", 
    "aquatic", "predator", "toothed", "backbone", "breathes", 
    "venomous", "fins", "legs", "tails", "domestic", "catsize", "class_type"
]

def load_online_data():
    # Use the URL library to open a connection and fetch the web data.
    # Read the content and decode it from bytes to a standard UTF-8 text string.
    # Split the massive text string into individual lines.
    # Pass the lines into the CSV reader function to parse columns correctly.
    # Convert the reader object into a clean Python list of rows.
    # Extract the first row to serve as our features header list.
    # Extract the remaining rows to serve as our training dataset.
    # Slice or clean the data rows if necessary to isolate features and labels.
    # Return both the header list and the cleaned training dataset list.
    with urllib.request.urlopen(DATA_URL) as response:
        rawtext = response.read().decode("utf-8")
    lines = rawtext.strip().split("\n")
    reader = csv.reader(lines)
    rawdata = list(reader)
    data = [
        row[1:]for row in rawdata if len(row) > 0
    ]
    cleanheader = header[1:]
    return cleanheader, data