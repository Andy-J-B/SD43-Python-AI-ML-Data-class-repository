import csv
import urllib.request

# Directly fetching a clean copy of the UCI Zoo Dataset hosted on GitHub
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data"

# The original dataset doesn't have an inline header row, so we define it manually based on documentation
header = [
    "animal_name", "hair", "feathers", "eggs", "milk", "airborne", 
    "aquatic", "predator", "toothed", "backbone", "breathes", 
    "venomous", "fins", "legs", "tails", "domestic", "catsize", "class_type"
]

def load_online_data():
    with urllib.request.urlopen(DATA_URL) as response:
        raw_text = response.read().decode('utf-8')
    
    lines = raw_text.strip().split('\n')
    reader = csv.reader(lines)
    raw_data = list(reader)
    
    # Cleaning Step: Strip out the unique animal name (index 0) so the AI doesn't accidentally 
    # cheat by memorizing the name instead of looking at the physical traits.
    cleaned_data = [row[1:] for row in raw_data if len(row) > 0]
    
    # We also slice the header to match our cleaned row structure (removing 'animal_name')
    cleaned_header = header[1:]
    
    return cleaned_header, cleaned_data