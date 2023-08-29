import json
import re
from tqdm import tqdm

# Load the JSON file
with open('/home/chengyili/project/CT-CLIP/Otter/mimic-it/convert-it/output/MED.json', 'r') as file:
    data = json.load(file)

# Extract all top-level keys from the JSON data
keys = data.keys()

# Define a regular expression pattern to match the required keys
pattern = re.compile(r'^MED_IMG_A_(\d+)_.*$')

# Filter the keys that contain "A_725"
# Using tqdm to show progress
patient_keys_set = set([key for key in tqdm(keys) if pattern.match(key) and "A_725" in key])

# Convert the set back to a list and sort it
patient_keys_sorted = sorted(patient_keys_set)

# Print the sorted list of patient keys containing "A_725"
print(patient_keys_sorted)

# Print the length of the list
print(len(patient_keys_sorted))
