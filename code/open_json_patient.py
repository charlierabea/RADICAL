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

# Filter the keys that match the pattern and have a patient number between 1 and 10000
# Extract only the A_xxxx portion and ensure no duplicates using set
# Using tqdm to show progress
patient_keys_set = set(['A_' + pattern.match(key).group(1) for key in tqdm(keys) if pattern.match(key) and 1 <= int(pattern.match(key).group(1)) <= 10000])

# Convert the set back to a list and sort it
patient_keys_sorted = sorted(patient_keys_set, key=lambda x: int(x.split('_')[1]))

# Print the sorted list of unique patient keys
print(patient_keys_sorted)

# Print the length of the list
print(len(patient_keys_sorted))
