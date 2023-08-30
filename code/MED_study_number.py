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

# Create a dictionary to store the count of each patient number
patient_count_dict = {}

# Iterate over the keys
for key in tqdm(keys):
    match = pattern.match(key)
    if match and 1 <= int(match.group(1)) <= 10000:
        patient_key = 'A_' + match.group(1)
        patient_count_dict[patient_key] = patient_count_dict.get(patient_key, 0) + 1

# Sort the dictionary by patient number for better visualization
sorted_patient_count = sorted(patient_count_dict.items(), key=lambda x: int(x[0].split('_')[1]))

# Print the sorted list of unique patient keys with their counts
for patient, count in sorted_patient_count:
    print(f"{patient}: {count} times")

# Print the total number of unique patient numbers
print(f"Total unique patient numbers: {len(sorted_patient_count)}")

# Print the 100 patient numbers that appear the least times
least_frequent_500 = sorted(patient_count_dict.items(), key=lambda x: x[1])[:500]
print("\n100 Patient Numbers that Appear the Least Times:")
for patient, count in least_frequent_500:
    print(f"{patient}: {count} times")
