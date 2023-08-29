import json

# Load the JSON file
with open('/home/chengyili/project/CT-CLIP/Otter/mimic-it/convert-it/output/MED.json', 'r') as file:
    data = json.load(file)

# Print the top-level keys
print(len(data.keys()))
