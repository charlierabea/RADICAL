import json
from tqdm import tqdm

# Load the JSON file
with open('/home/chengyili/project/CT-CLIP/Otter/mimic-it/convert-it/output/MED.json', 'r') as file:
    data = json.load(file)

# Identify truncated images
truncated_images = [key for key, value in tqdm(data.items()) if value is None]

# Print the truncated images and their count
print("Truncated Images:", truncated_images)
print("Number of Truncated Images:", len(truncated_images))
