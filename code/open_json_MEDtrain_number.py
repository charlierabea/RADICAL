import json

# Read the JSON from a file
with open('/home/chengyili/project/CT-CLIP/Otter/mimic-it/convert-it/output/MED_train.json', 'r') as file:
    data = json.load(file)

# Extract image_ids list lengths for each MED_INS and store them with the corresponding MED_INS key
lengths = [(key, len(entry["image_ids"])) for key, entry in data["data"].items()]

# Sort the tuples based on the length and get the shortest 100
shortest_100 = sorted(lengths, key=lambda x: x[1])[:100]

# Print the MED_INS and the corresponding length for the shortest 100
for med_ins, length in shortest_100:
    print(f"{med_ins}: {length}")
