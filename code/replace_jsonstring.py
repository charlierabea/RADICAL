import json

# Load the JSON data from the file
with open("/local2/chengyili/data/output/eval_instruction.json", "r") as file:
    data = json.load(file)

# Convert the JSON data to a string
data_str = json.dumps(data)

# Replace all occurrences of "MED" with "eval"
data_str = data_str.replace("MED", "eval")

# Convert the string back to a dictionary
data = json.loads(data_str)

# Save the modified data back to the file
with open("/local2/chengyili/data/output/eval_instruction.json", "w") as file:
    json.dump(data, file, indent=4)
