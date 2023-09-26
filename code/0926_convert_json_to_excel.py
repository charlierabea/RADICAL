import json
import pandas as pd

# Load the JSON data from the file
with open("/raid/jupyter-alz.ee09/data/0916_MED_instruction2_baseline.json", "r") as file:
    data = json.load(file)

# Extract relevant keys and answers
eval_data = {}
for key, value in data["data"].items():
    if key.startswith("MED_INS_"):
        eval_data[key] = value["answer"]

# Now, eval_data contains the "eval_INS_<number>" as keys and their corresponding "answer" as values.
# Convert the dictionary to a DataFrame
df = pd.DataFrame(list(eval_data.items()), columns=["Key", "Answer"])

# Save the DataFrame to an Excel file
output_file_path = "/raid/jupyter-alz.ee09/Excel/0926_train_gt.xlsx"
df.to_excel(output_file_path, index=False)
