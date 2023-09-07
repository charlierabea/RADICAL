import pandas as pd
import json

# 1. Load the Excel file into a DataFrame.
df = pd.read_excel("/Users/charliethebear/Documents/Lab/2023_summer/excel/0904_train_generated_captions_scores.xlsx")

# 2. Load the JSON file into a Python dictionary.
with open("/Volumes/My_Passport/otter/training_data/MED_instruction.json", "r") as json_file:
    data_json = json.load(json_file)

# 3. Process the DataFrame to extract patient and study information.
def extract_patient_study(row):
    # Get the corresponding instruction from the JSON data
    instruction_data = data_json["data"].get(row["id"], None)
    if instruction_data:
        # Extract patient and study info from one of the image_ids
        image_id = instruction_data["image_ids"][0]
        patient = image_id.split("_")[3]
        study = image_id.split("_")[5]
        return patient, study
    else:
        # If the ID doesn't exist in the JSON data, return NaN values
        return (pd.NA, pd.NA)

df["patient"], df["study"] = zip(*df.apply(extract_patient_study, axis=1))

# 4. Save the updated DataFrame back to the Excel file.
df.to_excel("/Users/charliethebear/Documents/Lab/2023_summer/excel/0906_train_study_captions_scores.xlsx", index=False)
