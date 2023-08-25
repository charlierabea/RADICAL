import pandas as pd
import json
import os

# Load the Excel file
excel_path = '/home/chengyili/data/CT/0824_test.xlsx'
df = pd.read_excel(excel_path)

data = {}
instruction_text = ("You are an AI assistant specialized in radiology topics. "
                    "\n\n You are provided with brain CT slices from a single study. "
                    "The number of slices is usually around 30 when it's the coronal section, "
                    "or 60 when sagittal section is added. \n Please generate image caption based on image")

# Define the destination directory
destination_directory = '/home/chengyili/data/CT/test/'

# Map each study to its set of images
image_ids_map = {}
for image_file in os.listdir(destination_directory):
    if image_file.endswith('.bmp'):
        # Extract the Patient and Study part of the filename (e.g., "A_3_Study_1" from "A_3_Study_1_80248_1.bmp")
        key = "_".join(image_file.split('_')[:-2])
        if key not in image_ids_map:
            image_ids_map[key] = []
        image_ids_map[key].append(image_file)
        
# Build the JSON structure
for index, row in df.iterrows():
    key = f"A_{row['Patient']}_Study_{row['Study']}"
    image_ids = image_ids_map.get(key, [])
    
    ins_id = f"MED_INS_{str(index).zfill(5)}"
    data[ins_id] = {
        "instruction": instruction_text,
        "answer": row['Description'],
        "image_ids": image_ids,
        "rel_ins_ids": []
    }

# Save the resulting JSON
output_path = '/home/chengyili/data/CT/test.json'
with open(output_path, 'w') as f:
    json.dump(data, f, indent=4)