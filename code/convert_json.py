import pandas as pd
import json
import os
from tqdm import tqdm

# Load the Excel file
excel_path = '/home/chengyili/project/CT-CLIP/Otter/mimic-it/convert-it/output/0824_2011-Glaucoma_anonymized.xlsx'
df = pd.read_excel(excel_path)

data = {}
instruction_text = ("You are an AI assistant specialized in radiology topics. "
                    "\n\n You are provided with brain CT slices from a single study. "
                    "The number of slices is usually around 30 when it's the coronal section, "
                    "or 60 when sagittal section is added. \n Please generate image caption based on image"
                    "You would have to look at the images to see if there's the following conditions: \n\n"
                    "brain atrophy/meningioma/fracture/soft tissue swelling/hydrocephalus/low density patches/enlargement of the ventricular system/enlargement of the sulci/calcified plaques/midline shift/intracranial hepatoma/epidural hepatoma/subdural hepatoma/lacunar infarction/cortical infarction/subcortical infarction/herniation\tarteriosclerotic/encephalopathy/encephalomalacia/wall calcification of cavernous ICA\n\n")

# Define the destination directory
destination_directory = '/home/chengyili/data/CT/mved/'

def reformat_image_id(image_id: str) -> str:
    """
    Reformat the image ID from the format "A_3_Study_1_80248_5.bmp" 
    to "DC_IMG_A_3_Study_1_80248_5".
    """
    ID = image_id.split('.')[0]
    parts = ID.split('_')
    new_format = f"MED_IMG_{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}_{parts[5]}"
    return new_format

# Map each study to its set of images
image_ids_map = {}
for image_file in tqdm(os.listdir(destination_directory), desc="Mapping images"):
    if image_file.endswith('.bmp'):
        # Extract the Patient and Study part of the filename (e.g., "A_3_Study_1" from "A_3_Study_1_80248_1.bmp")
        patient = int(image_file.split('_')[1])
        if patient < 1273:
            key = "_".join(image_file.split('_')[:-2])
            if key not in image_ids_map:
                image_ids_map[key] = []
            image_ids_map[key].append(reformat_image_id(image_file))

# Filter and select 26 images with the same second-to-last component for each study
for key, ids in image_ids_map.items():
    component_counts = {}
    for image_id in ids:
        component = image_id.split('_')[-2]
        if component not in component_counts:
            component_counts[component] = []
        component_counts[component].append(image_id)

    # Find the component with the most images and select the first 26 from it
    most_common_component = max(component_counts.keys(), key=lambda k: len(component_counts[k]))
    image_ids_map[key] = component_counts[most_common_component][:26]
        
# Build the JSON structure
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Building JSON"):
    key = f"A_{row['Patient']}_Study_{row['Study']}"
    image_ids = image_ids_map.get(key, [])
    
    ins_id = f"MED_INS_{str(index).zfill(5)}"
    data[ins_id] = {
        "instruction": instruction_text,
        "answer": row['Description'],
        "image_ids": image_ids,
        "rel_ins_ids": []
    }

# Build the final structure including meta
result = {
    "meta": {
        "version": "0.0.1",
        "time": "2023-08",
        "author": "big_data_center"
    },
    "data": data
}

# Save the resulting JSON
output_path = '/home/chengyili/project/CT-CLIP/Otter/mimic-it/convert-it/output/MED_train2.json'
with open(output_path, 'w') as f:
    json.dump(result, f, indent=4)
