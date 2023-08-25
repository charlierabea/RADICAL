import os
import shutil
from tqdm import tqdm

source_directory = '/home/chengyili/data/CT/1/'
destination_directory = '/home/chengyili/data/CT/alzheimer_2011-Glaucoma_flat/'

if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Compute total number of images for tqdm
total_images = sum([len(files) for r, d, files in os.walk(source_directory) if any(file.endswith('.bmp') for file in files)])

with tqdm(total=total_images, desc="Copying and renaming images") as pbar:
    # Traverse through the directory and copy & rename images
    for patient in os.listdir(source_directory):
        patient_path = os.path.join(source_directory, patient)
        if os.path.isdir(patient_path):
            for study in os.listdir(patient_path):
                study_path = os.path.join(patient_path, study)
                for image in os.listdir(study_path):
                    if image.endswith('.bmp'):
                        patient_number = patient.split('_')[1]
                        study_number = study.split('_')[1]
                        new_image_name = f"{patient_number}_{study_number}_{image}"
                        source_image_path = os.path.join(study_path, image)
                        dest_image_path = os.path.join(destination_directory, new_image_name)
                        shutil.copy2(source_image_path, dest_image_path)
                        pbar.update(1)
