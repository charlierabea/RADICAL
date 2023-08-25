#!/bin/bash

# Destination directory for the moved files
destination_dir="/home/chengyili/data/CT/mved"

# Iterate through the 9 source directories
for i in {5..9}; do
    source_dir="/home/chengyili/data/CT/$i"
    # Iterate through each patient directory (like A_10) inside the source_dir
    for patient_dir in "$source_dir"/A_*/; do
        # Extract the patient number from the directory name (e.g., A_10)
        patient_number=$(basename "$patient_dir")
        # Iterate through each study_number directory inside patient_number directory
        for study_dir in "$patient_dir"Study_*/; do
            # Extract the study number from the directory name
            study_number=$(basename "$study_dir")
            # Iterate through each image file inside study_number directory
            for image_file in "$study_dir"*.bmp; do
                # Extract the image name from the file path
                image_name=$(basename "$image_file")
                # Create the new filename pattern
                new_filename="${patient_number}_${study_number}_${image_name}"
                # Move the image file to the destination directory with the new filename
                mv "$image_file" "$destination_dir/$new_filename"
                echo "Moved: $image_file to $destination_dir/$new_filename"
            done
        done
    done
done

echo "All files moved successfully."
