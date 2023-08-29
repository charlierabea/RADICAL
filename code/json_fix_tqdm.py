file_path = '/home/chengyili/project/CT-CLIP/Otter/mimic-it/convert-it/output/MED_eval.json'
insert_position = 19619798029
comma = '}'

# Read the file contents
with open(file_path, 'r') as file:
    content = file.read()

# Insert the comma at the specified position
corrected_content = content[:insert_position] + comma

# Write the corrected content back to the file
with open(file_path, 'w') as file:
    file.write(corrected_content)
