input_file_path = '/home/chengyili/project/CT-CLIP/Otter/mimic-it/convert-it/output/MED.json'
output_file_path = '/home/chengyili/project/CT-CLIP/Otter/mimic-it/convert-it/output/fixed_MED.json'

with open(input_file_path, 'r') as file:
    content = file.read()

# Find occurrences of `""`, which indicate missing commas
comma_missing_positions = [pos for pos, char in enumerate(content[:-1]) if content[pos:pos+2] == '""']

print(f"Number of missing commas found: {len(comma_missing_positions)}")
print(f"Positions of missing commas: {comma_missing_positions}")

# Insert commas at the detected positions
for pos in reversed(comma_missing_positions):
    content = content[:pos+1] + ', ' + content[pos+1:]

# Write the corrected content to a new file
with open(output_file_path, 'w') as file:
    file.write(content)
