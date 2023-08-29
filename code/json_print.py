file_path = '/home/chengyili/project/CT-CLIP/Otter/mimic-it/convert-it/output/MED_train.json'
position = 0
context_range = 1000  # Print 500 characters before and after the position

with open(file_path, 'r') as file:
    file.seek(max(position - context_range, 0))  # Move to the position - context_range
    context = file.read(2 * context_range)  # Read the range around the position

print(context)
