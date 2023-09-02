file_path = '/local2/chengyili/data/output/eval_instruction.json'
position = 8623733
context_range = 100  # Print 500 characters before and after the position

with open(file_path, 'r') as file:
    file.seek(max(position - context_range, 0))  # Move to the position - context_range
    context = file.read(2 * context_range)  # Read the range around the position

print(context)
