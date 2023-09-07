import pandas as pd
import re

def extract_reporter_name(text):
    """
    Extract reporter's name from the given text and format it as 'last name, first name'.
    """
    # Check if text is a string
    if not isinstance(text, str):
        return None

    # Using regex to extract the name pattern from the string with case-insensitive match
    match = re.search(r'Reported By (.*?)(/|\(|License)', text, re.IGNORECASE)
    if not match:
        return None

    name = match.group(1).strip().replace("Dr.", "").strip()
    names = [part.strip() for part in name.split(",")]
    
    if len(names) == 2:
        # If the name is already in the format "Last, First"
        return f"{names[0].lower()}, {names[1].lower()}"
    elif len(names) == 1:
        # If the name is in the format "Last First Middle" or "First Last" or "Last First M."
        parts = names[0].split()
        if len(parts) == 3:
            # Format "Last First Middle"
            return f"{parts[0].lower()}, {parts[1].lower()}-{parts[2].lower()}"
        elif len(parts) == 2 and len(parts[1]) > 1:
            # Format "First Last"
            return f"{parts[1].lower()}, {parts[0].lower()}"
        elif len(parts) == 2:
            # Format "Last First M."
            return f"{parts[0].lower()}, {parts[1].lower()}"
    return None





def merge_excels_and_extract_reporter(input_file_a, input_file_b, output_file):
    # Load the two Excel files
    df_a = pd.read_excel(input_file_a)
    df_b = pd.read_excel(input_file_b)

    # Merge the dataframes based on the 'Patient' and 'Study' columns
    merged_df = df_a.merge(df_b[['Patient', 'Study', 'Processed_Text']], on=['Patient', 'Study'], how='left')

    # Extract and format the reporter's name and add it to the 'reporter' column
    merged_df['reporter'] = merged_df['Processed_Text'].apply(lambda x: extract_reporter_name(x) if pd.notnull(x) else None)

    # Drop the 'Processed_Text' column
    merged_df.drop(columns=['Processed_Text'], inplace=True)

    # Save the merged dataframe back to Excel
    merged_df.to_excel(output_file, index=False)

# Usage example
merge_excels_and_extract_reporter("/Users/charliethebear/Documents/Lab/2023_summer/excel/0906_eval_study_captions_scores.xlsx", "/Users/charliethebear/Documents/Lab/2023_summer/excel/0714_2011-Glaucoma_fulltxt2.xlsx", "/Users/charliethebear/Documents/Lab/2023_summer/excel/0906_eval_study_reporters_scores.xlsx")
