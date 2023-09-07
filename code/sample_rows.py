import pandas as pd

def sample_rows_with_reporter(input_file, output_file, n=100):
    """
    Randomly sample n rows where the "reporter" column is not empty from an excel file and store in another excel file.
    
    Parameters:
    - input_file: path to the input Excel file.
    - output_file: path to the output Excel file.
    - n: number of rows to sample. Default is 100.
    """
    # Read the Excel file into a DataFrame
    df = pd.read_excel(input_file)
    
    # Filter rows where "reporter" column is not empty
    filtered_df = df[df['reporter'].notna()]
    
    # Sample n rows from the filtered DataFrame
    sampled_df = filtered_df.sample(n=n)
    
    # Save the sampled DataFrame to a new Excel file
    sampled_df.to_excel(output_file, index=False)

# Usage example
sample_rows_with_reporter("/Users/charliethebear/Documents/Lab/2023_summer/excel/0906_eval_study_reporters_scores.xlsx", "/Users/charliethebear/Documents/Lab/2023_summer/excel/0906_eval_100_sample.xlsx")

