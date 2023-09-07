import pandas as pd

def summarize_reporter_scores(input_file, output_file):
    # Load the data
    df = pd.read_excel(input_file)

    # Define the columns to be summarized
    columns_to_summarize = ['Bleu_1.1', 'Bleu_2.1', 'Bleu_3.1', 'Bleu_4.1', 'CIDEr.1']

    # Group by reporter and calculate count, mean, and std
    summary = df.groupby('reporter')[columns_to_summarize].agg(['count', 'mean', 'std'])

    # Save the summarized data to a new Excel file
    summary.to_excel(output_file)

# Usage example
summarize_reporter_scores("/Users/charliethebear/Documents/Lab/2023_summer/excel/0906_eval_study_reporters_scores.xlsx", "/Users/charliethebear/Documents/Lab/2023_summer/excel/0906_eval_reporters_sorting.xlsx")
