import pandas as pd

# Load the Excel file
file_path = "/Users/charliethebear/Documents/Lab/2023_summer/excel/finaleval_ALL_CFcaption.xlsx"
with pd.ExcelFile(file_path) as xls:
    writer = pd.ExcelWriter("/Users/charliethebear/Documents/Lab/2023_summer/excel/finaleval_ALL_sortedCFcaption.xlsx")  # Output file
    for sheet_name in xls.sheet_names:
        # Read the sheet
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Convert columns to string and handle NaN values
        df['Pred sentence'] = df['Pred sentence'].astype(str).replace('nan', '', regex=True)
        df['GT sentence'] = df['GT sentence'].astype(str).replace('nan', '', regex=True)
        
        # Group by 'ID' and concatenate sentences
        grouped = df.groupby('ID').agg(
            Pred_sentence=('Pred sentence', ' '.join),
            GT_sentence=('GT sentence', ' '.join)
        ).reset_index()
        
        # Write the processed data to a new sheet in the output file
        grouped.to_excel(writer, sheet_name=sheet_name, index=False)
        
    writer.close()
