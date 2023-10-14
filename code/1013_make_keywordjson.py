# import openpyxl
import json
import pandas as pd

def process_sheet(df):
    synonym_dict = {}
    for index, row in df.iterrows():
        synonyms = [(" "+ str(x).lower()) for x in row if not pd.isna(x)]
        for synonym in synonyms:
            synonym_dict[synonym] = synonyms
    return synonym_dict

def excel_to_json(filename):
    # Load the Excel file using pandas
    xls = pd.ExcelFile(filename)

    # Process each sheet and store them in the dictionary
    output_dict = {}
    for sheet_name in ['degree', 'landmark', 'feature', 'impression']:
        df = xls.parse(sheet_name)
        output_dict[sheet_name] = process_sheet(df)

    return output_dict

filename = '/Users/charliethebear/Documents/Lab/2023_summer/excel/customized/1013_RADICALchecked_json.xlsx'
data = excel_to_json(filename)

# Write the data to 'output.json'
with open('/Users/charliethebear/Documents/Lab/2023_summer/excel/customized/1013_RADICALchecked.json', 'w') as f:
    json.dump(data, f)


