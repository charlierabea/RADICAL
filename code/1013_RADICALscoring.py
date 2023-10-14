import pandas as pd
import json
# /Users/charliethebear/Documents/Lab/2023_summer/excel/customized/1013_RADICALchecked.json
# /Users/charliethebear/Documents/Lab/2023_summer/excel/CFscore/finaleval_ALL_CFprecisioncaption.xlsx
# /Users/charliethebear/Documents/Lab/2023_summer/excel/CFscore/finaleval_ALL_RADICALprecision.xlsx

def extract_keywords_from_text(text, json_data):
    text = str(text).lower()
    keywords = set()
    for key in json_data:
        if key in text:
            keywords.add(key.strip())    
    return list(keywords)

def map_to_synonyms(key, json_data):
    for main_key, synonyms in json_data.items():
        # print(synonyms)
        if key in synonyms:
            print(key)
            return set(synonyms)
    return {key}  # Return a set with the key itself

def map_to_representative_synonym(key, json_data):
    for main_key, synonyms in json_data.items():
        if key in synonyms:
            return main_key  # Return the primary keyword
    return key

def calculate_precision_recall(gt_set, out_set, json_data):
    if not gt_set and not out_set:
        return None, None
    if not gt_set:
        return 0, None
    if not out_set:
        return None, 0
    # Pre-process keywords to their representative synonyms
    gt_representative = {map_to_representative_synonym((' ' + keyword), json_data) for keyword in gt_set}
    out_representative = {map_to_representative_synonym((' ' + keyword), json_data) for keyword in out_set}

    # True positive calculation considering synonyms
    tp = sum(1 for gt_key in gt_representative if any(gt_key in out_synonyms for out_synonyms in map(lambda x: map_to_synonyms((x), json_data), out_representative)))
    
    print(tp)

    fp = len(out_representative) - tp
    fn = len(gt_representative) - tp

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / len(gt_representative) if gt_representative else 0

    return precision, recall

    
    return precision, recall

def process_excel_sheet(sheet_df, json_data):
    for col_prefix in ['degree', 'landmark', 'feature', 'impression']:
        sheet_df[f'gt_{col_prefix}'] = sheet_df['gt'].apply(lambda x: list(set(extract_keywords_from_text(x, json_data[col_prefix]))))
        sheet_df[f'out_{col_prefix}'] = sheet_df['parsed_output'].apply(lambda x: list(set(extract_keywords_from_text(x, json_data[col_prefix]))))
        
        sheet_df[f'prec_{col_prefix}'], sheet_df[f'recall_{col_prefix}'] = zip(*sheet_df.apply(lambda row: calculate_precision_recall(set(row[f'gt_{col_prefix}']), set(row[f'out_{col_prefix}']), json_data[col_prefix]), axis=1))

    def bonus1_calc(row):
        if not any(row[f'gt_{col}'] for col in ['degree', 'landmark', 'feature']):
            return None
        return 1 if all([row[f'recall_{col}'] == 1 for col in ['degree', 'landmark', 'feature']]) else 0


    sheet_df['bonus1'] = sheet_df.apply(bonus1_calc, axis=1)
    
    return sheet_df

def process_excel(filename, json_data):
    # Load all sheets from the Excel file
    xls = pd.read_excel(filename, sheet_name=None)

    # Process each sheet
    for sheet_name, sheet_df in xls.items():
        xls[sheet_name] = process_excel_sheet(sheet_df, json_data)

    # Save modified sheets back to the Excel file
    with pd.ExcelWriter("/Users/charliethebear/Documents/Lab/2023_summer/excel/CFscore/finaleval_ALL_RADICALprecision.xlsx") as writer:
        for sheet_name, sheet_df in xls.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
    

# Load JSON data
with open('/Users/charliethebear/Documents/Lab/2023_summer/excel/customized/1013_RADICALchecked.json', 'r') as f:
    json_data = json.load(f)

filename = '/Users/charliethebear/Documents/Lab/2023_summer/excel/CFscore/finaleval_ALL_CFprecisioncaption.xlsx'
process_excel(filename, json_data)

