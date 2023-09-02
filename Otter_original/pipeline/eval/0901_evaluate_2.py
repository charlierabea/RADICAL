import pandas as pd
import json
from pycocotools.coco import COCO
from tqdm import tqdm
from pycocoevalcap.eval import COCOEvalCap
from coco_metric import compute_cider, postprocess_captioning_generation

def load_data_from_excel(file_path):
    data = pd.read_excel(file_path)
    data['parsed_output'] = data['parsed_output'].astype(str)
    return data

def format_data_for_evaluation(data):
    annotations = []
    results = []
    images = []
    for index, row in data.iterrows():
        annotations.append({
            'image_id': row['id'],
            'id': index,
            'caption': row['gt']
        })
        results.append({
            'image_id': row['id'],
            'caption': row['parsed_output']
        })
        images.append({
            'id': row['id'],
            'file_name': str(row['id']) + '.jpg'  # Dummy filename, adjust if needed
        })
    return annotations, results, images


def save_data_to_excel(data, file_path):
    print("Saving data to:", file_path)
    data.to_excel(file_path, index=False)
    print("Data saved successfully!")

def evaluate(data, annotations_path, results_path):
    annotations, results, images = format_data_for_evaluation(data)

    # Save to temporary JSON files
    with open(annotations_path, 'w') as f:
        json.dump({"annotations": annotations, "images": images}, f)  # Added 'images' key
    with open(results_path, 'w') as f:
        json.dump(results, f)

    coco = COCO(annotations_path)
    cocoRes = coco.loadRes(results_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()

    bleu_1_scores, bleu_2_scores, bleu_3_scores, bleu_4_scores, cider_scores = [], [], [], [], []

    for img_id in tqdm(data['id'], desc="Fetching scores"):
        bleu_1_scores.append(cocoEval.imgToEval[img_id]['Bleu_1'])
        bleu_2_scores.append(cocoEval.imgToEval[img_id]['Bleu_2'])
        bleu_3_scores.append(cocoEval.imgToEval[img_id]['Bleu_3'])
        bleu_4_scores.append(cocoEval.imgToEval[img_id]['Bleu_4'])
        cider_scores.append(cocoEval.imgToEval[img_id]['CIDEr'])

    data['Bleu_1'] = bleu_1_scores
    data['Bleu_2'] = bleu_2_scores
    data['Bleu_3'] = bleu_3_scores
    data['Bleu_4'] = bleu_4_scores
    data['CIDEr'] = cider_scores
    return data

# Paths for temporary files
annotations_path = 'temp_annotations.json'
results_path = 'temp_results.json'

# Load data
excel_file_path = '/home/chengyili/project/CT-CLIP/Otter_original/0901_eval_generated_captions.xlsx'
data = load_data_from_excel(excel_file_path)

# Evaluate
data = evaluate(data, annotations_path, results_path)

# Save updated data to Excel
save_data_to_excel(data, '/home/chengyili/project/CT-CLIP/Otter_original/0901_eval_generated_captions_scores.xlsx')
