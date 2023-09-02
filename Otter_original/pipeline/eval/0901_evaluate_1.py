import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from coco_metric import compute_cider, postprocess_captioning_generation
import json
import tempfile
import os
from tqdm import tqdm

# def compute_bleu(reference, candidate):
#     reference = [reference.split()]
#     candidate = candidate.split()
#     smoothing = SmoothingFunction().method1
#     return sentence_bleu(reference, candidate, smoothing_function=smoothing)

def compute_bleu(reference, candidate):
    # Convert both reference and candidate to strings
    reference = str(reference)
    candidate = str(candidate)
    reference = [reference.split()]
    candidate = candidate.split()
    smoothing = SmoothingFunction().method1
    return sentence_bleu(reference, candidate, smoothing_function=smoothing)


def compute_cider_score(reference, candidate):
    # Write references and candidates to temporary files
    ref_handle, ref_file = tempfile.mkstemp()
    cand_handle, cand_file = tempfile.mkstemp()

    # Reference data retains the dictionary structure
    ref_data = {
        "annotations": [{'image_id': i, 'id': i, 'caption': ref} for i, ref in enumerate([reference])],
        "images": [{'id': i} for i in range(len([reference]))]
    }

    # Candidate data is just a list of annotations
    cand_data = [{'image_id': i, 'id': i, 'caption': cand} for i, cand in enumerate([candidate])]

    with open(ref_file, 'w') as f:
        json.dump(ref_data, f)

    with open(cand_file, 'w') as f:
        json.dump(cand_data, f)

    # Let's print the content of the files to see the structure
    # print("Reference Data:")
    # with open(ref_file, 'r') as f:
    #     print(f.read())

    # print("\nCandidate Data:")
    # with open(cand_file, 'r') as f:
    #     print(f.read())

    # Call compute_cider using these files
    result = compute_cider(cand_file, ref_file)

    # Remove temporary files
    os.close(ref_handle)
    os.close(cand_handle)

    # Return the CIDEr score
    return result.get('CIDEr', 0.0)


def main():
    # Read the Excel file
    df = pd.read_excel('/home/chengyili/project/CT-CLIP/Otter_original/0901_eval_generated_captions.xlsx')
    
    # Extract columns
    references = df['gt'].astype(str).tolist()
    candidates = df['parsed_output'].astype(str).tolist()
    
    # Compute scores with tqdm progress bars
    bleu_scores = [compute_bleu(ref, cand) for ref, cand in tqdm(zip(references, candidates), total=len(references), desc="Computing BLEU Scores")]
    cider_scores = [compute_cider_score(ref, cand) for ref, cand in tqdm(zip(references, candidates), total=len(references), desc="Computing CIDEr Scores")]
    
    # Add scores to the dataframe
    df['BLEU Score'] = bleu_scores
    df['CIDEr Score'] = cider_scores
    
    # Save the updated dataframe to a new Excel file
    df.to_excel('/home/chengyili/project/CT-CLIP/Otter_original/0901_eval_generated_captions_scores.xlsx', index=False)

    # Print average scores
    print(f"Average BLEU Score: {sum(bleu_scores) / len(bleu_scores)}")
    print(f"Average CIDEr Score: {sum(cider_scores) / len(cider_scores)}")

if __name__ == "__main__":
    main()
