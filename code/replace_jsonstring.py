import json

# Load the JSON data from the file
with open("/raid/jupyter-alz.ee09/data/0916_eval_instruction2_baseline.json", "r") as file:
    data = json.load(file)

# Modify the dictionary directly
data["data"]["eval_INS_18875"]["answer"] = "> No obvious territorial acute infarction can be identified in this study. However, small acute infarction may be inapparent on initial CT. Suggest arranging MR study for correlation if clinically suspected \n\n> No remarkable finding of paranasal sinuses and bilateral mastoids \n\n> No significant acute hemorrhage, space occupying lesion, midline shift, or dilatation of the ventricular system noted in this study \n\n> Mild cerebral tissue loss \n\n> Mild nonspecific low density areas noted in bilateral periventricular white matter and centrum semiovale, without significant mass effect, favoring subcortical arteriosclerotic encephalopathy \n\n> No obvious bony destruction identified"

data["data"]["eval_INS_16032"]["answer"] = "> > Old lacunar infarction noted over left thalamus \n\n> Hyperdense mucus retension noted over left posterior ethmoid sinus \n\n> No obvious territorial acute infarction can be identified in this study. However, small acute infarction may be inapparent on initial CT. Suggest arranging MR study for correlation if clinically suspected \n\n> No significant acute hemorrhage, space occupying lesion, midline shift, or dilatation of the ventricular system noted in this study \n\n> Mild cerebral tissue loss \n\n> No obvious bony destruction identified \n\n> Calcification of bilateral ICAs and VAs"

# Save the modified data back to the file
with open("/raid/jupyter-alz.ee09/data/0916_eval_instruction2_baseline.json", "w") as file:
    json.dump(data, file, indent=4)
