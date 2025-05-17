import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# === CONFIGURATION ===
MODEL_NAME = 'dkleczek/bert-base-polish-uncased-v1'
CHECKPOINT_PATH = 'best_model_brier.bin'  # path to your fine-tuned weights
LABELS = ['PSL-TD', 'Polska2050-TD', 'PiS', 'KO', 'Konfederacja', 'Lewica', 'Razem']  # replace with your actual classes
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === LOAD TOKENIZER & MODEL ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS)
)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === INPUT TEXT ===n
text = '''
Prawo do abrocji to podstawowe prawo kobiet, wierzę w socjaldemokrację, nienawidzę neoliberalnej propagandy. Dobre finansowanie badań i rozwoju, dobre finansowanie służby zdrowia. Mówimy nie dla prywatyzacji służby zdrowia. Kocham państwa skandynawskie, powinniśmy brać z nich przykład. Koniecznie musimy zapobiec zmniejszeniu składki na służbę zdrowia. Koniecznie musimy wprowadzić wysokie progresywne podatki.
'''

# === TOKENIZE & INFER ===
encoding = tokenizer(
    text,
    max_length=512,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
input_ids = encoding['input_ids'].to(DEVICE)
attention_mask = encoding['attention_mask'].to(DEVICE)

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

# === DISPLAY RESULTS ===
for label, prob in zip(LABELS, probs):
    print(f"{label}: {prob:.4f}")