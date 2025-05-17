import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F

################################################
# CONSTANTS
BATCH_SIZE    = 8
EPOCHS        = 4
LEARNING_RATE = 2e-5
MAX_GRAD_NORM = 1.0
NUM_WORKERS   = 4
MAX_LENGTH    = 512
ALPHA         = 0.5   # weight for CE vs. Brier
TRAINING_DATASET_PATH   = "DATASET/TRAINING_DATASET_RAW_2024_TO_MAY_2025.csv"
VALIDATION_DATASET_PATH = "DATASET/VALIDATION_DATASET_2024_TO_MAY_2025.csv"
################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Model & Tokenizer ===
model_name = 'dkleczek/bert-base-polish-uncased-v1'
tokenizer  = AutoTokenizer.from_pretrained(model_name)

# === Dataset ===
class SpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_data(path, label_encoder=None, fit_encoder=False):
    df    = pd.read_csv(path)
    texts = df['statement'].astype(str).tolist()
    clubs = df['club'].tolist()
    if fit_encoder:
        label_encoder.fit(clubs)
    labels = label_encoder.transform(clubs)
    return texts, labels

# Prepare data
label_encoder = LabelEncoder()
train_texts, train_labels = load_data(TRAINING_DATASET_PATH, label_encoder, fit_encoder=True)
val_texts,   val_labels   = load_data(VALIDATION_DATASET_PATH, label_encoder)
num_labels = len(label_encoder.classes_)

train_ds = SpeechDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
val_ds   = SpeechDataset(val_texts,   val_labels,   tokenizer, MAX_LENGTH)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# === Model, Optimizer & Scheduler ===
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# === Loss Helpers ===
def brier_loss(logits, labels, num_labels):
    """Mean squared error between softmax probabilities and one-hot labels."""
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(labels, num_labels).float()
    return torch.mean(torch.sum((probs - one_hot)**2, dim=1))

# === Training & Evaluation ===
def train_epoch(model, loader, optimizer, scheduler, device, num_labels):
    model.train()
    total_loss, total_correct = 0.0, 0

    total_batches = len(loader)
    for i, batch in enumerate(loader, 1):
        if i % 100 == 0:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] Processing batch {i}/{total_batches}", flush=True)
            
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        ce = F.cross_entropy(logits, labels)
        br = brier_loss(logits, labels, num_labels)
        loss = ALPHA * ce + (1 - ALPHA) * br

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += torch.sum(preds == labels).item()

    avg_loss = total_loss / len(loader.dataset)
    avg_acc  = total_correct / len(loader.dataset)
    return avg_loss, avg_acc

def eval_model(model, loader, device, num_labels):
    model.eval()
    total_ce, total_correct = 0.0, 0
    brier_list, margin_list = [], []
    
    # using manual progress printing instead of tqdm for docker logs compability
    total_batches = len(loader)
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            if i % 100 == 0:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{now}] Processing batch {i}/{total_batches}", flush=True)

            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Cross-entropy for reporting
            ce = F.cross_entropy(logits, labels, reduction='sum')
            total_ce += ce.item()

            preds = torch.argmax(logits, dim=1)
            total_correct += torch.sum(preds == labels).item()

            # Brier per batch
            probs = F.softmax(logits, dim=1).cpu().numpy()
            one_hot = F.one_hot(labels, num_labels).cpu().numpy()
            batch_brier = np.mean(np.sum((probs - one_hot)**2, axis=1))
            brier_list.append(batch_brier)

            # Margin
            top2 = torch.topk(logits, k=2, dim=1).values
            batch_margin = (top2[:,0] - top2[:,1]).mean().item()
            margin_list.append(batch_margin)

    n = len(loader.dataset)
    avg_ce     = total_ce / n
    avg_acc    = total_correct / n
    avg_brier  = np.mean(brier_list)
    avg_margin = np.mean(margin_list)
    return avg_ce, avg_acc, avg_brier, avg_margin

# === Main Loop ===
best_brier = float('inf')

for epoch in range(1, EPOCHS+1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, num_labels)
    val_ce, val_acc, val_brier, val_margin = eval_model(model, val_loader, device, num_labels)

    print(f" ▶ Train  → loss: {train_loss:.4f} | acc: {train_acc:.4f}")
    print(f" ▶ Valid  → CE:   {val_ce:.4f} | acc: {val_acc:.4f} "
          f"| Brier: {val_brier:.4f} | Margin: {val_margin:.4f}")

    # Always save last epoch
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    torch.save(model.state_dict(), f'model_epoch{epoch}_{ts}.bin')

    # Save best by Brier
    if val_brier < best_brier:
        print(f"   ↳ New best Brier ({val_brier:.4f} < {best_brier:.4f}), saving checkpoint.")
        torch.save(model.state_dict(), 'best_model_brier.bin')
        best_brier = val_brier

print(f"\nTraining complete. Best validation Brier: {best_brier:.4f}")
