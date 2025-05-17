import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from datetime import datetime

################################################
# CONSTANTS
BATCH_SIZE    = 8
EPOCHS        = 4
LEARNING_RATE = 2e-5
MAX_GRAD_NORM = 1.0
NUM_WORKERS   = 4
MAX_LENGTH    = 512
TRAINING_DATASET_PATH   = "DATASET/TRAINING_DATASET_RAW_2024_TO_MAY_2025.csv"
VALIDATION_DATASET_PATH = "DATASET/VALIDATION_DATASET_2024_TO_MAY_2025.csv"
################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Model & Tokenizer ===
model_name = 'sdadas/polish-distilroberta'
tokenizer  = AutoTokenizer.from_pretrained(model_name)

# === Data Loading & Label Encoding ===
class SpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_data(path, label_encoder=None, fit_encoder=False):
    df    = pd.read_csv(path)
    texts = df['statement'].astype(str).tolist()
    clubs = df['club'].tolist()
    if fit_encoder:
        label_encoder.fit(clubs)
    labels = label_encoder.transform(clubs)
    return texts, labels

# Prepare label encoder and data
label_encoder = LabelEncoder()
train_texts, train_labels = load_data(TRAINING_DATASET_PATH, label_encoder, fit_encoder=True)
val_texts, val_labels     = load_data(VALIDATION_DATASET_PATH, label_encoder)
num_labels = len(label_encoder.classes_)

# Initialize model with correct number of labels
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
).to(device)
print(f"Loaded {model_name} with {num_labels} classes")

# Create DataLoaders
def create_data_loaders(train_texts, train_labels, val_texts, val_labels, tokenizer, batch_size):
    train_ds = SpeechDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_ds   = SpeechDataset(val_texts,   val_labels,   tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader

train_loader, val_loader = create_data_loaders(
    train_texts, train_labels, val_texts, val_labels,
    tokenizer, BATCH_SIZE
)

# === Optimizer & Scheduler ===
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# === Training & Evaluation Functions ===
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    losses, correct = [], 0
    total_batches = len(loader)
    for i, batch in enumerate(loader, 1):
        if i % 100 == 0:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] Processing batch {i}/{total_batches}", flush=True)

        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss, logits = outputs.loss, outputs.logits
        losses.append(loss.item())
        preds = torch.argmax(logits, dim=1)
        correct += torch.sum(preds == labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()
    return np.mean(losses), correct.double() / len(loader.dataset)

def eval_model(model, loader, device):
    model.eval()
    losses, correct = [], 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss, logits = outputs.loss, outputs.logits
            losses.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            correct += torch.sum(preds == labels)
    return np.mean(losses), correct.double() / len(loader.dataset)

# === Main Training Loop ===
best_acc = 0
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
    val_loss, val_acc     = eval_model(model,   val_loader,   device)
    print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
    print(f" Val  loss: {val_loss:.4f},  Val acc: {val_acc:.4f}")
    # Checkpointing
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    torch.save(model.state_dict(), f'model_epoch{epoch+1}_{ts}.bin')
    if val_acc > best_acc:
        torch.save(model.state_dict(), 'best_model.bin')
        best_acc = val_acc

print("Training complete.")
