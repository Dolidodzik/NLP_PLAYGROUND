import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from datetime import datetime
import sys


################################################
# CONSTANTS
BATCH_SIZE = 256
EPOCHS = 15
LEARNING_RATE = 2e-5
MAX_GRAD_NORM = 1.0
NUM_WORKERS=4
################################################


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_name = 'distilroberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=6).to(device)
model.load_state_dict(torch.load('somewhat_trained.bin', map_location=device)) # comment this line if want to start training from official model

# =================== LOADING DATA ========================= #
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['rating'] = df['rating'].astype(int)
    texts = df['review_text'].tolist()
    labels = df['rating'].tolist()
    return texts, labels

train_texts, train_labels = load_data('DATASETS/TRAINING_DATASET.csv')
val_texts, val_labels = load_data('DATASETS/VALIDATION_DATASET.csv')

print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")
print("type of train_texts: ", type(train_texts))
print("type of val_labels: ", type(val_labels))

# =================== Dataset Class ========================= #
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
# =================== Loading data into  ========================= #
def create_data_loaders(tokenizer, batch_size=8):
    train_dataset = ReviewDataset(train_texts, train_labels, tokenizer)
    val_dataset = ReviewDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=NUM_WORKERS
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=NUM_WORKERS
    )
    
    return train_loader, val_loader

train_loader, val_loader = create_data_loaders(tokenizer)

# =================== Training Setup ========================= #
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=200,
    num_training_steps=total_steps
)

# =================== Training Single Epoch ========================= #
def train_epoch(model, data_loader, optimizer, scheduler, device):
    model = model.train()
    losses = []
    correct_predictions = 0
    
    for batch in tqdm(data_loader, desc="Training", file=sys.stdout, dynamic_ncols=True):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    accuracy = correct_predictions.double() / len(data_loader.dataset)
    avg_loss = np.mean(losses)
    
    return avg_loss, accuracy

# =================== Evaluation Function ========================= #
def eval_model(model, data_loader, device):
    model = model.eval()
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluation", file=sys.stdout, dynamic_ncols=True):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
    
    accuracy = correct_predictions.double() / len(data_loader.dataset)
    avg_loss = np.mean(losses)
    
    return avg_loss, accuracy

# =================== Training Loop ========================= #
best_accuracy = 0
print(f"starting training, best accuracy so far {best_accuracy}")
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("-" * 10)
    
    train_loss, train_acc = train_epoch(
        model,
        train_loader,
        optimizer,
        scheduler,
        device
    )
    
    print(f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}")
    
    val_loss, val_acc = eval_model(
        model,
        val_loader,
        device
    )
    
    print(f"Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}")
    
    # saving every model with generic timestamp filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    epoch_filename = f"EPOCH_{epoch+1}_{timestamp}.bin"
    torch.save(model.state_dict(), epoch_filename)
    print(f"Saved {epoch_filename}")

    # saving the best model separately
    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model.bin')
        best_accuracy = val_acc