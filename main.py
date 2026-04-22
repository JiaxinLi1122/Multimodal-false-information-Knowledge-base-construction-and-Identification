import torch
import pandas as pd
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import random
import os
import json

from mult_models import *
from dataset import *


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- Config (single source of truth) ---
with open("./config/config.json", 'r', encoding='utf-8') as f:
    cfg = json.load(f)

# --- Seed first, before any random operations ---
set_seed(cfg['seed'])

# --- Device ---
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# --- Image transform ---
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Tokenizer ---
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

MAX_LEN   = cfg['max_len']
BATCH_SIZE = cfg['batch_size']
EPOCHS     = cfg['epochs']
data_dir   = "./data/twitter/"

# --- Load raw CSVs ---
df_all_train = pd.read_csv(data_dir + "train_posts_clean.csv")
df_test      = pd.read_csv(data_dir + "test_posts.csv")

# --- Stratified train / val split from training data ---
# Test set is kept completely separate until final evaluation.
df_train, df_val = train_test_split(
    df_all_train,
    test_size=cfg['val_split_ratio'],
    random_state=cfg['seed'],
    stratify=df_all_train['label']
)
df_train = df_train.reset_index(drop=True)
df_val   = df_val.reset_index(drop=True)

print(f"Split sizes — train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")

# --- Datasets ---
train_dataset = FakeNewsDataset(df_train, data_dir + "images_train/", image_transform, tokenizer, MAX_LEN)
val_dataset   = FakeNewsDataset(df_val,   data_dir + "images_train/", image_transform, tokenizer, MAX_LEN)
test_dataset  = FakeNewsDataset(df_test,  data_dir + "images_test/",  image_transform, tokenizer, MAX_LEN)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_dataloader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_dataloader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# --- Model ---
final_model = Text_Concat_Vision(cfg)
final_model = final_model.to(device)

# --- Loss ---
loss_fn = nn.BCELoss()

# --- Optimizer ---
optimizer = AdamW(final_model.parameters(), lr=cfg['l_r'], eps=cfg['eps'])

# --- Scheduler (based on actual epochs, not a hardcoded constant) ---
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# --- Train ---
os.makedirs('./saved_models', exist_ok=True)
MODEL_SAVE_PATH = './saved_models/best_model.pt'

train(
    model=final_model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    epochs=EPOCHS,
    device=device,
    save_best=True,
    model_save_path=MODEL_SAVE_PATH,
    patience=cfg['early_stopping_patience'],
    min_delta=cfg['early_stopping_min_delta']
)

# --- Final evaluation on held-out test set (run only once, after training) ---
print("\n" + "="*60)
print("Final Test Set Evaluation")
print("="*60)
final_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
m = evaluate(final_model, loss_fn, test_dataloader, device)

print(f"Loss:      {m['loss']:.6f}")
print(f"Accuracy:  {m['accuracy']:.2f}%")
print(f"Precision: {m['precision']:.2f}%  (fake class)")
print(f"Recall:    {m['recall']:.2f}%  (fake class)")
print(f"F1-Score:  {m['f1']:.2f}%  (fake class)")

print("\nClassification Report:")
print(classification_report(m['all_labels'], m['all_preds'],
                            target_names=['real (0)', 'fake (1)'], digits=4))

cm = confusion_matrix(m['all_labels'], m['all_preds'])
print("Confusion Matrix (rows=actual, cols=predicted):")
print(f"                  Pred:real  Pred:fake")
print(f"  Actual: real    {cm[0][0]:^9}  {cm[0][1]:^9}")
print(f"  Actual: fake    {cm[1][0]:^9}  {cm[1][1]:^9}")
print(f"\n  True Negatives (TN): {cm[0][0]}  — real correctly identified")
print(f"  False Positives (FP): {cm[0][1]}  — real misclassified as fake")
print(f"  False Negatives (FN): {cm[1][0]}  — fake missed")
print(f"  True Positives (TP): {cm[1][1]}  — fake correctly detected")
