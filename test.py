import torch
import pandas as pd
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import time
import json

from mult_models import Text_Concat_Vision, evaluate
from dataset import FakeNewsDataset


start = time.time()

with open("./config/config.json", 'r', encoding='utf-8') as f:
    cfg = json.load(f)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

df_test = pd.read_csv("./data/twitter/test_posts.csv")
test_dataset = FakeNewsDataset(df_test, "./data/twitter/images_test/",
                               image_transform, tokenizer, cfg['max_len'])
test_dataloader = DataLoader(test_dataset, batch_size=cfg['batch_size'],
                             shuffle=False, num_workers=0)

model = Text_Concat_Vision(cfg)
model.load_state_dict(torch.load('./saved_models/best_model.pt', map_location=device))
model = model.to(device)
loss_fn = nn.BCELoss()

m = evaluate(model, loss_fn, test_dataloader, device)

end = time.time()

print("=" * 60)
print("Test Set Evaluation")
print("=" * 60)
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
print(f"\n  TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}")
print(f"\nTest time: {end - start:.2f}s")
