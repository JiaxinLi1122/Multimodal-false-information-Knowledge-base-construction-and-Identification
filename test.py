import torch
import pandas as pd
import numpy as np
import transformers
import torchvision
from torchvision import models, transforms
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import time
import os
import re
import math
import json
from mult_models import *
from dataset import *

start = time.time()
df_test = pd.read_csv("./data/twitter/test_posts.csv")

if torch.cuda.is_available():       
    device = torch.device("cuda")
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

image_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

transformed_dataset_val = FakeNewsDataset(df_test, "./data/twitter/images_test/", image_transform, tokenizer, 500)

val_dataloader = DataLoader(transformed_dataset_val, batch_size=8,
                        shuffle=True, num_workers=0)


with open("./config/config.json",'r',encoding='utf-8') as f:
    parameter_dict_model= json.load(f)



model = Text_Concat_Vision(parameter_dict_model)
model.eval()
model.load_state_dict(torch.load('./saved_models/best_model.pt'),False)
model = model.to(device) 
loss_fn = nn.BCELoss()


val_accuracy = []
val_loss = []
predict = []
    # 每个损失之后
for batch in val_dataloader:
    img_ip , text_ip, label = batch["image_id"], batch["BERT_ip"], batch['label']
            
    b_input_ids, b_attn_mask = tuple(t.to(device) for t in text_ip)

    imgs_ip = img_ip.to(device)

    b_labels = label.to(device)

        # 计算logits
    with torch.no_grad():
        logits = model(text=[b_input_ids, b_attn_mask], image=imgs_ip)
        b_labels=b_labels.to(torch.float32)
            
        # 计算loss
    loss = loss_fn(logits, b_labels)
    val_loss.append(loss.item())
    predict.append(logits)

    logits[logits<0.5] = 0
    logits[logits>=0.5] = 1

        # 计算准确率
    accuracy = (logits == b_labels).cpu().numpy().mean() * 100
    val_accuracy.append(accuracy)

    # 计算平均准确率和验证集损失
val_loss = np.mean(val_loss)
val_accuracy = np.mean(val_accuracy)
end = time.time()
print("Model accuracy:",val_accuracy)
print("Test time: ",end-start)
