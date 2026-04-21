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


df_train = pd.read_csv("./data/twitter/train_posts_clean.csv")
df_test = pd.read_csv("./data/twitter/test_posts.csv")

if torch.cuda.is_available():       
    device = torch.device("cuda")
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
# 图像转换
image_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

# 实例化 BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# 固定最大长度
MAX_LEN = 500
data_dir = "./data/twitter/"

# 读取数据
transformed_dataset_train = FakeNewsDataset(df_train, data_dir+"images_train/", image_transform, tokenizer, MAX_LEN)

transformed_dataset_val = FakeNewsDataset(df_test, data_dir+"images_test/", image_transform, tokenizer, MAX_LEN)

train_dataloader = DataLoader(transformed_dataset_train, batch_size=8,
                        shuffle=True, num_workers=0)

val_dataloader = DataLoader(transformed_dataset_val, batch_size=8,
                        shuffle=True, num_workers=0)


# 损失
loss_fn = nn.BCELoss()

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

with open("./config/config.json",'r',encoding='utf-8') as f:
    parameter_dict_model= json.load(f)

with open("./config/config_opt.json",'r',encoding='utf-8') as f:
    parameter_dict_opt= json.load(f)


# 设置随机种子
set_seed(7)

final_model = Text_Concat_Vision(parameter_dict_model)

final_model = final_model.to(device) 

# 优化器
optimizer = AdamW(final_model.parameters(),
                  lr=parameter_dict_opt['l_r'],
                  eps=parameter_dict_opt['eps'])

# training steps总数
total_steps = len(train_dataloader) * 50

# 学习率衰减
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0, # 默认值
                                            num_training_steps=total_steps)


train(model=final_model,loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,train_dataloader=train_dataloader, val_dataloader=val_dataloader,
      epochs=1, evaluation=True,device=device,save_best=True)