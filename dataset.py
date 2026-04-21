import torch
import pandas as pd
import numpy as np
import transformers
import torchvision
from torchvision import transforms
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

import torch.nn.functional as F
from transformers import BertModel
import random
import time
import os
import re

# 预处理
def text_preprocessing(text):
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class FakeNewsDataset(Dataset):

    def __init__(self, df, root_dir, image_transform, tokenizer, MAX_LEN):
        """
        参数:
            csv_file (string):包含文本和图像名称的csv文件的路径
            root_dir (string):目录
            transform(可选):图像变换
        """
        self.csv_data = df
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.tokenizer_bert = tokenizer
        self.MAX_LEN = MAX_LEN

    def __len__(self):
        return self.csv_data.shape[0]
    
    # 定义BERT文本预处理函数
    def pre_processing_BERT(self, sent):

        # 创建空列表储存输出
        input_ids = []
        attention_mask = []

        # 使用BERT分词器对文本进行处理
        encoded_sent = self.tokenizer_bert.encode_plus(
            text=text_preprocessing(sent),  # 预处理
            add_special_tokens=True,        # `[CLS]`&`[SEP]`
            max_length=self.MAX_LEN,        # 截断/填充的最大长度
            padding='max_length',           # 句子填充最大长度
            return_attention_mask=True,     # 返回attention mask
            truncation=True                 # 截断文本
            )
        
        # 获取处理后的文本
        input_ids = encoded_sent.get('input_ids')
        attention_mask = encoded_sent.get('attention_mask')
        
        # 将处理后的文本转换为tensor
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)

        # 返回处理后的tensor
        return input_ids, attention_mask
     
        
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # 获取图像路径并打开图像
        img_name = self.root_dir + self.csv_data['image_id'][idx] + '.jpg'
        image = Image.open(img_name).convert("RGB")
        # 对图像进行变换
        image = self.image_transform(image)

        # 获取文本
        text = self.csv_data['post_text'][idx]

        # 对文本进行BERT预处理
        tensor_input_id, tensor_input_mask = self.pre_processing_BERT(text)

        # 获取标签
        label = self.csv_data['label'][idx]

        # 如果标签为fake则转换为1，否则为0
        if label == 'fake':
            label = '1'
        else:
            label = '0'
        label = int(label)
        
        # 将标签转换为tensor
        label = torch.tensor(label)

        # 将图像、BERT处理后的文本和标签组成dictionary并返回该dictionary
        sample = {
                  'image_id'  :  image, 
                  'BERT_ip'   : [tensor_input_id, tensor_input_mask],
                  'label'     :  label
                  }

        return sample