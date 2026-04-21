from importlib_metadata import re
import torch
import numpy as np
import transformers
import torchvision
from torchvision import models, transforms
import torch.nn as nn
from transformers import BertModel
import pandas as pd
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import time
import os
import re
import math

device = torch.device("cuda")

# 文本Bert基本模型
class TextEncoder(nn.Module):

    def __init__(self, text_fc2_out=32, text_fc1_out=2742, dropout_p=0.4, fine_tune_module=False):

        super(TextEncoder, self).__init__()
        
        self.fine_tune_module = fine_tune_module

        # 实例化
        self.bert = BertModel.from_pretrained(
                    'bert-base-uncased',
                    return_dict=True)

        self.text_enc_fc1 = torch.nn.Linear(768, text_fc1_out)

        self.text_enc_fc2 = torch.nn.Linear(text_fc1_out, text_fc2_out)

        self.dropout = nn.Dropout(dropout_p)

        self.fine_tune()
        
    def forward(self, input_ids, attention_mask):
        # 输入BERT
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(
            torch.nn.functional.relu(
                self.text_enc_fc1(out['pooler_output']))
        )    
        
        x = self.dropout(
            torch.nn.functional.relu(
                self.text_enc_fc2(x))
        ) 
        
        return x
    
    def fine_tune(self):
        for p in self.bert.parameters():
            p.requires_grad = self.fine_tune_module
            

# 视觉vgg19预训练模型
class VisionEncoder(nn.Module):
   
    def __init__(self, img_fc1_out=2742, img_fc2_out=32, dropout_p=0.4, fine_tune_module=False):
        super(VisionEncoder, self).__init__()
        
        self.fine_tune_module = fine_tune_module
        
        # 实例化
        vgg = models.vgg19(pretrained=True)
        vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:1])
        
        self.vis_encoder = vgg

        self.vis_enc_fc1 = torch.nn.Linear(4096, img_fc1_out)

        self.vis_enc_fc2 = torch.nn.Linear(img_fc1_out, img_fc2_out)

        self.dropout = nn.Dropout(dropout_p)

        self.fine_tune()
        
    def forward(self, images):
        x = self.vis_encoder(images)

        x = self.dropout(
            torch.nn.functional.relu(
                self.vis_enc_fc1(x))
        )

        x = self.dropout(
            torch.nn.functional.relu(
                self.vis_enc_fc2(x))
        )

        return x
    
    def fine_tune(self):
        for p in self.vis_encoder.parameters():
            p.requires_grad = False

        # 如果进行微调，则只微调卷积块2到4
        for c in list(self.vis_encoder.children())[5:]:
            for p in c.parameters():
                p.requires_grad = self.fine_tune_module

#LanguageAndVisionConcat
class Text_Concat_Vision(torch.nn.Module):

    def __init__(self,
        model_params
    ):
        super(Text_Concat_Vision, self).__init__()
        
        self.text_encoder = TextEncoder(model_params['text_fc2_out'], model_params['text_fc1_out'], model_params['dropout_p'], model_params['fine_tune_text_module'])
        self.vision_encode = VisionEncoder(model_params['img_fc1_out'], model_params['img_fc2_out'], model_params['dropout_p'], model_params['fine_tune_vis_module'])

        self.fusion = torch.nn.Linear(
            in_features=(model_params['text_fc2_out'] + model_params['img_fc2_out']), 
            out_features=model_params['fusion_output_size']
        )
        self.fc = torch.nn.Linear(
            in_features=model_params['fusion_output_size'], 
            out_features=1
        )
        self.dropout = torch.nn.Dropout(model_params['dropout_p'])


    #def forward(self, text, image, label=None):
    def forward(self, text, image, label=None):

        ## text to Bert
        text_features = self.text_encoder(text[0], text[1])
        ## image to vgg
        image_features = self.vision_encode(image)

        ## 连接image & text 
        combined_features = torch.cat(
            [text_features, image_features], dim = 1
        )

        combined_features = self.dropout(combined_features)
        
        fused = self.dropout(
            torch.relu(
            self.fusion(combined_features)
            )
        )
        
        prediction = torch.sigmoid(self.fc(fused))

        prediction = prediction.squeeze(-1)

        prediction = prediction.float()

        return prediction

def train(model, loss_fn, optimizer, scheduler, train_dataloader, val_dataloader=None, epochs=4, evaluation=False, device='cpu', 
        save_best=False):
    
    # 开始 training_loop
    best_acc_val = 0
    print("Start training...\n")
    for epoch_i in range(epochs):

        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*100)

        # 记录时间
        t0_epoch, t0_batch = time.time(), time.time()

        # 重置损失
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # 训练模式
        model.train()

        # batch训练
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1

            img_ip , text_ip, label = batch["image_id"], batch["BERT_ip"], batch['label']
            
            b_input_ids, b_attn_mask = tuple(t.to(device) for t in text_ip)
            
            imgs_ip = img_ip.to(device)
            
            b_labels = label.to(device)

            # 梯度清零
            model.zero_grad()

            # 向前传播
            logits = model(text=[b_input_ids, b_attn_mask], image=imgs_ip)

            # 计算、累计损失
            b_labels=b_labels.to(torch.float32)
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # 反向传播
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 更新参数和学习率
            optimizer.step()
            scheduler.step()


            # 每20个batch打印损失和时间
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch

                # 打印训练情况
                print(f"epoch{epoch_i + 1:^7} | batch{step:^7} | loss{batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | elapsed{time_elapsed:^9.2f}")
                
                # 重置
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # 计算整个训练集的平均损失
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*100)
        if evaluation == True:
            model_save_path='./saved_models/best_model.pt'
            # 在每个epoch训练完成后，测试模型的性能
            val_loss, val_accuracy, predict = evaluate(model, loss_fn, val_dataloader, device)
            
            time_elapsed = time.time() - t0_epoch
            print(f" {epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("predict:",predict)
            print("-"*70)
            
            # 保存best_model
            if save_best: 
                if val_accuracy > best_acc_val:
                    best_acc_val = val_accuracy
                    torch.save(model.state_dict(), model_save_path)
                    
        print("\n")
    
    print("Training complete!")
    
    
    
def evaluate(model, loss_fn, val_dataloader, device):
    model.eval()

    # 记录损失和准确率
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

    return val_loss, val_accuracy,predict