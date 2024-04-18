import random
import pandas as pd
import numpy as np
import os
import re
import glob
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore')

def seed_everything(seed):
    random.seed(seed) ##random module의 시드 고정
    os.environ['PYTHONHASHSEED'] = str(seed) #해시 함수의 랜덤성 제어, 자료구조 실행할 때 동일한 순서 고정
    np.random.seed(seed) #numpy 랜덤 숫자 일정
    torch.manual_seed(seed) # torch라이브러리에서 cpu 텐서 생성 랜덤 시드 고정
    torch.cuda.manual_seed(seed) # cuda의 gpu텐서에 대한 시드 고정
    torch.backends.cudnn.deterministic = True # 백엔드가 결정적 알고리즘만 사용하도록 고정
    torch.backends.cudnn.benchmark = True # CuDNN이 여러 내부 휴리스틱을 사용하여 가장 빠른 알고리즘 동적으로 찾도록 설정
class CustomDataset(Dataset):
    def __init__(self, df, path_col,  mode='train'):
        self.df = df
        self.path_col = path_col
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.mode == 'train':
            row = self.df.iloc[idx]
            image = read_image(row[self.path_col])/256.
            label = row['class']
            data = {
                'image':image,
                'label':label
            }
            return data
        elif self.mode == 'val':
            row = self.df.iloc[idx]
            image = read_image(row[self.path_col])/256.
            label = row['class']
            data = {
                'image':image,
                'label':label
            }
            return data
        elif self.mode == 'inference':
            row = self.df.iloc[idx]
            image = read_image(row[self.path_col])/256.
            data = {
                'image':image,
            }
            return data

    def train_transform(self, image):
        pass

class CustomCollateFn:
    def __init__(self, transform, mode):
        self.mode = mode
        self.transform = transform

    def __call__(self, batch):
        if self.mode=='train':
            pixel_values = torch.stack([self.transform(data['image']) for data in batch])
            label = torch.LongTensor([data['label'] for data in batch])
            return {
                'pixel_values':pixel_values,
                'label':label,
            }
        elif self.mode=='val':
            pixel_values = torch.stack([self.transform(data['image']) for data in batch])
            label = torch.LongTensor([data['label'] for data in batch])
            return {
                'pixel_values':pixel_values,
                'label':label,
            }
        elif self.mode=='inference':
            pixel_values = torch.stack([self.transform(data['image']) for data in batch])
            return {
                'pixel_values':pixel_values,
            }

class CustomModel(nn.Module):
    def __init__(self, model):
        super(CustomModel, self).__init__()
        self.model = model
        self.clf = nn.Sequential(
            nn.Tanh(),
            nn.LazyLinear(25),
        )

#     @torch.compile
    def forward(self, x, label=None):
        x = self.model(x).pooler_output
        x = self.clf(x)
        loss = None
        if label is not None:
            loss = nn.CrossEntropyLoss()(x, label)
        probs = nn.LogSoftmax(dim=-1)(x)
        return probs, loss

class LitCustomModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = CustomModel(model)
        self.validation_step_output = []

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-5)
        return opt

    def training_step(self, batch, batch_idx=None):
        x = batch['pixel_values']
        label = batch['label']
        probs, loss = self.model(x, label)
        self.log(f"train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx=None):
        x = batch['pixel_values']
        label = batch['label']
        probs, loss = self.model(x, label)
        self.validation_step_output.append([probs,label])
        return loss

    def predict_step(self, batch, batch_idx=None):
        x = batch['pixel_values']
        probs, _ = self.model(x)
        return probs

    def validation_epoch_end(self, step_output):
        pred = torch.cat([x for x, _ in self.validation_step_output]).cpu().detach().numpy().argmax(1)
        label = torch.cat([label for _, label in self.validation_step_output]).cpu().detach().numpy()
        score = f1_score(label,pred, average='macro')
        self.log("val_score", score)
        self.validation_step_output.clear()
        return score