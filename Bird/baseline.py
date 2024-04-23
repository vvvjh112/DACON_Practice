#!/usr/bin/env python
# coding: utf-8

# ## Import

# In[1]:


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

# In[2]:
USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)