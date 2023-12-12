import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


train_lst= ['index', '송하인_격자공간고유번호', '수하인_격자공간고유번호', '물품_카테고리', '운송장_건수']
test_lst = ['index', '송하인_격자공간고유번호', '수하인_격자공간고유번호', '물품_카테고리']

print(train.head())


train = train.drop('index',axis = 1)
test = test.drop('index',axis = 1)

from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.metrics import *

cate_lst = ['물품_카테고리', '송하인_격자공간고유번호', '수하인_격자공간고유번호']

# for i in cate_lst:
#     train[i] = LabelEncoder().fit_transform(train[i])
#     test[i] = LabelEncoder().fit_transform(test[i])
#     train[i] = train[i].astype('category')
#     test[i] = test[i].astype('category')

print(train.info())

import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

train.groupby(['물품_카테고리']).sum().plot.bar(y = '운송장_건수')
plt.show()


from xgboost import XGBRegressor