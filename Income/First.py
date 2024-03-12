import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import *
from sklearn.preprocessing import StandardScaler,LabelEncoder,TargetEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# row 생략 없이 출력
pd.set_option('display.max_rows', None)
# col 생략 없이 출력
pd.set_option('display.max_columns', None)

train = pd.read_csv('dataset/train.csv').drop('ID',axis = 1)
test = pd.read_csv('dataset/test.csv').drop('ID',axis = 1)
submission = pd.read_csv('dataset/sample_submission.csv')

print(train.isna().sum())
print(test.isna().sum())
# 0 1

# 비슷한 조건에서 train 데이터셋에 해당 값이 빈도 수가 제일 많음
test = test.fillna('Child 18+ never marr Not in a subfamily')


train = train.loc[train['Gains'] < 99999]

logscale_columns = ['Gains', 'Losses', 'Dividends', 'Income']
numeric_columns = train.select_dtypes(include=['int64', 'float64']).columns
category_columns = train.select_dtypes(exclude=['int64', 'float64']).columns
standardscale_columns = [x for x in numeric_columns if x not in logscale_columns]

#스케일링
train[logscale_columns] = np.log1p(train[logscale_columns])
test[logscale_columns] = np.log1p(test[logscale_columns])

ss = StandardScaler()
train[standardscale_columns] = ss.fit_transform(train[standardscale_columns])
test[standardscale_columns] = ss.transform(test[standardscale_columns])