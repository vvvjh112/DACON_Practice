## 전처리는 baseline을 바탕으로 랜덤포레스트 및 gridSearchCV 사용해서 점수 올려볼 계획

import pandas as pd
import numpy as np
import os
import glob
import random

import warnings
warnings.filterwarnings("ignore")

#출력 설정
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

train = pd.read_csv('train/train.csv')
submission = pd.read_csv('sample_submission.csv')


def preprocess_data(data, is_train=True):
    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]

    if is_train == True:

        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
        temp['Target2'] = temp['TARGET'].shift(-48 * 2).fillna(method='ffill')
        temp = temp.dropna()

        return temp.iloc[:-96]

    elif is_train == False:

        temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]

        return temp.iloc[-48:, :]


df_train = preprocess_data(train)

df_test = []

for i in range(81):
    file_path = 'test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)

X_test = pd.concat(df_test)


#학습 데이터 df_train / 테스트 데이터 X_test

from sklearn import ensemble

N_ESTIMATORS = 1000
rf = ensemble.RandomForestRegressor(n_estimators=N_ESTIMATORS,
                                    max_features=1, random_state=0,
                                    max_depth = 5,
                                    verbose=True,
                                    n_jobs=-1)
rf.fit(df_train, X_test)

rf_preds = []
for estimator in rf.estimators_:
    rf_preds.append(estimator.predict(test))
rf_preds = np.array(rf_preds)

for i, q in enumerate(np.arange(0.1, 1, 0.1)):
    y_pred = np.percentile(rf_preds, q * 100, axis=0)
    submission.iloc[:, i] = np.ravel(y_pred)