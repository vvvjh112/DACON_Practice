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


from sklearn.model_selection import train_test_split
X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -2], test_size=0.3, random_state=0)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -1], test_size=0.3, random_state=0)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

from lightgbm import LGBMRegressor


# Get the model and the predictions in (a) - (b)
def LGBM(q, X_train, Y_train, X_valid, Y_valid, X_test):
    # (a) Modeling
    model = LGBMRegressor(objective='quantile', alpha=q,
                          n_estimators=10000, bagging_fraction=0.7, learning_rate=0.027, subsample=0.7)

    model.fit(X_train, Y_train, eval_metric=['quantile'],
              eval_set=[(X_valid, Y_valid)], early_stopping_rounds=300, verbose=500)

    # (b) Predictions
    pred = pd.Series(model.predict(X_test).round(2))
    return pred, model


def train_data(X_train, Y_train, X_valid, Y_valid, X_test):
    LGBM_models = []
    LGBM_actual_pred = pd.DataFrame()

    for q in quantiles:
        print(q)
        pred, model = LGBM(q, X_train, Y_train, X_valid, Y_valid, X_test)
        LGBM_models.append(model)
        LGBM_actual_pred = pd.concat([LGBM_actual_pred, pred], axis=1)

    LGBM_actual_pred.columns = quantiles

    return LGBM_models, LGBM_actual_pred

#1일 뒤
models_1, results_1 = train_data(X_train_1, Y_train_1, X_valid_1, Y_valid_1, X_test)

#2일 뒤
models_2, results_2 = train_data(X_train_2, Y_train_2, X_valid_2, Y_valid_2, X_test)

submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2.sort_index().values

submission.to_csv('baseline.csv', index=False)