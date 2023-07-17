import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, VotingRegressor
import numpy as np

df = pd.read_csv('./train.csv') # train 데이터 path

X_train = df.drop(['ID','착과량(int)'], axis=1)
y_train = df['착과량(int)']


col_saesoon = X_train.iloc[:, 4:93]
col_yeoprok = X_train.iloc[:, 93:]

saesoon_mean = pd.DataFrame()
for i in col_saesoon.iterrows():
    temp = pd.DataFrame()
    for o in range(5, len(i[1]), 5):
        temp = pd.concat([temp, pd.Series(i[1][o-5:o].mean())], ignore_index=True, axis=1)
    temp = pd.concat([temp, pd.Series(i[1][85:].mean())], ignore_index=True, axis=1)
    pd.concat([saesoon_mean,temp])


saesoon_mean = saesoon_mean.reset_index(drop=True)

yeoprok_mean = pd.DataFrame()
for i in col_yeoprok.iterrows():
    temp = pd.DataFrame()
    for o in range(5, len(i[1]), 5):
        temp = pd.concat([temp, pd.Series(i[1][o-5:o].mean())], ignore_index=True, axis=1)
    temp = pd.concat([temp, pd.Series(i[1][85:].mean())], ignore_index=True, axis=1)
    # yeoprok_mean = yeoprok_mean.append(temp)
    pd.concat([yeoprok_mean,temp])

yeoprok_mean = yeoprok_mean.reset_index(drop=True)
yeoprok_mean = yeoprok_mean.div(X_train['수고(m)'], axis=0)

X_train_new = pd.concat([X_train.iloc[:, :4], saesoon_mean, yeoprok_mean], ignore_index=True, axis=1)
X_train_new['cntzero'] = col_saesoon.apply(lambda x: x[x.isin([0.000])].count(), axis=1)

print(X_train_new.columns)