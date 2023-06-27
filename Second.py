import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

pd.set_option('mode.chained_assignment',  None) # <==== 경고를 끈다

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('submission.csv')

columns = ['hour', 'hour_bef_temperature', 'hour_bef_precipitation', 'hour_bef_windspeed', 'hour_bef_humidity']     #요인
X_train = train[columns]
y_train = train['count']        # 변화할 값
X_test = test[columns]


X_train['hour_bef_temperature'] = X_train['hour_bef_temperature'].fillna(value = X_train['hour_bef_temperature'].mean())
X_train['hour_bef_precipitation'] = X_train['hour_bef_precipitation'].fillna(value = X_train['hour_bef_temperature'].mean())
X_train['hour_bef_windspeed'] = X_train['hour_bef_windspeed'].fillna(value = X_train['hour_bef_temperature'].mean())
X_train['hour_bef_humidity'] = X_train['hour_bef_humidity'].fillna(value = X_train['hour_bef_temperature'].mean())

X_test['hour_bef_temperature'] = X_test['hour_bef_temperature'].fillna(value = X_test['hour_bef_temperature'].mean())
X_test['hour_bef_precipitation'] = X_test['hour_bef_precipitation'].fillna(value = X_test['hour_bef_temperature'].mean())
X_test['hour_bef_windspeed'] = X_test['hour_bef_windspeed'].fillna(value = X_test['hour_bef_temperature'].mean())
X_test['hour_bef_humidity'] = X_test['hour_bef_humidity'].fillna(value = X_test['hour_bef_temperature'].mean())

# 모델 객체 생성
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor

import numpy as np

model_dict = {'DT':DecisionTreeRegressor(),
             'RF':RandomForestRegressor(),
             'LGB':lgb.LGBMRegressor(),
             'XGB':xgb.XGBRegressor(),
             'KNN':KNeighborsRegressor()}

#시각화는 생략.


# K-FOLD
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# k_fold = KFold(n_splits=5, shuffle= True, random_state=10)
#
# score = {}
#
# for model_name in model_dict.keys():
#     model = model_dict[model_name]
#
#     score[model_name] = np.mean(
#         cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', n_jobs=-1, cv=k_fold))*-1
#
#
# pd.Series(score).plot(kind = 'bar')

# plt.show()

# 예측
# LGB = lgb.LGBMRegressor()
# LGB.fit(X_train, y_train)
#
#
# LGB_predict = LGB.predict(X_test)
#
# # submission 파일에 예측값 대입 및 내보내기
# submission['count'] = LGB_predict
# submission.to_csv('LGB.csv', index=False)



from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#결측치 처리
train_isna_sum = train.isna().sum()
na_columns = train_isna_sum[train_isna_sum != 0].index

test_isna_sum = test.isna().sum()
na_columns1 = test_isna_sum[test_isna_sum != 0].index

def fill_bicycle_na(df, column):
    df[column] = df[column].fillna(value=df[column].mean())

for col in na_columns:
    fill_bicycle_na(train, col)

for col in na_columns1:
    fill_bicycle_na(test, col)

fill_bicycle_na(train, 'hour_bef_precipitation')
# print(X_train.isna().sum())
#

# #KNN
# model = KNeighborsRegressor(n_jobs = -1)
#
# #대여시각과 기온을 요인으로 대여량 예측
# column = ['hour', 'hour_bef_temperature']
# X_train = train[column]
# y_train = train['count']
# X_test = test[column]
#
# #이웃 수를 여러가지로 평가를 위해
# model_5 = KNeighborsRegressor(n_jobs = -1, n_neighbors = 5)
# model_7 = KNeighborsRegressor(n_jobs = -1, n_neighbors = 7)
# model_9 = KNeighborsRegressor(n_jobs = -1, n_neighbors = 9)
#
kfold = KFold(n_splits = 5, shuffle = True, random_state = 10)
#
# print(np.mean(cross_val_score(model_5, X_train, y_train, cv = kfold, scoring = 'neg_mean_squared_error')))
# #-2154.0119004848657
# print(np.mean(cross_val_score(model_7, X_train, y_train, cv = kfold, scoring = 'neg_mean_squared_error')))
# #-2053.407982413414
# print(np.mean(cross_val_score(model_9, X_train, y_train, cv = kfold, scoring = 'neg_mean_squared_error')))
# #-1987.690754979273
#
# # model_9가 오차가 젤 적다
# model_9.fit(X_train, y_train)
# submission['count'] = model_9.predict(X_test)
# submission.to_csv('knn_9.csv', index = False)
#
#
# model.fit(X_train, y_train)
# submission['count'] = model.predict(X_test)
# submission.to_csv('knn_5.csv', index = False)


#랜덤포레스트
from sklearn.model_selection import GridSearchCV
model = RandomForestRegressor()
X_train = train.drop(['id', 'count'], axis = 1)
y_train = train['count']
X_test = test.drop('id', axis = 1)

param = {'min_samples_split': [30, 50, 70],
        'max_depth': [5, 6, 7],
        'n_estimators': [50, 150, 250]}

gs = GridSearchCV(estimator=model, param_grid=param, scoring = 'neg_mean_squared_error', cv = 3)

# gs.fit(X_train, y_train)
#
# submission['count'] = gs.predict(X_test)
# submission.to_csv('gridsearch.csv', index = False)

# print(gs.best_params_)
#{'max_depth': 7, 'min_samples_split': 30, 'n_estimators': 150}
models = {}
models["model_100"] = RandomForestRegressor(n_estimators=100, n_jobs = -1, random_state=10)
models["model_200"] = RandomForestRegressor(n_estimators=200, n_jobs = -1, random_state=10)
models["model_300"] = RandomForestRegressor(n_estimators=300, n_jobs = -1, random_state=10)
models["model_500"] = RandomForestRegressor(n_estimators=500, n_jobs = -1, random_state=10)


pd.DataFrame({'n_estimators':[100, 200, 300, 500], 'score':[-1546, -1537, -1529, -1521]}).plot('n_estimators', 'score')

# for i in models.keys():
#     score = cross_val_score(models[i], X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
#     print(np.mean(score))
#-1546.8865621858963
#-1537.2335820896296
#-1529.93820386781
#-1521.7993785402248

#분류기가 500일때 젤 좋음
models2={}
models2["model_sam_30"] = RandomForestRegressor(n_estimators=500, n_jobs = -1, random_state=10, min_samples_split=30)
models2["model_sam_50"] = RandomForestRegressor(n_estimators=500, n_jobs = -1, random_state=10, min_samples_split=50)
models2["model_sam_100"] = RandomForestRegressor(n_estimators=500, n_jobs = -1, random_state=10, min_samples_split=100)

# for i in models2.keys():
#     score = cross_val_score(models2[i], X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
#     print(np.mean(score))
# -1647.5599150463909
# -1746.5733867422455
# -2009.23405613022

final_model = RandomForestRegressor(n_estimators=500, n_jobs = -1, random_state=10, min_samples_split=30)
final_model.fit(X_train, y_train)
submission['count'] = final_model.predict(X_test)
submission.to_csv('estimators_500_samples_30.csv', index = False)