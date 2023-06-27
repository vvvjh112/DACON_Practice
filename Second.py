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



