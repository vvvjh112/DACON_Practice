import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


train_lst= ['index', '송하인_격자공간고유번호', '수하인_격자공간고유번호', '물품_카테고리', '운송장_건수']
test_lst = ['index', '송하인_격자공간고유번호', '수하인_격자공간고유번호', '물품_카테고리']

print(train.head())


train = train.drop('index',axis = 1)
test = test.drop('index',axis = 1)

import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# train.groupby(['물품_카테고리']).sum('운송장_건수').plot.bar(y = '운송장_건수')
# plt.show()

#결측값
print(train.isna().sum())
print(test.isna().sum())
#결측값 없음

train1 = train.copy()
test1 = test.copy()
from sklearn.preprocessing import *
train1['물품_카테고리'] = LabelEncoder().fit_transform(train1['물품_카테고리'])
test1['물품_카테고리'] = LabelEncoder().fit_transform(test1['물품_카테고리'])
cate_lst = ['물품_카테고리', '송하인_격자공간고유번호', '수하인_격자공간고유번호']

for i in cate_lst:
    train[i] = LabelEncoder().fit_transform(train[i])
    test[i] = LabelEncoder().fit_transform(test[i])
    train[i] = train[i].astype('category')
    test[i] = test[i].astype('category')


from sklearn.metrics import *
from sklearn.model_selection import *
from xgboost import XGBRegressor
from sklearn.ensemble import *
from sklearn.linear_model import *
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor

trainX = train.drop('운송장_건수',axis = 1)
trainY = train['운송장_건수']

train1X = train1.drop('운송장_건수',axis = 1)
train1Y = train1['운송장_건수']

model1 = XGBRegressor()
model2 = LGBMRegressor(metric = 'mse')
model3 = RandomForestRegressor()
model4 = LinearRegression()
model5 = DecisionTreeRegressor()


# param1 = {'max_depth':[3,4,5], 'n_estimators':[100,300,500], 'eta':[0.01,0.02,0.05,0.1]}
# param2 = {'max_depth':[3,4,5],'num_leaves' : [30,40,50]}
# param3 = {'max_depth' : ['none',3,4,5], 'n_estimators' : [100,300,500,600]}
#
#
# k_fold = KFold(n_splits=9)
# grid1 = GridSearchCV(model1,param1,n_jobs=-1,cv =k_fold,scoring='neg_mean_squared_error')
# grid2 = GridSearchCV(model2,param2,n_jobs=-1,cv =k_fold,scoring='neg_mean_squared_error')
# grid3 = GridSearchCV(model3,param3,n_jobs=-1,cv =k_fold,scoring='neg_mean_squared_error')
#
# print(train1.info())
# grid1.fit(train1X,train1Y)
# grid2.fit(trainX,trainY)
# grid3.fit(trainX,trainY)
#
#
#
# from math import sqrt
# print('best params:', grid1.best_params_)
# print('best estimator:', grid1.best_estimator_)
# print('best rmse :', sqrt(-(grid1.best_score_)))
#
# print('best params:', grid2.best_params_)
# print('best estimator:', grid2.best_estimator_)
# print('best rmse :', sqrt(-(grid2.best_score_)))
#
# print('best params:', grid3.best_params_)
# print('best estimator:', grid3.best_estimator_)
# print('best rmse :', sqrt(-(grid3.best_score_)))

model = XGBRegressor(eta = 0.05, max_depth = 5, n_estimators = 300)
x = train1.drop('운송장_건수',axis = 1)
y = train1['운송장_건수']

trX,teX,trY,teY = train_test_split(x,y,test_size=0.2,random_state=200)
model.fit(trX,trY)
pred = model.predict(teX)

print(r2_score(teY,pred))
print(mean_squared_error(teY,pred,squared=False))



