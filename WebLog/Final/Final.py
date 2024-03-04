import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import joblib
import torch
import random

# sessionID : 세션 ID
# userID : 사용자 ID
# TARGET : 세션에서 발생한 총 조회수
# browser : 사용된 브라우저
# OS : 사용된 기기의 운영체제
# device : 사용된 기기
# new : 첫 방문 여부 (0: 첫 방문 아님, 1: 첫 방문)
# quality : 세션의 질 (거래 성사를 기준으로 측정된 값, 범위: 1~100)
# duration : 총 세션 시간 (단위: 초)
# bounced : 이탈 여부 (0: 이탈하지 않음, 1: 이탈함)
# transaction : 세션 내에서 발생의 거래의 수
# transaction_revenue : 총 거래 수익
# continent : 세션이 발생한 대륙
# subcontinent : 세션이 발생한 하위 대륙
# country : 세션이 발생한 국가
# traffic_source : 트래픽이 발생한 소스
# traffic_medium : 트래픽 소스의 매체
# keyword : 트래픽 소스의 키워드, 일반적으로 traffic_medium이 organic, cpc인 경우에 설정
# referral_path : traffic_medium이 referral인 경우 설정되는 경로

# RMSE


#plt 한글출력
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# 한글 폰트 경로 지정
font_path = '/Library/Fonts/AppleGothic.ttf'  # 예시로 AppleGothic 폰트 사용

# 한글 폰트 설정
# plt.rcParams['font.family'] = 'AppleGothic'

# row 생략 없이 출력
pd.set_option('display.max_rows', None)
# col 생략 없이 출력
pd.set_option('display.max_columns', None)

train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')
submission = pd.read_csv('Data/sample_submission.csv')


print(train.info())


#파생변수
train['분당수익'] = train['transaction_revenue'] / (train['duration'].replace(0, 1) / 60)
test['분당수익'] = test['transaction_revenue'] / (test['duration'].replace(0, 1) / 60)

train['분당거래'] = train['transaction'] / (train['duration'].replace(0, 1) / 60)
test['분당거래'] = test['transaction'] / (test['duration'].replace(0, 1) / 60)

train['평균거래액'] = train['transaction_revenue'] / train['transaction'].replace(0, 1)
test['평균거래액'] = test['transaction_revenue'] / test['transaction'].replace(0, 1)

train['퀄리티지수'] = train['quality'] * train['duration']
test['퀄리티지수'] = test['quality'] * test['duration']


#결측값
train.fillna('-',inplace=True)
test.fillna('-',inplace=True)
# print(train.isna().sum())
print(test.isna().sum())


from sklearn.model_selection import *
from sklearn.preprocessing import *
from category_encoders import TargetEncoder

#인코딩
train['new'] = train['new'].astype(object)
train['bounced'] = train['bounced'].astype(object)
test['new'] = test['new'].astype(object)
test['bounced'] = test['bounced'].astype(object)

category_features = train.select_dtypes(include="object").columns.tolist()
mask = train[category_features].nunique()<=10
category_enc = train[category_features].nunique().loc[mask].index.tolist()
target_enc = train[category_features].nunique().loc[-mask].index.tolist()

for i in category_enc:
    train[i] = train[i].astype('category')
    test[i] = test[i].astype('category')

for i in target_enc:
    te = TargetEncoder(cols = i)
    train[i] = te.fit_transform(train[i], train['TARGET'])
    test[i] = te.transform(test[i])

#합계는 스케일링 하자
mm = StandardScaler()
numeric = train.select_dtypes(exclude=["object", "category"]).drop(['TARGET'], axis=1).columns.tolist()
train[numeric] = mm.fit_transform(train[numeric])
test[numeric] = mm.transform(test[numeric])

# 숫자가 커서 스케일링 후 파생변수 생성
train['거래확률'] = train['quality'] / train['transaction']
test['거래확률'] = test['quality'] / test['transaction']



#데이터 분리
x = train.drop(['sessionID','userID','TARGET'],axis=1)
y = train['TARGET']

test= test.drop(['sessionID','userID'],axis =1)

trainX, testX, trainY, testY = train_test_split(x,y,test_size=0.2,random_state=2000)

print(trainX.columns)

#Catboost / LGBM
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import *
import model_tuned as mt


#LGBM
# 옵튜나
# lgbm , lgbm_study = mt.lgbm_modeling(trainX,trainY,testX,testY)
# print(lgbm.feature_importances_)
# pred = lgbm.predict(testX)
# print("점수 ", mean_squared_error(testY,pred,squared=False))
# best_params = lgbm_study.best_params
# print("최적 하이퍼파라미터:", best_params)

hp = {'num_leaves': 355, 'colsample_bytree': 0.8474389094719293, 'reg_alpha': 0.7592006668666332, 'reg_lambda': 7.225166341237797, 'max_depth': 12,
      'learning_rate': 0.003945880517624002, 'n_estimators': 1417, 'min_child_samples': 28, 'subsample': 0.6070270926393032}


#Cat
# cat, cat_study = mt.cat_modeling(trainX,trainY,testX,testY,category_enc)
# print("점수는 : ",mean_squared_error(testY,cat.predict(testX),squared=False))
# print("베스트 파라미터~!~",cat_study.best_params)


import os
# 모델 저장할 폴더 생성
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2000)
lgbm_scores = []
for i, (tri , vai) in tqdm(enumerate(cv.split(x, y)), total=10):
    x_train = x.iloc[tri]
    y_train = y.iloc[tri]

    x_valid = x.iloc[vai]
    y_valid = y.iloc[vai]

    lgbm_model = LGBMRegressor(**hp, objective='rmse', metric='rmse', verbosity=-1, n_jobs=-1)
    lgbm_model.fit(x_train, y_train)

    pred = lgbm_model.predict(x_valid)
    score = mean_squared_error(y_valid,pred,squared=False)
    lgbm_scores.append(score)
    # 모델 저장
    joblib.dump(lgbm_model, f"model/lgbm_model/{i}_lgbm_model.pkl")
#
#
lgbm_pred_list = []
lgbm_score_list = []
for i in range(10):
    model = joblib.load(f"model/lgbm_model/{i}_lgbm_model.pkl")
    pred = model.predict(testX)
    lgbm_score_list.append(mean_squared_error(testY,pred,squared=False))
    pred = model.predict(test)
    lgbm_pred_list.append(pred)
    submission['lgbm'+str(i)] = pred

print("LGBM pred 리스트 : ", lgbm_score_list)

#cat
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2000)
cat_scores = []
for i, (tri , vai) in tqdm(enumerate(cv.split(x, y)), total=10):
    x_train = x.iloc[tri]
    y_train = y.iloc[tri]

    x_valid = x.iloc[vai]
    y_valid = y.iloc[vai]
    chp = {'od_wait': 1968, 'learning_rate': 0.1434520185514101, 'reg_lambda': 38.281084288725815, 'random_strength': 12.867710796532581,
    'depth': 12, 'min_data_in_leaf': 24,
    'leaf_estimation_iterations': 7, 'bagging_temperature': 0.45930339623048033}
    cat_model = CatBoostRegressor(**chp,random_state= 2000,eval_metric='RMSE', cat_features=category_enc,task_type='GPU')
    cat_model.fit(x_train, y_train)
    pred = cat_model.predict(x_valid)
    score = mean_squared_error(y_valid,pred,squared=False)
    cat_scores.append(score)
    # 모델 저장
    joblib.dump(cat_model, f"model/cat_model/{i}_cat_model.pkl")

cat_pred_list = []
cat_score_list = []
for i in range(10):
    model = joblib.load(f"model/cat_model/{i}_cat_model.pkl")
    pred = model.predict(testX)
    cat_score_list.append(mean_squared_error(testY,pred,squared=False))
    pred = model.predict(test)
    cat_pred_list.append(pred)
    submission['cat'+str(i)] = pred

print("cat pred 리스트 : ",cat_score_list)


#앙상블
submission['TARGET1'] = 0
submission['TARGET2'] = 0

for i in range(10):
    submission['TARGET1'] = submission['TARGET1']+submission['cat'+str(i)]
    submission['TARGET2'] = submission['TARGET2'] + submission['lgbm' + str(i)]

submission['TARGET1'] = submission['TARGET1']/10
submission['TARGET2'] = submission['TARGET2']/10
submission['TARGET'] = submission['TARGET1']*0.5 + submission['TARGET2']*0.5
submission = submission[['sessionID','TARGET']]


#TARGET값 0보다 작은거 0으로 보정하기
import datetime
title = 'ENSEMBLE'+str(datetime.datetime.now().month)+'_'+str(datetime.datetime.now().day)+'_'+str(datetime.datetime.now().hour)+'_'+str(datetime.datetime.now().minute)+'.csv'
submission.loc[submission['TARGET'] < 0.0, 'TARGET'] = 0.0
submission.to_csv(title,index=False)