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

# train = pd.read_csv('Data/train_duration.csv')
# test = pd.read_csv('Data/test_duration.csv')
train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')
submission = pd.read_csv('Data/sample_submission.csv')


print(train.info())

# 사용된 브라우저별 총 거래 수익 - 완료
# 브라우저별 평균 거래
# 국가별 총 거래 수익 및 거래 수
# 브라우저별 이탈율
# 디바이스별
# OS별
# test셋어 없는 데이터들 제외
# 전체 데이터셋의 세션유지시간 비율

#파생변수
train['분당수익'] = train['transaction_revenue'] / (train['duration'].replace(0, 1) / 60)
test['분당수익'] = test['transaction_revenue'] / (test['duration'].replace(0, 1) / 60)

train['분당거래'] = train['transaction'] / (train['duration'].replace(0, 1) / 60)
test['분당거래'] = test['transaction'] / (test['duration'].replace(0, 1) / 60)

train['평균거래액'] = train['transaction_revenue'] / train['transaction'].replace(0, 1)
test['평균거래액'] = test['transaction_revenue'] / test['transaction'].replace(0, 1)

train['퀄리티지수'] = train['quality'] * train['duration']
test['퀄리티지수'] = test['quality'] * test['duration']


# train['거래지수'] = train['transaction'] * train['duration']
# test['거래지수'] = test['transaction'] * test['duration']

# train['거래지수'] = train['transaction'] * train['quality']
# test['거래지수'] = test['transaction'] * test['quality']

#시각화 이전 그룹화
# 키워드 비율 브라우저 비율

group_os = train.groupby(['OS']).mean('TARGET')

group_browser = train.groupby(['browser']).mean('TARGET')[['TARGET']].reset_index('browser')
group_browser = group_browser[~group_browser['browser'].str.startswith(';__CT_JOB_ID__:')]

group_device = train.groupby(['device']).mean('TARGET')

group_new = train.groupby(['new']).mean('TARGET')

group_bounced = train.groupby(['bounced']).mean('TARGET')

group_country = train.groupby(['country']).mean('TARGET')

group_source = train.groupby(['traffic_source']).mean('TARGET')

plt.title('OS별 평균 조회수')
plt.xticks(fontsize = 7, rotation = 45, ha = 'right')
sns.lineplot(x=group_os.index, y= 'TARGET', data=group_os, marker = 'o')
# plt.show()

plt.title('브라우저별 평균 조회수')
plt.xticks(fontsize = 7, rotation = 45, ha = 'right')
sns.lineplot(x='browser', y='TARGET', data=group_browser, marker='o')
# plt.show()

plt.title('디바이스별 평균 조회수')
sns.lineplot(x=group_device.index, y= 'TARGET', data=group_device, marker = 'o')
# plt.show()

plt.title('신규여부별 평균 조회수')
sns.barplot(x=group_new.index, y= 'TARGET', data=group_new)
# plt.show()

plt.title('이탈여부별 평균 조회수')
sns.barplot(x=group_bounced.index, y= 'TARGET', data=group_bounced)
# plt.show()

plt.title('나라별 평균 조회수')
plt.xticks(fontsize = 7, rotation = 45, ha = 'right')
sns.lineplot(x=group_country.index, y= 'TARGET', data=group_country, marker = 'o')
# plt.show()

plt.title('소스별 평균 조회수')
plt.xticks(fontsize = 7, rotation = 45, ha = 'right')
sns.lineplot(x=group_source.index, y= 'TARGET', data=group_source, marker = 'o')
# plt.show()

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
    # train[i] = LabelEncoder().fit_transform(train[i])
    # test[i] = LabelEncoder().fit_transform(test[i])
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

train['거래확률'] = train['quality'] / train['transaction']
test['거래확률'] = test['quality'] / test['transaction']



#데이터 분리
x = train.drop(['sessionID','userID','TARGET'],axis=1)
y = train['TARGET']

test= test.drop(['sessionID','userID'],axis =1)

trainX, testX, trainY, testY = train_test_split(x,y,test_size=0.2,random_state=2000)

print(trainX.columns)

#Catboost / xgboost / LGBM /
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from sklearn.metrics import *
import model_tuned as mt

#pycaret
# mt.compare_model(train.drop(['sessionID','userID'],axis=1),'TARGET')


#LGBM
# 옵튜나
# lgbm , lgbm_study = mt.lgbm_modeling(trainX,trainY,testX,testY)
# lgbm_predict = lgbm.predict(test)
# submission['TARGET'] = lgbm_predict
# lgbm.fit(trainX,trainY)
# print(lgbm.feature_importances_)
# pred = lgbm.predict(testX)
# print("점수 ", mean_squared_error(testY,pred,squared=False))
# best_params = lgbm_study.best_params
# print("최적 하이퍼파라미터:", best_params)

#1차
# hp = {'num_leaves': 343, 'colsample_bytree': 0.8799056412183298, 'reg_alpha': 0.9188535423836559, 'reg_lambda': 0.5661962662592113, 'max_depth': 13, 'learning_rate': 0.0022962143663780715, 'n_estimators': 2574, 'min_child_samples': 38, 'subsample': 0.9721091518154885}
hp = {'num_leaves': 355, 'colsample_bytree': 0.8474389094719293, 'reg_alpha': 0.7592006668666332, 'reg_lambda': 7.225166341237797, 'max_depth': 12, 'learning_rate': 0.003945880517624002, 'n_estimators': 1417, 'min_child_samples': 28, 'subsample': 0.6070270926393032}
# lm = LGBMRegressor(**hp)
# lm.fit(trainX,trainY)
# print(lm.feature_importances_)
# print(trainX.columns)
# pred = lm.predict(testX)
# print("점수 ", mean_squared_error(testY,pred,squared=False))
# print((trainX[numeric.append('TARGET')]).corr())

# pred = lm.predict(test)
# submission['TARGET'] = pred

#Cat
# cat, cat_study = mt.cat_modeling(trainX,trainY,testX,testY,category_enc)
# cat.save_model('cat_optuna.bin')
# chp = {'iterations': 17324, 'od_wait': 1638, 'learning_rate': 0.11197043807190474, 'reg_lambda': 76.4152624339764, 'random_strength': 26.281635365200945, 'depth': 13, 'min_data_in_leaf': 20, 'leaf_estimation_iterations': 3, 'bagging_temperature': 1.0080089583639982}
# iterations 12384
# cat = CatBoostRegressor(**chp)
# cat.fit(train_pool)
# cat.save_model('cat.bin')
# cat = CatBoostRegressor().load_model('cat.bin')


# print("점수는 : ",mean_squared_error(testY,cat.predict(testX),squared=False))
# 2.3203520016255856
# print("베스트 파라미터~!~",cat_study.best_params)
# cat_pred = cat.predict(test)

# submission['TARGET'] = cat_pred

import os
# 모델 저장할 폴더 생성
# cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2000)
# lgbm_scores = []
# for i, (tri , vai) in tqdm(enumerate(cv.split(x, y)), total=10):
#     x_train = x.iloc[tri]
#     y_train = y.iloc[tri]
#
#     x_valid = x.iloc[vai]
#     y_valid = y.iloc[vai]
#
#     lgbm_model = LGBMRegressor(**hp, objective='rmse', metric='rmse', verbosity=-1, n_jobs=-1)
#     lgbm_model.fit(x_train, y_train)
#
#     pred = lgbm_model.predict(x_valid)
#     score = mean_squared_error(y_valid,pred,squared=False)
#     lgbm_scores.append(score)
#     # 모델 저장
#     joblib.dump(lgbm_model, f"model/lgbm_model/{i}_lgbm_model.pkl")
#
#
lgbm_pred_list = []
lgbm_score_list = []
for i in range(10):
    model = joblib.load(f"model/lgbm_model/best/{i}_lgbm_model.pkl")
    pred = model.predict(testX)
    lgbm_score_list.append(mean_squared_error(testY,pred,squared=False))
    pred = model.predict(test)
    lgbm_pred_list.append(pred)
    submission['lgbm'+str(i)] = pred

print("LGBM pred 리스트 : ", lgbm_score_list)

# for i in range(10):
#    submission['TARGET1'] =submission['TARGET1']+submission['lgbm'+str(i)]
# submission['TARGET1'] = submission['TARGET1']/10
# submission = submission[['sessionID','TARGET']]


#cat
# cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2000)
# cat_scores = []
# for i, (tri , vai) in tqdm(enumerate(cv.split(x, y)), total=10):
#     x_train = x.iloc[tri]
#     y_train = y.iloc[tri]
#
#     x_valid = x.iloc[vai]
#     y_valid = y.iloc[vai]
#     chp = {'od_wait': 1968, 'learning_rate': 0.1434520185514101, 'reg_lambda': 38.281084288725815, 'random_strength': 12.867710796532581,
#     'depth': 12, 'min_data_in_leaf': 24,
#     'leaf_estimation_iterations': 7, 'bagging_temperature': 0.45930339623048033}
#     cat_model = CatBoostRegressor(**chp,random_state= 2000,eval_metric='RMSE', cat_features=category_enc,task_type='GPU')
#     cat_model.fit(x_train, y_train)
#     # 2.337508400777979
#     pred = cat_model.predict(x_valid)
#     score = mean_squared_error(y_valid,pred,squared=False)
#     cat_scores.append(score)
#     # 모델 저장
#     joblib.dump(cat_model, f"model/cat_model/{i}_cat_model.pkl")

cat_pred_list = []
cat_score_list = []
for i in range(10):
    model = joblib.load(f"model/cat_model/best/{i}_cat_model.pkl")
    pred = model.predict(testX)
    cat_score_list.append(mean_squared_error(testY,pred,squared=False))
    pred = model.predict(test)
    cat_pred_list.append(pred)
    submission['cat'+str(i)] = pred

print("cat pred 리스트 : ",cat_score_list)
#
# for i in range(10):
#     submission['TARGET'] =submission['TARGET']+submission[str(i)]
# submission['TARGET'] = submission['TARGET']/10
# submission = submission[['sessionID','TARGET']]

# print(submission.head(5))
submission['TARGET1'] = 0
submission['TARGET2'] = 0
#앙상블
for i in range(10):
    submission['TARGET1'] = submission['TARGET1']+submission['cat'+str(i)]
    submission['TARGET2'] = submission['TARGET2'] + submission['lgbm' + str(i)]

submission['TARGET1'] = submission['TARGET1']/10
submission['TARGET2'] = submission['TARGET2']/10
submission['TARGET'] = submission['TARGET1']*0.5 + submission['TARGET2']*0.5
submission = submission[['sessionID','TARGET']]


#TARGET값 0보다 작은거 0으로 보정하기
import datetime
# title = 'LGBM'+str(datetime.datetime.now().month)+'_'+str(datetime.datetime.now().day)+'_'+str(datetime.datetime.now().hour)+'_'+str(datetime.datetime.now().minute)+'.csv'
# title = 'CAT'+str(datetime.datetime.now().month)+'_'+str(datetime.datetime.now().day)+'_'+str(datetime.datetime.now().hour)+'_'+str(datetime.datetime.now().minute)+'.csv'
title = 'ENSEMBLE'+str(datetime.datetime.now().month)+'_'+str(datetime.datetime.now().day)+'_'+str(datetime.datetime.now().hour)+'_'+str(datetime.datetime.now().minute)+'.csv'
submission.loc[submission['TARGET'] < 0.0, 'TARGET'] = 0.0
submission.to_csv(title,index=False)


#### 분포를 확인해보고 로그를 취하거나 / 강조하고싶은 변수에 제곱을 하자
# 현재까진 catboost (random 2000)으로 파생변수 분당3가지 넣은거와 조회수 합계들 - 2.3485293369289479 -> 2.93468
# 같은 조건으로 2.33398496 나옴 LGBM 1차 파라미터로
# 퀄리티지수 추가하고 LGBM 2.3319800399778257  1차 파라미터
# 분당거래만 뺏을 때 LGBM 1차 파라미터 2.3296616
# 파생변수 transaction * quality 2.3299864
# 파생변수 거래확률 추가했을 때 1차파라미터로 2.3266463  - 2.92915
# 분당거래 추가해서 2.3249634 - 2.9247839215
# 유니크 값 10 초과되는거 타겟인코딩으로 변경 2.303577 - 2.91257
# kfold 적용하고 점수 미세하게 올라감 2.91196 -> random_state 안준거
# 2.054163016161751
# 0
# 2.032664352344181
# 1


# 일요일에 {'num_leaves': 355, 'colsample_bytree': 0.8474389094719293, 'reg_alpha': 0.7592006668666332, 'reg_lambda': 7.225166341237797, 'max_depth': 12, 'learning_rate': 0.003945880517624002, 'n_estimators': 1417, 'min_child_samples': 28, 'subsample': 0.6070270926393032}
# 3_2_20_58 이거로 제출 해보고 안되면 앙상블 0.5씩 곱해서 -> 2.9090 ---- LGBM 저 파라미터로 10fold random 안줌
# 10폴드 앙상블 cat(randomstate 2000) * 0.3 + lgbm * 0.7 - 2.88769
# 10폴드 앙상블 cat(randomstate 2000) * 0.5 + lgbm * 0.5 - 2.88287

# 거래지수 추가한거
# LGBM pred 리스트 :  [2.0582841553150115, 2.0474506439635443, 2.0554619230734743, 2.0737051592670426, 2.067120451937934, 2.05349359475045, 2.0647940860115495, 2.0604713449347836, 2.0542843824840964, 2.055631298823072]
# cat pred 리스트 :  [1.724911391120229, 1.6274775553243577, 1.6361485706635073, 1.695098060623455, 1.6817830938334575, 1.6349693170089903, 1.6655808747324166, 1.640391710877164, 1.6697469291043257, 1.627725729717484]

# 거래지수 추가 안한거
# LGBM pred 리스트 :  [2.0583462705387308, 2.0445648587698835, 2.050186533331076, 2.074643395682001, 2.0670479172523026, 2.0532032131989006, 2.067437046106775, 2.058081177740565, 2.0555337486251783, 2.0580238149671066]
# cat pred 리스트 :  [1.6982026171470224, 1.6275310143238024, 1.6272505218106543, 1.6808082065930543, 1.6941476340618242, 1.6221042280099807, 1.6496311222020559, 1.6328456953697037, 1.6738964325869656, 1.6220761641524304]