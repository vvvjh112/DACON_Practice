import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# 총 거래 수익/세션 분당
# 분당 거래의 수
# 사용된 브라우저별 총 거래 수익
# 국가별 총 거래 수익 및 거래 수
# 총 거래수익 / 총 거래의 수
# 브라우저별 이탈율
# test셋어 없는 데이터들 제외

# RMSE


#plt 한글출력
# plt.rcParams['font.family'] ='Malgun Gothic'
# plt.rcParams['axes.unicode_minus'] =False

# 한글 폰트 경로 지정
font_path = '/Library/Fonts/AppleGothic.ttf'  # 예시로 AppleGothic 폰트 사용

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'

# row 생략 없이 출력
pd.set_option('display.max_rows', None)
# col 생략 없이 출력
pd.set_option('display.max_columns', None)

train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')
submission = pd.read_csv('Data/sample_submission.csv')


print(train.info())

#파생변수 출력
train['perM'] = train['transaction_revenue'] / (train['duration'].replace(0, 1) / 60)
test['perM'] = test['transaction_revenue'] / (test['duration'].replace(0, 1) / 60)



#시각화 이전 그룹
# OS별, 브라우저별, 디바이스별, 첫방문별, 이탈여부, 나라별, 소스별
group_os = train.groupby(['OS']).mean('TARGET')
# group_os = group_os[['TARGET']]

group_browser = train.groupby(['browser']).mean('TARGET')[['TARGET']].reset_index('browser')
group_browser = group_browser[~group_browser['browser'].str.startswith(';__CT_JOB_ID__:')]


group_device = train.groupby(['device']).mean('TARGET')
# group_device = group_device[['TARGET']]

group_new = train.groupby(['new']).mean('TARGET')
# group_new = group_new[['TARGET']]

group_bounced = train.groupby(['bounced']).mean('TARGET')
# group_bounced = group_bounced[['TARGET']]

group_country = train.groupby(['country']).mean('TARGET')
# group_country = group_country[['TARGET']]

group_source = train.groupby(['traffic_source']).mean('TARGET')
# group_source = group_source[['TARGET']]

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
# print(test.isna().sum())

from sklearn.model_selection import *
from sklearn.preprocessing import *
#라벨 인코딩
categorical_features = ["browser", "OS", "device", "continent", "subcontinent", "country", "traffic_source", "traffic_medium", "keyword", "referral_path"]
for i in categorical_features:
    train[i] = LabelEncoder().fit_transform(train[i])
    test[i] = LabelEncoder().fit_transform(test[i])

#데이터 분리
x = train.drop(['sessionID','userID','TARGET'],axis=1)
y = train['TARGET']

test= test.drop(['sessionID','userID'],axis =1)

trainX, testX, trainY, testY = train_test_split(x,y,test_size=0.2)

#Catboost / xgboost / LGBM /
import model_tuned as mt
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import *

# mt.compare_model(train.drop(['sessionID','userID'],axis=1),'TARGET')




#옵튜나 모델링
# lgbm , lgbm_study = mt.lgbm_modeling(trainX,trainY,testX,testY)
# lgbm_predict = lgbm.predict(test)
# submission['TARGET'] = lgbm_predict

cat, cat_study = mt.cat_modeling(trainX,trainY,testX,testY)
cat_predict = cat.predict(test)
submission['TARGET'] = cat_predict




#TARGET값 0보다 작은거 0으로 보정하기
submission.loc[submission['TARGET'] < 0.0, 'TARGET'] = 0.0
submission.to_csv('Cat_First_2133.csv',index=False)