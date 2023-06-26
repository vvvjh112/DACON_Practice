import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# train 파일의 데이터를 학습시켜서 모델을 통해 test의 컬럼들의 데이터를 이용하여 예측하여 submission파일에 출력


#출력 설정
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission = pd.read_csv("submission.csv")

#상단 데이터 5개 확인
# print(train.head())
# print(test.head())
# print(submission.head())


#컬럼확인
# print(train.columns)
# print(test.columns)
# print(submission.columns)

# 사분위수 및 평균 등 확인
# print(test.describe())

#결측치 확인
# print(train.isna().sum())
# print(test.isna().sum())
# print(train.isna().sum())


#다음 그래프로 8시와 6시에 대여량이 증가하는걸 확인 가능 (출퇴근 영향)
# plt.plot('hour','count','*',data=train)
# plt.show()

#강수 관련 컬럼의 값 분포..?
# print(train['hour_bef_precipitation'].value_counts())

# 강수 컬럼 결측치 확인
# print(train['hour_bef_precipitation'].isna().sum())

#컬럼 기온 습도 등등 평균 확인
# print(train['hour_bef_temperature'].mean())
# print(train['hour_bef_windspeed'].mean())
# print(train['hour_bef_humidity'].mean())
# print(train['hour_bef_visibility'].mean())
# print(train['hour_bef_ozone'].mean())
# print(train['hour_bef_pm10'].mean())
# print(train['hour_bef_pm2.5'].mean())

#결측치 값 저장
train_isna = train.isna().sum()

#결측치가 없는 컬럼
train_na_col = train_isna[train_isna > 0].index
# print(train_na_col)

#train 프레임의 결측치를 평균값으로 대체
train = train.fillna(train.mean())

# print(train.isna().sum())  # 결측치가 사라진 것 확인가능.
# print(test['hour_bef_precipitation'].value_counts())


### test 프레임 ###
test['hour_bef_precipitation'].fillna(value = '0', inplace=True)
# print(test.isna().sum())

# 평균
# print(test['hour_bef_temperature'].mean())
# print(test['hour_bef_windspeed'].mean())
# print(test['hour_bef_humidity'].mean())
# print(test['hour_bef_visibility'].mean())
# print(test['hour_bef_ozone'].mean())
# print(test['hour_bef_pm10'].mean())
# print(test['hour_bef_pm2.5'].mean())




test_isna = test.isna().sum()



# test 결측치를 평균값으로 대체
# test = test.fillna(test.mean()) 로 진행할 시 오류나서 na 컬럼만 대체
def fill_bicycle_na(df, column):
    df[column] = df[column].fillna(value=df[column].mean())

na_columns = test_isna[test_isna != 0].index
for col in na_columns:
    fill_bicycle_na(test,col)


#train 데이터에서 시간과 대여수 그래프
# plt.plot('hour', 'count', 'o', data=train)
# plt.show()

#히트맵
# plt.figure(figsize = (12,12))
# sns.heatmap(train.corr(), annot = True)
# plt.show()

#컬럼
xfet = ['hour', 'hour_bef_temperature', 'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility','hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5']

xfets = train[xfet].columns

# 각 컬럼 컬럼들과 대여량 과의 관계 시각화
# for i in xfets:
#     sns.lmplot(x=i, y='count', data=train)
#     plt.show()
    

############################################################################

# 모델 import
from sklearn.linear_model import LinearRegression               #선형회귀
from sklearn.ensemble import RandomForestRegressor              #랜덤포레스트
from sklearn.model_selection import GridSearchCV                #GVsearch  -- 머신러닝 모델에서 성능향상 / 최적의 하이퍼 파라미터 찾음
from sklearn.model_selection import RandomizedSearchCV          #RandomizedSearch
from sklearn.ensemble import GradientBoostingRegressor          #그래디언트 부스팅

features = ['hour', 'hour_bef_temperature', 'hour_bef_ozone', 'hour_bef_windspeed', 'hour_bef_humidity']

# X독립변수  Y 종속변수
X_train = train[features]
y_train = train['count']
X_test = test[features]

#선형회귀모델 선택
# lr = LinearRegression()
# lr.fit(X_train, y_train)

#예측
# lr_predict = lr.predict(X_test)

#submission 파일에 예측값 대입 및 내보내기
# submission['count'] = lr_predict
# submission.to_csv('lr.csv', index=False)

#랜덤포레스트
# rf = RandomForestRegressor()
#
# rf_model = RandomForestRegressor(n_estimators=100)
#
# rf_model.fit(X_train,y_train)
#
# rf_predict = rf_model.predict(X_test)
#
# submission['count'] = rf_predict
# submission.to_csv('rf.csv', index=False)


#GridSearchCV
# param = {'min_samples_split': range(1,5),
#         'max_depth':range(8,12,2),
#         'n_estimators': range(250,450,50)} # 찾고자 하는 파라미터
#
# gs = GridSearchCV(estimator=rf, param_grid=param, scoring='neg_mean_squared_error',cv=3)  #cv = fold 횟수  scoring 은 회귀분석이기 때문에 "neg~~"
#
# gs.fit(X_train,y_train)
#
# print('최고 정확도 : ', gs.best_score_)
# print('최고 파라미터 : ', gs.best_params_)
#
#
# rf_gs_predict = gs.predict(X_test)
# submission['count'] = rf_gs_predict
# submission.to_csv('rf_gs.csv', index=False)


# 랜덤포레스트 최적화
# rs_model = RandomForestRegressor()
# param = {'min_samples_split': range(12,15),
#         'max_depth': range(8,11),
#         'n_estimators': range(222,225)}
#
# rs = RandomizedSearchCV(estimator=rs_model, param_distributions=param, scoring = 'neg_mean_squared_error', cv=3)
#
# rs.fit(X_train, y_train)
#
# print('최고 정확도 : ', rs.best_score_)
# print('최고 파라미터 : ', rs.best_params_)
#
# rf_rs_predict = rs.predict(X_test)
#
# submission['count'] = rf_rs_predict
# submission.to_csv('rf_rs.csv', index=False)