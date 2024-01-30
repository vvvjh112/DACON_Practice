import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#plt 한글출력
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# row 생략 없이 출력
pd.set_option('display.max_rows', None)
# col 생략 없이 출력
pd.set_option('display.max_columns', None)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
countrywide = pd.read_csv('external_open/countrywide_accident.csv') #대구 제외한 교통사고 정보
submission = pd.read_csv('sample_submission.csv')


#test셋에는 없는 값들 확인
train_column = train.columns
test_column = test.columns
# print(list(set(train_column)-set(test_column)))
#['사고유형 - 세부분류', '경상자수', '피해운전자 상해정도', '사망자수', '부상자수', '중상자수', '가해운전자 차종', '피해운전자 성별', '법규위반', '가해운전자 상해정도', '가해운전자 연령', '피해운전자 연령', '피해운전자 차종', '가해운전자 성별']
lst = {}
for i in test_column:
    set_train = set(train[i].unique())
    set_test = set(test[i].unique())
    lst[i] = set_train-set_test

#안개가 test엔 없는것 확인 - 삭제
train = train[train['기상상태']!= '안개']

#날짜는 추후 편의를 위해 datetime 타입으로 변환
train['사고일시'] = pd.to_datetime(train['사고일시'],format="%Y-%m-%d %H")
test['사고일시'] = pd.to_datetime(test['사고일시'],format="%Y-%m-%d %H")

#시군구 분리
pattern = r'(\S+) (\S+) (\S+)'

train[['도시', '구', '동']] = train['시군구'].str.extract(pattern)
train = train.drop(columns=['시군구','도시'])

test[['도시', '구', '동']] = test['시군구'].str.extract(pattern)
test = test.drop(columns=['시군구','도시'])

#도로형태 분리
pattern = '(.+) - (.+)'

train[['도로형태1', '도로형태2']] = train['도로형태'].str.extract(pattern)
test[['도로형태1', '도로형태2']] = test['도로형태'].str.extract(pattern)

train = train.drop('도로형태',axis = 1)
test = test.drop('도로형태',axis = 1)

#시간
train['연'] = train['사고일시'].dt.year
train['월'] = train['사고일시'].dt.month
train['일'] = train['사고일시'].dt.day
train['시간'] = train['사고일시'].dt.hour
test['월'] = test['사고일시'].dt.month
test['시간'] = test['사고일시'].dt.hour

#파생컬럼 계절 추가
def season(x):
    if 3 <= x['월'] <= 5:
        return '봄'
    elif 6 <= x['월'] <= 8:
        return '여름'
    elif 9 <= x['월'] <= 11:
        return '가을'
    else:
        return '겨울'

train['계절'] = train.apply(season,axis=1)
test['계절'] = test.apply(season,axis=1)


#공휴일 체크
holiday_2019 = ['20190101', '20190204', '20190205', '20190206', '20190301', '20190505', '20190506', '20190512', '20190606', '20190815', '20190912', '20190913', '20190914', '20191003', '20191009', '20191225']
holiday_2020 = ['20200101', '20200124', '20200125', '20200126', '20200127', '20200301', '20200415', '20200430', '20200505', '20200606', '20200815', '20200817', '20200930', '20201001', '20201002', '20201003', '20201009', '20201225']
holiday_2021 = ['20210101', '20210211', '20210212', '20210213', '20210301', '20210505', '20210519', '20210606', '20210815', '20210816', '20210920', '20210921', '20210922', '20211003', '20211004', '20211009', '20211011', '20211225']
holiday_2022 = ['20220101', '20220131', '20220201', '20220202', '20220301', '20220309', '20220505', '20220508', '20220601', '20220606', '20220815', '20220909', '20220910', '20220911', '20220912', '20221003', '20221009', '20221010', '20221225']

def check_holiday(x):
    year = x.year
    date = x.strftime('%Y%m%d')
    if year == 2019:
        if date in holiday_2019:
            return 1
    elif year == 2020:
        if date in holiday_2020:
            return 1
    elif year == 2021:
        if date in holiday_2021:
            return 1
    elif year == 2022:
        if date in holiday_2022:
            return 1
    return 0

train['공휴일'] = train['사고일시'].apply(check_holiday)
test['공휴일'] = test['사고일시'].apply(check_holiday)

# print(train.head(20))
# print(test.head(20))
#평일이면 평일로구분 주말 or 공휴일이면 휴일로 구분하는것도 괜찮을듯 싶은데
#요일 + 공휴일 합쳐서 휴무 여부로 변경
def restday(x):
    if (x['공휴일'] == 1)or(x['요일']=='토요일')or(x['요일']=='일요일'):
        return 1
    else:
        return 0

# train['휴무여부'] = train.apply(restday, axis=1)
# test['휴무여부'] = test.apply(restday, axis=1)

#기상상태, 요일별, 월별, 공휴일 ECLO 시각화해보기
group_year = train.groupby(['연']).mean('ECLO')
group_year = group_year[['ECLO']]

group_month = train.groupby(['월']).mean('ECLO')
group_month = group_month[['ECLO']]

group_day = train.groupby(['일']).mean('ECLO')
group_day = group_day[['ECLO']]

group_hour = train.groupby(['시간']).mean('ECLO')
group_hour = group_hour[['ECLO']]

group_holi = train.groupby(['공휴일']).mean('ECLO')
group_holi = group_holi[['ECLO']]

#기상상태, 노면상태, 사고유형, 도로형태 시각화
group_weather = train.groupby(['기상상태']).mean('ECLO')
group_weather = group_weather[['ECLO']]

group_surface = train.groupby(['노면상태']).mean('ECLO')
group_surface = group_surface[['ECLO']]

group_acc = train.groupby(['사고유형']).mean('ECLO')
group_acc = group_acc[['ECLO']]

group_road1 = train.groupby(['도로형태1']).mean('ECLO')
group_road1 = group_road1[['ECLO']]

group_road2 = train.groupby(['도로형태2']).mean('ECLO')
group_road2 = group_road2[['ECLO']]

def return_days(x):
    if x == '월요일':
        return 0
    elif x == '화요일':
        return 1
    elif x == '수요일':
        return 2
    elif x == '목요일':
        return 3
    elif x == '금요일':
        return 4
    elif x == '토요일':
        return 5
    elif x == '일요일':
        return 6

def reverse_days(x):
    if x == 0:
        return '월요일'
    elif x == 1:
        return '화요일'
    elif x == 2:
        return '수요일'
    elif x == 3:
        return '목요일'
    elif x == 4:
        return '금요일'
    elif x == 5:
        return '토요일'
    elif x == 6:
        return '일요일'

group_days = train.groupby(['요일']).mean('ECLO')
group_days = group_days[['ECLO']]
group_days = group_days.reset_index()
group_days['요일'] = group_days['요일'].apply(return_days)
group_days = group_days.sort_values('요일',ascending=True)
group_days['요일'] = group_days['요일'].apply(reverse_days)

gy = group_year.plot(title='연별 평균',kind='line',marker='o')
gy.set_xticks(group_year.index)
# plt.show()

gm = group_month.plot(title='월별 평균',kind='line',marker='o')
gm.set_xticks(group_month.index)
# plt.show()

gd = group_day.plot(title='일별 평균',kind='line',marker='o')
gd.set_xticks(group_day.index)
# plt.show()

gh = group_hour.plot(title='시간별 평균',kind='line',marker='o')
gh.set_xticks(group_hour.index)
# plt.show()

gds = group_days.plot(title='요일별 평균',kind='line',marker='o',x='요일')
# plt.show()

gholi = group_holi.plot(title='공휴일 평균',kind='bar')
# plt.show()

gw = group_weather.plot(title='기상상태',kind = 'bar')
# plt.show()

gsurface = group_surface.plot(title='노면상태',kind = 'bar')
# plt.show()

gacc = group_acc.plot(title='사고형태',kind = 'bar')
# plt.show()

gr1 = group_road1.plot(title='도로형태1',kind = 'bar')
# plt.show()

gr2 = group_road2.plot(title='도로형태2',kind = 'bar')
# plt.show()

#결측값 확인
# print(train.isna().sum())
# print(test.isna().sum())
#결측값
# 피해운전자 차종       991
# 피해운전자 성별       991
# 피해운전자 연령       991
# 피해운전자 상해정도     991




#컬럼을 test에 맞춰 수정할것이기 때문에 카피해서 작업진행
train_1 = train.copy()
test_1 = test.copy()


#test에는 없는 컬럼들 삭제해보고 진행해보자 우선.
lst = ['사고유형 - 세부분류', '경상자수', '피해운전자 상해정도', '사망자수', '부상자수', '중상자수', '가해운전자 차종', '피해운전자 성별', '법규위반', '가해운전자 상해정도', '가해운전자 연령', '피해운전자 연령', '피해운전자 차종', '가해운전자 성별']

#요일, 공휴일은 의미 있으나 연 월 일은 의미 없음
train_1 = train_1.drop(['연','일','사고일시','ID'],axis=1)
train_1 = train_1.drop(lst,axis=1)
test_1 = test_1.drop(['ID','사고일시'],axis=1)

print(test_1.head())

#타겟인코딩, 라벨인코딩, 원핫인코딩
#우선 라벨인코딩만
from sklearn.preprocessing import LabelEncoder
Label_lst = ['요일','기상상태','도로형태1','도로형태2','노면상태','사고유형','구','동','계절']
temp = []
for i in Label_lst:
    lb = LabelEncoder()
    temp.append(lb)
    train_1[i] = lb.fit_transform(train[i])
    test_1[i] = lb.fit_transform(test[i])

# print(train_1.head())
# print(test_1.head())
#모델링
from pycaret.regression import *
#2024.01.19 pycaret
# clf = setup(data=train_1, target='ECLO', train_size=0.8)
# best_model = compare_models()
# compare_models(n_select = 5, sort = 'RMSLE')
#            RMSLE    MAPE  TT (Sec)
# huber     0.4461  0.5274     0.795
# gbr       0.4588  0.6225     0.654
# br        0.4594  0.6230     0.176
# ridge     0.4596  0.6230     0.122
# lr        0.4599  0.6232     0.159
# lightgbm  0.4600  0.6230     0.187
# omp       0.4602  0.6247     0.127
# catboost  0.4625  0.6247     1.604
# lasso     0.4657  0.6332     0.125
# en        0.4657  0.6332     0.124
# llar      0.4657  0.6332     0.123
# dummy     0.4657  0.6332     0.123
# xgboost   0.4699  0.6327     0.883
# lar       0.4909  0.7742     0.123
# knn       0.5011  0.6567     0.250
# rf        0.5072  0.6809     2.009
# par       0.5428  0.6392     0.137
# et        0.5477  0.7087     2.195
# dt        0.6254  0.7800     0.159
# ada       0.6855  1.3009     0.577
# model_py_1 = create_model('huber')
# tuned_huber = tune_model(model_py_1,optimize = 'RMSLE')
# print(tuned_huber)
# final_model = finalize_model(tuned_huber)
# prediction = predict_model(final_model, data = test_1)
# result = prediction['prediction_label']
# submission['ECLO'] = result
#0.43585

#2024.01.19 2차
# clf = setup(data=train_1, target='ECLO', train_size=0.8)
# best_model = compare_models()
# compare_models(n_select = 5, sort = 'RMSLE')
#            RMSLE    MAPE  TT (Sec)
# huber     0.4545  0.5356     0.200
# gbr       0.4589  0.6251     0.298
# lightgbm  0.4606  0.6261     0.077
# catboost  0.4625  0.6263     1.632
# lar       0.4637  0.6349     0.011
# br        0.4637  0.6348     0.015
# lr        0.4637  0.6349     0.015
# ridge     0.4637  0.6349     0.012
# en        0.4663  0.6357     0.012
# lasso     0.4663  0.6357     0.013
# llar      0.4663  0.6357     0.013
# omp       0.4663  0.6357     0.013
# dummy     0.4663  0.6356     0.013
# xgboost   0.4708  0.6339     0.331
# rf        0.5090  0.6847     1.049
# knn       0.5102  0.6773     0.043
# et        0.5429  0.7019     0.810
# ada       0.5653  0.9575     0.067
# dt        0.6237  0.7756     0.029
# par       0.7689  1.4496     0.022

# model_py_2 = create_model('gbr')
# tuned_gbr = tune_model(model_py_2,optimize = 'RMSLE')
# print(tuned_gbr)
# final_model = finalize_model(tuned_gbr)
# prediction = predict_model(final_model, data = test_1)
# result = prediction['prediction_label']
# submission['ECLO'] = result
#0.44427

#3차
# clf = setup(data=train_1, target='ECLO', train_size=0.8)
# best_model = compare_models()
# compare_models(n_select = 5, sort = 'RMSLE')
#            RMSLE    MAPE  TT (Sec)
# huber     0.4532  0.5319     0.193
# gbr       0.4569  0.6198     0.378
# lightgbm  0.4589  0.6220     0.086
# ridge     0.4617  0.6298     0.012
# lar       0.4617  0.6298     0.012
# br        0.4617  0.6297     0.014
# lr        0.4617  0.6298     0.015
# catboost  0.4624  0.6238     1.612
# en        0.4643  0.6306     0.013
# omp       0.4643  0.6306     0.012
# lasso     0.4643  0.6306     0.013
# llar      0.4643  0.6306     0.012
# dummy     0.4643  0.6306     0.013
# xgboost   0.4730  0.6359     0.077
# rf        0.4909  0.6786     1.300
# knn       0.5088  0.6710     0.049
# et        0.5107  0.6867     1.042
# ada       0.5414  0.8801     0.076
# dt        0.6484  0.8080     0.034
# par       0.8049  1.9689     0.019

# RMSLE 계산 함수 정의
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_log_error
def rmsle(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

# RMSLE를 스코어 함수로 만들기
rmsle_scorer = make_scorer(rmsle, greater_is_better=False)


#모델링
x = train_1.drop('ECLO',axis = 1)
y = train_1['ECLO']

from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(x,y,test_size=0.2,random_state=2023)

from sklearn.model_selection import GridSearchCV

#LGBM
from lightgbm import LGBMRegressor
# model_lgbm = LGBMRegressor(learning_rate=0.01,min_child_samples=30,n_estimators=400,num_leaves=31,reg_alpha=0.5,reg_lambda=0)
model_lgbm = LGBMRegressor(learning_rate=0.01,n_estimators=300, num_leaves=31, min_child_samples=40,reg_alpha=0.1,reg_lambda=0.0)
lgbm_param = {
    'num_leaves': [31, 40, 50],
    'learning_rate': [0.01, 0.05 ,0.1],
    'n_estimators': [300,400,500,600],
    'min_child_samples': [20,30,40],
    'reg_alpha': [0.0, 0.1, 0.5],
    'reg_lambda': [0.0, 0.1, 0.5]
    }
# lgbm_grid = GridSearchCV(model_lgbm,param_grid=lgbm_param,n_jobs=-1,scoring=rmsle_scorer,cv=4)
# lgbm_grid.fit(trainX,trainY)
# best_lgbm = lgbm_grid.best_estimator_
# print('최적의 하이퍼 파라미터는:', lgbm_grid.best_params_)
# {'learning_rate': 0.01, 'min_child_samples': 30, 'n_estimators': 400, 'num_leaves': 31, 'reg_alpha': 0.5, 'reg_lambda': 0.0}



#Linear Rgeression
from sklearn.linear_model import LinearRegression
# model_lr = LinearRegression()
# model_lr.fit(trainX,trainY)

#huber
from sklearn.linear_model import HuberRegressor
model_huber = HuberRegressor(alpha= 0.0001, epsilon= 1.5, max_iter= 500)
model_huber.fit(trainX,trainY)
huber_param = {
    'epsilon': [1.0, 1.5, 2.0],  # 적절한 값으로 조정
    'alpha': [0.0001, 0.001, 0.01],  # 적절한 값으로 조정
    'max_iter': [500, 1000, 1500]  # 적절한 값으로 조정
}
# huber_grid = GridSearchCV(model_huber,param_grid=huber_param,n_jobs=-1,scoring=rmsle_scorer,cv=4)
# huber_grid.fit(trainX,trainY)
# best_huber = huber_grid.best_estimator_
# print('최적의 하이퍼 파라미터는:', huber_grid.best_params_)


#XGB
from xgboost import XGBRegressor
model_xgb = XGBRegressor(learning_rate = 0.01, max_depth = 3, n_estimators = 800)

xgb_param = {
    'learning_rate' : [0.01,0.05,0.1],
    'n_estimators' : [100,400,600,800],
    'max_depth' : [3,5,7],
}

# print("xgb 그리드 서치 시작")
# xgb_grid = GridSearchCV(model_xgb,param_grid=xgb_param,n_jobs=-1,scoring=rmsle_scorer,cv=4)
# xgb_grid.fit(trainX,trainY)
# best_xgb = xgb_grid.best_estimator_
# print('최적의 하이퍼 파라미터 : ', xgb_grid.best_params_)


#피처 중요도
#model.feature_importances_
#lgbm
model_lgbm.fit(trainX,trainY)
model_xgb.fit(trainX,trainY)
print(model_lgbm.feature_importances_)
# print(model_huber.feature_importances_)
print((model_xgb.feature_importances_)*100)
# [1403    329      244       707    0    1322    3480    496      1076      280      2663]
#  요일   기상상태  노면상태   사고유형  도시    구      동    도로형태1  도로형태2   공휴일     시간
# [ 7.440489   4.2729135  6.7964053 42.735065   3.75589    4.9095335
#   6.273524   5.5323863  4.342402   5.859895   4.264044   3.817453 ]

#모델링 후 예측
#LGBM
# model_lgbm.fit(trainX,trainY)
# pred_lgbm_1 = model_lgbm.predict(testX)
#
#
# score_lgbm = mean_squared_log_error(testY,pred_lgbm_1,squared=False)
# print(score_lgbm)
# 0.4566
# #실제예측
# pred_lgbm_2 = model_lgbm.predict(test_1)
# submission['ECLO'] = pred_lgbm_2
#0.4439

#XGB
# model_xgb.fit(trainX,trainY)
# pred_xgb_1 = model_xgb.predict(testX)
#
# score_xgb = mean_squared_log_error(testY,pred_xgb_1,squared=False)
# print(score_xgb)
# 0.4567
# 실제
# pred_xgb_2 = model_xgb.predict(test_1)
# submission['ECLO'] = pred_xgb_2
#0.44026

#LinearRegression
# pred_lr = model_lr.predict(testX)
# score_lr = mean_squared_log_error(testY,pred_lr,squared=False)
# print(score_lr)
#0.4614305671790214

#Huber
# pred_huber_1 = model_huber.predict(testX)
# score_huber = mean_squared_log_error(testY,pred_huber_1,squared=False)
# print(score_huber)
# 0.4482
# pred_huber_2= model_huber.predict(test_1)
# submission['ECLO'] = pred_huber_2
#0.4333

#AutoML
from supervised.automl import AutoML
#
# automl = AutoML(
#     mode="Compete",
#     algorithms=['LightGBM', 'Xgboost', 'CatBoost', 'Neural Network', 'Extra Trees'],
#     n_jobs=-1,
#     total_time_limit=43200,
#     eval_metric="rmse",
#     ml_task="regression",
#     features_selection=True,  # 특성 선택 활성화
#     boost_on_errors=True,  # 오류에 대한 부스팅 활성화
#
# )
# automl.fit(trainX, trainY)
# pred_auto = automl.predict(test_1)
# submission['ECLO'] = pred_auto
# submission.loc[submission['ECLO'] < 0.0, 'ECLO'] = 0.0
# 0.4282 _1_29_15_47.csv


#XGB, LGBM, Huber 앙상블
# final = (pred_lgbm_2 + pred_xgb_2 + pred_huber_2)/3
# submission['ECLO'] = final
# 0.4365 - ensemble1_29_15_50


# csv파일 도출
import datetime
# title = str(round(score_huber,5))+'_'+str(datetime.datetime.now().month)+'_'+str(datetime.datetime.now().day)+'_'+str(datetime.datetime.now().hour)+'_'+str(datetime.datetime.now().minute)+'.csv'
# title = '_'+str(datetime.datetime.now().month)+'_'+str(datetime.datetime.now().day)+'_'+str(datetime.datetime.now().hour)+'_'+str(datetime.datetime.now().minute)+'.csv'
# submission.to_csv(title,index=False)

#다른지역 추가 전에 xgb linear 모델링 후 비교해보고 앙상블 해보자
#다른 지역 추가해보자 우선 광역시 위주로

#단지 예측 뿐 아니라 train 셋에 있는 데이터를 바탕으로 사고를 줄일 수 있는 방법 제시.
#카메라랑 사고 그래프 보여주면서 카메라가 효과적 이런거


#현재 huber가 제일 점수가 잘 나옴
#현재 컬럼들은 세이브 하고, 학습에 다른 광역시 자료 추가해서 다음 파일로 진행.
#추가데이터는 정확도가 떨어짐
#앙상블 시도해보고 그래도 떨어지면 파생컬럼 추가 고려 / 제출 초과해서 내일 해봐야함