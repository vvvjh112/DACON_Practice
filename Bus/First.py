import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

#데이터를 바탕으로 도착시간을 예측해야 함
pd.set_option('display.max_columns', None) ## 모든 열을 출력한다.
pd.set_option('display.max_rows', None)

#데이터 탐색
#date, route_id, vh_id, route_nm 컬럼은 예측에 필요하지 않을 것으로 생각됨.
#현재 도착시간을 다음 도착시간에서 빼서 예상 소요시간 컬럼을 추가하면 좋을 듯 함.
#현재정류장은 중요하지 않을 듯 함.

#좌표를 거리로 변환해주는 모델도 있음.
#주요 랜드마크 확인도 스코어 올리는데 도움이 될 듯 함.
# -> 거리 계산해서 반경 몇미터 이내에 존재하면 컬럼 추가해서 핫플 여부

#이상값 체크 / 시각화
#모델 선택 및 하이퍼 파라미터 최적화
#스코어 확인
#앙상블 필요 여부 판단


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# print(train.columns)
def df_del(df):
    temp = df.copy()
    temp = temp.drop(['date','route_id','vh_id','route_nm','now_station','next_station'],axis=1)
    return temp

from haversine import haversine
pd.set_option('mode.chained_assignment',  None) # <==== 경고를 끈다
#주요구역이 포함되어 있는지 검사
def dist_check(df):
    # 해당 주요 장소의 임의 지역 위도, 경도
    up = (33.506286, 126.490312)  # 제주국제공항 근처
    right = (33.493521, 126.895326)  # 성산일출봉 근처
    down = (33.246742, 126.562387)  # 서귀포시 근처
    center = (33.379724, 126.545315)  # 성산일출봉 근처
    pointer = [[(33.506286, 126.490312)],[(33.493521, 126.895326)],[(33.246742, 126.562387)],[(33.379724, 126.545315)]]

    temp = df.copy()
    def tmp(df,i):
        lat = (df['now_latitude'][i],df['now_longitude'][i])
        lat2 = (df['next_latitude'][i],df['next_longitude'][i])
        if haversine(lat,(33.506286, 126.490312),unit='km') <5 or haversine(lat,(33.493521, 126.895326),unit='km') <5 or haversine(lat,(33.246742, 126.562387),unit='km') <5 or haversine(lat,(33.379724, 126.545315),unit='km')<5:
            return 1
        if haversine(lat2,(33.506286, 126.490312),unit='km') <5 or haversine(lat2,(33.493521, 126.895326),unit='km')<5 or haversine(lat2,(33.246742, 126.562387),unit='km')<5 or haversine(lat2,(33.379724, 126.545315),unit='km')<5:
            return 1
        return 0
    temp['hot']=11
    for i in range(df.shape[0]):
        temp['hot'][i] = tmp(temp,i)

        # tmmp = temp['now_arrive_time'][i]
        # tmmp = tmmp[:2]
        # temp['now_arrive_time'][i] = int(tmmp)

    return temp

#
# tmp = df_del(train)
# tmp = dist_check(tmp)
# tmp = tmp.drop(['now_latitude','now_longitude','next_latitude','next_longitude'],axis=1)
# print(tmp.columns)
#
#
# X_train = tmp.drop(['id', 'next_arrive_time'],axis=1)
#
# #데이터타입 변경
# X_train.set_index('now_arrive_time',inplace=True)
# X_train = X_train.astype('int64')
# X_train.reset_index(inplace=True)

# y_train = tmp['next_arrive_time']
#Index(['id', 'now_arrive_time', 'distance', 'next_arrive_time', 'hot'], dtype='object')



##### 내가한 전처리는 효과가 별로인 것 같음..
#2차시도
#핫플은 유지하고  음..

print(train_data.head())
station_encoder = LabelEncoder() # 인코더 생성

_station = list(train_data['now_station'].values) + list(train_data['next_station'].values) # train_data 의 모든 정류장 이름
station_set = set(_station)
# print(len(station_set))
station_encoder.fit(list(station_set)) # 인코딩

# 모든 학습, 시험 데이터의 정류장 정보 치환
train_data['now_station'] = station_encoder.transform(train_data['now_station'])
train_data['next_station'] = station_encoder.transform(train_data['next_station'])
test_data['now_station'] = station_encoder.transform(test_data['now_station'])
test_data['next_station'] = station_encoder.transform(test_data['next_station'])
# print(train_data.head())

# train_data['next_arrive_time'].hist()
# plt.show()

## train data

train_data['date'] = pd.to_datetime(train_data['date']) # date 값을 datetime으로
train_data['weekday'] = train_data['date'].dt.weekday  # Monday 0, Sunday 6
train_data['weekday'] = train_data['weekday'].apply(lambda x: int(0) if x < 5 else int(1))
# 0 ~ 5 는 월요일 ~ 금요일이므로 평일이면 0, 주말이면 1을 설정하였다

train_data = pd.get_dummies(train_data, columns=['weekday']) # 평일/주말에 대해 One-hot Encoding

train_data = train_data.drop('date', axis=1) # 필요없는 date 칼럼을 drop
# print(train_data.head())

## test data
# 시험데이터도 마찬가지로 처리해준다.

test_data['date'] = pd.to_datetime(test_data['date'])
test_data['weekday'] = test_data['date'].dt.weekday  # Monday 0, Sunday 6
test_data['weekday'] = test_data['weekday'].apply(lambda x: 0 if x < 5 else 1)
test_data = pd.get_dummies(test_data, columns=['weekday'])

test_data = test_data.drop('date', axis=1)
test_data.head()

train_data['time_group']='group' #time_group 변수를 미리 생성

train_data.loc[ (train_data['now_arrive_time']>='05시') & (train_data['now_arrive_time']<'12시') ,['time_group'] ]= 'morning' # 05~11시
train_data.loc[ (train_data['now_arrive_time']>='12시') & (train_data['now_arrive_time']<'18시') ,['time_group'] ]= 'afternoon' #12~17시
train_data.loc[ (train_data['now_arrive_time']>='18시') | (train_data['now_arrive_time']=='00시'),['time_group'] ]= 'evening' #18~00시

train_data = pd.get_dummies(train_data,columns=['time_group']) # 원 핫 인코딩을 수행
train_data = train_data.drop('now_arrive_time', axis=1) # 필요없는 now_arrive_time drop

test_data['time_group']='group' #time_group 변수를 미리 생성

test_data.loc[ (test_data['now_arrive_time']>='05시') & (test_data['now_arrive_time']<'12시') ,['time_group'] ]= 'morning' # 05~11시
test_data.loc[ (test_data['now_arrive_time']>='12시') & (test_data['now_arrive_time']<'18시') ,['time_group'] ]= 'afternoon' #12~17시
test_data.loc[ (test_data['now_arrive_time']>='18시') | (test_data['now_arrive_time']=='00시'),['time_group'] ]= 'evening' #18~00시

test_data = pd.get_dummies(test_data,columns=['time_group']) # 원 핫 인코딩을 수행
test_data = test_data.drop('now_arrive_time', axis=1) # 필요없는 now_arrive_time drop



train_data= dist_check(train_data)
test_data = dist_check(test_data)

train_data = pd.get_dummies(train_data, columns=['hot']) # 핫플 통과 여부에 대해 One-hot Encoding
# train_data = train_data.drop('hot', axis=1) # 필요없는 칼럼을 drop
# #
test_data = pd.get_dummies(test_data, columns=['hot']) # 핫플 통과 여부에 대해 One-hot Encoding
# test_data = test_data.drop('hot', axis=1) # 필요없는 칼럼을 drop

train_data = train_data.drop(['id', 'route_nm', 'next_latitude', 'next_longitude',
                              'now_latitude', 'now_longitude'], axis=1)
test_data = test_data.drop(['route_nm', 'next_latitude', 'next_longitude',
                              'now_latitude', 'now_longitude'], axis=1)

train_data = train_data[train_data['next_arrive_time'] <= 700]

print(train_data.head())
print(test_data.head())


# 학습 데이터 칼럼에서 목표치인 next_arrive_time만 제거하여 선택한다.
input_var = list(train_data.columns)
input_var.remove('next_arrive_time')

Xtrain = train_data[input_var] # 학습 데이터 선택
Ytrain = train_data['next_arrive_time'] # target 값인 Y 데이터 선택

Xtest = test_data[input_var] # 시험 데이터도 선택
#이상값 체크 및 모델 스코어 비교 해보자

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# import lightgbm as lgb
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor

import numpy as np

model_dict = {'DT':DecisionTreeRegressor(),
             'RF':RandomForestRegressor(),
             # 'LGB':lgb.LGBMRegressor(),
             'XGB':xgb.XGBRegressor(),
             'KNN':KNeighborsRegressor()}

#시각화는 생략.



from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=5, shuffle= True, random_state=10)

score = {}

for model_name in model_dict.keys():
    model = model_dict[model_name]

    score[model_name] = np.mean(
        cross_val_score(model, Xtrain, Ytrain, scoring='neg_mean_squared_error', n_jobs=-1, cv=k_fold))*-1


# pd.Series(score).plot(kind = 'bar')
# plt.ylim(0,5000)
# plt.show()

#DT, RF, XGB가 비슷 LGB가 좀 떨어짐 KNN은 나가리

from sklearn.model_selection import GridSearchCV

#
def get_best_params(model, param):
    grid_model = GridSearchCV(model, param_grid=param, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    grid_model.fit(Xtrain, Ytrain)
    rmse = np.sqrt(-1* grid_model.best_score_)
    print('최적 평균 RMSE 값:', np.round(rmse, 4))
    print('최적 파라미터:', grid_model.best_params_)
    return grid_model.best_estimator_
#
xg_param = {
        #  'max_depth':range(2,10,2),
        #
        # 'n_estimators': range(400,1050,100)
        # 'n_estimators' : [100,200,300,400,500],
        'learning_rate' : [0.01,0.05,0.1,0.15],
        'max_depth' : [4,6,8,10,15],
        'gamma' : [0,1,2,3],
        'colsample_bytree' : [0.8,0.9],
}
#
# xgb = xgb.XGBRegressor(colsample_bytree= 0.8, gamma= 0, learning_rate= 0.15, max_depth= 10)
#
# xgb = xgb.XGBRegressor()
# best_xgb = get_best_params(xgb,xg_param)
# #
# print(best_xgb)

model = xgb.XGBRegressor(random_state=110, verbosity=0, nthread=23, n_estimators=980, max_depth=4)
kfold = KFold(n_splits=8, shuffle=True, random_state=777)
n_iter = 0
cv_score = []

def rmse(target, pred):
    return np.sqrt(np.sum(np.power(target - pred, 2)) / np.size(pred))


for train_index, test_index in kfold.split(Xtrain, Ytrain):
    # K Fold가 적용된 train, test 데이터를 불러온다
    X_train, X_test = Xtrain.iloc[train_index, :], Xtrain.iloc[test_index, :]
    Y_train, Y_test = Ytrain.iloc[train_index], Ytrain.iloc[test_index]

    # 모델 학습과 예측 수행
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    print(pred)

    # 정확도 RMSE 계산
    n_iter += 1
    score = rmse(Y_test, pred)
    print(score)
    cv_score.append(score)
print('\n교차 검증별 RMSE :', np.round(cv_score, 4))
print('평균 검증 RMSE :', np.mean(cv_score))

result = model.predict(Xtest) # 시험 데이터 예측

test_data['next_arrive_time'] = result # next_arrive_time 예측 결과로 추가
test_data[['id', 'next_arrive_time']].to_csv('Second_YJH_Edit.csv',index=False, float_format='%.14f') # csv로 변환

# 변수 중요도 평가
n_feature = X_train.shape[1] #주어진 변수들의 갯수를 구함
index = np.arange(n_feature)

plt.barh(index, model.feature_importances_, align='center') #
plt.yticks(index, input_var)
plt.ylim(-1, n_feature)
plt.xlabel('feature importance', size=15)
plt.ylabel('feature', size=15)
plt.show()
np.sum(model.feature_importances_)