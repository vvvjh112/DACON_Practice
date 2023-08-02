import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

        tmmp = temp['now_arrive_time'][i]
        tmmp = tmmp[:2]
        temp['now_arrive_time'][i] = int(tmmp)

    return temp

tmp = df_del(train)
tmp = dist_check(tmp)
tmp = tmp.drop(['now_latitude','now_longitude','next_latitude','next_longitude'],axis=1)
print(tmp.columns)


X_train = tmp.drop(['id', 'next_arrive_time'],axis=1)
#데이터타입 변경
X_train.set_index('now_arrive_time',inplace=True)
X_train = X_train.astype('int64')
X_train.reset_index(inplace=True)


print("데이터 타입" , X_train.dtypes)
y_train = tmp['next_arrive_time']
#Index(['id', 'now_arrive_time', 'distance', 'next_arrive_time', 'hot'], dtype='object')
#테스트
print(tmp.head())
# print("\n")
# print(X_train.head())

##### 내가한 전처리는 효과가 별로인 것 같음..
#2차시도
#핫플은 유지하고  음..




#이상값 체크 및 모델 스코어 비교 해보자

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



# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score
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
#
# plt.show()

#DT, RF, XGB가 비슷 LGB가 좀 떨어짐 KNN은 나가리

from sklearn.model_selection import GridSearchCV


def get_best_params(model, param):
    grid_model = GridSearchCV(model, param_grid=param, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    grid_model.fit(X_train, y_train)
    rmse = np.sqrt(-1* grid_model.best_score_)
    print('최적 평균 RMSE 값:', np.round(rmse, 4))
    print('최적 파라미터:', grid_model.best_params_)
    return grid_model.best_estimator_

xg_param = {
        #  'max_depth':range(2,10,2),
        #
        # 'n_estimators': range(400,1050,100)
        'n_estimators' : [100,200,300,400,500],
        # 'learning_rate' : [0.01,0.05,0.1,0.15],
        # 'max_depth' : [3,5,7,10,15],
        # 'gamma' : [0,1,2,3],
        # 'colsample_bytree' : [0.8,0.9],
}

xgb = xgb.XGBRegressor(colsample_bytree= 0.8, gamma= 0, learning_rate= 0.15, max_depth= 10)

best_xgb = get_best_params(xgb,xg_param)

print(best_xgb)