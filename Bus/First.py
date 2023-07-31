import pandas as pd
import numpy as np
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

    return temp

tmp = df_del(train)
tmp = dist_check(tmp)
tmp = tmp.drop(['now_latitude','now_longitude','next_latitude','next_longitude'],axis=1)
print(tmp.columns)
#Index(['id', 'now_arrive_time', 'distance', 'next_arrive_time', 'hot'], dtype='object')

#이상값 체크 및 모델 스코어 비교 해보자
