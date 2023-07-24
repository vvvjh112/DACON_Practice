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

#이상값 체크 / 시각화
#모델 선택 및 하이퍼 파라미터 최적화
#스코어 확인
#앙상블 필요 여부 판단


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.columns)
# print(train)
