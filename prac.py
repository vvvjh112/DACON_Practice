import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

print(train.isna().sum())  # 결측치가 사라진 것 확인가능.