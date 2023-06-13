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
plt.plot('hour','count','*',data=train)
plt.show()