#train 데이터
#착과량(int) / 수고(m) / 수관폭1(min) / 수관폭2(max) / 수관폭평균 / 220901 ~ 221128 새순 및 엽록소 - 89데이터 * 2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#출력 설정
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

# plt.plot('착과량(int)', '수고(m)', 'o', data=train)

# plt.plot('착과량(int)', '수관폭평균', 'o', data=train)

train_a = train.copy()

print(train_a.columns)
train_a['나무면적'] = train_a['수관폭평균'] * train_a['수고(m)']



xfet = ['수고(m)', '수관폭평균', '수관폭1(min)', '수관폭2(max)', '나무면적']

xfets = train_a[xfet].columns

# 각 컬럼 컬럼들과 대여량 과의 관계 시각화
# for i in xfets:
#     sns.lmplot(x=i, y='착과량(int)', data=train_a)
#     plt.show()
col_saesoon = train.iloc[:, 4:93]
col_yeoprok = train.iloc[:, 93:]
col_saesoon.plot()
plt.show()
col_yeoprok.plot()
plt.show()