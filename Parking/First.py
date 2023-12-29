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
gender = pd.read_csv('age_gender_info.csv')
submission = pd.read_csv('sample_submission.csv')

train.rename(columns={'도보 10분거리 내 지하철역 수(환승노선 수 반영)': '지하철', '도보 10분거리 내 버스정류장 수': '버스'},inplace=True)
test.rename(columns={'도보 10분거리 내 지하철역 수(환승노선 수 반영)': '지하철', '도보 10분거리 내 버스정류장 수': '버스'},inplace=True)

# ['단지코드', '총세대수', '임대건물구분', '지역', '공급유형', '전용면적', '전용면적별세대수', '공가수',
#        '자격유형', '임대보증금', '임대료', '지하철', '버스', '단지내주차면수', '등록차량수']
# ['단지코드', '총세대수', '임대건물구분', '지역', '공급유형', '전용면적', '전용면적별세대수', '공가수',
#        '자격유형', '임대보증금', '임대료', '지하철', '버스, '단지내주차면수']
# ['지역', '10대미만(여자)', '10대미만(남자)', '10대(여자)', '10대(남자)', '20대(여자)',
#        '20대(남자)', '30대(여자)', '30대(남자)', '40대(여자)', '40대(남자)', '50대(여자)',
#        '50대(남자)', '60대(여자)', '60대(남자)', '70대(여자)', '70대(남자)', '80대(여자)',
#        '80대(남자)', '90대(여자)', '90대(남자)', '100대(여자)', '100대(남자)']

#평가지표 MAE

# print(train.info())
# print(train.head())
#
# a = train['지역'].unique()
# b = test['지역'].unique()
# print(a)
# print(b)
# print(set(a)-set(b))

#결측값
train['지하철'].fillna(0,inplace=True)
train['버스'].fillna(0,inplace=True)

train[['임대보증금','임대료']] = train[['임대보증금', '임대료']].fillna("0").replace("-", "0").astype(int)

test['지하철'].fillna(0,inplace=True)
test['버스'].fillna(0,inplace=True)

test[['임대보증금','임대료']] = test[['임대보증금', '임대료']].fillna("0").replace("-", "0").astype(int)
#test에 자격유형 결측값 있음
#C2411 = A / C2253 = C
test.loc[196, '자격유형'] = 'A' #C2411
test.loc[258, '자격유형'] = 'C' #C2253
# print(test[test.isna().any(axis=1)])
# print(test[test['단지코드']=='C2253'])

# print(train.isna().sum())
print(test.isna().sum())

#라벨인코딩
from sklearn.preprocessing import LabelEncoder
#단지코드는 나중에
label_lst = ['임대건물구분', '지역', '공급유형','자격유형']

for i in label_lst:
    train[i] = LabelEncoder().fit_transform(train[i])
    test[i] = LabelEncoder().fit_transform(test[i])
print(train.info())
#임대료에 - 를 0으로 바꿔줘야함
#자격유형 B, F, O가 test에는 없음 - 삭제해도 될 듯 / test에 nan값 있음
#지하철/정류장 수도 2개를 넘어가면 2, 1개면 1, 없으면 0 3가지로 해도 괜찮을 듯
#공급유형 - test에는 '장기전세', '공공분양', '공공임대(5년)' 없음
#test에 서울특별시 없음
#수도권 / 광역시 / 비수도권 구분

#상관계수
# 상관계수 계산
import seaborn as sns

correlation_matrix = train.drop('단지코드',axis=1).corr()
#
# # 히트맵 그리기
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
#
# # 플롯 제목 추가
# plt.title('Correlation Heatmap')
#
# # 그래프 표시
# plt.show()

# 컬럼별 상관계수 산점도 하나씩 보기
# for i in (train.drop('단지코드',axis=1).columns):
#
#     #상관계수 산점도
#     sns.scatterplot(x=i, y='등록차량수', data=train.drop('단지코드',axis=1))
#
#     # 플롯 제목 추가
#     plt.title('Correlation Heatmap')
#
#     # 그래프 표시
#     plt.show()

cor_lst = abs(correlation_matrix['등록차량수']).sort_values(ascending=False).head(6).reset_index(name='상관계수')
cor_lst = np.array(cor_lst['index'])[1:]
print(cor_lst)
#상관계수 높은거 5개 ['단지내주차면수' '임대료' '공급유형' '임대건물구분' '임대보증금']

#중복값
#이상값 (unique)
#병합할 컬럼 있는지

from sklearn.metrics import *

#train/test 구분 및 독립변수 타겟변수 설정
from sklearn.model_selection import train_test_split

x = train.drop('등록차량수',axis = 1)
y = train['등록차량수']


#여러모델 평균값 앙상블 (xgb,randomforest,lgbm)
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

#pycaret
from pycaret.regression import *

clf = setup(data=train, target='등록차량수', train_size=0.8)
best_model = compare_models()
compare_models(n_select = 5, sort = 'MAE')
et = create_model('et')
tuned_et = tune_model(et,optimize = 'MAE')
final_model = finalize_model(tuned_et)
prediction = predict_model(final_model, data = test)
result = prediction['prediction_label']
submission['num'] = result
submission.to_csv('result4.csv',index=False)
#302점,,,
#catboost

