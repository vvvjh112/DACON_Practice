import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# row 생략 없이 출력
pd.set_option('display.max_rows', None)
# col 생략 없이 출력
pd.set_option('display.max_columns', None)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
gender = pd.read_csv('age_gender_info.csv')
submission = pd.read_csv('sample_submission.csv')

# ['단지코드', '총세대수', '임대건물구분', '지역', '공급유형', '전용면적', '전용면적별세대수', '공가수',
#        '자격유형', '임대보증금', '임대료', '도보 10분거리 내 지하철역 수(환승노선 수 반영)',
#        '도보 10분거리 내 버스정류장 수', '단지내주차면수', '등록차량수']
# ['단지코드', '총세대수', '임대건물구분', '지역', '공급유형', '전용면적', '전용면적별세대수', '공가수',
#        '자격유형', '임대보증금', '임대료', '도보 10분거리 내 지하철역 수(환승노선 수 반영)',
#        '도보 10분거리 내 버스정류장 수', '단지내주차면수']
# ['지역', '10대미만(여자)', '10대미만(남자)', '10대(여자)', '10대(남자)', '20대(여자)',
#        '20대(남자)', '30대(여자)', '30대(남자)', '40대(여자)', '40대(남자)', '50대(여자)',
#        '50대(남자)', '60대(여자)', '60대(남자)', '70대(여자)', '70대(남자)', '80대(여자)',
#        '80대(남자)', '90대(여자)', '90대(남자)', '100대(여자)', '100대(남자)']

# print(train.info())
# print(train.head())

#임대료에 - 를 0으로 바꿔줘야함
#자격유형 BFO가 test에는 없음 - 삭제해도 될 듯
a = train['자격유형'].value_counts().reset_index(name = 'name').sort_values('자격유형',ascending=True)
b = test['자격유형'].value_counts().reset_index(name = 'name').sort_values('자격유형',ascending=True)
print(a)
print(b)
#지하철/정류장 수도 2개를 넘어가면 2, 1개면 1, 없으면 0 3가지로 해도 괜찮을 듯

#결측값
#중복값
#이상값
#상관계수

#여러모델 평균값 앙상블 (xgb,randomforest,lgbm)
#catboost
#pycaret
