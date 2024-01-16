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

# from sklearn.metrics import mean_squared_log_error, make_scorer
# def rmsle(y_actual, y_pred):
#     diff = np.log(y_pred + 1) - np.log(y_actual + 1)
#     mean_error = np.mean(np.square(diff))
#
#     return np.sqrt(mean_error)
#
# rmsle_score = make_scorer(rmsle, greater_is_better=False)


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
location_pattern = r'(\S+) (\S+) (\S+)'

train[['도시', '구', '동']] = train['시군구'].str.extract(location_pattern)
train = train.drop(columns=['시군구'])

test[['도시', '구', '동']] = test['시군구'].str.extract(location_pattern)
test = test.drop(columns=['시군구'])

#공휴일 체크
# print(train['사고일시'].dt.year.unique())
# print(test['사고일시'].dt.year.unique())
# train[2019 2020 2021], test[2022]
#공휴일 api를 통해 가져오기(대체휴무일때문)
import requests
from datetime import datetime
import json
from pandas import json_normalize

def getholiday(year):
    today_year = year

    KEY = "JWgg0HGk6X1%2FiSamZNl29O5awvu46mP%2BwM%2Fj8WNoLfNNfMeo2zhjPECwNdheapXHpIKbEZ0GCg1sWUm%2BrTdBfg%3D%3D"
    url = (
        "http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo?_type=json&numOfRows=50&solYear="
        + str(today_year)
        + "&ServiceKey="
        + str(KEY)
    )
    response = requests.get(url)
    if response.status_code == 200:
        json_ob = json.loads(response.text)
        holidays_data = json_ob["response"]["body"]["items"]["item"]
        dataframe = json_normalize(holidays_data)
    # dateName = dataframe.loc[dataframe["locdate"] == int(today), "dateName"]
    # print(dateName)
    result = dataframe["locdate"].astype(str)
    return result.to_list()

holiday_2019 = getholiday(2019)
holiday_2020 = getholiday(2020)
holiday_2021 = getholiday(2021)
holiday_2022 = getholiday(2022)

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

#기상상태, 요일별, 월별, 공휴일 ECLO 시각화해보기
train['연'] = train['사고일시'].dt.year
train['월'] = train['사고일시'].dt.month
train['일'] = train['사고일시'].dt.day
train['시간'] = train['사고일시'].dt.hour

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

group_road = train.groupby(['도로형태']).mean('ECLO')
group_road = group_road[['ECLO']]

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

gr = group_road.plot(title='도로형태',kind = 'bar')
# plt.show()

print(train.info())
print(test.info())

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
train_1 = train_1.drop(['연','월','일','사고일시'],axis=1)
train_1 = train_1.drop(lst,axis=1)
print(train_1.info())

#타겟인코딩, 라벨인코딩, 원핫인코딩

#피처 중요도
#단지 예측 뿐 아니라 train 셋에 있는 데이터를 바탕으로 사고를 줄일 수 있는 방법 제시.
#카메라랑 사고 그래프 보여주면서 카메라가 효과적 이런거