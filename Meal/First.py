import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

#데이터 불러오기
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

#오타수정
train.at[1142, '중식메뉴'] = '쌀밥/곤드레밥/찰현미밥 된장찌개 돼지고추장불고기 버섯잡채 삼색물만두무침 겉절이김치/양념장 견과류샐러드*요거트D'
train['중식메뉴'] = train['중식메뉴'].str.replace('삽겹', '삼겹')

#요일 수정
train['일자'] = pd.to_datetime(train['일자'])
test['일자'] = pd.to_datetime(test['일자'])

train['요일'] = train['일자'].dt.day_name().str[:2].map({'Mo' : 5, 'Tu' : 4, 'We' : 3, 'Th' : 2, 'Fr' : 1})
test['요일'] = test['일자'].dt.day_name().str[:2].map({'Mo' : 5, 'Tu' : 4, 'We' : 3, 'Th' : 2, 'Fr' : 1})

# train['month'] = train['일자'].dt.month
# test['month'] = test['일자'].dt.month

print(train.columns)

#메뉴 전처리 시도
train = train.drop(['조식메뉴'],axis=1)
train['중식메뉴'] = train['중식메뉴'].str.split(' ')
train['석식메뉴'] = train['석식메뉴'].str.split(' ')


test = test.drop(['조식메뉴'],axis=1)
test['중식메뉴'] = test['중식메뉴'].str.split(' ')
test['석식메뉴'] = test['석식메뉴'].str.split(' ')

def get_token(data) :
    tokens = []
    for token in data :
        s_list = []
        for t in token :
            if t.startswith('(N') :
                s_list.append(t)
            elif (t.startswith('(') == False) & (len(t) > 1) :
                s_list.append(t)
            else :
                pass
        tokens.append(s_list)
    return tokens

train['중식메뉴'] = get_token(train['중식메뉴'])
train['석식메뉴'] = get_token(train['석식메뉴'])

test['중식메뉴'] = get_token(test['중식메뉴'])
test['석식메뉴'] = get_token(test['석식메뉴'])

idx = train[train['석식계']<1].index
train = train.drop(idx)

train = train.drop(586)

# print(train.loc[[586],:])
# print(train['석식메뉴'].head(566))
def main_menu(df,mode):
    temp1 = list(df['중식메뉴'])
    temp2 = list(df['석식메뉴'])
    # print(temp2[565])
    # print(temp1[565])
    if (mode == 1):  #밥 추출
        for i in range(len(temp1)):
            temp1[i] = temp1[i][0]
            temp2[i] = temp2[i][0]

        df['점심_밥'] = temp1
        df['석식_밥'] = temp2
        
    if (mode == 2):  #국 추출
        for i in range(len(temp1)):
            temp1[i] = temp1[i][1]
            temp2[i] = temp2[i][1]

        df['점심_국'] = temp1
        df['석식_국'] = temp2

    if (mode == 3):  #메인메뉴 추출
        # +
        for i in range(len(temp1)):
            temp1[i] = temp1[i][2]
            temp2[i] = temp2[i][2]

        df['점심_메인'] = temp1
        df['석식_메인'] = temp2
    return df

#메인메뉴는 두개긴 한데 우선 하나만 해서 점수 뽑아보자
train = main_menu(train,1)
train = main_menu(train,2)
train = main_menu(train,3)

test = main_menu(test,1)
test = main_menu(test,2)
test = main_menu(test,3)

train = train.drop(['중식메뉴','석식메뉴'],axis=1)
test = test.drop(['중식메뉴','석식메뉴'],axis=1)


#컬럼순서 변경
train = train[['일자', '요일', '본사정원수', '본사휴가자수', '본사출장자수', '본사시간외근무명령서승인건수',
       '현본사소속재택근무자수', '점심_밥', '점심_국', '점심_메인', '석식_밥',  '석식_국',  '석식_메인','중식계','석식계']]
test = test[['일자', '요일', '본사정원수', '본사휴가자수', '본사출장자수', '본사시간외근무명령서승인건수',
       '현본사소속재택근무자수', '점심_밥', '점심_국', '점심_메인', '석식_밥',  '석식_국',  '석식_메인']]


# print(train.head())
# print(test.head())

#라벨링 하고 pyc뭐시기 써보자


#라벨링
def label_encoder(df):
    encoder1 = LabelEncoder() # 인코더 생성
    encoder2 = LabelEncoder()  # 인코더 생성
    encoder3 = LabelEncoder()  # 인코더 생성
    encoder4 = LabelEncoder()  # 인코더 생성
    encoder5 = LabelEncoder()  # 인코더 생성
    encoder6 = LabelEncoder()  # 인코더 생성

    category1 = list(df['점심_밥'].values) # 카테고리
    category2 = list(df['점심_국'].values) # 카테고리
    category3 = list(df['점심_메인'].values) # 카테고리

    category4 = list(df['석식_밥'].values) # 카테고리
    category5 = list(df['석식_국'].values) # 카테고리
    category6 = list(df['석식_메인'].values) # 카테고리

    category_set1 = set(category1)
    category_set2 = set(category2)
    category_set3 = set(category3)

    category_set4 = set(category4)
    category_set5 = set(category5)
    category_set6 = set(category6)

    encoder1.fit(list(category_set1)) # 인코딩
    encoder2.fit(list(category_set2)) # 인코딩
    encoder3.fit(list(category_set3)) # 인코딩
    encoder4.fit(list(category_set4)) # 인코딩
    encoder5.fit(list(category_set5)) # 인코딩
    encoder6.fit(list(category_set6)) # 인코딩

    # 모든 학습, 시험 데이터의 정류장 정보 치환
    df['점심_밥'] = encoder1.transform(df['점심_밥'])
    df['점심_국'] = encoder2.transform(df['점심_국'])
    df['점심_메인'] = encoder3.transform(df['점심_메인'])
    df['석식_밥'] = encoder4.transform(df['석식_밥'])
    df['석식_국'] = encoder5.transform(df['석식_국'])
    df['석식_메인'] = encoder6.transform(df['석식_메인'])

    return df

train = label_encoder(train)
test = label_encoder(test)

print(train.head())
# print(test.head())


#pycaret
from pycaret.regression import *
Xtrain = train[['요일', '본사정원수','본사휴가자수','본사출장자수', '본사시간외근무명령서승인건수','현본사소속재택근무자수', '점심_밥','점심_국', '점심_메인', '중식계']]
reg = setup(session_id = 1, data = Xtrain, target = '중식계', normalize = True, transformation=True)

pycaret_regression_models = compare_models(n_select=25, sort='MAE', include = ['lr','lasso','ridge','en','lar','llar','omp','br','ard','par','ransac','tr',
                                                                              'huber','kr','svm','knn','dt','rf','et','ada','gbr','mlp','xgboost','lightgbm','catboost'])