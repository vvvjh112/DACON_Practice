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

def main_menu(df,mode):

    temp1 = list(df['중식메뉴'])
    temp2 = list(df['석식메뉴'])
    if (mode == 1):  #밥 추출
        for i in range(len(temp1)):
            temp1[i] = temp1[i][0]
            temp2[i] = temp1[i][0]

        df['점심_밥'] = temp1
        df['석식_밥'] = temp2
        
    if (mode == 2):  #국 추출
        for i in range(len(temp1)):
            temp1[i] = temp1[i][0]
            temp2[i] = temp1[i][0]

        df['점심_국'] = temp1
        df['석식_국'] = temp2

    if (mode == 3):  #메인메뉴 추출
        for i in range(len(temp1)):
            temp1[i] = temp1[i][0]
            temp2[i] = temp1[i][0]

        df['점심_국'] = temp1
        df['석식_국'] = temp2
    return df

#메인메뉴는 두개긴 한데 우선 하나만 해서 점수 뽑아보자
train = main_menu(train,1)
train = main_menu(train,2)
train = main_menu(train,3)






# print(train[train['석식계']== 0])

print(train['점심_밥'].head())
print(train.head())
print(test.head())