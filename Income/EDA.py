import pandas as pd
import numpy as np
import model_tuned as mt
import matplotlib.pyplot as plt
import seaborn as sns

# Age (나이): 가능한 값: 정수형 변수로, 양의 정수값으로 표현될 것으로 예상됩니다.
# Gender (성별): 가능한 값: 남성, 여성, 그 외 성별 정의
# Education_Status (교육 수준): 가능한 값: 초등학교, 중학교, 고등학교, 대학교 이상 등
# Employment_Status (고용 상태): 가능한 값: 고용 중, 실업 중, 자영업, 무급 가족 고용 등
# Working_Week (Yearly) (연간 근무 주): 가능한 값: 연간 근무 주의 수를 나타내는 연속형 변수
# Industry_Status (산업 분야): 가능한 값: 제조업, 서비스업, 농업 등
# Occupation_Status (직업 상태): 가능한 값: 사무직, 기술직, 서비스직 등
# Race (인종): 가능한 값: 백인, 흑인, 아시안, 히스패닉 등
# Hispanic_Origin (히스패닉 출신): 가능한 값: 히스패닉, 비히스패닉 등
# Martial_Status (결혼 상태): 가능한 값: 결혼, 이혼, 싱글 등
# Household_Status (가구 상태): 가능한 값: 가구주, 부모님과 동거, 비혼인듯 등
# Household_summary (가구 개요): 가능한 값: 가구 구성원의 수 등
# Citizenship (시민권): 가능한 값: 시민권 보유 여부 등
# Birth_Country (출생 국가): 가능한 값: 미국, 다른 국가 등
# Birth_Country (Father) (부의 출생 국가): 가능한 값: 미국, 다른 국가 등
# Birth_Country (Mother) (모의 출생 국가): 가능한 값: 미국, 다른 국가 등
# Tax_Status (세금 상태): 가능한 값: 납세자, 비납세자 등
# Gains (이득): 가능한 값: 경제적 이득을 나타내는 연속형 변수
# Losses (손실): 가능한 값: 경제적 손실을 나타내는 연속형 변수
# Dividends (배당금): 가능한 값: 배당금 수입을 나타내는 연속형 변수
# Incom_Status (소득 상태): 가능한 값: 소득 수준의 범주 또는 소득 대비 생활비 비율 등
# Income : 목표변수 , 1시간당 소득

# row 생략 없이 출력
pd.set_option('display.max_rows', None)
# col 생략 없이 출력
pd.set_option('display.max_columns', None)


train = pd.read_csv('dataset/train.csv').drop('ID',axis = 1)
test = pd.read_csv('dataset/test.csv').drop('ID',axis = 1)

# train = train.loc[train['Gains'] < 99999]
# train = train.loc[train['Losses'] < 4356]
# train = train.loc[train['Dividends']<45000]

print(train.head())

# print(train.info())
# print(test.info())

numeric_columns = train.select_dtypes(include=['int64', 'float64']).columns
category_columns = train.select_dtypes(exclude=['int64', 'float64']).columns
print("수치형 컬럼 : ",numeric_columns)
print("카테고리 컬럼 : ",category_columns)

print(train[train['Income']==0].head(500))

#상관계수 (수치)
correlation_with_target = train[numeric_columns].corr()

# 상관 계수 히트맵 그리기
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_with_target, annot=True, cmap='coolwarm', fmt=".2f", cbar=False)
plt.title('Correlation with Income Heatmap')
plt.show()

#Working_Week 0.42로 가장 상관계수가 높으며 나머지는 영향이 거의 없다고 봐도 무방하다

def plot_distribution(data, column):
    plt.figure(figsize=(10, 6))

    # 박스 플롯 그리기
    plt.subplot(1, 2, 1)
    sns.boxplot(data=data[column], color='blue')
    plt.title(f'Boxplot of {column}')
    plt.xlabel(column)

    # 히스토그램 그리기
    plt.subplot(1, 2, 2)
    sns.histplot(data=data[column], bins=30, kde=True, edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

#
for i in numeric_columns:
    print(train[i].describe())
    plot_distribution(train, i)

#Gains 이상치 처리 필요 99999

#Gains, Losses, Dividends, Income 로그 스케일링
columns_to_transform = ['Gains', 'Losses', 'Dividends', 'Income']
train[columns_to_transform] = np.log1p(train[columns_to_transform])


for i in numeric_columns:
    print(train[i].describe())
    plot_distribution(train, i)

def scatter_plot(df, target_column, feature_column):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=feature_column, y=target_column, color='skyblue')
    plt.title(f'Scatter plot of {target_column} vs {feature_column}')
    plt.xlabel(feature_column)
    plt.ylabel(target_column)
    plt.show()

for i in numeric_columns:
    scatter_plot(train,'Income',i)
#
for column in category_columns:
    unique_count = train[column].nunique()
    print(f"{column}: {unique_count}")

# Gender: 2
# Education_Status: 17
# Employment_Status: 8
# Industry_Status: 24
# Occupation_Status: 15
# Race: 5
# Hispanic_Origin: 10
# Martial_Status: 7
# Household_Status: 31
# Household_Summary: 8
# Citizenship: 5
# Birth_Country: 43
# Birth_Country (Father): 43
# Birth_Country (Mother): 43
# Tax_Status: 6
# Income_Status: 3


def plot_category_count(df, category_column):
    # 카테고리 컬럼의 카운트 계산
    category_counts = df[category_column].value_counts(ascending = False)
    print(category_counts/train.shape[0])
    # 카운트를 그래프로 시각화
    plt.figure(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values)
    plt.title(f'Count of Each {i}')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45,fontsize = 8)  # x 축 레이블 회전
    plt.show()

for i in category_columns:
    plot_category_count(train,i)

#Gender
# F    0.5236
# M    0.4764

# Education_Status
# High graduate                     0.32470
# College                           0.18970
# Children                          0.11040
# Bachelors degree                  0.09780
# High Junior                       0.04100
# High Sophomore                    0.03995
# Associates degree (Vocational)    0.03705
# Associates degree (Academic)      0.03045
# High Freshman                     0.02775
# Middle (7-8)                      0.02740
# Masters degree                    0.02525
# Elementary (5-6)                  0.01585
# High Senior                       0.01210
# Elementary (1-4)                  0.00750
# Professional degree               0.00555
# Doctorate degree                  0.00435
# Kindergarten                      0.00320

# Employment_Status
# Children or Armed Forces         0.55710
# Full-Time                        0.32940
# Not Working                      0.06105
# Choice Part-Time                 0.02965
# Part-Time (Usually Part-Time)    0.00790
# Seeking Full-Time                0.00605
# Part-Time (Usually Full-Time)    0.00500
# Seeking Part-Time                0.00385

# Industry_Status
# Not in universe or children                     0.23440
# Retail                                          0.15745
# Manufacturing (Durable)                         0.07875
# Manufacturing (Non-durable)                     0.06115
# Education                                       0.05205
# Business & Repair                               0.04235
# Medical (except Hospitals)                      0.04190
# Construction                                    0.04160
# Hospitals                                       0.04105
# Finance Insurance & Real Estate                 0.03635
# Transportation                                  0.03465
# Public Administration                           0.03205
# Other professional services                     0.02385
# Wholesale                                       0.02250
# Personal Services (except Private Household)    0.02145
# Social Services                                 0.01835
# Entertainment                                   0.01390
# Agriculture                                     0.01340
# Utilities & Sanitary                            0.01010
# Communications                                  0.00975
# Private Household Services                      0.00625
# Mining                                          0.00535
# Forestry & Fisheries                            0.00130
# Armed Forces                                    0.00005

# Occupation_Status
# Unknown                             0.23440
# Admin Support (include Clerical)    0.13545
# Services                            0.11565
# Craft & Repair                      0.09345
# Sales                               0.08460
# Professional                        0.07440
# Machine Operators & Inspectors      0.06915
# Management                          0.05555
# Handlers/Cleaners                   0.04185
# Transportation                      0.03450
# Technicians & Support               0.02790
# Farming & Forestry & Fishing        0.01480
# Protective Services                 0.01300
# Private Household Services          0.00525
# Armed Forces                        0.00005

# Race
# White                           0.84225
# Black                           0.10610
# Asian/Pacific                   0.02555
# Other                           0.01480
# Native American/Aleut/Eskimo    0.01130

# Hispanic_Origin
#  All other                    0.88845
#  Mexican-American             0.03270
#  Mexican (Mexicano)           0.03120
#  Central or South American    0.01625
#  Puerto Rican                 0.01215
#  Other Spanish                0.01005
#  Cuban                        0.00380
#  NA                           0.00295
#  Chicano                      0.00135
#  Do not know                  0.00110

# Martial_Status
# Married                         0.47770
# Single                          0.36900
# Divorced                        0.08165
# Widowed                         0.03780
# Separated                       0.02180
# Married (Spouse Absent)         0.00855
# Married (Armed Force Spouse)    0.00350

# Household_Status
# Householder                                                               0.30435
# Spouse of householder                                                     0.23970
# Child <18 never marr not in subfamily                                     0.13350
# Nonfamily householder                                                     0.12325
# Child 18+ never marr Not in a subfamily                                   0.09300
# Secondary individual                                                      0.04225
# Other Rel 18+ never marr not in subfamily                                 0.00975
# Other Rel 18+ ever marr not in subfamily                                  0.00770
# Child 18+ ever marr Not in a subfamily                                    0.00590
# Child 18+ ever married Responsible Person of subfamily                    0.00505
# Child 18+ never married Responsible Person of subfamily                   0.00480
# Grandchild <18 never married child of subfamily Responsible Person        0.00440
# Responsible Person of unrelated subfamily                                 0.00435
# Other Relative 18+ ever married Responsible Person of subfamily           0.00385
# Other Relative 18+ spouse of subfamily Responsible Person                 0.00365
# Grandchild 18+ never marr not in subfamily                                0.00315
# Grandchild <18 never marr not in subfamily                                0.00215
# Child under 18 of Responsible Person of unrelated subfamily               0.00195
# Other Rel <18 never marr not in subfamily                                 0.00180
# Other Relative <18 never married child of subfamily Responsible Person    0.00135
# In group quarters                                                         0.00130
# Other Relative 18+ never married Responsible Person of subfamily          0.00095
# Child 18+ spouse of subfamily Responsible Person                          0.00080
# Child <18 never married Responsible Person of subfamily                   0.00040
# Child <18 ever marr not in subfamily                                      0.00015
# Spouse of Responsible Person of unrelated subfamily                       0.00015
# Grandchild 18+ ever marr not in subfamily                                 0.00015
# Grandchild 18+ spouse of subfamily Responsible Person                     0.00005
# Grandchild 18+ ever married Responsible Person of subfamily               0.00005
# Child <18 ever married Responsible Person of subfamily                    0.00005
# Other Relative <18 ever married Responsible Person of subfamily           0.00005

# Household_Summary
# Householder                             0.42760
# Spouse of householder                   0.23970
# Child under 18 never married            0.13395
# Child 18 or older                       0.10960
# Nonrelative of householder              0.04870
# Other relative of householder           0.03905
# Group Quarters- Secondary individual    0.00120
# Child under 18 ever married             0.00020

# Citizenship
# Native                                         0.89125
# Foreign-born (Non-US Citizen)                  0.06290
# Foreign-born (Naturalized US Citizen)          0.02940
# Native (Born Abroad)                           0.00985
# Native (Born in Puerto Rico or US Outlying)    0.00660


# Birth_Country
# US                              0.89125
# Mexico                          0.02700
# Unknown                         0.01650
# Puerto-Rico                     0.00585
# Philippines                     0.00560
# Germany                         0.00450
# Canada                          0.00375
# El-Salvador                     0.00340
# Cuba                            0.00290
# India                           0.00275
# Dominican-Republic              0.00245
# England                         0.00235
# Poland                          0.00225
# Jamaica                         0.00225
# Columbia                        0.00180
# Italy                           0.00175
# South Korea                     0.00155
# Vietnam                         0.00155
# Ecuador                         0.00155
# Japan                           0.00150
# Portugal                        0.00145
# Nicaragua                       0.00140
# China                           0.00140
# Guatemala                       0.00140
# Haiti                           0.00125
# Iran                            0.00120
# Peru                            0.00110
# Ireland                         0.00105
# Hong Kong                       0.00080
# Outlying-U S (Guam USVI etc)    0.00075
# France                          0.00070
# Honduras                        0.00070
# Greece                          0.00065
# Laos                            0.00055
# Taiwan                          0.00055
# Thailand                        0.00050
# Cambodia                        0.00040
# Trinadad&Tobago                 0.00040
# Yugoslavia                      0.00035
# Scotland                        0.00035
# Panama                          0.00020
# Hungary                         0.00020
# Holand-Netherlands              0.00015

# Birth_Country (Father)
# US                              0.82815
# Mexico                          0.04225
# Unknown                         0.02905
# Puerto-Rico                     0.01030
# Italy                           0.00985
# Canada                          0.00640
# Philippines                     0.00605
# Poland                          0.00550
# Dominican-Republic              0.00495
# Germany                         0.00460
# El-Salvador                     0.00415
# Cuba                            0.00390
# England                         0.00345
# China                           0.00345
# India                           0.00340
# Jamaica                         0.00280
# Ireland                         0.00235
# Portugal                        0.00235
# Columbia                        0.00225
# Ecuador                         0.00210
# Haiti                           0.00185
# South Korea                     0.00170
# Guatemala                       0.00165
# Greece                          0.00165
# Vietnam                         0.00160
# Nicaragua                       0.00150
# Japan                           0.00140
# Scotland                        0.00135
# Peru                            0.00130
# Iran                            0.00125
# Hungary                         0.00110
# Yugoslavia                      0.00090
# France                          0.00085
# Cambodia                        0.00065
# Laos                            0.00060
# Honduras                        0.00060
# Outlying-U S (Guam USVI etc)    0.00060
# Trinadad&Tobago                 0.00055
# Taiwan                          0.00050
# Hong Kong                       0.00040
# Thailand                        0.00035
# Holand-Netherlands              0.00025
# Panama                          0.00010

# Birth_Country (Mother)
# US                              0.82970
# Mexico                          0.04245
# Unknown                         0.02585
# Puerto-Rico                     0.00970
# Canada                          0.00795
# Italy                           0.00760
# Philippines                     0.00640
# Germany                         0.00585
# Poland                          0.00520
# El-Salvador                     0.00495
# England                         0.00440
# Dominican-Republic              0.00375
# Cuba                            0.00365
# India                           0.00335
# China                           0.00300
# Ireland                         0.00290
# Jamaica                         0.00280
# Columbia                        0.00225
# Ecuador                         0.00220
# Portugal                        0.00220
# South Korea                     0.00205
# Vietnam                         0.00195
# Japan                           0.00185
# Guatemala                       0.00180
# Nicaragua                       0.00175
# Haiti                           0.00170
# Peru                            0.00130
# Iran                            0.00130
# France                          0.00125
# Greece                          0.00115
# Hungary                         0.00090
# Yugoslavia                      0.00085
# Honduras                        0.00085
# Scotland                        0.00075
# Outlying-U S (Guam USVI etc)    0.00070
# Trinadad&Tobago                 0.00070
# Thailand                        0.00060
# Cambodia                        0.00055
# Taiwan                          0.00055
# Hong Kong                       0.00050
# Laos                            0.00040
# Panama                          0.00020
# Holand-Netherlands              0.00020

# Tax_Status
# Married Filling Jointly both under 65 (MFJ)                 0.42940
# Single                                                      0.27735
# Nonfiler                                                    0.19370
# Head of Household (HOH)                                     0.06085
# Married Filling Jointly both over 65 (MFJ)                  0.02345
# Married Filling Jointly one over 65 & one under 65 (MFJ)    0.01525

# Income_Status
# Under Median    0.66185
# Unknown         0.30130
# Over Median     0.03685


# for i in category_columns:
#     group = train.groupby([i]).mean('Income')
#     plt.title(f'{i}_Mean of Imcome')
#     plt.xticks(fontsize=7, rotation=45, ha='right')
#     sns.lineplot(x=group.index, y='Income', data=group, marker='o')
#     plt.show()


#14살까지는 소득 0
#에듀케이션이 child면 소득 0
#employment_status 가 not working 이면 0