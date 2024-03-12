import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sessionID : 세션 ID
# userID : 사용자 ID
# TARGET : 세션에서 발생한 총 조회수
# browser : 사용된 브라우저
# OS : 사용된 기기의 운영체제
# device : 사용된 기기
# new : 첫 방문 여부 (0: 첫 방문 아님, 1: 첫 방문)
# quality : 세션의 질 (거래 성사를 기준으로 측정된 값, 범위: 1~100)
# duration : 총 세션 시간 (단위: 초)
# bounced : 이탈 여부 (0: 이탈하지 않음, 1: 이탈함)
# transaction : 세션 내에서 발생의 거래의 수
# transaction_revenue : 총 거래 수익
# continent : 세션이 발생한 대륙
# subcontinent : 세션이 발생한 하위 대륙
# country : 세션이 발생한 국가
# traffic_source : 트래픽이 발생한 소스
# traffic_medium : 트래픽 소스의 매체
# keyword : 트래픽 소스의 키워드, 일반적으로 traffic_medium이 organic, cpc인 경우에 설정
# referral_path : traffic_medium이 referral인 경우 설정되는 경로


#plt 한글출력
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# row 생략 없이 출력
pd.set_option('display.max_rows', None)
# col 생략 없이 출력
pd.set_option('display.max_columns', None)


train = pd.read_csv('../Data/train.csv')

print(train.info())


# 데이터프레임에서 수치형 데이터만 선택
numeric_columns = train.select_dtypes(include=['int64', 'float64']).columns
category_columns = train.select_dtypes(exclude=['int64', 'float64']).columns


#상관계수 (수치)
# 'TARGET'과 다른 수치형 열 간의 상관 계수 계산
correlation_with_target = train[numeric_columns].corr()['TARGET'].drop('TARGET')

# 상관 계수 히트맵 그리기
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_with_target.to_frame(), annot=True, cmap='coolwarm', fmt=".2f", cbar=False)
plt.title('Correlation with TARGET Heatmap')
plt.show()

# quality와 druation은 0.6 이상의 양의 상관관계이며, bounced는 음의 상관관계를 갖고 있음
# quality와 duration은 수치형으로 곱하여 둘의 특성을 동시에 고려하는 변수를 파생하여도 좋을것 같음.

# 데이터프레임의 특정 열에 대한 박스 플롯과 히스토그램을 그림
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


# 주어진 데이터프레임에서 타겟 열과 다른 연속형 열 간의 산점도를 그립니다.
def scatter_plot(df, target_column, feature_column):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=feature_column, y=target_column, color='skyblue')
    plt.title(f'Scatter plot of {target_column} vs {feature_column}')
    plt.xlabel(feature_column)
    plt.ylabel(target_column)
    plt.show()

for i in numeric_columns:
    scatter_plot(train,'TARGET',i)

for column in category_columns:
    unique_count = train[column].nunique()
    print(f"{column}: {unique_count}")


#시각화 이전 그룹화
group_os = train.groupby(['OS']).mean('TARGET')

group_browser = train.groupby(['browser']).mean('TARGET')[['TARGET']].reset_index('browser')
group_browser = group_browser[~group_browser['browser'].str.startswith(';__CT_JOB_ID__:')]

group_device = train.groupby(['device']).mean('TARGET')

group_new = train.groupby(['new']).mean('TARGET')

group_bounced = train.groupby(['bounced']).mean('TARGET')

group_country = train.groupby(['country']).mean('TARGET')

group_source = train.groupby(['traffic_source']).mean('TARGET')

plt.title('OS별 평균 조회수')
plt.xticks(fontsize = 7, rotation = 45, ha = 'right')
sns.lineplot(x=group_os.index, y= 'TARGET', data=group_os, marker = 'o')
# plt.show()

plt.title('브라우저별 평균 조회수')
plt.xticks(fontsize = 7, rotation = 45, ha = 'right')
sns.lineplot(x='browser', y='TARGET', data=group_browser, marker='o')
# plt.show()

plt.title('디바이스별 평균 조회수')
sns.lineplot(x=group_device.index, y= 'TARGET', data=group_device, marker = 'o')
# plt.show()

plt.title('신규여부별 평균 조회수')
sns.barplot(x=group_new.index, y= 'TARGET', data=group_new)
# plt.show()

plt.title('이탈여부별 평균 조회수')
sns.barplot(x=group_bounced.index, y= 'TARGET', data=group_bounced)
# plt.show()

plt.title('나라별 평균 조회수')
plt.xticks(fontsize = 7, rotation = 45, ha = 'right')
sns.lineplot(x=group_country.index, y= 'TARGET', data=group_country, marker = 'o')
# plt.show()

plt.title('소스별 평균 조회수')
plt.xticks(fontsize = 7, rotation = 45, ha = 'right')
sns.lineplot(x=group_source.index, y= 'TARGET', data=group_source, marker = 'o')
# plt.show()

