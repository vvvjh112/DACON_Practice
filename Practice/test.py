import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('open/train.csv')

# print(train.info())
# print(train.head())


def year_month_mean_target_plot(df,time_col,target_col):
    # 시간 열을 datetime 형식으로 변환
    df[time_col] = pd.to_datetime(df[time_col])

    # 연도와 월을 추출하여 새로운 열 생성
    df['year'] = df[time_col].dt.year
    df['month'] = df[time_col].dt.month
    # 연도별 월 평균 타겟값 계산
    monthly_mean = df.groupby(['year', 'month'])[target_col].mean().reset_index()

    # 시각화: 연도별 월 평균 타겟값
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=monthly_mean, x='month', y=target_col, hue='year', marker='o')
    # sns.scatterplot(data=monthly_mean, x='month', y=target_col, hue='year', marker='o')
    plt.xticks(range(1, 13))  # x 라벨을 1부터 12까지 1단위로 설정
    plt.title('month_target of mean')
    plt.xlabel('Month')
    plt.ylabel('mean of target')
    plt.legend(title='year')
    plt.grid(True)
    plt.show()


def monthly_mean_target_plot(df, time_col, target_col):
    # 시간 열을 datetime 형식으로 변환
    df[time_col] = pd.to_datetime(df[time_col])

    # 연도-월 형식의 새로운 열 생성
    df['y-m'] = df[time_col].dt.strftime('%Y-%m')

    # 월별 평균 타겟값 계산
    monthly_mean = df.groupby(['y-m'])[target_col].mean()

    # 시각화: 월별 평균 타겟값
    plt.figure(figsize=(15, 8))
    plt.plot(monthly_mean.index, monthly_mean.values)
    plt.title('Monthly average {}'.format(target_col))
    plt.xlabel('Month')
    plt.ylabel('Mean {}'.format(target_col))
    plt.grid(True)
    plt.show()


def weekday_mean_target_plot(df, time_col, target_col):
    # 시간 열을 datetime 형식으로 변환
    df[time_col] = pd.to_datetime(df[time_col])

    # 요일 정보를 추출하여 새로운 열 생성 (0: 월요일, 1: 화요일, ..., 6: 일요일)
    df['weekday'] = df[time_col].dt.weekday

    # 요일별 평균 타겟값 계산
    weekday_mean = df.groupby(['weekday'])[target_col].mean()

    # 요일 이름 설정
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # 시각화: 요일별 평균 타겟값
    plt.figure(figsize=(10, 6))
    plt.plot(weekdays, weekday_mean.values)
    plt.title('Weekday average {}'.format(target_col))
    plt.xlabel('Weekday')
    plt.ylabel('Mean {}'.format(target_col))
    plt.grid(True)
    plt.show()


weekday_mean_target_plot(train,'timestamp','price(원/kg)')
monthly_mean_target_plot(train,'timestamp','price(원/kg)')
year_month_mean_target_plot(train,'timestamp','price(원/kg)')
# 몇주차인지
# data_TG['week'] = data_TG['timestamp'].dt.isocalendar().week