import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

import warnings

from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

#데이터읽기
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X = train.drop(columns=['ID', 'MEDV'])
y = train['MEDV']
test = test.drop(columns=['ID'])

#데이터 전처리
from sklearn.preprocessing import StandardScaler

# StandardScaler
# 분석 시 변수들의 스케일이 다른 경우 컬럼 별 단위 또는 범위를 통일시켜주기 위해 표준화를 해줍니다.
#
# 표준화란 데이터 값들을 평균이 0이고 분산이 1인 정규 분포로 만드는 것입니다.

scaler = StandardScaler()

train_scaler = scaler.fit_transform(train[train.columns[1:-1]])
test_scaler = scaler.transform(test)

#모델링


# Model Hyperparameter Setting
# 대부분의 모델들은 사람이 직접 설정할 수 있는 Hyperparameter를 가지고 있습니다.
# 이런 Hyperparameter에 어떤 값이 설정되는가에 따라 모델의 성능은 크게 차이나게 됩니다.
# 본 Baseline에서 제공한 Ridge Regression 모델에서는 alpha를 Hyperparameter로 제공했습니다.
# alpha는 모델의 규제항으로, 모델의 오버피팅을 방지하는 역할을 합니다.

# model = Ridge(alpha=1.0) #alpha의 값을 바꿔 규제 정도를 조절할 수 있습니다.
#
# model.fit(train_scaler, train['MEDV'])
#
# pred = model.predict(test_scaler)
#
submit = pd.read_csv('./sample_submission.csv')
#
# submit['MEDV'] = pred
#
# submit.to_csv('./submit.csv', index=False)

#최적화 (교차검증)
# from sklearn.linear_model import RidgeCV
#
# alphas = [0.01, 0.05, 0.1, 0.2 ,0.7, 1.0, 2.0, 3.0, 4.0,  10.0,11.0,12.0,13.0, 100.0]
#
# ridge = RidgeCV(alphas=alphas, cv=5)
# ridge.fit(train_scaler, train['MEDV'])
# print("alpha: ", ridge.alpha_)
# print("best score: ", ridge.best_score_)
#
# model = Ridge(alpha=3.0) #alpha의 값을 바꿔 규제 정도를 조절할 수 있습니다.
#
# model.fit(train_scaler, train['MEDV'])
#
# pred = model.predict(test_scaler)
#
# submit = pd.read_csv('./sample_submission.csv')
#
# submit['MEDV'] = pred
#
# submit.to_csv('submit_alpha3.csv', index=False)

formula = """
MEDV ~ scale(CRIM) + scale(ZN) + scale(INDUS) + scale(NOX) + scale(RM) + scale(AGE) + scale(DIS) + scale(RAD) + scale(TAX) + scale(PTRATIO) + scale(B) + scale(LSTAT)
"""

import statsmodels.api as sm

model = sm.OLS.from_formula(formula, data=train)
result = model.fit()

pred = result.predict(test)
submit['MEDV'] = pred

submit.to_csv('ttt.csv',index=False)