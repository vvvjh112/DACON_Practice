import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder

matplotlib.rcParams['font.family'] ='Malgun Gothic'

matplotlib.rcParams['axes.unicode_minus'] =False



#데이터 불러오기
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

train['운송장_건수'].plot()
plt.show()
# idx = train[train['운송장_건수']>350].index
# train = train.drop(idx)
def pre(df):
    tmp = list(df['송하인_격자공간고유번호'])
    tmp1 = list(df['수하인_격자공간고유번호'])

    for i in range(len(tmp)):
        a = str(tmp[i])
        b = str(tmp1[i])
        tmp[i] = int(a[0:5])
        tmp1[i] = int(b[0:5])

    df['송하인_격자공간고유번호'] = tmp
    df['수하인_격자공간고유번호'] = tmp1

    df = df.drop(['index'],axis=1)
    return df

train = pre(train)
test = pre(test)

#라벨링
encoder = LabelEncoder() # 인코더 생성

category = list(train['물품_카테고리'].values) # 카테고리
category_set = set(category)
# print(len(station_set))
encoder.fit(list(category_set)) # 인코딩

# 모든 학습, 시험 데이터의 정류장 정보 치환
train['물품_카테고리'] = encoder.transform(train['물품_카테고리'])
test['물품_카테고리'] = encoder.transform(test['물품_카테고리'])

print(train.head())
print(test.head())

#데이터 생성
Xtrain = train.drop(['운송장_건수'],axis=1)
Ytrain = train['운송장_건수']


#모델 평가
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor

import numpy as np

model_dict = {'DT':DecisionTreeRegressor(),
             'RF':RandomForestRegressor(),
             'LGB':lgb.LGBMRegressor(),
             'XGB':xgb.XGBRegressor(),
             'KNN':KNeighborsRegressor()}


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=5, shuffle= True, random_state=10)

score = {}

# for model_name in model_dict.keys():
#     model = model_dict[model_name]
#
#     score[model_name] = np.mean(
#         cross_val_score(model, Xtrain, Ytrain, scoring='neg_mean_squared_error', n_jobs=-1, cv=k_fold))*-1


# pd.Series(score).plot(kind = 'bar')
# plt.ylim(0,100)
# plt.show()


#LGBM 선택

#RandomSearch / GridSearch
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
# model = lgb.LGBMRegressor(learning_rate=0.05,max_depth=7,min_child_samples=10,num_leaves=60)
model = lgb.LGBMRegressor()
params = {
    'n_estimators': [30,50,60,70,80,90,100,200],
    # 'learning_rate': [0.1, 0.05, 0.01,0.005],
    # 'max_depth': [4,5,6,7,8,9,10],
    # 'num_leaves' : [20,30,50,60,70,80], #디폴트 31
    # 'min_child_samples' : [10,20,30,40]
}
grid_search = GridSearchCV(model, params, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(Xtrain,Ytrain)

print("Best Score : ",grid_search.best_score_)


print("Best Param : ",grid_search.best_params_)



#디폴트값으로 점수 측정
model = lgb.LGBMRegressor(learning_rate=0.05,max_depth=7,min_child_samples=10,num_leaves=60,n_estimators=70)
kfold = KFold(n_splits=8, shuffle=True, random_state=777)
n_iter = 0
cv_score = []

def rmse(target, pred):
    return np.sqrt(np.sum(np.power(target - pred, 2)) / np.size(pred))


for train_index, test_index in kfold.split(Xtrain, Ytrain):
    # K Fold가 적용된 train, test 데이터를 불러온다
    X_train, X_test = Xtrain.iloc[train_index, :], Xtrain.iloc[test_index, :]
    Y_train, Y_test = Ytrain.iloc[train_index], Ytrain.iloc[test_index]

    # 모델 학습과 예측 수행
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    print(pred)

    # 정확도 RMSE 계산
    n_iter += 1
    score = rmse(Y_test, pred)
    # print(score)
    cv_score.append(score)

print('\n교차 검증별 RMSE :', np.round(cv_score, 4))
print('평균 검증 RMSE :', np.mean(cv_score))
#
result = model.predict(test)
submission['운송장_건수'] = result
submission.to_csv('prac6.csv',index=False)


#중요도 시각화
input_var = list(train.columns)
input_var.remove('운송장_건수')

print(input_var)

n_feature = X_train.shape[1] #주어진 변수들의 갯수를 구함
index = np.arange(n_feature)

plt.barh(index, model.feature_importances_, align='center') #
plt.yticks(index, input_var)
plt.ylim(-1, n_feature)
plt.xlabel('feature importance', size=15)
plt.ylabel('feature', size=15)
plt.show()
np.sum(model.feature_importances_)