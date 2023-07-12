## 전처리는 baseline을 바탕으로 랜덤포레스트 및 gridSearchCV 사용해서 점수 올려볼 계획
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

train = pd.read_csv('train/train.csv')
submission = pd.read_csv('sample_submission.csv')
submission.set_index('id',inplace=True)
def transform(dataset, target, start_index, end_index, history_size,
                      target_size, step):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index, 48):
        indices = range(i-history_size, i, step)
        data.append(np.ravel(dataset[indices].T))
        labels.append(target[i:i+target_size])
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

# x_col =['DHI','DNI','WS','RH','T','TARGET']
x_col =['TARGET']
y_col = ['TARGET']

dataset = train.loc[:,x_col].values
label = np.ravel(train.loc[:,y_col].values)



past_history = 48 * 2
future_target = 48 * 2

### transform train
train_data, train_label = transform(dataset, label, 0,None, past_history,future_target, 1)
print(train_data, "//",train_label)

### transform test
test = []
for i in range(81):
    data = []
    tmp = pd.read_csv(f'test/{i}.csv')
    tmp = tmp.loc[:, x_col].values
    tmp = tmp[-past_history:,:]
    data.append(np.ravel(tmp.T))
    data = np.array(data)
    test.append(data)
test = np.concatenate(test, axis=0)

from sklearn import ensemble
N_ESTIMATORS = 1000
rf = ensemble.RandomForestRegressor(n_estimators=N_ESTIMATORS,
                                    max_features=1, random_state=0,
                                    max_depth = 5,
                                    verbose=True,
                                    n_jobs=-1)
rf.fit(train_data, train_label)

rf_preds = []
for estimator in rf.estimators_:
    rf_preds.append(estimator.predict(test))
rf_preds = np.array(rf_preds)

for i, q in enumerate(np.arange(0.1, 1, 0.1)):
    y_pred = np.percentile(rf_preds, q * 100, axis=0)
    submission.iloc[:, i] = np.ravel(y_pred)

# submission.to_csv(f'submission.csv')

#GridSearchCV
# param = {'min_samples_split': range(1,5),
#         'max_depth':range(8,12,2),
#         'n_estimators': range(250,450,50)} # 찾고자 하는 파라미터
#
# gs = GridSearchCV(estimator=rf, param_grid=param, scoring='neg_mean_squared_error',cv=3)  #cv = fold 횟수  scoring 은 회귀분석이기 때문에 "neg~~"
#
# gs.fit(train_data, train_label)
#
# print('최고 정확도 : ', gs.best_score_)
# print('최고 파라미터 : ', gs.best_params_)


rf = ensemble.RandomForestRegressor(n_estimators=400,
                                    max_features=1, random_state=0,
                                    max_depth = 10,
                                    verbose=True,
                                    n_jobs=-1, min_samples_split=3)
rf.fit(train_data, train_label)

rf_preds = []
for estimator in rf.estimators_:
    rf_preds.append(estimator.predict(test))
rf_preds = np.array(rf_preds)

for i, q in enumerate(np.arange(0.1, 1, 0.1)):
    y_pred = np.percentile(rf_preds, q * 100, axis=0)
    submission.iloc[:, i] = np.ravel(y_pred)

# submission.to_csv(f'rf_gs.csv')

#RandomizedSearchCV 를 이용해서 더 튜닝 해보기
# rs_model = RandomForestRegressor()
# param = {'min_samples_split': range(12,15),
#         'max_depth': range(8,11),
#         'n_estimators': range(222,225)}
# print("시작")
# rs = RandomizedSearchCV(estimator=rs_model, param_distributions=param, scoring = 'neg_mean_squared_error', cv=3)
#
# rs.fit(train_data, train_label)
#
# print('최고 정확도 : ', rs.best_score_)
# print('최고 파라미터 : ', rs.best_params_)

# rf = ensemble.RandomForestRegressor(n_estimators=224,
#                                     max_features=1, random_state=0,
#                                     max_depth = 8,
#                                     verbose=True,
#                                     n_jobs=-1, min_samples_split=13)
# rf.fit(train_data, train_label)
#
# rf_preds = []
# for estimator in rf.estimators_:
#     rf_preds.append(estimator.predict(test))
# rf_preds = np.array(rf_preds)
#
# for i, q in enumerate(np.arange(0.1, 1, 0.1)):
#     y_pred = np.percentile(rf_preds, q * 100, axis=0)
#     submission.iloc[:, i] = np.ravel(y_pred)
#
# submission.to_csv(f'rf_rs.csv')