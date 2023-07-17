import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, VotingRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('./train.csv') # train 데이터 path

X_train = df.drop(['ID','착과량(int)'], axis=1)
y_train = df['착과량(int)']
y_train = pd.DataFrame(y_train)

col_saesoon = X_train.iloc[:, 4:93]
col_yeoprok = X_train.iloc[:, 93:]


def Orange_pre(main_df):
    df = pd.DataFrame()
    for i in main_df.iterrows():
        temp = pd.DataFrame()
        for o in range(5, len(i[1]), 5):
            temp = pd.concat([temp, pd.Series(i[1][o - 5:o].mean())], ignore_index=True, axis=1)
        temp = pd.concat([temp, pd.Series(i[1][85:].mean())], ignore_index=True, axis=1)
        pd.concat([df, temp])

    df = df.reset_index(drop=True)
    return df

saesoon_mean = Orange_pre(col_saesoon)
yeoprok_mean = Orange_pre(col_yeoprok)



yeoprok_mean = yeoprok_mean.div(X_train['수고(m)'], axis=0)

X_train_new = pd.concat([X_train.iloc[:, :4], saesoon_mean, yeoprok_mean], ignore_index=True, axis=1)
X_train_new['cntzero'] = col_saesoon.apply(lambda x: x[x.isin([0.000])].count(), axis=1)




def Orange_pre(main_df):
    df = pd.DataFrame()
    for i in main_df.iterrows():
        temp = pd.DataFrame()
        for o in range(5, len(i[1]), 5):
            temp = pd.concat([temp, pd.Series(i[1][o - 5:o].mean())], ignore_index=True, axis=1)
        temp = pd.concat([temp, pd.Series(i[1][85:].mean())], ignore_index=True, axis=1)
        pd.concat([df, temp])

    df = df.reset_index(drop=True)
    return df

saesoon_mean = Orange_pre(col_saesoon)
yeoprok_mean = Orange_pre(col_yeoprok)

yeoprok_mean = yeoprok_mean.div(X_train['수고(m)'], axis=0)

X_train_new = pd.concat([X_train.iloc[:, :4], saesoon_mean, yeoprok_mean], ignore_index=True, axis=1)
X_train_new['cntzero'] = col_saesoon.apply(lambda x: x[x.isin([0.000])].count(), axis=1)

X_train_new.rename(columns= {0: 'A', 1: 'B', 2:'C', 3:'D'}, inplace=True)

##############################################

test = pd.read_csv('./test.csv') # test 데이터 path
test.drop(['ID'], axis=1, inplace=True)

tcol_saesoon = test.iloc[:, 4:93]
tcol_yeoprok = test.iloc[:, 93:]

tsaesoon_mean = Orange_pre(tcol_yeoprok)

tyeoprok_mean = Orange_pre(tcol_saesoon)

tyeoprok_mean = tyeoprok_mean.reset_index(drop=True)
tyeoprok_mean = tyeoprok_mean.div(test['수고(m)'], axis=0)

test_new = pd.concat([test.iloc[:, :4], tsaesoon_mean, tyeoprok_mean], ignore_index=True, axis=1)
test_new['cntzero'] = tcol_saesoon.apply(lambda x: x[x.isin([0.000])].count(), axis=1)

test_new.rename(columns= {0: 'A', 1: 'B', 2:'C', 3:'D'}, inplace=True)

############################################################
rf = RandomForestRegressor()

#GridSearchCV
# param = {'min_samples_split': range(2,7),
#         'max_depth':range(2,12,2),
#         'n_estimators': range(150,450,50)} # 찾고자 하는 파라미터
#
# gs = GridSearchCV(estimator=rf, param_grid=param, scoring='neg_median_absolute_error',cv=3)  #cv = fold 횟수  scoring 은 회귀분석이기 때문에 "neg~~"
#
# gs.fit(X_train_new, y_train)
#
# print('최고 정확도 : ', gs.best_score_)
# print('최고 파라미터 : ', gs.best_params_)

# rf = RandomForestRegressor(max_depth=8,min_samples_split=6,n_estimators=150)
#
# rf.fit(X_train_new,y_train)
# pred = rf.predict(test_new)


from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()

model.fit(X_train_new, y_train)
pred = model.predict(test_new)

sample_submission = pd.read_csv('./sample_submission.csv')
sample_submission['착과량(int)'] = pred
sample_submission.to_csv("./DT.csv", index = False)