import pandas as pd
import numpy as np
from tqdm import tqdm

def NMAE(true, pred):
    mae = np.mean(np.abs(true-pred))
    score = mae / np.mean(np.abs(true))
    return score

def reduce_col3(df,trainortest):
    sprout=df.iloc[:,-178:-89] # 새순
    chlor=df.iloc[:,-89:] # 엽록소(chlorophyll)

    # 9일씩 묶어서 평균
    sarr=np.array(sprout.columns)
    sarr=np.append(sarr,'d').reshape(10,9)
    sdate_dic={
        'period_1 새순':sarr[0],
        'period_2 새순':sarr[1],
        'period_3 새순':sarr[2],
        'period_4 새순':sarr[3],
        'period_5 새순':sarr[4],
        'period_6 새순':sarr[5],
        'period_7 새순':sarr[6],
        'period_8 새순':sarr[7],
        'period_9 새순':sarr[8],
        'period_10 새순':sarr[9][:-1]
    }

    carr=np.array(chlor.columns)
    carr=np.append(carr,'d').reshape(10,9)
    cdate_dic={
        'period_1 엽록소':carr[0],
        'period_2 엽록소':carr[1],
        'period_3 엽록소':carr[2],
        'period_4 엽록소':carr[3],
        'period_5 엽록소':carr[4],
        'period_6 엽록소':carr[5],
        'period_7 엽록소':carr[6],
        'period_8 엽록소':carr[7],
        'period_9 엽록소':carr[8],
        'period_10 엽록소':carr[9][:-1]
    }

    new=pd.DataFrame()
    for period,dates in sdate_dic.items():
        new[period]=sprout[dates].mean(axis=1)

    #for period,dates in cdate_dic.items():
    #    new[period]=chlor[dates].mean(axis=1)
    ## 엽록소 걍 버림

    new['새순diff']=df['2022-09-01 새순']-df['2022-11-28 새순']
    new['새순max']=df['2022-09-01 새순']
    new['새순min']=df['2022-11-28 새순']

    # 다른 칼럼들도 예쁘게
    ## ex) 착과량(int) -> 착과량

    if trainortest=='train' :
        ## train은 착과량 포함
        new['착과량']=df['착과량(int)']

    new['나무부피']=df['수고(m)']*df['수관폭평균']
    return new

#train, test 셋 불러오기
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 앞에서 만든 함수 적용 및 X, y split
X=reduce_col3(train,'train').drop('착과량',axis=1)
y=reduce_col3(train,'train')['착과량']


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate
import time

#시간 체크용
start = time.time()

rf = RandomForestRegressor(n_jobs = -1, random_state = 0)
param_grid = {
        'max_depth':  [3, 4, 5, 6, 7],
        'bootstrap': [True, False],
        'max_features': ['auto', 'sqrt', 'log2'],
        'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']
    }

ss=ShuffleSplit(test_size=0.3,random_state=0)
grid_rf_cv=GridSearchCV(rf,
                     return_train_score=True,
                       param_grid=param_grid,
                       cv=ss,
                       verbose=0,
                       scoring='neg_mean_absolute_error',
                       n_jobs=-1)

grid_rf_cv.fit(X,y)

# end=time.time()
# print('수행 시간: {0:.3f}'.format(end - start))
# print('최적의 매개변수 조합: ', grid_rf_cv.best_params_)
# print('최고의 교차 검증 점수: ', grid_rf_cv.best_score_)
#
#
best_rf = grid_rf_cv.best_estimator_
# y_pred_rf = best_rf.predict(X)
# print(f'NMAE : {NMAE(y,y_pred_rf)}')

X_test=reduce_col3(test,'test')
test_pred_rf = best_rf.predict(X_test)

import xgboost as xgb

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV

xgbr=xgb.XGBRegressor()
param_grid = {
        'learning_rate':[0.001,0.01,0.1,0.2],
        'max_depth':  [3, 4, 5, 6, 7]
    }

ss=ShuffleSplit(test_size=0.3,random_state=0)
grid_xgb_cv=GridSearchCV(xgbr,return_train_score=True,
                       param_grid=param_grid,
                       cv=ss,
                       verbose=0,
                       scoring='neg_mean_absolute_error',
                       n_jobs=-1)

grid_xgb_cv.fit(X,y)

print('최적의 매개변수 조합: ', grid_xgb_cv.best_params_)
print('최고의 교차 검증 점수: ', grid_xgb_cv.best_score_)

best_xgb=grid_xgb_cv.best_estimator_
y_pred_xgb=best_xgb.predict(X)
print(f'NMAE : {NMAE(y,y_pred_xgb)}')

test_pred_xgb = best_xgb.predict(X_test)

final_pred = 0.5*test_pred_rf + 0.5*test_pred_xgb

sample_submission = pd.read_csv('sample_submission.csv')
sample_submission['착과량(int)'] = final_pred
sample_submission.to_csv('rf_xg.csv', index=False)
