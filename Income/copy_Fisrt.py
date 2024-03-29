import pandas as pd
import numpy as np
import model_tuned as mt
import os, random, optuna, datetime
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import *
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping
from catboost import CatBoostRegressor
from category_encoders import TargetEncoder
from sklearn.ensemble import VotingRegressor

RANDOM_SEED = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# row 생략 없이 출력
pd.set_option('display.max_rows', None)
# col 생략 없이 출력
pd.set_option('display.max_columns', None)



train = pd.read_csv('dataset/train.csv').drop('ID',axis = 1)
test = pd.read_csv('dataset/test.csv').drop('ID',axis = 1)
submission = pd.read_csv('dataset/sample_submission.csv')


# print(train.isna().sum())
# print(test.isna().sum())
# 0 1

# 비슷한 조건에서 train 데이터셋에 해당 값이 빈도 수가 제일 많음
test = test.fillna('Child 18+ never marr Not in a subfamily')

# 중복값 제거
train = train.drop_duplicates()

#이상치 극단적인 끝 값 제거
train = train.loc[train['Gains'] < 99999]
train = train.loc[train['Losses'] < 4356]
# train = train.loc[train['Dividends']<45000]

train = train.drop(['Losses'],axis = 1)
test = test.drop(['Losses'], axis = 1)

#파생변수
train['working_time/Dividends'] = train['Working_Week (Yearly)'] / (train['Dividends'].replace(0,1))
test['working_time/Dividends'] = test['Working_Week (Yearly)'] / (test['Dividends'].replace(0,1))

logscale_columns = ['Gains', 'Losses', 'Dividends', 'Income']
numeric_columns = train.select_dtypes(include=['int64', 'float64']).columns
standardscale_columns = [x for x in numeric_columns if x not in logscale_columns]

category_columns = train.select_dtypes(exclude=['int64', 'float64']).columns
mask = train[category_columns].nunique()<=10
category_enc = train[category_columns].nunique().loc[mask].index.tolist()
target_enc = train[category_columns].nunique().loc[-mask].index.tolist()

#스케일링
train[[ 'Dividends']] = np.log1p(train[[ 'Dividends']])
test[[ 'Dividends']] = np.log1p(test[[ 'Dividends']])

#추후 돌려놓기 위함
test_age = test['Age']

ss = StandardScaler()
train[standardscale_columns] = ss.fit_transform(train[standardscale_columns])
test[standardscale_columns] = ss.transform(test[standardscale_columns])

#인코딩
le = LabelEncoder()

# for i in target_enc:
#     te = TargetEncoder(cols = i)
#     train[i] = te.fit_transform(train[i], train['Income'])
#     test[i] = te.transform(test[i])
#
# train[category_enc] = train[category_enc].astype('category')
# test[category_enc] = test[category_enc].astype('category')
train[category_columns] = train[category_columns].astype('category')
test[category_columns] = test[category_columns].astype('category')

x = train.drop('Income',axis = 1)
y = train['Income']

trainX, testX, trainY, testY = train_test_split(x,y,test_size=0.2,random_state=RANDOM_SEED)


cat, cat_study = mt.cat_modeling(trainX,trainY,testX,testY,list(category_columns))
# cat = CatBoostRegressor(**cat_param,cat_features=list(category_columns))
# cat.fit(trainX,trainY)
# pred = cat.predict(test)
# submission['Income'] = pred
#
# print(mean_squared_error(testY,cat.predict(testX),squared=False))
cat_param = {'depth': 4, 'learning_rate': 0.07476093452252774, 'random_strength': 0.019414095664808752, 'border_count': 12, 'l2_leaf_reg': 0.020185588392668135, 'leaf_estimation_iterations': 6, 'leaf_estimation_method': 'Newton', 'bootstrap_type': 'MVS', 'grow_policy': 'SymmetricTree', 'min_data_in_leaf': 78, 'one_hot_max_size': 2}
#581.45770

cat_param = cat_study.best_params

lgbm, lgbm_study = mt.lgbm_modeling(trainX,trainY,testX,testY)
# print(lgbm.feature_importances_)
# print(mean_squared_error(testY,lgbm.predict(testX),squared=False))
# pred = lgbm.predict(test)

# lgbm_param = {'num_leaves': 14, 'colsample_bytree': 0.812193189553036, 'reg_alpha': 0.4930962498343851, 'reg_lambda': 0.8673443634562641, 'max_depth': 11, 'learning_rate': 0.006791051445122969, 'n_estimators': 1828, 'min_child_samples': 27, 'subsample': 0.4522636111143746}
# 579.2563185183457 파생변수 있을 때

#lgbm 최적 파라미터
lgbm_param = {'num_leaves': 472, 'colsample_bytree': 0.7367140734280581, 'reg_alpha': 0.5235571646798937, 'reg_lambda': 3.04295394947452, 'max_depth': 9, 'learning_rate': 0.004382890500796395, 'n_estimators': 1464, 'min_child_samples': 27, 'subsample': 0.5414477150306246}
#577.0274964472734 파생변수 없을 때 -- > 541.86065

lgbm_param = lgbm_study.best_params

# lgbm = LGBMRegressor(**lgbm_param,random_state=42)
# lgbm.fit(trainX,trainY)
# pred = lgbm.predict(test)


kf = KFold(n_splits=5)
models = []
#
#
for train_index, test_index in tqdm(kf.split(x), total=kf.get_n_splits()):
    # model = LGBMRegressor(random_state=RANDOM_SEED, **lgbm_param, verbose = -1)
    model = VotingRegressor(estimators=[('lgbm',LGBMRegressor(random_state=RANDOM_SEED,**lgbm_param,verbose = -1)), ('catboost',CatBoostRegressor(random_state=RANDOM_SEED,**cat_param,cat_features=list(category_columns),verbose = False))])
    ktrainX, ktrainY = x.iloc[train_index], y.iloc[train_index]
    ktestX, ktestY = x.iloc[test_index], y.iloc[test_index]
    model.fit(ktrainX, ktrainY)
    models.append(model)
#
pred_list = []
score_list = []
test_list = []
lgbm_feature_importances = []
cat_feature_importances = []
for model in models:
    pred_list.append(model.predict(test))
    score_list.append(model.predict(ktestX))
    test_list.append(model.predict(testX))

    lgbm_model = model.named_estimators_['lgbm']
    catboost_model = model.named_estimators_['catboost']

    lgbm_importance = lgbm_model.feature_importances_
    catboost_importance = catboost_model.feature_importances_

    lgbm_feature_importances.append(lgbm_importance)
    cat_feature_importances.append(catboost_importance)

pred = np.mean(pred_list, axis=0)
score = np.mean(score_list, axis = 0)
test_score = np.mean(test_list, axis = 0)
average_lgbm_feature_importance = np.mean(lgbm_feature_importances, axis=0)
average_catboost_feature_importance = np.mean(cat_feature_importances, axis=0)

print("평균 점수 : ", mean_squared_error(ktestY, score,squared=False))
print("평균 점수 : ", mean_squared_error(testY, test_score,squared=False))

#피처중요도
for column_name, lgbm_importance, catboost_importance in zip(train.columns, average_lgbm_feature_importance, average_catboost_feature_importance):
    print("피처(컬럼) 이름:", column_name)
    print("LGBM 중요도:", lgbm_importance)
    print("CatBoost 중요도:", catboost_importance)
    print()

test['Income'] = pred
test['Age'] = test_age
test.loc[(test['Education_Status'] == 'children') | (test['Age'] <= 14) | (test['Employment_Status'] == 'not working'), 'Income'] = 0
submission['Income'] = test['Income']
title = 'Voting_CAT+LGBM'+str(datetime.datetime.now().month)+'_'+str(datetime.datetime.now().day)+'_'+str(datetime.datetime.now().hour)+'_'+str(datetime.datetime.now().minute)+'.csv'
submission.loc[submission['Income'] < 0.0, 'Income'] = 0.0
# submission.to_csv(title,index=False)




# lgbm , cat





#14살까지는 소득 0
#에듀케이션이 child면 소득 0
#employment_status 가 not working 이면 0

#5Fold 적용해서
#lgbm_param = {'num_leaves': 472, 'colsample_bytree': 0.7367140734280581, 'reg_alpha': 0.5235571646798937, 'reg_lambda': 3.04295394947452, 'max_depth': 9, 'learning_rate': 0.004382890500796395, 'n_estimators': 1464, 'min_child_samples': 27, 'subsample': 0.5414477150306246}
#랜덤42로 평균 522.2806 -> 539.11004


#5Fold Cat+LGBM VotingRegressor
#cat_param = {'depth': 4, 'learning_rate': 0.07476093452252774, 'random_strength': 0.019414095664808752, 'border_count': 12, 'l2_leaf_reg': 0.020185588392668135, 'leaf_estimation_iterations': 6, 'leaf_estimation_method': 'Newton', 'bootstrap_type': 'MVS', 'grow_policy': 'SymmetricTree', 'min_data_in_leaf': 78, 'one_hot_max_size': 2}
#lgbm_param = {'num_leaves': 472, 'colsample_bytree': 0.7367140734280581, 'reg_alpha': 0.5235571646798937, 'reg_lambda': 3.04295394947452, 'max_depth': 9, 'learning_rate': 0.004382890500796395, 'n_estimators': 1464, 'min_child_samples': 27, 'subsample': 0.5414477150306246}
# 537.8688 -> 537.93272