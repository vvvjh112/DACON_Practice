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
# train = train.loc[train['Dividends']<40000]
# train = train.loc[train['Income'] < 9500]

# train['G-L'] = train['Gains'] - train['Losses']
# test['G-L'] = test['Gains'] - test['Losses']

# train = train.drop(['Birth_Country (Father)','Birth_Country (Mother)'],axis = 1)
# test = test.drop(['Birth_Country (Father)','Birth_Country (Mother)'],axis = 1)


train['work*age'] = train['Working_Week (Yearly)'] * train['Age']
test['work*age'] = test['Working_Week (Yearly)'] * test['Age']

train['Gain*Div'] = train['Gains'] * train['Dividends']
test['Gain*Div'] = test['Gains'] * test['Dividends']

train['Losses-Div'] = train['Losses'] - train['Dividends']
test['Losses-Div'] = test['Losses'] - test['Dividends']



train = train.drop(['Gains','Losses'],axis = 1)
test = test.drop(['Gains','Losses'], axis = 1)

# 모델별 학습데이터 생성
lgbm_train = train.copy()
lgbm_test = test.copy()

cat_train = train.copy()
cat_test = test.copy()

#추후 돌려놓기 위함
test_age = test['Age']


#파생변수
def create_age_group(age):
    if age < 20:
        return '10대 미만'
    elif 20 <= age < 30:
        return '20대'
    elif 30 <= age < 40:
        return '30대'
    elif 40 <= age < 50:
        return '40대'
    elif 50 <= age < 60:
        return '50대'
    elif 60 <= age < 70:
        return '60대'
    else:
        return '70대 이상'




#고용상태 및 교육수준 조합
# train['Employment_and_Education'] = train['Employment_Status'] + '_' + train['Education_Status']
# test['Employment_and_Education'] = test['Employment_Status'] + '_' + test['Education_Status']


logscale_columns = ['Gains', 'Losses', 'Dividends', 'Income']

#lgbm
#파생변수

# lgbm_train = lgbm_train.drop(['Race'],axis =1)
# lgbm_test = lgbm_test.drop(['Race'],axis =1)

# lgbm_train['AgeGroup'] = lgbm_train['Age'].apply(create_age_group)
# lgbm_test['AgeGroup'] = lgbm_test['Age'].apply(create_age_group)

# lgbm_train['ESI'] = lgbm_train['Gains'] - lgbm_train['Losses']
# lgbm_test['ESI'] = lgbm_test['Gains'] - lgbm_test['Losses']

lgbm_train = lgbm_train.drop(['Losses-Div'],axis=1)
lgbm_test = lgbm_test.drop(['Losses-Div'],axis = 1)

lgbm_numeric_columns = lgbm_train.select_dtypes(include=['int64', 'float64']).columns
lgbm_standardscale_columns = [x for x in lgbm_numeric_columns if x not in logscale_columns]

lgbm_category_columns = lgbm_train.select_dtypes(exclude=['int64', 'float64']).columns
mask = lgbm_train[lgbm_category_columns].nunique()<=10
lgbm_category_enc = lgbm_train[lgbm_category_columns].nunique().loc[mask].index.tolist()
lgbm_target_enc = lgbm_train[lgbm_category_columns].nunique().loc[-mask].index.tolist()



#스케일링
lgbm_train[['Dividends']] = np.log1p(lgbm_train[[ 'Dividends']])
lgbm_test[['Dividends']] = np.log1p(lgbm_test[[ 'Dividends']])


ss = StandardScaler()
lgbm_train[lgbm_standardscale_columns] = ss.fit_transform(lgbm_train[lgbm_standardscale_columns])
lgbm_test[lgbm_standardscale_columns] = ss.transform(lgbm_test[lgbm_standardscale_columns])

#인코딩
# for i in lgbm_target_enc:
#     te = TargetEncoder(cols = i)
#     lgbm_train[i] = te.fit_transform(lgbm_train[i], lgbm_train['Income'])
#     lgbm_test[i] = te.transform(lgbm_test[i])
# #
# lgbm_train[lgbm_category_enc] = train[lgbm_category_enc].astype('category')
# lgbm_test[lgbm_category_enc] = test[lgbm_category_enc].astype('category')

lgbm_train[lgbm_category_columns] = lgbm_train[lgbm_category_columns].astype('category')
lgbm_test[lgbm_category_columns] = lgbm_test[lgbm_category_columns].astype('category')




#cat
#파생변수

# cat_train = cat_train.drop(['Birth_Country (Father)','Birth_Country (Mother)'],axis = 1)
# cat_test = cat_test.drop(['Birth_Country (Father)','Birth_Country (Mother)'],axis = 1)
#
# cat_train['AgeGroup'] = cat_train['Age'].apply(create_age_group)
# cat_test['AgeGroup'] = cat_test['Age'].apply(create_age_group)
#
# cat_train['ESI'] = cat_train['Gains'] - cat_train['Losses']
# cat_test['ESI'] = cat_test['Gains'] - cat_test['Losses']

cat_numeric_columns = cat_train.select_dtypes(include=['int64', 'float64']).columns
cat_standardscale_columns = [x for x in cat_numeric_columns if x not in logscale_columns]

cat_category_columns = cat_train.select_dtypes(exclude=['int64', 'float64']).columns
mask = cat_train[cat_category_columns].nunique()<=10
cat_category_enc = cat_train[cat_category_columns].nunique().loc[mask].index.tolist()
cat_target_enc = cat_train[cat_category_columns].nunique().loc[-mask].index.tolist()

#스케일링
cat_train[[ 'Dividends']] = np.log1p(cat_train[[ 'Dividends']])
cat_test[['Dividends']] = np.log1p(cat_test[[ 'Dividends']])


ss = StandardScaler()
cat_train[cat_standardscale_columns] = ss.fit_transform(cat_train[cat_standardscale_columns])
cat_test[cat_standardscale_columns] = ss.transform(cat_test[cat_standardscale_columns])

#인코딩
cat_train[cat_category_columns] = cat_train[cat_category_columns].astype('category')
cat_test[cat_category_columns] = cat_test[cat_category_columns].astype('category')




lgbm_x = lgbm_train.drop('Income', axis = 1)
cat_x = cat_train.drop('Income', axis = 1)

y = train['Income']


ltrainX, ltestX, ltrainY, ltestY = train_test_split(lgbm_x,y,test_size=0.2,random_state=RANDOM_SEED)
ctrainX, ctestX, ctrainY, ctestY = train_test_split(cat_x,y,test_size=0.2,random_state=RANDOM_SEED)


# cat, cat_study = mt.cat_modeling(ctrainX,ctrainY,ctestX,ctestY,list(cat_category_columns))
# cat = CatBoostRegressor(**cat_param,cat_features=list(category_columns))
# cat.fit(trainX,trainY)
# pred = cat.predict(test)
# submission['Income'] = pred
#
# print(mean_squared_error(testY,cat.predict(testX),squared=False))

# cat_param = {'depth': 4, 'learning_rate': 0.07476093452252774, 'random_strength': 0.019414095664808752, 'border_count': 12, 'l2_leaf_reg': 0.020185588392668135, 'leaf_estimation_iterations': 6, 'leaf_estimation_method': 'Newton', 'bootstrap_type': 'MVS', 'grow_policy': 'SymmetricTree', 'min_data_in_leaf': 78, 'one_hot_max_size': 2}
#581.45770

#테스트
# cat_param = cat_study.best_params
# cat_param = {'depth': 4, 'learning_rate': 0.3780882464586928, 'random_strength': 0.018017309563986762, 'border_count': 181, 'l2_leaf_reg': 57.55942688579224, 'leaf_estimation_iterations': 5, 'leaf_estimation_method': 'Newton', 'bootstrap_type': 'MVS', 'grow_policy': 'SymmetricTree', 'min_data_in_leaf': 97, 'one_hot_max_size': 1}
cat_param = {'depth': 8, 'learning_rate': 0.05891535596662352, 'random_strength': 0.013037605369033879, 'border_count': 138, 'l2_leaf_reg': 1.0645707429462432, 'leaf_estimation_iterations': 7, 'leaf_estimation_method': 'Gradient', 'bootstrap_type': 'MVS', 'grow_policy': 'SymmetricTree', 'min_data_in_leaf': 93, 'one_hot_max_size': 1}


# lgbm, lgbm_study = mt.lgbm_modeling(ltrainX,ltrainY,ltestX,ltestY)
# print(lgbm.feature_importances_)
# print(mean_squared_error(testY,lgbm.predict(testX),squared=False))
# pred = lgbm.predict(test)


#lgbm 최적 파라미터
# lgbm_param = {'num_leaves': 472, 'colsample_bytree': 0.7367140734280581, 'reg_alpha': 0.5235571646798937, 'reg_lambda': 3.04295394947452, 'max_depth': 9, 'learning_rate': 0.004382890500796395, 'n_estimators': 1464, 'min_child_samples': 27, 'subsample': 0.5414477150306246}
#577.0274964472734 파생변수 없을 때 -- > 541.86065

# lgbm_param = lgbm_study.best_params
lgbm_param =  {'num_leaves': 1002, 'colsample_bytree': 0.823235304584475, 'reg_alpha': 0.03917551797755455, 'reg_lambda': 9.600923728838367, 'max_depth': 9, 'learning_rate': 0.007434521362412546, 'n_estimators': 1415, 'min_child_samples': 37, 'subsample': 0.4909937758395281}
# lgbm = LGBMRegressor(**lgbm_param,random_state=42)
# lgbm.fit(trainX,trainY)
# pred = lgbm.predict(test)


kf = KFold(n_splits=5)
models = []
lgbm_models = []
cat_models = []

for train_index, test_index in tqdm(kf.split(lgbm_x), total=kf.get_n_splits()):
    model = LGBMRegressor(random_state=RANDOM_SEED, **lgbm_param, verbose = -1)
    # model = VotingRegressor(estimators=[('lgbm',LGBMRegressor(random_state=RANDOM_SEED,**lgbm_param,verbose = -1)), ('catboost',CatBoostRegressor(random_state=RANDOM_SEED,**cat_param,cat_features=list(category_columns),verbose = False))])
    lktrainX, lktrainY = lgbm_x.iloc[train_index], y.iloc[train_index]
    lktestX, lktestY = lgbm_x.iloc[test_index], y.iloc[test_index]
    model.fit(lktrainX, lktrainY)
    lgbm_models.append(model)

for train_index, test_index in tqdm(kf.split(cat_x), total=kf.get_n_splits()):
    model = CatBoostRegressor(random_state=RANDOM_SEED, **cat_param, cat_features=list(cat_category_columns),verbose=False)
    # model = VotingRegressor(estimators=[('lgbm',LGBMRegressor(random_state=RANDOM_SEED,**lgbm_param,verbose = -1)), ('catboost',CatBoostRegressor(random_state=RANDOM_SEED,**cat_param,cat_features=list(category_columns),verbose = False))])
    cktrainX, cktrainY = cat_x.iloc[train_index], y.iloc[train_index]
    cktestX, cktestY = cat_x.iloc[test_index], y.iloc[test_index]
    model.fit(cktrainX, cktrainY)
    cat_models.append(model)
#
pred_list = []
lgbm_pred_list = []
cat_pred_list = []
score_list = []
lgbm_score_list = []
cat_score_list = []
test_list = []
lgbm_test_list = []
cat_test_list = []
lgbm_feature_importances = []
cat_feature_importances = []
for model in lgbm_models:
    lgbm_pred_list.append(model.predict(lgbm_test))
    lgbm_score_list.append(model.predict(lktestX))
    lgbm_test_list.append(model.predict(ltestX))
    lgbm_feature_importances.append(model.feature_importances_)

for model in cat_models:
    cat_pred_list.append(model.predict(cat_test))
    cat_score_list.append(model.predict(cktestX))
    cat_test_list.append(model.predict(ctestX))
    cat_feature_importances.append(model.feature_importances_)


lgbm_pred = np.mean(lgbm_pred_list, axis=0)
cat_pred = np.mean(cat_pred_list, axis = 0)
lgbm_score = np.mean(lgbm_score_list, axis = 0)
cat_score = np.mean(cat_score_list, axis = 0)
lgbm_test_score = np.mean(lgbm_test_list, axis = 0)
cat_test_score = np.mean(cat_test_list, axis = 0)
average_lgbm_feature_importance = np.mean(lgbm_feature_importances, axis=0)
average_catboost_feature_importance = np.mean(cat_feature_importances, axis=0)

print("lgbm ktestY 평균 점수 : ", mean_squared_error(lktestY, lgbm_score,squared=False))
print("lgbm testY 평균 점수 : ", mean_squared_error(ltestY, lgbm_test_score,squared=False))

print("cat ktestY 평균 점수 : ", mean_squared_error(cktestY, cat_score,squared=False))
print("cat testY 평균 점수 : ", mean_squared_error(ctestY, cat_test_score,squared=False))

print("ktestY 평균 점수 : ", mean_squared_error(cktestY, (cat_score*0.5)+(lgbm_score*0.5),squared=False))
print("testY 평균 점수 : ", mean_squared_error(ctestY, (cat_test_score*0.5)+(lgbm_test_score*0.5),squared=False))

print("총 ktest 평균 점수 : ", (mean_squared_error(cktestY, cat_score,squared=False) + mean_squared_error(lktestY, lgbm_score,squared=False))/2 )
print("총 testY 평균 점수 : ", (mean_squared_error(ctestY, cat_test_score,squared=False)+mean_squared_error(ltestY, lgbm_test_score,squared=False))/2)

#피처중요도
for column_name, lgbm_importance, catboost_importance in zip(train.columns, average_lgbm_feature_importance, average_catboost_feature_importance):
    print("피처(컬럼) 이름:", column_name)
    print("LGBM 중요도:", lgbm_importance)
    print("CatBoost 중요도:", catboost_importance)
    print()

pred = (lgbm_pred * 0.5) + (cat_pred * 0.5)

test['Income'] = pred
test['Age'] = test_age
test.loc[(test['Education_Status'] == 'children') | (test['Age'] <= 14) | (test['Employment_Status'] == 'not working'), 'Income'] = 0
test.loc[(test['Industry_Status'] == 'Not in universe or children') | (test['Industry_Status'] == 'Armed Forces') | (test['Occupation_Status'] == 'Unknown'), 'Income'] = 0
submission['Income'] = test['Income']
title = 'Voting_CAT+LGBM'+str(datetime.datetime.now().month)+'_'+str(datetime.datetime.now().day)+'_'+str(datetime.datetime.now().hour)+'_'+str(datetime.datetime.now().minute)+'.csv'
submission.loc[submission['Income'] < 0.0, 'Income'] = 0.0
submission.to_csv(title,index=False)




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