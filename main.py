import pandas as pd
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('train.csv')

test = pd.read_csv('test.csv')

# train.info()

train.fillna(0,inplace=True)
test.fillna(0,inplace=True)

# print(train.isnull().sum())
# print(test.isnull().sum())

train_x = train.drop(['count'],axis=1)
train_y = train['count']

model = RandomForestRegressor(n_estimators=100)
model.fit(train_x,train_y)

pred = model.predict(test)

submission = pd.read_csv("submission.csv")
submission['count'] = pred

submission.to_csv("베이스라인.csv",index = False)