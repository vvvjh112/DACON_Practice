import pandas as pd
import numpy as np

train = pd.read_csv('train/train.csv')
submission = pd.read_csv('sample_submission.csv')
submission.set_index('id',inplace=True)

#출력 설정
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def transform(dataset, target, start_index, end_index, history_size, target_size, step):
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

print(train_data)
print(train_label)

print(train.head())