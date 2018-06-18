import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


pred_list = []
for i in [30, 31, 32, 33]:
    pred = np.load('logs/log.%03d/predictions.npy' % i)
    pred = softmax(pred.T).T
    print(i, np.argmax(pred, axis=1))
    pred_list.append(pred)

prediction = np.ones_like(pred_list[0])

# アンサンブル
for pred in pred_list:
    prediction *= pred
prediction = prediction ** (1.0 / len(pred_list))


train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/sample_submission.csv')

# DataFrameのラベルをインデックスに変換
le = LabelEncoder()
le.fit(np.unique(train_df.label))
train_df['label_idx'] = le.transform(train_df['label'])

# Top3の出力をラベルに変換
prediction_tensor = torch.from_numpy(prediction)
_, indices = prediction_tensor.topk(3)

predicted_labels = le.classes_[indices]
predicted_labels = [' '.join(lst) for lst in predicted_labels]
test_df['label'] = predicted_labels
test_df.to_csv('submission.csv', index=False)
