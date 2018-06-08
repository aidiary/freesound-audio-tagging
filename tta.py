import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import torch
from train import test_time_augmentation
from dataset import AudioDataset
from model import AlexNet, LeNet


cuda = torch.cuda.is_available()
if cuda:
    print('cuda available!')

device = torch.device('cuda' if cuda else 'cpu')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir')
    parser.add_argument('model')
    args = parser.parse_args()

    # load dataset
    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/sample_submission.csv')

    le = LabelEncoder()
    le.fit(np.unique(train_df.label))
    train_df['label_idx'] = le.transform(train_df['label'])
    num_classes = len(le.classes_)

    test_dataset = AudioDataset(test_df, './data/audio_test', test=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, 128, shuffle=False)

    # load model
    model = AlexNet(num_classes).to(device)
    model.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))

    # test time augmentation
    tta_predictions = test_time_augmentation(test_loader, model, num_aug=5)
    np.save(os.path.join(args.log_dir, 'tta_predictions.npy'), tta_predictions.cpu().numpy())

    # Top3の出力を持つラベルに変換
    _, indices = tta_predictions.topk(3)
    predicted_labels = le.classes_[indices]
    predicted_labels = [' '.join(lst) for lst in predicted_labels]
    test_df['label'] = predicted_labels
    test_df.to_csv(os.path.join(args.log_dir, 'tta_submission.csv'), index=False)


if __name__ == '__main__':
    main()
