import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import torch
from train import test_time_augmentation
from dataset import AudioDataset
from model import AlexNet2d, AlexNet1d, ConvLSTM, ResNet


cuda = torch.cuda.is_available()
if cuda:
    print('cuda available!')

device = torch.device('cuda' if cuda else 'cpu')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir', type=str,
                        help='input log directory')
    parser.add_argument('model_file', type=str,
                        help='input model file')
    parser.add_argument('--feature', type=str,
                        choices=['melgram', 'mfcc'], default='mfcc',
                        help='feature')
    parser.add_argument('--model_type', type=str,
                        choices=['alex1d', 'alex2d', 'lstm', 'resnet'], default='alex2d',
                        help='convolution type of the model')
    args = parser.parse_args()

    print('log_dir:', args.log_dir)
    print('model_file:', args.model_file)
    print('feature:', args.feature)
    print('model_type:', args.model_type)

    # load dataset
    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/sample_submission.csv')

    le = LabelEncoder()
    le.fit(np.unique(train_df.label))
    train_df['label_idx'] = le.transform(train_df['label'])
    num_classes = len(le.classes_)

    test_dataset = AudioDataset(
        test_df,
        './data/audio_test',
        test=True,
        feature=args.feature,
        model_type=args.model_type
    )

    test_loader = torch.utils.data.DataLoader(test_dataset, 128, shuffle=False)

    # load model
    if args.model_type == 'alex2d':
        model = AlexNet2d(num_classes).to(device)
    elif args.model_type == 'alex1d':
        model = AlexNet1d(num_classes).to(device)
    elif args.model_type == 'lstm':
        model = ConvLSTM(num_classes).to(device)
    elif args.model_type == 'resnet':
        model = ResNet([2, 2, 2, 2])
    else:
        print('Invalid model_type: %s' % args.model_type)
        exit(1)

    print(model)

    # 学習済みモデルをロード
    model.load_state_dict(torch.load(args.model_file, map_location=lambda storage, loc: storage))

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
