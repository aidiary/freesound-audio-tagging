import os
import librosa
import numpy as np
import torch.utils.data


def random_crop(y, max_length=176400):
    """音声波形を固定長にそろえる

    max_lengthより長かったらランダムに切り取る
    max_lengthより短かったらランダムにパディングする
    """
    if len(y) > max_length:
        max_offset = len(y) - max_length
        offset = np.random.randint(max_offset)
        y = y[offset:max_length + offset]
    else:
        if max_length > len(y):
            max_offset = max_length - len(y)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        y = np.pad(y, (offset, max_length - len(y) - offset), 'constant')
    return y


class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, df, wav_dir, test=False,
                 sr=None, max_length=4.0, window_size=0.02, hop_size=0.01,
                 n_feature=64, feature='mfcc', model_type='alex2d'):
        if not os.path.exists(wav_dir):
            print('ERROR: not found %s' % wav_dir)
            exit(1)
        self.df = df
        self.wav_dir = wav_dir
        self.test = test
        self.sr = sr
        self.max_length = max_length     # sec
        self.window_size = window_size   # sec
        self.hop_size = hop_size         # sec
        self.n_feature = n_feature
        self.feature = feature
        self.model_type = model_type

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        fpath = os.path.join(self.wav_dir, self.df.fname[index])
        y, sr = librosa.load(fpath, sr=self.sr)
        if sr is None:
            print('WARNING:', fpath)
            sr = 44100

        # ランダムクロップ
        y = random_crop(y, int(self.max_length * sr))

        # 特徴抽出
        n_fft = int(self.window_size * sr)
        hop_length = int(self.hop_size * sr)

        if self.feature == 'mfcc':
            feature = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=self.n_feature)
        elif self.feature == 'melgram':
            feature = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=self.n_feature)
        else:
            print('Invalid feature name: %s' % self.feature)
            exit(1)

        data = torch.from_numpy(feature).float()
        s = data.size()

        if self.model_type == 'alex2d' or self.model_type == 'resnet':
            # Conv2dの場合は (channel, features, frames)
            data.resize_(1, s[0], s[1])
        elif self.model_type == 'alex1d' or self.model_type == 'lstm':
            # Conv1dの場合は (features, frames)
            data.resize_(s[0], s[1])
        else:
            print('Invalid conv type: %s' % self.model_type)
            exit(1)

        mean = data.mean()
        std = data.std()
        if std != 0:
            data.add_(-mean)
            data.div_(std)

        if self.test:
            # テストモードのときは正解ラベルがないのでデータだけ返す
            return data
        else:
            # label
            label = self.df.label_idx[index]

            return data, label
