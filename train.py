import os
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

from dataset import AudioDataset
from model import LeNet, VGG11
from tensorboardX import SummaryWriter
from tqdm import tqdm


cuda = torch.cuda.is_available()
if cuda:
    print('cuda available!')

device = torch.device('cuda' if cuda else 'cpu')
num_workers = 16


def train(train_loader, model, criterion, optimizer):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), desc='train'):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(data)

        loss = criterion(outputs, target)
        running_loss += loss.item()

        # train_acc
        pred = outputs.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        loss.backward()
        optimizer.step()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    return train_loss, train_acc


def test(test_loader, model, criterion):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader), desc='test'):
            data, target = data.to(device), target.to(device)

            outputs = model(data)

            # val_loss
            loss = criterion(outputs, target)
            running_loss += loss.item()

            # val_acc
            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    val_loss = running_loss / len(test_loader)
    val_acc = correct / total

    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='logs', metavar='LD',
                        help='output log directory')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='training and valid batch size')
    parser.add_argument('--valid_ratio', type=float, default=0.1, metavar='VR',
                        help='the ratio of validation data')
    parser.add_argument('--arch', default='LeNet',
                        help='network architecture: LeNet')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed')

    args = parser.parse_args()

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    # load data
    train_df = pd.read_csv('./data/train.csv')

    # add label index to df
    le = LabelEncoder()
    le.fit(np.unique(train_df.label))
    train_df['label_idx'] = le.transform(train_df['label'])
    np.save('labels.npy', le.classes_)
    num_classes = len(le.classes_)

    train_dataset = AudioDataset(train_df, './data/audio_train')

    # split training data
    num_train = len(train_dataset)

    indices = list(range(num_train))
    split = int(args.valid_ratio * num_train)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        args.batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=num_workers
    )

    val_loader = torch.utils.data.DataLoader(
        train_dataset,
        args.batch_size,
        sampler=valid_sampler,
        drop_last=True,
        num_workers=num_workers
    )

    # build model
    if args.arch == 'LeNet':
        model = LeNet(num_classes)
    elif args.arch == 'VGG11':
        model = VGG11(num_classes)
    else:
        print('ERROR: not found model %s' % args.arch)
        exit(1)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    writer = SummaryWriter(args.log_dir)

    model_file = None

    for epoch in range(1, args.epochs + 1):
        loss, acc = train(train_loader, model, criterion, optimizer)
        val_loss, val_acc = test(val_loader, model, criterion)

        # logging
        writer.add_scalar('train/loss', loss, epoch)
        writer.add_scalar('train/acc', acc, epoch)
        writer.add_scalar('valid/loss', val_loss, epoch)
        writer.add_scalar('valid/acc', val_acc, epoch)

        print('Epoch [%d/%d] loss: %.5f acc: %.5f val_loss: %.5f val_acc: %.5f'
              % (epoch, args.epochs, loss, acc, val_loss, val_acc))

        if val_acc > best_acc:
            print('val_acc improved from %.5f to %.5f' % (best_acc, val_acc))
            best_acc = val_acc

            # remove the old model file
            if model_file is not None:
                os.remove(model_file)

            model_file = os.path.join(args.log_dir, 'epoch%03d-%.3f-%.3f.pth' % (epoch, val_loss, val_acc))
            torch.save(model.state_dict(), model_file)


if __name__ == '__main__':
    main()
