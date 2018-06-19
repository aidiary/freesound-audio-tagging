#python train.py --log_dir=logs/log.034 --seed 1234 --epochs 100 --feature mfcc --conv_type 2d
python train.py --log_dir=logs/log.038 --seed 1234 --epochs 100 --feature mfcc --conv_type lstm
#python train.py --log_dir=logs/log.036 --seed 1234 --epochs 100 --feature melgram --conv_type 2d
python train.py --log_dir=logs/log.039 --seed 1234 --epochs 100 --feature melgram --conv_type lstm

# test time augmentation
#python tta.py logs/log.034 logs/log.034/epoch082-1.128-0.776.pth --feature mfcc --conv_type 2d
#python tta.py logs/log.035 logs/log.035/epoch080-1.223-0.741.pth --feature mfcc --conv_type 1d
#python tta.py logs/log.036 logs/log.036/epoch099-1.332-0.707.pth --feature melgram --conv_type 2d
#python tta.py logs/log.037 logs/log.037/epoch099-1.597-0.637.pth --feature melgram --conv_type 1d

tar cvzf log.tar.gz logs/log.034 logs/log.035 logs/log.036 logs/log.037
gdrive upload log.tar.gz
sudo shutdown -h now
