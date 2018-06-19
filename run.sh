python train.py --log_dir=logs/log.034 --seed 1234 --epochs 100 --feature mfcc --conv_type 2d
python train.py --log_dir=logs/log.035 --seed 1234 --epochs 100 --feature mfcc --conv_type 1d
python train.py --log_dir=logs/log.036 --seed 1234 --epochs 100 --feature melgram --conv_type 2d
python train.py --log_dir=logs/log.037 --seed 1234 --epochs 100 --feature melgram --conv_type 1d

tar cvzf log.tar.gz logs/log.034 logs/log.035 logs/log.036 logs/log.037
gdrive upload log.tar.gz
sudo shutdown -h now
