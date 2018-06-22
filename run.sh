python train.py --log_dir=logs/log.040 --seed 1234 --epochs 150 --feature mfcc --conv_type 2d
python train.py --log_dir=logs/log.041 --seed 1234 --epochs 150 --feature mfcc --conv_type 1d
python train.py --log_dir=logs/log.042 --seed 1234 --epochs 150 --feature melgram --conv_type 2d
python train.py --log_dir=logs/log.043 --seed 1234 --epochs 150 --feature melgram --conv_type 1d
python train.py --log_dir=logs/log.044 --seed 1234 --epochs 150 --feature mfcc --conv_type lstm
python train.py --log_dir=logs/log.045 --seed 1234 --epochs 150 --feature melgram --conv_type lstm

# test time augmentation
# python tta.py logs/log.040 logs/log.040/.pth --feature mfcc --conv_type 2d
# python tta.py logs/log.041 logs/log.041/.pth --feature mfcc --conv_type 1d
# python tta.py logs/log.042 logs/log.042/.pth --feature melgram --conv_type 2d
# python tta.py logs/log.043 logs/log.043/.pth --feature melgram --conv_type 1d
# python tta.py logs/log.044 logs/log.044/.pth --feature mfcc --conv_type lstm
# python tta.py logs/log.045 logs/log.045/.pth --feature melgram --conv_type lstm

tar cvzf log.tar.gz logs/log.40 logs/log.041 logs/log.042 logs/log.043 logs/log.044 logs/log.045
gdrive upload log.tar.gz
sudo shutdown -h now
