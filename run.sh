python train.py --log_dir=logs/log.047 --seed 1234 --epochs 150 --feature mfcc --model_type resnet

# test time augmentation
# python tta.py logs/log.040 logs/log.040/.pth --feature mfcc --model_type resnet

tar cvzf log.047.tar.gz logs/log.047
gdrive upload log.047.tar.gz
sudo shutdown -h now
