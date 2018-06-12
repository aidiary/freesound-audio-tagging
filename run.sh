python train.py --log_dir=logs/log.034 --seed 1234 --epochs 100

tar cvzf log.034.tar.gz logs/log.034
gdrive upload log.034.tar.gz
sudo shutdown -h now
