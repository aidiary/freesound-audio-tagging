python train.py --log_dir=logs/log.030 --seed 1234 --epochs 100

tar cvzf log.030.tar.gz logs/log.030
gdrive upload log.030.tar.gz
sudo shutdown -h now
