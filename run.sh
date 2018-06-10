python train.py --log_dir=logs/log.032 --seed 1234 --epochs 100

tar cvzf log.032.tar.gz logs/log.032
gdrive upload log.032.tar.gz
sudo shutdown -h now
