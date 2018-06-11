python train.py --log_dir=logs/log.033 --seed 1234 --epochs 100

tar cvzf log.033.tar.gz logs/log.033
gdrive upload log.033.tar.gz
sudo shutdown -h now
