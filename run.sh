# 001: LeNet, Adam(0.001), epochs=32
python train.py --log_dir=logs/log.001 --seed 1234

# upload log
tar cvzf logs.001.tar.gz logs/logs.001
gdrive upload logs.001.tar.gz
sudo shutdown -h now
