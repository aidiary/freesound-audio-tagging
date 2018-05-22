# 001: LeNet, Adam(0.001), epochs=32
python train.py --log_dir=logs/log.001 --seed 1234

# upload log
tar cvzf logs_freesound.tar.gz logs
gdrive upload logs_freesound.tar.gz
sudo shutdown -h now
