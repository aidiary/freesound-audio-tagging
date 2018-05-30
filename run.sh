# 001: LeNet, Adam(0.001), epochs=32
#python train.py --log_dir=logs/log.020.1 --seed 1234 --epochs 100
python train.py --log_dir=logs/log.020.2 --seed 1235 --epochs 100
python train.py --log_dir=logs/log.020.3 --seed 1236 --epochs 100
python train.py --log_dir=logs/log.020.4 --seed 1237 --epochs 100
python train.py --log_dir=logs/log.020.5 --seed 1238 --epochs 100

# upload log
tar cvzf logs.020.tar.gz logs/log.020.*
gdrive upload logs.020.tar.gz
sudo shutdown -h now
