#python train.py --log_dir=logs/log.1 --seed 1234
python train.py --log_dir=logs/log.2 --seed 1234 --arch VGG11
tar cvzf logs_freesound.tar.gz logs
gdrive upload logs_freesound.tar.gz
sudo shutdown -h now
