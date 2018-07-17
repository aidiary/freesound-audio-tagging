import os
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(description='extract open smile features from wave files.')
    parser.add_argument('wav_dir', metavar='wav_dir', type=str, help='input wav directory')
    parser.add_argument('output', metavar='output', type=str, help='output feature file')
    parser.add_argument('--config', metavar='config', type=str,
                        default='config/IS09_emotion.conf', help='open smile config file')
    args = parser.parse_args()

    print('wav_dir:', args.wav_dir)
    print('output_file:', args.output)
    print('config_file:', args.config)

    target_files = sorted(os.listdir(args.wav_dir))
    print(len(target_files))

    if os.path.exists(args.output):
        os.remove(args.output)

    for f in target_files:
        fpath = os.path.join(args.wav_dir, f)
        if 'IS09_emotion.conf' in args.config:
            cmd = 'SMILExtract -C %s -I %s -csvoutput %s -N %s' % (args.config, fpath, args.output, f)
        elif 'emobase.conf' in args.config:
            cmd = 'SMILExtract -C %s -I %s -O %s -classlabel -1' % (args.config, fpath, args.output)
        elif 'emo_large.conf' in args.config:
            cmd = 'SMILExtract -C %s -I %s -O %s -N %s -classlabel -1 -classes numeric' % (args.config, fpath, args.output, f)
        else:
            print('invalid config file: %s' % args.config)
            exit(1)

        subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    main()
