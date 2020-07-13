import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', type=int, default=1)
    args = parser.parse_args()

    if args.gui == 1:
        print('running with gui')
        os.system('python main.py')
    elif args.gui == 0:
        print('no gui \npress "q" to exit')
        os.system('''
python detect_drowsiness.py --shape-predictor predictor.dat --alarm alarm.wav
''')
