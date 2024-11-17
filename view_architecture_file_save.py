import argparse
from Tools.View import viewArchitectureAI

PATH_TFLITE = './Checkpoint/save/'
MY_PORT = 1712

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path_file', type=str, default='', help='Path file')
args = parser.parse_args()

# Pretrain
if args.path_file == '':
    viewArchitectureAI(PATH_TFLITE, port=MY_PORT)
else: 
    viewArchitectureAI(args.path_file, port=MY_PORT)
    