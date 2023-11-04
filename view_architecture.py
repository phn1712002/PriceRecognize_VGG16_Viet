import argparse
from Tools.View import viewArchitectureAI

PATH_TFLITE = './Checkpoint/save/'

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path_file', type=str, default='', help='Path file')
args = parser.parse_args()

# Pretrain
if args.path_file == '':
    viewArchitectureAI(PATH_TFLITE)
else: 
    viewArchitectureAI(args.path_file)
    