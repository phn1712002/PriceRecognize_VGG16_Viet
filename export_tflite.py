import tensorflow as tf, argparse
from Architecture.Model import VGG16
from Dataset.CreateDataset import PriceRecognize_Dataset_Vietnamese
from Tools.Json import loadJson
from Tools.Folder import createFolder

# Environment Variables
PATH_CONFIG = './config.json'
PATH_DATASET = './Dataset/'
PATH_LOGS = './Checkpoint/logs/'
PATH_TENSORBOARD = './Checkpoint/tensorboard/'
PATH_TFLITE = './Checkpoint/export/'

# Create Folder
createFolder(PATH_LOGS)
createFolder(PATH_TENSORBOARD)
createFolder(PATH_TFLITE)


# Argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name_file', type=str, default='', help='Path file model')
args = parser.parse_args()

# Get config
config = loadJson(PATH_CONFIG)
if not config == None:
    keys_to_check = ['config_model']
    if all(key in config for key in keys_to_check):
        config_model = config['config_model']
    else:
        raise RuntimeError("Error config")
        
# Load dataset
train_dataset_raw, _, _ = PriceRecognize_Dataset_Vietnamese(path=PATH_DATASET)()

# Init class names
class_names = tf.keras.preprocessing.text.Tokenizer()
class_names.fit_on_texts(train_dataset_raw[1])

# Create model
model = VGG16(class_names=class_names, 
              **config_model).build(True)

# Load Weights
if args.name_file == '':
    model = model.loadWeights(path=PATH_LOGS)
else: 
    model = model.loadWeights(path=PATH_LOGS, name_file=args.name_file)
        
# Export TFLite
model.exportTFlite(path_export=PATH_TFLITE)
