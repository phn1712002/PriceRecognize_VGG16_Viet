
import tensorflow as tf, warnings, wandb, argparse
from Architecture.Model import VGG16
from Dataset.CreateDataset import PriceRecognize_Dataset_Vietnamese
from Architecture.Pipeline import PriceRecognize_VGG16
from Optimizers.OptimizersVGG16 import CustomOptimizers
from jlclient import jarvisclient
from jlclient.jarvisclient import *
from Tools.Json import loadJson
from Tools.Callbacks import createCallbacks
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
parser.add_argument('--pretrain_config', type=bool, default=False, help='Pretrain model VGG16 in logs training in dataset')
parser.add_argument('--name_file_pretrain', type=str, default='', help='Name file pretrain model')
parser.add_argument('--export_tflite', type=bool, default=False, help='Export to tflite')
args = parser.parse_args()

# Get config
config = loadJson(PATH_CONFIG)
if not config == None:
    keys_to_check = ['config_wandb', 'config_jarvislabs','config_model', 'config_opt', 'config_other', 'config_train', 'config_dataset']
    if all(key in config for key in keys_to_check):
        config_wandb = config['config_wandb']
        config_model = config['config_model']
        config_opt = config['config_opt']
        config_other = config['config_other']
        config_train = config['config_train']
        config_dataset = config['config_dataset']
        config_jarvislabs = config['config_jarvislabs']
    else:
        raise RuntimeError("Error config")
        
# Turn off warning
if not config_other['warning']:
    warnings.filterwarnings('ignore')
    
# Load dataset
train_dataset_raw, dev_dataset_raw, test_dataset_raw = PriceRecognize_Dataset_Vietnamese(path=PATH_DATASET)()

# Init class names
class_names = tf.keras.preprocessing.text.Tokenizer()
class_names.fit_on_texts(train_dataset_raw[1])

# Augments Image
config_augments = None
if config_dataset['using_augments']: config_augments = config_dataset['config_augments']

# Create pipeline 
pipeline = PriceRecognize_VGG16(class_names=class_names, config_model=config_model)

train_dataset = PriceRecognize_VGG16(class_names=class_names, 
                                     config_augments=config_augments, 
                                     config_model=config_model)(dataset=train_dataset_raw, batch_size=config_dataset['batch_size_train'])

dev_dataset = PriceRecognize_VGG16(class_names=class_names, 
                                   config_model=config_model)(dataset=dev_dataset_raw, batch_size=config_dataset['batch_size_dev'])

test_dataset = PriceRecognize_VGG16(class_names=class_names, 
                                   config_model=config_model)(dataset=test_dataset_raw, batch_size=config_dataset['batch_size_test'])

# Create optimizers
opt_VGG16 = CustomOptimizers(**config_opt)()

# Callbacks
callbacks_VGG16= createCallbacks(PATH_TENSORBOARD=PATH_TENSORBOARD, 
                                PATH_LOGS=PATH_LOGS, 
                                config=config, 
                                train_dataset=train_dataset,
                                test_dataset=test_dataset,
                                dev_dataset=dev_dataset, 
                                pipeline=pipeline)

# Create model
model = VGG16(class_names=class_names, 
              opt=opt_VGG16,
              **config_model).build(config_other['summary'])

# Pretrain
if args.pretrain_config:
    if args.name_file_pretrain == '':
        model = model.loadWeights(path=PATH_LOGS)
    else: 
        model = model.loadWeights(path=PATH_LOGS, name_file=args.name_file_pretrain)
        
# Train model
model.fit(train_dataset=train_dataset, 
          dev_dataset=dev_dataset, 
          callbacks=callbacks_VGG16,
          epochs=config_train['epochs'])

# Export to tflite
if args.export_tflite:
    model.exportTFlite(path_export=PATH_TFLITE)

# Off Wandb
wandb.finish()

# Stop Jarvislabs
if config_jarvislabs['using']:
    jarvisclient.token = config_jarvislabs['token']
    jarvisclient.user_id = config_jarvislabs['user_id']
    instance = User.get_instances()[0]
    instance.pause()