# Testing
import tensorflow as tf, cv2, keyboard
from Dataset.CreateDataset import PriceRecognize_Dataset_Vietnamese
from Architecture.Pipeline import PriceRecognize_VGG16
from Tools.Json import loadJson


# Environment Variables
PATH_CONFIG = './config.json'
PATH_DATASET = './Dataset/'

# Get config
config = loadJson(PATH_CONFIG)
if not config == None:
    keys_to_check = ['config_model', 'config_dataset']
    if all(key in config for key in keys_to_check):
        config_model = config['config_model']
        config_dataset = config['config_dataset']
    else:
        raise RuntimeError("Error config")
    
# Load dataset
train_dataset_raw, _, _ = PriceRecognize_Dataset_Vietnamese(path=PATH_DATASET)()

# Init class names
class_names = tf.keras.preprocessing.text.Tokenizer()
class_names.fit_on_texts(train_dataset_raw[1])

# Augments Image
config_augments = None
if config_dataset['using_augments']: config_augments = config_dataset['config_augments']

# Create pipeline 
pipeline = PriceRecognize_VGG16(class_names=class_names, config_model=config_model)

# Create dataset
train_dataset = PriceRecognize_VGG16(class_names=class_names, 
                                     config_augments=config_augments, 
                                     config_model=config_model)(dataset=train_dataset_raw, batch_size=config_dataset['batch_size_train'])
# Check Pipeline
while True:
    for X, Y in train_dataset.take(1):
        print(f"X_Shape: {X.shape}, Y_Shape: {Y.shape}")
        if not X.shape[0] == 1:
            index = 0
            X = X[index]
            Y = Y[index]
            Y = tf.expand_dims(Y, axis=0)
            print(pipeline.decoderLable(Y.numpy()))
            cv2.imshow('Webcam', X.numpy())
            key = cv2.waitKey(0)
            if key == 13:  # Check for Enter key (ASCII code 13)
                cv2.destroyAllWindows()
                break
    else:
        continue
    break