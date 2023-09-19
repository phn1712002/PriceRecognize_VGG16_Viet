
import tensorflow as tf
from Dataset.CreateDataset import PriceRecognize_Dataset_Vietnamese
from Architecture.Pipeline import PriceRecognize_VGG16
from Architecture.Model import VGG16_TFLite
from Tools.Json import saveJson
from Tools.Folder import createFolder
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Environment Variables
PATH_CONFIG = './config.json'
PATH_DATASET = './Dataset/'
PATH_REPORT = './Report/'
PATH_EXPORT = './Checkpoint/export/'
NAME_TFLITE = 'VGG16'

# Create Folder
createFolder(PATH_REPORT)

# Load dataset
_, _, test_dataset_raw = PriceRecognize_Dataset_Vietnamese(path=PATH_DATASET)()

# Model
model = VGG16_TFLite(path=PATH_EXPORT, name_file=NAME_TFLITE).build()

# Create dataset
test_dataset = PriceRecognize_VGG16(class_names=model.class_names, 
                                   config_model=model.getConfig())(dataset=test_dataset_raw, batch_size=10)
target_names = list(model.class_names.word_index.keys())
# Predict evaluate 
y_true, y_pred = model.predict_on_evaluate(test_dataset)
     
# Calculate classification report
path_classification_report = PATH_REPORT + 'classification_report.json'
classrp_dict = classification_report(y_true, y_pred, output_dict=True, target_names=target_names)
saveJson(path=path_classification_report, data=classrp_dict)
print(f"Save classification report in {path_classification_report}")

# Calculate multilabel confusion matrix
mcm = multilabel_confusion_matrix(y_true, y_pred)
path_multilabel_confusion_matrix = PATH_REPORT + 'multilabel_confusion_matrix.json'
mcm_dict = {}
for index in range(len(mcm)): mcm_dict.update({f'{target_names[index]}': mcm[index].tolist()})
saveJson(path=path_multilabel_confusion_matrix, data=mcm_dict)
print(f"Save multilabel confusion matrix in {path_multilabel_confusion_matrix}")

