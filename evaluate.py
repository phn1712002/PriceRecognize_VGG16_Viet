
import tensorflow as tf
from Dataset.CreateDataset import PriceRecognize_Dataset_Vietnamese
from Architecture.Pipeline import PriceRecognize_VGG16
from Architecture.Model import VGG16_TFLite
from Tools.Json import loadJson, saveJson
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

# Evaluate 
y_true, y_pred = model.predict_in_evaluate(test_dataset)
     
# Output Info
path_classification_report = PATH_REPORT + 'classification_report.json'
data_classification_report = classification_report(y_true, y_pred, output_dict=True)
saveJson(path=path_classification_report, data=data_classification_report)
print(f"Save classification report in {path_classification_report}")

# Calculate multilabel confusion matrix
mcm = multilabel_confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
for i, class_name in enumerate(list(model.class_names.word_index.keys())):
    plt.subplot(2, 3, i + 1)
    sns.heatmap(mcm[i], annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix for {class_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')

plt.tight_layout()
plt.show()

# Save the plot as an image
path_multilabel_confusion_matrix = PATH_REPORT + 'multilabel_confusion_matrix.png'
plt.savefig(path_multilabel_confusion_matrix)
print(f"Save multilabel confusion matrix in {path_multilabel_confusion_matrix}")

