
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

# Create Folder
createFolder(PATH_REPORT)

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
train_dataset_raw, _, test_dataset_raw = PriceRecognize_Dataset_Vietnamese(path=PATH_DATASET)()

# Init class names
class_names = tf.keras.preprocessing.text.Tokenizer()
class_names.fit_on_texts(train_dataset_raw[1])

# Create pipeline 
pipeline = PriceRecognize_VGG16(class_names=class_names, config_model=config_model)

# Create dataset
test_dataset = PriceRecognize_VGG16(class_names=class_names, 
                                   config_model=config_model)(dataset=test_dataset_raw, batch_size=config_dataset['batch_size_test'])

# Model
model = VGG16_TFLite().build()

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
for i, class_name in enumerate(class_names.classes_):
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
print(f"Save multilabel confusion matrix in {path_classification_report}")

