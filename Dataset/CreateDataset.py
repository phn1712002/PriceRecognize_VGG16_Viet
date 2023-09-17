import os
import numpy as np
from imutils import paths
from sklearn.model_selection import train_test_split
 
class PriceRecognize_Dataset_Vietnamese:
    def __init__(self, path='./Dataset/'):
        self.path = path
        self._train_dataset = None
        self._dev_dataset = None
        self._test_dataset = None
        
    def __call__(self, random_state=1712):
        list_path_image = list(paths.list_images(self.path + 'train/'))
        lable = []
        for path_image in list_path_image:
            lable.append(path_image.split(os.path.sep)[-2])
            
        X_train, X_dev, y_train, y_dev = train_test_split(list_path_image, lable, test_size=0.2, random_state=random_state)
        self._train_dataset = X_train, y_train
        self._dev_dataset = X_dev, y_dev
        return self._train_dataset, self._dev_dataset, self._test_dataset