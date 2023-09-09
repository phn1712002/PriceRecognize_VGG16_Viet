import wandb
from keras.callbacks import Callback
from Tools.Weights import getPathWeightsNearest
from Architecture.Pipeline import PriceRecognize_VGG16

class CustomCallbacksWandB(Callback):
    def __init__(self, pipeline: PriceRecognize_VGG16, path_logs='./Checkpoint/logs/', dev_dataset=None):
        super().__init__()
        self.dev_dataset = dev_dataset
        self.path_logs = path_logs
        self.pipeline = pipeline
        self.__last_name_update = None
        
    def on_epoch_end(self, epoch: int, logs=None):
        # Cập nhật file weights model to cloud wandb
        path_file_update = getPathWeightsNearest(self.path_logs)
        if self.__last_name_update != path_file_update: 
            self.__last_name_update = path_file_update
            wandb.save(path_file_update)
        
        