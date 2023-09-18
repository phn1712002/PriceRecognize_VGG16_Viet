import wandb, os
import tensorflow as tf
import numpy as np
from keras.callbacks import Callback
from Tools.Weights import getPathWeightsNearest
from Architecture.Pipeline import PriceRecognize_VGG16

class CustomCallbacksWandB(Callback):
    def __init__(self, pipeline: PriceRecognize_VGG16, path_logs='./Checkpoint/logs/', dataset=None):
        super().__init__()
        self.dataset = dataset
        self.path_logs = path_logs
        self.pipeline = pipeline
        self.__last_name_update = None
        
    def on_epoch_end(self, epoch: int, logs=None):
        # Sao lưu logs wandb
        wandb.log(logs)
        
        # Sao lưu một mẫu âm thanh kiểm tra
        tableOutputPredict = wandb.Table(columns=["Epoch", "Input", "Output Target", "Output Pred"])
        for X, Y in self.dataset.take(1):
            if not X.shape[0] == 1:
                index = np.random.randint(low=0, high=X.shape[0] - 1)
                X = X[index]
                X = tf.expand_dims(X, axis=0)
                Y = Y[index]
                Y = tf.expand_dims(Y, axis=0)
        
        Y_pred = self.model.predict_on_batch(X)
        output_pred = self.pipeline.decoderLable(Y_pred)
        output_target = self.pipeline.decoderLable(Y)
        
        image_input_wandb = wandb.Image(tf.squeeze(X).numpy())
            
        tableOutputPredict.add_data(epoch + 1, image_input_wandb, output_target, output_pred)
        wandb.log({'Predict': tableOutputPredict}) 
        
        
        # Sao lưu h5
        path_file_update = getPathWeightsNearest(self.path_logs)
        wandb.save(path_file_update)