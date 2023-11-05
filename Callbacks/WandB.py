import wandb, os 
import tensorflow as tf
import numpy as np
from keras.callbacks import Callback
from Architecture.Pipeline import PriceRecognize_VGG16

class CustomCallbacksWandB(Callback):
    def __init__(self, pipeline: PriceRecognize_VGG16,
                 name_save='model.h5',
                 monitor='val_loss',
                 verbose=1,
                 save_freq=1, 
                 mode='min',
                 path_logs='./Checkpoint/logs/', 
                 dataset=None):
        
        # Public
        super().__init__()
        self.dataset = dataset
        self.path_logs = path_logs
        self.pipeline = pipeline
        
        self.name_save = name_save
        self.history_checkpoint = []
        self.monitor = monitor     
        self.mode = mode
        self.verbose = verbose
        if save_freq == 'epoch':
            self.save_freq = 1
        else:
            if save_freq > 0: self.save_freq = save_freq
            else: self.save_freq = 1
    
        # Private
        self._epoch_save = self.save_freq
        
    def on_epoch_end(self, epoch: int, logs=None):
        
        # Fix epoch 0
        current_epoch = epoch + 1
        
        # Save logs 
        wandb.log(logs)
        
        # Save graph
        if current_epoch == 1: wandb.run.summary["graph"] = wandb.Graph.from_keras(self.model)
        
        
        # Save history_checkpoint
        self.history_checkpoint.append(logs[self.monitor])
        
        # Checkpoint
        save_model_checkpoint = False
        if self.mode == 'min':
            save_model_checkpoint = min(self.history_checkpoint) >= logs[self.monitor]
        elif self.mode == 'max':
            save_model_checkpoint = max(self.history_checkpoint) <= logs[self.monitor]
        
        
            
        # Save weight 
        if current_epoch >=  self._epoch_save and save_model_checkpoint:
            path = self.path_logs + self.name_save.format(epoch=current_epoch)
            self._epoch_save += current_epoch + self.save_freq
            self.model.save_weights(path)
            wandb.save(path)
            if self.verbose == 1:
                print(f"Save weights epoch {current_epoch} - {path}" )
        """  
        # Print one data in dev
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
            
        tableOutputPredict.add_data(current_epoch, image_input_wandb, output_target, output_pred)
        wandb.log({'Predict': tableOutputPredict}) 
        """ 
        
       