import wandb, os, time
from Callbacks.WandB import CustomCallbacksWandB
from keras.callbacks import ModelCheckpoint, TensorBoard
from wandb.keras import WandbCallback

def CreateCallbacks(PATH_TENSORBOARD, PATH_LOGS, config, train_dataset, dev_dataset, pipeline):
    NAME_TIME = time.strftime("%Y%m%d-%H%M%S-")
    tensorBoard_callbacks = TensorBoard(log_dir=PATH_TENSORBOARD)
    checkpoint_callbacks = ModelCheckpoint(filepath=PATH_LOGS + NAME_TIME + 'weights-{epoch:03d}.h5', save_weights_only=True, **config['config_train']['checkpoint'])
    callbacks_model = [tensorBoard_callbacks, checkpoint_callbacks]
    if config['config_wandb']['using'] == True:
        os.environ['WANDB_API_KEY'] = config['config_wandb']['api_key']
        wandb.login()
        wandb.tensorboard.patch(root_logdir=PATH_TENSORBOARD)
        config_update = config.copy()
        config_update.pop('config_wandb')
        wandb.init(project=config['config_wandb']['project'],
                name=NAME_TIME + config['config_wandb']['name'],
                sync_tensorboard=config['config_wandb']['sync_tensorboard'],
                config=config_update)
        checkpoint_WandB = CustomCallbacksWandB(pipeline=pipeline, path_logs=PATH_LOGS, dev_dataset=dev_dataset)
        checkpoint_WandB_log = WandbCallback(training_data=train_dataset,
                                            validation_data=dev_dataset,
                                            save_model=False, 
                                            save_graph=True, 
                                            log_weights=True, 
                                            log_gradients=True, 
                                            log_evaluation=True)
        callbacks_model.append(checkpoint_WandB)
        callbacks_model.append(checkpoint_WandB_log)
    
    return callbacks_model