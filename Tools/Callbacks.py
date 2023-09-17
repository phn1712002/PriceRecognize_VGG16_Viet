import wandb, os, time
from Callbacks.WandB import CustomCallbacksWandB
from keras.callbacks import ModelCheckpoint, TensorBoard
from wandb.keras import WandbCallback, WandbModelCheckpoint

def createCallbacks(PATH_TENSORBOARD, PATH_LOGS, config, train_dataset, test_dataset ,dev_dataset, pipeline):
    NAME_TIME = time.strftime("%Y%m%d-%H%M%S-")
    NAME_STRUCTURE = '{epoch:02d}.h5'
    
    tensorBoard_callbacks = TensorBoard(log_dir=PATH_TENSORBOARD)
    callbacks_model = [tensorBoard_callbacks]
    
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
        print_output_WandB = CustomCallbacksWandB(pipeline=pipeline, 
                                                 path_logs=PATH_LOGS, 
                                                 dataset=test_dataset)
        
        log_WandB = WandbCallback(training_data=train_dataset, 
                                  validation_data=dev_dataset, 
                                  save_graph=True,
                                  save_model=False, 
                                  log_weights=True,
                                  log_gradients=True, 
                                  log_evaluation=True)
        
        checkpoint_callbacks = WandbModelCheckpoint(filepath=PATH_LOGS,
                                                    save_weights_only=True, 
                                                    **config['config_train']['checkpoint'])
        
        callbacks_model.append(print_output_WandB)
        callbacks_model.append(log_WandB)
        callbacks_model.append(checkpoint_callbacks)
    else:
        checkpoint_callbacks = ModelCheckpoint(filepath=PATH_LOGS + NAME_TIME + NAME_STRUCTURE, 
                                               save_weights_only=True, 
                                               **config['config_train']['checkpoint'])
        callbacks_model.append(checkpoint_callbacks)
    return callbacks_model