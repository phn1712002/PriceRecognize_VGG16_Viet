import tensorflow as tf, os, datetime
from keras import optimizers, losses, Model, Input, applications, metrics
from keras.layers import Flatten, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from Tools.Json import saveJson

class CustomModel:
    def __init__(self, model=None, opt=None, loss=None):
        self.model = model
        self.opt = opt
        self.loss = loss

class VGG16(CustomModel):
    def __init__(self, 
                 class_names:Tokenizer,
                 name='VGG16',
                 transfer_learning=False,
                 num_layers=2,
                 hidden_layers=4096,
                 rate_dropout=0.5,
                 image_size=(128, 128, 3),
                 opt=optimizers.Adam(), 
                 loss=losses.CategoricalCrossentropy(from_logits=False)
                 ):
        super().__init__(model=None, opt=opt, loss=loss)
        self.transfer_learning = transfer_learning
        self.name = name
        self.image_size = image_size
        self.num_layers = num_layers
        self.hidden_layers = hidden_layers
        self.rate_dropout = rate_dropout
        self.class_names = class_names
        self.num_lables = len(class_names.index_word) + 1

        
    def build(self, summary=False):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            input = Input(shape=self.image_size, name='input')
            
            # Transfer Learning
            if self.transfer_learning:
                model_vgg16_conv = applications.VGG16(weights='imagenet', include_top=False)
            else: 
                model_vgg16_conv = applications.VGG16(weights=None, include_top=False)
            
            # Đóng băng các layers
            """
            for layer in model_vgg16_conv.layers:
                layer.trainable = False
            """
              
            output_vgg16_conv = model_vgg16_conv(input)
            
            # Thêm vào các FC layers 
            x = Flatten(name='flatten')(output_vgg16_conv)
            
            hidden_layers = None
            rate_dropout = None
            for i in range(0, self.num_layers):
                
                if isinstance(self.hidden_layers, list): 
                    if i > len(self.hidden_layers): hidden_layers = self.hidden_layers[-1]
                    else: hidden_layers = self.hidden_layers[i]
                else: hidden_layers = self.hidden_layers
                    
                if isinstance(self.rate_dropout, list): 
                    if i > len(self.rate_dropout): rate_dropout = self.rate_dropout[-1]
                    else: rate_dropout = self.rate_dropout[i]
                else: rate_dropout = self.rate_dropout

                x = Dense(hidden_layers, activation='relu', name=f'fc_{i + 1}')(x)
                x = Dropout(rate_dropout, name=f'dropout_{i + 1}')(x)
                    
                    
            output = Dense(self.num_lables, activation='softmax', name='output')(x)
            
            self.model = Model(inputs=input, outputs=output, name=self.name)
            self.model.compile(optimizer=self.opt, loss=self.loss)
        
        if summary:
            self.model.summary()
        return self
    
    def fit(self, train_dataset, dev_dataset=None, epochs=1, callbacks=None):

        self.model.fit(train_dataset,
                       validation_data=dev_dataset,
                       epochs=epochs,
                       callbacks=callbacks)
        
        return self
    
    def standardizedImage(self, image_tf):
        image = tf.image.resize(image_tf, (self.image_size[0], self.image_size[1]))
        image = tf.cast(image_tf/255.0, tf.float32)
        return image
    
    def decoderLable(self, output_tf):
        output_tf = tf.math.argmax(output_tf, axis=-1) 
        output = self.class_names.sequences_to_texts([output_tf.numpy()])
        return output[0]
    
    def getConfig(self):
        return {
            'name': self.name,
            'image_size': self.image_size,
            'num_layers': self.num_layers,
            'hidden_layers': self.hidden_layers,
            'rate_dropout': self.rate_dropout
        }
        
    def loadWeights(self, path=None, name_file=None):
        if name_file is None:
            check_load = False
            path_list = os.listdir(path)
            if len(path_list) > 0:
                path_list = [path + name for name in path_list]
                time = [datetime.datetime.fromtimestamp(os.path.getmtime(path)) for path in path_list]
                while not check_load:
                    nearest_time_index = time.index(max(time))
                    nearest_path = path_list[nearest_time_index]
                    check_h5 = nearest_path.rfind('.h5')
                    if not (check_h5 == -1):
                        self.model.load_weights(nearest_path)
                        print(f"Load file : {nearest_path}")
                        check_load = True
                    else:
                        path_list.pop(nearest_time_index)
                        time.pop(nearest_time_index)
                    if len(path_list) == 0:
                        break
        else:
            if os.path.exists(path + name_file):
                self.model.load_weights(path + name_file)
                print(f"Load file : {path + name_file}")
        
        return self   
    
    def exportTFlite(self, path_export='./Checkpoint/export/'):
        if os.path.exists(path_export):
            # Convert to tflite
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            tflite_model = converter.convert()
            
            # Get config model
            config_model = self.getConfig()
            config_class_names = self.class_names.to_json()
            
            # Path config
            path_tflite = path_export + self.name + '.tflite' 
            path_json_class_names = path_export + self.name + '_class_names.json'
            path_json_config = path_export + self.name + '.json'
            
            # Save
            saveJson(path=path_json_config, data=config_model)
            saveJson(path=path_json_class_names, data=config_class_names)  
            tf.io.write_file(filename=path_tflite, contents=tflite_model)
            
            print(f"Export model to tflite filename:{path_tflite} and json:{path_json_config} - {path_json_class_names}")
        else:
            raise RuntimeError('Error path')

