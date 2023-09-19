import cv2
import tensorflow as tf, os, datetime
from keras import optimizers, losses, Model, applications, metrics
from keras.layers import Flatten, Dense, Dropout
from keras.preprocessing import text
from Tools.Json import saveJson, loadJson

class CustomModel:
    def __init__(self, model=None, opt=None, loss=None):
        self.model = model
        self.opt = opt
        self.loss = loss

class VGG16(CustomModel):
    def __init__(self, 
                 class_names:text.Tokenizer,
                 name='VGG16',
                 freeze=True,
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
        self.freeze = freeze
        self.name = name
        self.image_size = tuple(image_size)
        self.num_layers = num_layers
        self.hidden_layers = hidden_layers
        self.rate_dropout = rate_dropout
        self.class_names = class_names
        self.num_lables = len(class_names.index_word) + 1

        
    def build(self, summary=False):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            #input = Input(shape=self.image_size, name='input')
            
            # Transfer Learning
            if self.transfer_learning:
                model_vgg16_conv = applications.VGG16(input_shape=self.image_size, weights='imagenet')
                # Đóng băng các layers
                if self.freeze:
                    for layer in model_vgg16_conv.layers:
                        layer.trainable = False
            else: 
                model_vgg16_conv = applications.VGG16(input_shape=self.image_size, weights=None)
                     
            #output_vgg16_conv = model_vgg16_conv(input)
            
            # Thêm vào các FC layers 
            x = Flatten(name='flatten')(model_vgg16_conv.get_layer('block5_pool').output)
            
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
            
            self.model = Model(inputs=model_vgg16_conv.inputs, outputs=output, name=self.name)
            self.model.compile(optimizer=self.opt, loss=self.loss, metrics=[metrics.CategoricalAccuracy()])
        
        if summary:
            self.model.summary()
        return self
    
    def fit(self, train_dataset, dev_dataset=None, epochs=1, callbacks=None):

        self.model.fit(train_dataset,
                       validation_data=dev_dataset,
                       epochs=epochs,
                       callbacks=callbacks)
        
        return self
       
    def predict(self, image_input):
        image_input_tf = self.standardizedImage(image_input)
        output_tf = self.model.predict_on_batch(image_input_tf)
        output = self.decoderLable(output_tf)
        return output

    def standardizedImage(self, image_input):
        image = tf.convert_to_tensor(cv2.resize(image_input, (self.image_size[0], self.image_size[1]), interpolation = cv2.INTER_AREA), dtype=tf.float32)
        image = tf.cast(image/255.0, tf.float32)
        return image
    
    def decoderLable(self, output_tf):
        output_tf = tf.math.argmax(output_tf, axis=-1) 
        output_decoder = self.class_names.sequences_to_texts([output_tf.numpy()])
        return output_decoder[0]
    
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
            return VGG16_TFLite(path=path_export, name_file=self.name).build()
        else:
            raise RuntimeError('Error path')


class VGG16_TFLite(VGG16):
    def __init__(self, path='./Checkpoint/export/', name_file='VGG16'):
        
        self.name_file = name_file
        self.path = path
        
        self.index_input = None
        self.index_ouput = None
        self.dtype_input = None
        
        if os.path.exists(path):
            path_json_class_names = path + name_file + '_class_names.json'
            path_json_config = path + name_file + '.json'
            
            config_model = loadJson(path=path_json_config)
            config_class_names = loadJson(path=path_json_class_names)
            class_names = text.tokenizer_from_json(config_class_names)
            super().__init__(class_names=class_names, **config_model)
        else:
            raise RuntimeError('Model load error')
    
    def build(self):
        self.model = tf.lite.Interpreter(model_path=self.path + self.name_file + '.tflite')
        self.index_input = self.model.get_input_details()[0]['index']
        self.dtype_input = self.model.get_input_details()[0]['dtype']
        self.index_ouput = self.model.get_output_details()[0]['index']
        return self
    
    def predict(self, image_input):
        image_input_tf = super().standardizedImage(image_input)
        output_tf  = self.__invoke(image_input_tf)
        output = super().decoderLable(output_tf)
        return output
    
    def predict_in_evaluate(self, dataset=None): 
        all_y_target = []
        all_y_pred = []
        for _, (X, y) in enumerate(dataset):
            output_tf  = self.__invoke(X)
            for i, _ in enumerate(output_tf):
                all_y_pred.append(tf.math.argmax(output_tf, axis=-1)[i].numpy())
                all_y_target.append(tf.math.argmax(y, axis=-1)[i].numpy())
            
        return all_y_target, all_y_pred
    
    def __invoke(self, input_tf):
        model = self.model
        
        if input_tf.shape[0] == 1: 
            input_tf = tf.expand_dims(input_tf, axis=0)
        else: 
            model.resize_tensor_input(self.index_input, tensor_size=input_tf.shape)
            
        model.allocate_tensors()
        model.set_tensor(self.index_input, tf.cast(input_tf, dtype=self.dtype_input))
        model.invoke()
        ouput = model.get_tensor(self.index_ouput)
        tf_output = tf.convert_to_tensor(ouput)
        return tf_output
        
        
