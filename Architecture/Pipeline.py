import cv2
import tensorflow as tf
from Architecture.Model import VGG16
from albumentations import Compose, ShiftScaleRotate, Flip, PixelDropout, RandomBrightness, RingingOvershoot
from keras.utils import to_categorical

class PriceRecognize_VGG16(VGG16):
    def __init__(self, class_names=None, config_augments=None, config_model=None):
        super().__init__(opt=None, loss=None, class_names=class_names,**config_model)
        
        self.check_augments = tf.constant(False, dtype=tf.bool)
        self.transforms = None
        self.config_augments = config_augments
        
        if not config_augments is None:
            self.transforms = Compose([
                PixelDropout(**config_augments['PixelDropout']),
                #RandomBrightness(**config_augments['RandomBrightness']),
                RingingOvershoot(**config_augments['RingingOvershoot']),
                ShiftScaleRotate(**config_augments['ShiftScaleRotate']),
                Flip(**config_augments['Flip'])
            ])
            self.check_augments = tf.constant(True, dtype=tf.bool)
        
    def augmentsImage(self, image=None):
        aug_data = self.transforms(image=image.numpy())
        return aug_data['image']
    
    def loadImage(self, image_path):
        iimage_path = image_path.numpy().decode()
        image = cv2.imread(image_path)
        return cv2.resize(image, (self.image_size[0], self.image_size[1]), interpolation=cv2.INTER_AREA)  # check here for image size
    
    def encoderLabel(self, label):
        label = label.numpy().decode()
        label_seq = self.tokenizer.texts_to_sequences([label])
        label_seq = [item for sublist in label_seq for item in sublist] 
        return to_categorical(label_seq, num_classes=self.num_classes, dtype='int32')

    def mapProcessing(self, path, label):
        image = tf.py_function(func=self.loadImage, inp=[path], Tout=tf.float32)
        if self.check_augments:
            image = tf.numpy_function(func=self.augmentsImage, inp=[image], Tout=tf.float32)
        image = tf.cast(image/255.0, tf.float32)
        label = tf.py_function(self.encoderLabel, inp=[label], Tout=tf.int32)
        return image, label
    
    def __call__(self, dataset=None, batch_size=1):
        data = (tf.data.Dataset.from_tensor_slices((dataset))
                .map(self.mapProcessing, num_parallel_calls=tf.data.AUTOTUNE)
                .shuffle(buffer_size=batch_size, reshuffle_each_iteration=True)
                .batch(batch_size)
                .prefetch(buffer_size=tf.data.AUTOTUNE))
        return data