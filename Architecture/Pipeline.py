import cv2
import tensorflow as tf
from Architecture.Model import VGG16
from albumentations import Compose, RandomBrightnessContrast, HueSaturationValue, RandomContrast, Rotate
from keras.utils import to_categorical

class PriceRecognize_VGG16(VGG16):
    def __init__(self, class_names=None, config_augments=None, config_model=None):
        super().__init__(opt=None, loss=None, class_names=class_names,**config_model)
        
        self.check_augments = tf.constant(False, dtype=tf.bool)
        self.transforms = None
        self.config_augments = config_augments
        
        if not config_augments is None:
            self.transforms = Compose([
                Rotate(**config_augments['Rotate']),
            ])
            self.check_augments = tf.constant(True, dtype=tf.bool)
        
    def augmentsImage(self, image=None):
        aug_data = self.transforms(image=image)
        aug_img = aug_data["image"]
        aug_img = tf.image.resize(aug_img, (self.image_size[0], self.image_size[1]))
        return aug_img
    
    def loadImage(self, image_path):
        image_path = image_path.numpy().decode()
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.image_size[0], self.image_size[1]), interpolation = cv2.INTER_AREA)
        return tf.convert_to_tensor(image, dtype=tf.float32)
    
    def encoderLable(self, lable):
        lable = lable.numpy().decode()
        lable = self.class_names.texts_to_sequences([lable])
        lable = tf.convert_to_tensor(lable)
        lable = tf.squeeze(lable)
        lable = to_categorical(lable, num_classes=self.num_lables, dtype='int32')
        return tf.convert_to_tensor(lable, dtype=tf.int32)
     
    def mapProcessing(self, path, lable):
        image = tf.py_function(func=self.loadImage, inp=[path], Tout=tf.float32)
        image = tf.cond(
            self.check_augments == True, 
            lambda: tf.numpy_function(func=self.augmentsImage, inp=[image], Tout=tf.float32),
            lambda: image
            )
        image = tf.cast(image/255.0, tf.float32)
        lable = tf.py_function(self.encoderLable, inp=[lable], Tout=tf.int32)
        return image, lable
    
    def __call__(self, dataset=None, batch_size=1):
        data = tf.data.Dataset.from_tensor_slices(dataset)
        data = (data.map(self.mapProcessing, num_parallel_calls=tf.data.AUTOTUNE)
                .shuffle(buffer_size=batch_size, reshuffle_each_iteration=True)
                .batch(batch_size)
                .prefetch(buffer_size=tf.data.AUTOTUNE))
        return data