import cv2
import tensorflow as tf
from Architecture.Model import VGG16
from albumentations import Compose, ShiftScaleRotate, Flip, GaussNoise
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer

class PriceRecognize_VGG16(VGG16):
    def __init__(self, class_names=None, config_augments=None, config_model=None):
        super().__init__(**config_model)  # provide necessary parameters as per the needs
        self.config_augments = config_augments
        self.transforms = self._create_transforms() if self.config_augments else None
        self.check_augments = tf.constant(bool(self.transforms), dtype=tf.bool)
        self.tokenizer = Tokenizer()

        self.tokenizer.fit_on_texts(class_names) if class_names else None
        self.num_classes = len(self.tokenizer.word_index)

    def _create_transforms(self):
        if isinstance(self.config_augments, dict) and all(key in self.config_augments for key in ['GaussNoise', 'ShiftScaleRotate', 'Flip']):
            return Compose([
                GaussNoise(**self.config_augments['GaussNoise']),
                ShiftScaleRotate(**self.config_augments['ShiftScaleRotate']),
                Flip(**self.config_augments['Flip'])
            ])
        else:
            return None

    def load_and_resize_image(self, image_path):
        image_path = image_path.numpy().decode()
        image = cv2.imread(image_path)
        return cv2.resize(image, (self.image_size[0], self.image_size[1]), interpolation=cv2.INTER_AREA)  # check here for image size

    def augment_image(self, image):
        aug_data = self.transforms(image=image.numpy())
        return aug_data['image'] 

    def encoderLabel(self, label): 
        label = label.numpy().decode()
        label_seq = self.tokenizer.texts_to_sequences([label])
        label_seq = [item for sublist in label_seq for item in sublist] 
        return to_categorical(label_seq, num_classes=self.num_classes, dtype='int32')

    def mapProcessing(self, path, label):
        image = tf.py_function(func=self.load_and_resize_image, inp=[path], Tout=tf.float32)
        if self.check_augments:
            image = tf.numpy_function(func=self.augment_image, inp=[image], Tout=tf.float32)
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
