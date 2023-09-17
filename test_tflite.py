import cv2
from Architecture.Model import VGG16_TFLite

path_image = 'D:/_Source/Working/AI/PriceRecognize_VGG16_Viet/Dataset/raw/archive/200000/2023-09-17_18-19-44.png'
image = cv2.imread(path_image)

model = VGG16_TFLite().build()
output = model.predict(image)
print(output)