from Architecture.Model import VGG16_TFLite
from Device.peripherals import Camera

cam = Camera(COM=0)
model = VGG16_TFLite(path='./Checkpoint/save/').build()
model.predict_with_showCam(cam)


    
    
