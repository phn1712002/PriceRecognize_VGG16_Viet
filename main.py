import keyboard
from Architecture.Model import VGG16_TFLite
from Device.adruino import UnoR3
from Device.peripherals import Camera

cam = Camera(COM=0)
model = VGG16_TFLite(path='./Checkpoint/save/').build()
board = UnoR3(COM="COM4")


while True:
    frame = cam.getFrame()
    currency = model.predict(frame)
    board.writeLCD(currency)
    
    if keyboard.is_pressed('esc'):
        break

    
    
