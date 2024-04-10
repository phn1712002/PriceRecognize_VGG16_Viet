import keyboard
from Architecture.Model import VGG16_TFLite
from Device.adruino import UnoR3
from Device.peripherals import Camera
from Tools.Json import loadJson

# Environment Variables
PATH_CONFIG = './config_system.json'

# Get config
config = loadJson(PATH_CONFIG)
if not config == None:
    keys_to_check = ['config_camera', 'config_arduino', 'config_model', 'config_other']
    if all(key in config for key in keys_to_check):
        config_camera = config['config_camera']
        config_arduino = config['config_arduino']
        config_model = config['config_model']
        config_other = config['config_other']
    else:
        raise RuntimeError("Error config")

# Include device
cam = Camera(**config_camera)
board = UnoR3(**config_arduino)
model = VGG16_TFLite(**config_model).build()

# Working
if config_other['view_cam_with_adruino']:
    model.predict_with_showCam(cam=cam, 
                               func_print_info=board.writeLCD, 
                               opt_speed_T_or_smooth_F=config_other['opt_speed_T_or_smooth_F'], 
                               key_stop=config_other['key_stop'])
else:
    while True:
        frames = cam.getFrame()
        currency = model.predict(frames)
        print(f"Currency : {currency}")
        board.writeLCD(currency)
        if keyboard.is_pressed(config_other['key_stop']): break
    
