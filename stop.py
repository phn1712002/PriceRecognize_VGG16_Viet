import psutil
import time
from Tools.Json import loadJson
from jlclient import jarvisclient
from jlclient.jarvisclient import *

# Environment Variables
PATH_CONFIG = './config.json'
sleep_duration = 5

# Get config
config = loadJson(PATH_CONFIG)
if not config == None:
    keys_to_check = ['config_jarvislabs']
    if all(key in config for key in keys_to_check):
        config_jarvislabs = config['config_jarvislabs']
    else:
        raise RuntimeError("Error config")

# Nhập PID cần kiểm tra
pid_to_check = input("Nhập PID: ")

try:
    pid_to_check = int(pid_to_check)
except ValueError:
    print("PID phải là một số nguyên.")
else:
    while True:
        if psutil.pid_exists(pid_to_check):
            process = psutil.Process(pid_to_check)
            time_now = time.strftime("%Y%m%d-%H%M%S")
            print(f"PID {pid_to_check} đang chạy tại thời gian {time_now}")
            time.sleep(sleep_duration)
        else:
            jarvisclient.token = config_jarvislabs['token']
            jarvisclient.user_id = config_jarvislabs['user_id']
            instance = User.get_instances()[0]
            instance.pause()

        
