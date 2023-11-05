import time, psutil, subprocess, keyboard
from Tools.Json import loadJson
from jlclient import jarvisclient
from jlclient.jarvisclient import *

# Environment Variables
PATH_CONFIG = './config.json'
sleep_duration = 60

# Execute 'ps -ef' command on a Linux terminal
processes = subprocess.run(['ps', '-ef'], stdout=subprocess.PIPE, text=True)
print(processes.stdout)  # Display the list of running processes

# Get configuration details
config = loadJson(PATH_CONFIG)
if not config == None:
    # Check for required keys in the configuration file
    keys_to_check = ['config_jarvislabs']
    if all(key in config for key in keys_to_check):
        config_jarvislabs = config['config_jarvislabs']
    else:
        raise RuntimeError("Error in configuration file")

# Enter the PID to check
print("-"*100)
pid_to_check = input("Enter PID: ")
print("-"*100)

try:
    pid_to_check = int(pid_to_check)
except ValueError:
    print("PID must be an integer.")
else:
    # Continuously monitor the entered PID
    while True:
        if psutil.pid_exists(pid_to_check):
            # Retrieve information about the process with the provided PID
            process = psutil.Process(pid_to_check)
            time_now = time.strftime("%Y%m%d-%H%M%S")
            print(f"PID {pid_to_check} is running at time {time_now}")
            time.sleep(sleep_duration)
        else:
            # If the provided PID is not running, pause the Jarvis instance
            jarvisclient.token = config_jarvislabs['token']
            jarvisclient.user_id = config_jarvislabs['user_id']
            instance = User.get_instances()[0]
            instance.pause()  # Pause Jarvis instance
