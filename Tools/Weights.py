import os
from datetime import datetime
from Architecture.Model import CustomModel

def loadNearest(class_model: CustomModel, path_folder_logs="./Checkpoint/logs/"): 
    """
        Hàm này có mục đích để khi huấn luyện model xong có thể load model lên tiếp huấn luyện tiếp tục
    Returns:
        bool: Trả về điều kiện load model 
    """
    check_load = False
    path_list = os.listdir(path_folder_logs)
    if len(path_list) > 0:
        path_list = [path_folder_logs + name for name in path_list]
        time = [datetime.fromtimestamp(os.path.getmtime(path)) for path in path_list]
        while not check_load:
            nearest_time_index = time.index(max(time))
            nearest_path = path_list[nearest_time_index]
            check_h5 = nearest_path.rfind('.h5')
            if not (check_h5 == -1):
                class_model.model.load_weights(nearest_path)
                print(f"Load file : {nearest_path}")
                check_load = True
            else:
                path_list.pop(nearest_time_index)
                time.pop(nearest_time_index)
            if len(path_list) == 0:
                break
    return class_model

def loadWeights(class_model: CustomModel, path=None):
    check_load = False
    if os.path.exists(path):
        class_model.model.load_weights(path)
        print(f"Load file : {path}")
    return class_model

def getPathWeightsNearest(path_folder_logs="./Checkpoint/logs/"):
    path_load = None
    check_load = False
    path_list = os.listdir(path_folder_logs)
    if len(path_list) > 0:
        path_list = [path_folder_logs + name for name in path_list]
        time = [datetime.fromtimestamp(os.path.getmtime(path)) for path in path_list]
        while not check_load:
            nearest_time_index = time.index(max(time))
            nearest_path = path_list[nearest_time_index]
            check_h5 = nearest_path.rfind('.h5')
            if not (check_h5 == -1):
                check_load = True
                path_load = nearest_path
            else:
                path_list.pop(nearest_time_index)
                time.pop(nearest_time_index)
            if len(path_list) == 0:
                break
    return path_load