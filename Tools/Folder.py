import os 

def createFolder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return True