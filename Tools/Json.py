import json, os
 
def loadJson(path='./config.json'):
    if os.path.exists(path):
        with open(path, 'r') as json_file:
                data_save = json.load(json_file)
        return data_save
    else:
        return None
    

def saveJson(path='./config.json', data=None):
    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    return True

    