import netron, keyboard, threading
from Tools.Folder import showFilesInDirectory


def _stopNetron(port):
    while True:
        if keyboard.is_pressed('esc'): 
            netron.stop(address=port) 
            break

def _viewNetron(name_file, port):
    print("Is pressed ESC stop view")
    netron.start(file=name_file, address=port)  


def viewArchitectureAI(path, port=8080, file_format_extension='.tflite'):
    # Get all name file in Directory
    files = showFilesInDirectory(path)
    
    # Check file exits
    if path is None: return None 
    
    # Check files is folder ?
    if files.__class__ is list:
        
        # Filtered all file .tflite
        filtered_files = [file for file in files if file[0].endswith(file_format_extension)]
        
        if len(filtered_files) > 1:
            # View index all file
            for index, (file, path) in enumerate(filtered_files):
                print(f"{index + 1}: {file}")
            
            # Select the view model
            while True:
                print(f"Select the index of file to view:")
                index_input = int(input("Index: "))
                index_input -= 1
            
                if 0 <= index_input < len(filtered_files):
                    name_file = filtered_files[index_input][1]
                    break
                else:
                    print('Error index, please re-enter')
        else:
            name_file = filtered_files[0][1]
    else:
        name_file = path  
         
    # View
    view_thread = threading.Thread(target=_viewNetron, args=(name_file, port, ))
    stop_thread = threading.Thread(target=_stopNetron, args=(port,))
    
    # Wait
    stop_thread.start()
    view_thread.start()
    view_thread.join()
    stop_thread.join()
    
    
    