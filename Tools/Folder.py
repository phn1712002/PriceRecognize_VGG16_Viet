import os 

def createFolder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return True

def showFilesInDirectory(path):
    file_list = []        
    if os.path.exists(path):
        if os.path.isdir(path):
            files = os.listdir(path)
            if files:
                for file in files:
                    file_path = os.path.join(path, file)  # Đường dẫn đầy đủ của file
                    file_list.append((file, file_path))  # Thêm tuple (tên file, đường dẫn) vào danh sách
            else:
                return file_list  # Trả về danh sách trống nếu thư mục không có tệp tin
        else:
            return path
    else:
        return None
    
    return file_list
