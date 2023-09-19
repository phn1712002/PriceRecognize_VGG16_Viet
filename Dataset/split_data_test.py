import os
import shutil
import random

# Đường dẫn đến thư mục gốc chứa các thư mục con
root_dir = './raw/'

# Đường dẫn đến thư mục đích cho tập train và test
train_dir = './train'
test_dir = './test'

# Tỷ lệ dữ liệu được sử dụng cho tập test
test_ratio = 0.1

# Lặp qua các thư mục con trong thư mục gốc
for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)
    
    # Kiểm tra xem mục con có phải là một thư mục không
    if os.path.isdir(subdir_path):
        # Lấy danh sách tất cả các tệp trong mục con này
        file_list = os.listdir(subdir_path)
        
        # Tính số lượng tệp cần sao chép cho tập test
        num_test_files = int(len(file_list) * test_ratio)
        
        # Lấy một danh sách ngẫu nhiên của các tệp để sao chép cho tập test
        test_files = random.sample(file_list, num_test_files)
        
        # Tạo thư mục con tương ứng trong thư mục đích cho tập train và test
        train_subdir = os.path.join(train_dir, subdir)
        test_subdir = os.path.join(test_dir, subdir)
        os.makedirs(train_subdir, exist_ok=True)
        os.makedirs(test_subdir, exist_ok=True)
        
        # Sao chép các tệp vào thư mục đích
        for file in file_list:
            src_path = os.path.join(subdir_path, file)
            if file in test_files:
                dest_path = os.path.join(test_subdir, file)
            else:
                dest_path = os.path.join(train_subdir, file)
            shutil.copy2(src_path, dest_path)

print("Hoàn thành sao chép và chia tập train và test.")
