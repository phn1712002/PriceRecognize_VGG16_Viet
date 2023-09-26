import cv2, keyboard, os, datetime, time

# Label: 00000 là không cầm tiền, còn lại là các mệnh giá
label = "1000"
path = "D:/_Source/Working/AI/PriceRecognize_VGG16_Viet/Dataset/raw/"

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Đặt chiều rộng tối đa
cap.set(4, 720)  # Đặt chiều cao tối đa

# Biến đếm, để chỉ lưu dữ liệu sau khoảng 60 frame, tránh lúc đầu chưa kịp cầm tiền lên
i = 0
count = 1

while True:
    # Capture frame-by-frame
    i += 1
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.resize(frame, dsize=None, fx=0.3, fy=0.3)

    # Hiển thị
    cv2.imshow('frame', frame)

    # Lưu dữ liệu khi nhấn Enter
    if cv2.waitKey(1) and keyboard.is_pressed('enter'):  # Enter key
        # Tạo thư mục nếu chưa có
        if not os.path.exists(path + str(label)):
            os.mkdir(path + str(label))

        current_time = datetime.datetime.now()
        file_name = current_time.strftime("%Y-%m-%d_%H-%M-%S.png")
        cv2.imwrite(path + str(label) + '/' + file_name, frame)
        i = 0  # Đặt lại biến đếm sau khi lưu hình ảnh
        print(f"Save pic {count}")
        count += 1
        time.sleep(1)
        
    # Thoát vòng lặp khi nhấn 'q'
    if cv2.waitKey(1) and keyboard.is_pressed('esc'):
        break

# Giải phóng tài nguyên và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()



