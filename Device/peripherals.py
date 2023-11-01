import cv2 

class Camera:
    def __init__(self, COM, resolution=(1280, 720), flip=False):
        self.COM = COM
        self.resolution = resolution
        self.flip = flip

        self.cap = cv2.VideoCapture(self.COM)
        if not self.cap.isOpened():
            raise RuntimeError("Camera error")
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()
        
    def getFrame(self):
        ret, image = self.cap.read()
        if not ret: image = None
        elif self.flip: image = cv2.flip(image, 1)
        return image
    
    def showFrame(self, frame, name_windown='Webcam', key_stop='q', delay=1):
        if not frame is None: 
            cv2.imshow(name_windown, frame)
            if cv2.waitKey(delay) & 0xFF == ord(key_stop): return True
        return False
