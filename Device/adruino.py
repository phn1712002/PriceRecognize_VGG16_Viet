from serial import Serial


class UnoR3:
    def __init__(self, COM, baud_rate=9600):  
        self.COM = COM  
        self.baud_rate = baud_rate  
        self.ser = Serial(COM, self.baud_rate, timeout=1)
    
    def __del__(self):
        self.ser.close()
    
    def writeLCD(self, messenger):
        messenger = str(messenger) + '\n'
        self.ser.write(messenger.encode())