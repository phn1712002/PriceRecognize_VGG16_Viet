#include <Arduino.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);  // Địa chỉ I2C 0x27, màn hình 16x2

void setup() {
  lcd.init();                      // Khởi tạo màn hình LCD
  lcd.backlight();                 // Bật đèn nền
  lcd.setCursor(0, 0);            // Đặt con trỏ về vị trí (0,0)
  Serial.begin(9600);             // Khởi động cổng Serial
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');  // Đọc dữ liệu từ Serial Monitor
    
    // Xóa màn hình
    lcd.clear();
    
    // In văn bản lên màn hình
    lcd.setCursor(0, 0);
    lcd.print("Menh gia tien:");
    lcd.setCursor(0, 1);
    lcd.print(input);
  }
}
