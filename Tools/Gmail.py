import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def sendEmail(sender_email, sender_password, receiver_email, subject, body, smtp_server='smtp.gmail.com', port=587):
    # Tạo một đối tượng message
    message = MIMEMultipart() 
    message['From'] = sender_email 
    message['To'] = receiver_email  
    message['Subject'] =  subject
  
    # Thêm body vào email
    message.attach(MIMEText(body, 'plain')) 
  
    # Tạo một session SMTP
    session = smtplib.SMTP(smtp_server, port) 
  
    # Bắt đầu phiên TLS cho bảo mật
    session.starttls() 
  
    # Authentication
    session.login(sender_email, sender_password) 
  
    # Chuyển đổi message sang dạng string và gửi
    text = message.as_string()
    session.sendmail(sender_email, receiver_email, text) 
  
    session.quit()
