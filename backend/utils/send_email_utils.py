from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import smtplib
from dotenv import load_dotenv

dotenv_path = "backend/.env"
load_dotenv(dotenv_path=dotenv_path)

EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')

async def send_email(email: str, subject: str, body: str):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg["To"] = email
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, email, msg.as_string())

        server.quit()

        print(f"Email sent successfully")
        return True
    
    except Exception as e:
         print(f"Error sending email: {e}")
         return False