import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from keras.models import load_model
age_gender_model = load_model('gender_model.h5')


path = 'data-new'

def findEncodings():
    encodeList = []
    classNames = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".JPG") or file.endswith(".png"):
                curImgPath = os.path.join(root, file)
                curImg = cv2.imread(curImgPath)
                curImg = cv2.resize(curImg, (225, 225))
                curImg = cv2.cvtColor(curImg, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(curImg)
                if len(face_locations) > 0:
                    encoded_face = face_recognition.face_encodings(curImg, face_locations)[0]
                    encodeList.append(encoded_face)
                    name = os.path.basename(root)
                    classNames.append(name)
    return encodeList, classNames

encoded_face_train, classNames = findEncodings()
def markAttendance(name):
    with open('Attendance.csv', 'a') as f:
        now = datetime.now()
        time = now.strftime('%I:%M:%S:%p')
        date = now.strftime('%d-%B-%Y')
        f.writelines(f'{name}, {time}, {date}\n')
def send_email(subject, message, image):
    # Configure SMTP settings
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    smtp_username = 'gdharanids@gmail.com'
    smtp_password = 'bgfgbvtfgnuuuyuy'
    recipient_email = 'dharani.21iamos121@iadc.ac.in'

    # Create a multipart message container
    msg = MIMEMultipart()
    msg['From'] = smtp_username
    msg['To'] = recipient_email
    msg['Subject'] = subject

    # Attach the message as plain text
    msg.attach(MIMEText(message, 'plain'))

    # Load the image without the bounding box
    img_without_bounding_box = cv2.imread(image)

    # Convert image data to bytes
    _, img_encoded = cv2.imencode('.jpg', img_without_bounding_box)
    img_data = img_encoded.tobytes()

    # Attach the image
    image_attachment = MIMEImage(img_data, name=os.path.basename(image))
    msg.attach(image_attachment)

    # Connect to SMTP server and send the email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.send_message(msg)


# Capture video from webcam
cap = cv2.VideoCapture(0)
print(encoded_face_train)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
    unknown_detected = False

    for encode_face,faceloc in zip(encoded_faces,faces_in_frame):
        matches = face_recognition.compare_faces(encoded_face_train, encode_face)
        faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
        matchIndex = np.argmin(faceDist)
        print(matchIndex)
        if not matches[matchIndex]:
            unknown_detected = True
            break

    # After the unknown face is detected
    if unknown_detected:
        # Save the snapshot without the bounding box
        now = datetime.now()
        timestamp = now.strftime('%Y%m%d%H%M%S')
        snapshot_filename = f'snapshot_{timestamp}.jpg'
        cv2.imwrite(snapshot_filename, img)

        # Perform gender detection
        face_img = img[y1:y2, x1:x2]  # Extract the face region
        face_img = cv2.resize(face_img, (128, 128))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = face_img.astype('float') / 255.0  # Normalize the pixel values
        face_img = np.expand_dims(face_img, axis=-1)  # Add channel dimension
        face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension

        gender_pred = age_gender_model.predict(face_img)
        gender = 'Male' if gender_pred[0] < 0.5 else 'Female'

        # Send email with snapshot, date, time, and gender
        subject = 'Unknown Face Detected'
        message = f"An unknown face was detected at {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        message += f"Gender: {gender}"
        send_email(subject, message, snapshot_filename)

    tolerance = 0.5

    for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
        matches = face_recognition.compare_faces(encoded_face_train, encode_face, tolerance=tolerance)
        faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
        matchIndex = np.argmin(faceDist)
        print(matchIndex)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper().lower()
            y1, x2, y2, x1 = faceloc
            # since we scaled down by 4 times
            y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
        else:
            name = 'Unknown'
            y1,x2,y2,x1 = faceloc
            # since we scaled down by 4 times
            y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1,y1),(x2,y2),(0,0,255),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,0,255), cv2.FILLED)
            cv2.putText(img, name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
