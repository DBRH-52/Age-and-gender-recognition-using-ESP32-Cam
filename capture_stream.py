#capture_stream.py
import os
import cv2
import requests
import numpy as np

stream_url = "http://192.168.21.13:81/stream" # bez :81 nie zadziala, ale z 8080 nie dziala
capture_video = cv2.VideoCapture(stream_url)
if not capture_video.isOpened():
    print("Error. Couldn't open video stream URL.")
    exit()
else:
    print("Stream opened successfully.")

predict_url = "http://192.168.21.13:81/predict"
'''capture_video = cv2.VideoCapture(predict_url)
if not capture_video.isOpened():
    print("Error. Couldn't open video predict stream URL.")
    exit()
else:
    print("Predict stream opened successfully.")'''

# wymuszenie MJPEG codec
#capture_video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
# Set FPS and buffer size
#capture_video.set(cv2.CAP_PROP_FPS, 30)  # ustawienie frame rate
#capture_video.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # ustawienie buffer size zeby nie bylo delaya

while True:
    return_capture_video_read, frame = capture_video.read()
    if not return_capture_video_read:
        print("Failed to capture frame.")
        break

    cv2.imshow("ESP32-CAM AI Thinker Camera Stream", frame) # lokalne

    # Preprocessing the imgae
    image_resized = cv2.resize(frame, (224,224))
    image_normalized = image_resized / 255.0
    image_expanded = np.expand_dims(image_normalized, axis=0)

    # Sending the frme to the server for prediction
    response = requests.post(predict_url, json={'image': image_expanded.tolist()}) # konwersja do listy
    #response = requests.post(stream_url, json={'image': image_expanded.tolist()})  # konwersja do listy

    if response.status_code == 200:
        prediction_data = response.json()
        print(f"Gender: {prediction_data['gender']}")
        print(f"Age: {prediction_data['age']}")

    # Exit when the user presses 'x' button
    if cv2.waitKey(1) & 0xFF == ord('x'): #0xFF - 255(10) ; ACII x = 120
        break

capture_video.release()
cv2.destroyAllWindows()