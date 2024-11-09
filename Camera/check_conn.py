import cv2
from time import sleep

def connect_stream(url):
    cap = None
    while cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            break
        print("Reconectando al stream...")
        sleep(2)  # Esperar antes de reintentar
    return cap

connect_stream("rtsp://p2pquinta:quintap2p@190.244.61.244:554/cam/realmonitor?channel=1&subtype=0")