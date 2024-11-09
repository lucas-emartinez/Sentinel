import cv2
import numpy as np
import threading

class CameraManager:
    def __init__(self):
        self.cams = []
        self.lock = threading.Lock()
        
    def initialize_camera(self, cam_index, user, password, ip, port, protocol):
        cap = cv2.VideoCapture(f"{protocol}://{user}:{password}@{ip}:{port}/cam/realmonitor?channel={cam_index}&subtype=0")
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if self.is_black_screen(frame):
                    print(f"Pantalla negra detectada en la cámara {cam_index}.")
                    cap.release()
                    return False
                else:
                    # Ensure the cams list has enough elements
                    while len(self.cams) < cam_index:
                        self.cams.append(None)
                    self.cams[cam_index - 1] = cap
                    print(f"Cámara {cam_index} inicializada correctamente")
                    return True
            else:
                print(f"Error al capturar el frame de la cámara {cam_index}.")
                cap.release()
                return False
        else:
            print(f"Error al iniciar la cámara {cam_index}")
            cap.release()
            return False
    
    def release_cameras(self):
        """Clean up resources"""
        self.running = False
        with self.lock:
            for cam in self.cams:
                if cam.isOpened():
                    cam.release()
        # Wait for buffer threads to finish
        for thread in self.buffer_threads.values():
            thread.join(timeout=1.0)
            
    def get_camera_frame(self, camera_number):
        # This method should be implemented to return a frame from the specified camera
        # You might need to adjust this based on how your camera system is set up
        if 1 <= camera_number <= len(self.cams):
            cap = self.get_camera(camera_number - 1)
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    return frame
        return None
        
    def is_black_screen(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        total_pixels = thresh.size
        black_pixels = np.count_nonzero(thresh == 0)
        return (black_pixels / total_pixels) * 100 > 80
            
    def get_camera(self, index):
        with self.lock:  # Asegurar el acceso sincronizado a self.cams
            return self.cams[index] if index < len(self.cams) else None

    def release_cameras(self):
        with self.lock:  # Sincronizar la liberación de cámaras
            for cam in self.cams:
                if cam.isOpened():
                    cam.release()
