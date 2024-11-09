from concurrent.futures import ThreadPoolExecutor
from camera import CameraManager
from model import load_model
from infer import ModelInference
from utils import draw_boxes, show_frame, create_combined_frame
import Bot.telegram as telegram
import cv2
import tracemalloc
import gc
from Memory.memory import MemoryData
from time import time
import queue
import platform
import threading

# Monitorización de memoria
tracemalloc.start()

class CameraProcessor:
    def __init__(self, memory, model):
        self.memory = memory
        self.model = model
        self.frame_queues = {}
        self.camera_manager = CameraManager()
        self.model_inference = ModelInference(model, memory)
        self.detection_counts = {}
        self.detection_timeframes = {}
        self.running = True
        self.active_cameras = []  # Track actually active cameras
        
        # Initialize settings
        network_settings = memory.get("network_settings")
        self.ip = network_settings['ip']
        self.port = network_settings['port']
        self.user = network_settings['username']
        self.password = network_settings['password']
        self.protocol = network_settings['protocol']
        
        # Camera settings
        self.NUM_CAMERAS = 8
        self.detection_threshold = 3
        self.detection_interval = 2
        
        # Initialize component
        self.token = memory.get_nested("bot.token")
        self.bot = telegram.TelegramBot(self.token, self.model_inference, self.camera_manager, memory)
                
        # Thread synchronization
        self.inference_lock = threading.Lock()
        
        # Initialize cameras
        self.initialize_cameras()
        
    def initialize_cameras(self):
        """Initialize cameras and track which ones are actually active"""
        for i in range(1, self.NUM_CAMERAS):
            if self.camera_manager.initialize_camera(i, self.user, self.password, 
                                                   self.ip, self.port, self.protocol):
                self.frame_queues[i] = queue.Queue(maxsize=10)
                self.detection_counts[i] = 0
                self.detection_timeframes[i] = 0
                self.active_cameras.append(i)
                print(f"Camera {i} added to active cameras list")
    
    def infer_and_process(self, cam_index, frame):
        """Process frame with model inference using thread-safe approach"""
        with self.inference_lock:
            try:
                results = self.model_inference.infer(frame)
                detected, boxes = draw_boxes(frame, 
                                             results, 
                                             self.detection_counts, 
                                          self.detection_timeframes, 
                                          self.model_inference.infer_threshold,
                                          cam_index)
                
                current_time = time()
                if detected:
                    if ((current_time - self.detection_timeframes[cam_index] <= self.detection_interval) and 
                        (self.detection_counts[cam_index] >= self.detection_threshold)):
                        self.detection_counts[cam_index] = 0
                        combined_frame = create_combined_frame(frame, boxes)
                        self.bot.process_detection(combined_frame, cam_index)
                    else:
                        self.detection_counts[cam_index] += 1
                
                return frame
            except Exception as e:
                print(f"Error in inference for camera {cam_index}: {e}")
                return frame
    
    def process_camera(self, cam_index):
        """Process individual camera feed with proper error handling"""
        print(f"Starting processing for camera {cam_index}")
        cam = self.camera_manager.get_camera(cam_index - 1)
        if cam is None:
            print(f"Camera {cam_index} not available.")
            return

        frame_count = 0
        last_processed_time = time()
        zoom_level = 1.0
        
        while self.running:
            try:
                if not cam.isOpened():
                    print(f"Camera {cam_index} is not open. Attempting to reinitialize...")
                    self.camera_manager.initialize_camera(cam_index, self.user, self.password, self.ip, self.port, self.protocol)
                    cam = self.camera_manager.get_camera(cam_index - 1)
                    if cam is None:
                        print(f"Failed to reinitialize camera {cam_index}. Exiting camera processing.")
                        return
                    continue

                ret, frame = cam.read()
                if not ret:
                    print(f"Failed to read frame from camera {cam_index}. Skipping this frame.")
                    continue

                # Resize for memory optimization
                frame_resized = cv2.resize(frame, (640, 480))
                
                # Proces every 2 fr
                if frame_count % 2 == 0:
                    frame_resized = self.infer_and_process(cam_index, frame_resized)
                    last_processed_time = time()
                    

                # Reset frame count to prevent overflow
                if frame_count >= 1000:
                    frame_count = 0
                                        
                # Handle display based on platform
                if platform.system() != "Darwin":  # Skip display on MacOS
                    show_frame(frame_resized, cam_index)
                    
                    # Check for 'q' key to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break
                    

            except Exception as e:
                print(f"Error processing camera {cam_index}: {e}")
                time.sleep(1)  # Add a small delay before retrying
            
        print(f"Camera {cam_index} processing stopped")
    
    def close_resources(self):
        """Properly clean up all resources"""
        self.running = False
        self.camera_manager.release_cameras()
        cv2.destroyAllWindows()
        gc.collect()
    
    def start(self):
        """Start processing with proper camera handling"""
        try:
            with ThreadPoolExecutor(max_workers=len(self.active_cameras) + 1) as executor:
                # Start Telegram bot
                executor.submit(self.bot.start)
                
                # Start camera processing for active cameras only
                camera_futures = []
                print(f"Active cameras: {self.active_cameras}")
                for cam_index in self.active_cameras:
                    print(f"Submitting camera {cam_index} for processing")
                    camera_futures.append(
                        executor.submit(self.process_camera, cam_index)
                    )
                
                # Wait for all cameras to finish
                for future in camera_futures:
                    future.result()
                    
        except Exception as e:
            print(f"Error in execution: {e}")
        finally:
            self.close_resources()
