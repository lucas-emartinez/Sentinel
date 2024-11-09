import cv2
from time import time
import numpy as np

# Función para dibujar cuadros y contar detecciones
def draw_boxes(frame, results, detection_count, detection_timeframes, infer_threshold, cam_index):
    detected = False  # Para verificar si se detectó alguna persona
    boxes = []  # Lista para almacenar las coordenadas de las cajas
    if results is None:
        return False, boxes
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 0 and box.conf[0] >= infer_threshold:
                detected = True
                detection_count[cam_index] += 1  # Incrementar el contador de detecciones
                detection_timeframes[cam_index] = time()  # Actualizar tiempo de detección
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                boxes.append((x1, y1, x2, y2, conf))  # Guardar la caja y la confianza
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Persona {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return detected, boxes  # Retorna si se detectó alguna persona y las cajas

def create_combined_frame(original_frame, boxes):
    """
    Creates a combined frame with:
    - Left side: Complete original frame
    - Right side: Resized detection area
    
    Args:
        original_frame: The complete original frame
        boxes: List of detection boxes (x1, y1, x2, y2, conf)
    """
    height, width = original_frame.shape[:2]
    
    # Create a blank canvas twice the width of original frame
    combined_frame = np.zeros((height, width * 2, 3), dtype=np.uint8)
    
    # Place the complete original frame on the left side
    combined_frame[:, :width] = original_frame
    
    # For the right side, we'll show the detection area
    if boxes:
        # Get the bounding box coordinates of the first detection
        x1, y1, x2, y2, conf = boxes[0]  # Using first detection for simplicity
        
        # Extract the region of interest (ROI)
        roi = original_frame[y1:y2, x1:x2]
        
        if roi.size > 0:  # Check if ROI is valid
            # Resize the ROI to fit the right half while maintaining aspect ratio
            roi_height, roi_width = roi.shape[:2]
            aspect_ratio = roi_width / roi_height
            
            # Calculate new dimensions to fit in right half
            if height * aspect_ratio <= width:
                # Height limited
                new_height = height
                new_width = int(height * aspect_ratio)
            else:
                # Width limited
                new_width = width
                new_height = int(width / aspect_ratio)
            
            # Resize ROI
            roi_resized = cv2.resize(roi, (new_width, new_height))
            
            # Calculate position to center the ROI in right half
            y_offset = (height - new_height) // 2
            x_offset = width + (width - new_width) // 2
            
            # Place the resized ROI in the right half
            combined_frame[y_offset:y_offset + new_height, 
                        x_offset:x_offset + new_width] = roi_resized
            
            # Draw a rectangle around the detection in the original frame
            cv2.rectangle(combined_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add confidence text
            conf_text = f"Conf: {conf:.2f}"
            cv2.putText(combined_frame, conf_text, (x_offset, y_offset - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return combined_frame
    
def show_frame(frame, cam_index):
    cv2.imshow(f"CAM {cam_index}", frame)

