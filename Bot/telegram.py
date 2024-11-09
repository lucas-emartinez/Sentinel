import threading
import telebot
import cv2
import torch
from queue import Queue
from collections import deque
import time
import os
import Memory.memory as memory
import tracemalloc
import torch
from infer import ModelInference
import subprocess
from datetime import datetime, timedelta


class TelegramBot:
    def __init__(self, token, model_inference: ModelInference, camera_manager, memory_data: memory.MemoryData):
        self.bot = telebot.TeleBot(token)
        self.model_inference = model_inference
        self.camera_manager = camera_manager
        self.memory_data = memory_data
        # Dictionary to store camera-specific frame buffers
        self.camera_buffers = {}  # {camera_id: deque()}
        # Control de rate limiting
        self.last_sent_time = datetime.now()
        self.messages_in_minute = 0
        self.MAX_MESSAGES_PER_MINUTE = 19
        self.MIN_INTERVAL = 3
        
        # Video configuration
        self.VIDEO_FPS = 10
        self.VIDEO_MAX_DURATION = 10  # seconds
        self.VIDEO_THRESHOLD = 5  # minimum frames for video
        self.MAX_BUFFER_SIZE = 30  # frames per camera
        
        # Lock for synchronization
        self.send_lock = threading.Lock()
        self.buffer_locks = {}  # {camera_id: threading.Lock()}
        
        self.register_handlers()
        
    def get_or_create_buffer(self, camera_id):
        """Get or create a buffer for a specific camera."""
        if camera_id not in self.camera_buffers:
            self.camera_buffers[camera_id] = deque(maxlen=self.MAX_BUFFER_SIZE)
            self.buffer_locks[camera_id] = threading.Lock()
        return self.camera_buffers[camera_id]
        
    def buffer_frame(self, frame, camera_id):
        """Add a frame to the camera-specific buffer."""
        buffer = self.get_or_create_buffer(camera_id)
        with self.buffer_locks[camera_id]:
            buffer.append({
                'frame': frame,
                'timestamp': datetime.now()
            })
        
    def process_detection(self, frame, camera_id):
        """Process a new detection from a specific camera."""
        self.buffer_frame(frame, camera_id)
        buffer = self.get_or_create_buffer(camera_id)
        
        # Check if we should send a message
        if len(buffer) >= self.VIDEO_THRESHOLD or (
            buffer and (datetime.now() - buffer[0]['timestamp']).total_seconds() >= 10
        ):
            subscribers = self.get_subscribers()
            self.send_detection_message(subscribers, camera_id)
            
    def can_send_message(self):
        """Verifica si podemos enviar un nuevo mensaje según el rate limiting."""
        now = datetime.now()
        # Resetear contador si ha pasado un minuto
        if (now - self.last_sent_time) > timedelta(minutes=1):
            self.messages_in_minute = 0
            
        return (self.messages_in_minute < self.MAX_MESSAGES_PER_MINUTE and 
                (now - self.last_sent_time).total_seconds() >= self.MIN_INTERVAL)
        
    def create_video_from_frames(self, frames, camera_id):
        """Creates a video from frames with camera-specific naming."""
        if not frames:
            return None
            
        temp_video_path = f'temp_detection_cam_{camera_id}.mp4'
        temp_avi_path = f'temp_detection_cam_{camera_id}.avi'
        
        height, width = frames[0]['frame'].shape[:2]
        
        try:
            # Try MJPG first as it's widely supported
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            
            out = cv2.VideoWriter(
                temp_avi_path,
                fourcc,
                self.VIDEO_FPS,
                (width, height),
                isColor=True
            )
            
            if not out.isOpened():
                # Fallback to more basic codec
                out.release()
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(
                    temp_avi_path,
                    fourcc,
                    self.VIDEO_FPS,
                    (width, height),
                    isColor=True
                )
            
            # Write frames with camera ID and timestamp
            for frame_data in frames:
                frame = frame_data['frame'].copy()
                timestamp = frame_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                # Add camera ID and timestamp to frame
                cv2.putText(frame, f"Camera {camera_id} - {timestamp}", 
                           (10, height - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                out.write(frame)
            
            out.release()
            
            # Convert to MP4
            try:
                mp4_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_mp4 = cv2.VideoWriter(
                    temp_video_path,
                    mp4_fourcc,
                    self.VIDEO_FPS,
                    (width, height),
                    isColor=True
                )
                
                cap = cv2.VideoCapture(temp_avi_path)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out_mp4.write(frame)
                
                cap.release()
                out_mp4.release()
                
            except Exception as e:
                print(f"Direct MP4 conversion failed for camera {camera_id}: {e}")
                if self._has_ffmpeg():
                    self._convert_with_ffmpeg(temp_avi_path, temp_video_path)
                else:
                    temp_video_path = temp_avi_path
            
            # Cleanup temporary AVI
            if os.path.exists(temp_avi_path) and os.path.exists(temp_video_path) and temp_video_path != temp_avi_path:
                os.remove(temp_avi_path)
                
            return temp_video_path
            
        except Exception as e:
            print(f"Error creating video for camera {camera_id}: {e}")
            for file in [temp_video_path, temp_avi_path]:
                if os.path.exists(file):
                    try:
                        os.remove(file)
                    except:
                        pass
            return None
    
    def _has_ffmpeg(self):
        """Check if ffmpeg is available."""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE)
            return True
        except:
            return False

    def _convert_with_ffmpeg(self, input_path, output_path):
        """Convert video using ffmpeg."""
        try:
            subprocess.run([
                'ffmpeg', '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-pix_fmt', 'yuv420p',
                output_path,
                '-y'
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"FFMPEG conversion failed: {e.stderr.decode()}")
            raise
        
    def send_detection_message(self, subscribers, camera_id, detection_type='video'):
        """Send detection message for a specific camera."""
        buffer = self.get_or_create_buffer(camera_id)
        
        if not buffer:
            return

        try:
            with self.buffer_locks[camera_id]:
                buffer_size = len(buffer)
                if buffer_size < 2:  # Need at least 2 frames for meaningful time span
                    return
                    
                time_span = (buffer[-1]['timestamp'] - 
                            buffer[0]['timestamp']).total_seconds()
                
                base_caption = (f"⚠️ Detección de intrusión - Cámara {camera_id}\n"
                              f"📸 {buffer_size} detecciones en {time_span:.1f} segundos\n"
                              f"🕒 Última detección: {buffer[-1]['timestamp'].strftime('%H:%M:%S')}")

                with self.send_lock:
                    if not self.can_send_message():
                        return

                    if detection_type == 'video' and buffer_size >= self.VIDEO_THRESHOLD:
                        # Create and send video
                        video_path = self.create_video_from_frames(list(buffer), camera_id)
                        if video_path:
                            with open(video_path, 'rb') as video:
                                for subscriber in subscribers:
                                    try:
                                        self.bot.send_video(subscriber, video, 
                                                          caption=f"{base_caption}\n🎥 Video de la secuencia")
                                        self.messages_in_minute += 1
                                        self.last_sent_time = datetime.now()
                                    except Exception as e:
                                        print(f"Error sending video from camera {camera_id} to {subscriber}: {e}")
                            os.remove(video_path)
                    else:
                        # Send latest image if not enough frames for video
                        latest_frame = buffer[-1]['frame']
                        temp_path = f'temp_detection_cam_{camera_id}.jpg'
                        cv2.imwrite(temp_path, latest_frame)
                        
                        with open(temp_path, 'rb') as photo:
                            for subscriber in subscribers:
                                try:
                                    self.bot.send_photo(subscriber, photo, 
                                                      caption=f"{base_caption}\n📸 Imagen instantánea")
                                    self.messages_in_minute += 1
                                    self.last_sent_time = datetime.now()
                                except Exception as e:
                                    print(f"Error sending photo from camera {camera_id} to {subscriber}: {e}")
                        os.remove(temp_path)
                    
                    # Clear buffer after sending
                    buffer.clear()

        except Exception as e:
            print(f"Error in send_detection_message for camera {camera_id}: {e}")

    def is_authorized(self, subcriber_id):
        subscribers = self.get_subscribers()
        return subcriber_id in subscribers

    def get_subscribers(self):
        return self.memory_data.get_nested("bot.subscribers") or []  # Return empty list if None
    
    def get_chat_id(self, message):
        if message.chat.type in ['group', 'supergroup']:
            return str(message.chat.id)
        else:
            return str(message.from_user.id)

    def register_handlers(self):
        @self.bot.message_handler(commands=['set'])
        def set_command(message):
            
            subcriber_id = self.get_chat_id(message)

            # Obtener el número de serie de los argumentos del mensaje
            command_parts = message.text.split(' ')
            serial_number = str(command_parts[1]) if len(command_parts) > 1 else None
            
            # Verificar si el número de serie fue proporcionado
            if serial_number is None:
                return

            # Verificar si el número de serie está autorizado
            authorized_sn = self.memory_data.get("sn") or []
            if serial_number in authorized_sn:
                # Verificar si el suscriptor ya está en la lista
                if not self.is_authorized(subcriber_id):
                    subscribers = self.get_subscribers()  # Obtener lista de suscriptores
                    subscribers.append(subcriber_id)  # Agregar el nuevo suscriptor
                    self.memory_data.set_nested("bot.subscribers", subscribers)  # Guardar cambios en rom.json
                    self.bot.reply_to(message, f"Suscrito a Sentinela: {serial_number}")
                else:
                    self.bot.reply_to(message, "Ya estás suscrito.")
            else:
                self.bot.reply_to(message, "No estás autorizado para usar este bot.")

        @self.bot.message_handler(commands=['activate'])
        def activate_command(message):
            # Verificar si el mensaje viene de un grupo o de una conversación privada
            subcriber_id = self.get_chat_id(message)

            if self.is_authorized(subcriber_id):
                # Si el mensaje viene de un grupo, responder en el grupo y avisar quien fue el que activo al bot
                self.model_inference.infer_activated = True
                self.memory_data.set_nested("inference.activated.status", True)
                if message.chat.type in ['group', 'supergroup']:
                    self.bot.reply_to(message, f"{message.from_user.first_name} ha activado al Sentinela.")
                else:
                    self.bot.reply_to(message, "Sentinela activado.")
            else:
                self.bot.reply_to(message, "No estás autorizado para usar este bot.")

        @self.bot.message_handler(commands=['remove'])
        def remove_command(message):
            # Verificar si el mensaje viene de un grupo o de una conversación privada
            subcriber_id = self.get_chat_id(message)

            if self.is_authorized(subcriber_id):
                subscribers = self.get_subscribers()
                # elimina al suscriptor de la lista
                subscribers.remove(subcriber_id)
                self.memory_data.set_nested("bot.subscribers", subscribers)
                # Si el mensaje viene de un grupo, responder en el grupo y avisar quien fue el que desuscribio al bot
                if message.chat.type in ['group', 'supergroup']:
                    self.bot.reply_to(message, f"Desuscripción exitosa. {message.from_user.first_name} ha desuscripto al Sentinela.")
                else:
                    self.bot.reply_to(message, "Desuscripción exitosa del Sentinela.")
                return True
            self.bot.reply_to(message, "No estás suscrito.")
            return False

        @self.bot.message_handler(commands=['deactivate'])
        def deactivate_command(message):
            subcriber_id = self.get_chat_id(message)
            if self.is_authorized(subcriber_id):
                self.memory_data.set_nested("inference.activated.status", False)
                if message.chat.type in ['group', 'supergroup']:
                    self.bot.reply_to(message, f"{message.from_user.first_name} ha desactivado el Sentinela.")
                else:
                    self.bot.reply_to(message, "Sentinela desactivado.")
            else:
                self.bot.reply_to(message, "No estás autorizado para usar este bot.")

        @self.bot.message_handler(commands=['suscriptors'])
        def suscriptors_command(message):
            # Verificar si el mensaje viene de un grupo o de una conversación privada
            subcriber_id = self.get_chat_id(message)

            if self.is_authorized(subcriber_id):
                subscribers = self.get_subscribers()
                self.bot.reply_to(message, f"Suscriptores: {', '.join(subscribers)}")
        
        @self.bot.message_handler(commands=['inference_status'])
        def inference_status_command(message):
            # Verificar si el mensaje viene de un grupo o de una conversación privada
            subcriber_id = self.get_chat_id(message)

            if self.is_authorized(subcriber_id):
                status = self.memory_data.get_nested("inference.activated.status")
                self.bot.reply_to(message, f"Estado de la inferencia: {'Activado' if status else 'Desactivado'}")
        
        @self.bot.message_handler(commands=['snapshot'])
        def snapshot_command(message):
            subscriber_id = self.get_chat_id(message)
            
            if self.is_authorized(subscriber_id):
                try:
                    # Extract camera number from the command
                    command_parts = message.text.split()
                    if len(command_parts) != 2:
                        self.bot.reply_to(message, "Usage: /snapshot <camera_number>")
                        return
                    
                    camera_number = int(command_parts[1])
                    
                    # Get the frame from the camera
                    frame = self.camera_manager.get_camera_frame(camera_number)
                    
                    if frame is not None:
                        # Save the frame as an image
                        temp_path = f'temp_snapshot_cam_{camera_number}.jpg'
                        cv2.imwrite(temp_path, frame)
                        
                        # Send the image
                        with open(temp_path, 'rb') as photo:
                            self.bot.send_photo(subscriber_id, photo, 
                                                caption=f"📸 Snapshot from Camera {camera_number}")
                        os.remove(temp_path)
                    else:
                        self.bot.reply_to(message, f"Error fetching frame from camera {camera_number}")
                        
                except Exception as e:
                    self.bot.reply_to(message, f"Error processing snapshot command: {e}")
            else:
                self.bot.reply_to(message, "No estás autorizado para usar este bot.")        
            
        @self.bot.message_handler(commands=['active_cams'])
        def active_cams_command(message):
            subcriber_id = self.get_chat_id(message)
            if self.is_authorized(subcriber_id):
                active_cam_indices = [] 
                for i in range(len(self.camera_manager.cams)):
                    if self.camera_manager.cams[i] is not None:
                        active_cam_indices.append(i + 1)
                if active_cam_indices:
                    self.bot.reply_to(message, f"Cámaras activas: {', '.join(str(i) for i in active_cam_indices)}")
                else:
                    self.bot.reply_to(message, "No hay cámaras activas en este momento.")
            else:
                self.bot.reply_to(message, "No estás autorizado para usar este bot.")

        
        # Comando para modificar el threshold de inferencia
        @self.bot.message_handler(commands=['set_criteria'])
        def set_threshold_command(message):
            # Verificar si el mensaje viene de un grupo o de una conversación privada
            subcriber_id = self.get_chat_id(message)
            if self.is_authorized(subcriber_id):
                try:
                    threshold = float(message.text.split(' ')[1])
                    # max should be 1 and min should be 0
                    if 0 <= threshold <= 1:
                        self.model_inference.infer_threshold = threshold
                        self.memory_data.set_nested("inference.threshold", threshold)
                        self.bot.reply_to(message, f"Threshold de detección actualizado a {threshold}")
                    else:
                        self.bot.reply_to(message, "El threshold debe estar entre 0 y 1")
                except (IndexError, ValueError):
                    self.bot.reply_to(message, "Por favor, proporciona un valor numérico válido para el threshold.")
            else:
                self.bot.reply_to(message, "No estás autorizado para usar este comando.")
        
        @self.bot.message_handler(commands=['mem_stat'])
        def mem_stat_command(message):
            # Verificar si el mensaje viene de un grupo o de una conversación privada
            subcriber_id = self.get_chat_id(message)
            
            if self.is_authorized(subcriber_id):
                mem_gpu = 0
                current, peak = tracemalloc.get_traced_memory()
                torch_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                mem_cpu_current = current / 1e6  # Convert to MB
                mem_cpu_peak = peak / 1e6  # Convert to MB
                mem_gpu = 0
                gpu_name = "CPU"
                if torch.cuda.is_available():
                    mem_gpu = torch.cuda.memory_allocated() / 1e9
                    gpu_name = torch.cuda.get_device_name()
                elif torch.backends.mps.is_available():
                    mem_gpu = torch.cuda.memory_allocated() / 1e9
                    gpu_name = "MPS"
                self.bot.reply_to(message, 
                                  f"Inference processor: {gpu_name}\n"
                                  f"Malloc GPU: {mem_gpu:.2f} GB\n"
                                  f"Uso de memoria ram: {mem_cpu_current:.2f} MB\n"
                                  f"Pico de uso de memoria ram: {mem_cpu_peak:.2f} MB\n")
                
        @self.bot.message_handler(commands=['help'])
        def help_command(message):
            subcriber_id = self.get_chat_id(message)

            if self.is_authorized(subcriber_id):
                self.bot.reply_to(message, "Comandos disponibles:\n"
                                         "/activate - Activa el sentinela\n"
                                         "/deactivate - Desactiva el sentinela\n"
                                         "/set - Setea el número de serie de tu sentinela\n"
                                         "/snapshot <camera_number> - Get an instant picture from a specific camera\n"
                                         "/active_cams - Get the list of active cameras\n"
                                         "/inference_status - Muestra el estado de la inferencia\n"
                                         "/remove - Desuscribirse de las notificaciones\n"
                                         "/suscriptors - Lista los suscriptores actuales\n"
                                         "/mem_stat - Muestra el estado de la memoria\n"
                                         "/set_criteria X - Setea el threshold de detección (0-1)\n"
                                         "/help - Mostrar los comandos disponibles\n"
                                         "/stop - Detener el bot")

    def start(self):
        self.bot.infinity_polling()
