﻿# Sentinel: Real-Time Surveillance Alert System

Welcome to **Sentinel**, a real-time surveillance alert system designed to detect human presence and provide early warnings via alerts. This project leverages computer vision to deliver enhanced security monitoring, alerting you in real time when activity is detected on camera.

---

## 🎯 Project Goals

1. **Human Detection and Tracking**: Continuously monitor and detect human presence in video streams while the system is active.
2. **Real-Time Alerts**: Send instant alerts with video clips or snapshots via a Telegram bot when activity is detected.
3. **Optimized Performance**: Use multithreading to ensure smooth, efficient processing of multiple camera feeds.
4. **Enhanced Accuracy**: Utilize a YOLOv8 Nano model for improved accuracy, specifically tuned for both day and night-time images.

---

## 📹 How It Works

- **Capture Video Streams**: The system captures feeds using the RTSP protocol, allowing multiple cameras to be monitored simultaneously.
- **Interval-Based Inference**: Rather than running inferences on every frame, it processes frames at specific intervals to optimize system resources.
- **Alert Triggers**: Upon detecting human presence, the system sends an alert via a Telegram bot, attaching a video clip or snapshot of the detected activity.

---

## 🚀 Why This Project?

Inspired by recent security issues in my neighborhood, Sentinel is designed to address a common security gap. Traditionally, surveillance systems serve primarily as review tools after an incident occurs. Sentinel’s real-time detection approach aims to transform passive surveillance systems into active, preventative security measures.

---

## 🛠️ Technology Stack

- **Computer Vision**: YOLOv11 Nano model, fine-tuned for high accuracy on nighttime and low-light conditions.
- **Multithreading**: Python’s `threading` library for efficient handling of multiple camera feeds.
- **Telegram Bot**: For real-time alerts sent directly to your mobile device with snapshots or video clips.
- **RTSP Protocol**: Captures video streams from compatible IP cameras.

---

## 🔧 Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/sentinel
    cd sentinel
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up your Telegram Bot**:
   - Create a bot on Telegram and get your `BOT_TOKEN`.
   - Add your bot’s token to the memory.json

4. **Configure memory.json**:
   - Add your parameters to the memory.json -> IP, PORT, USER, PASSWORD, INFERENCE THRESHOLD.

---

## 🚀 Usage

1. **Run the System**:
   ```bash
   python main.py
   ```

2. **Adjust Model:
   - Fine-tune the inference intervals as needed.

3. **Telegram Bot commands:
```bash
   - Fine-tune the inference intervals as needed.
   - Available commands:
        /activate - Activate sentinel inference
        /deactivate - Deactivate sentinel inference
        /set - Set Serial number of sentinel - If memory.json serial number matches the one that has been sended, the chat will be subscribed.
        /snapshot <camera_number> - Get an instant picture from a specific camera
        /active_cams - Get the list of active cameras
        /inference_status - Show the inference state
        /remove - Desuscribe from the bot
        /suscriptors - List all subscribers
        /mem_stat - Shows allocated memory
        /set_criteria X - Set inference threshold criteria -> sweet spot on 0.69-0.75 
        /help - Show avalaible commands
```
---


## 🙌 Acknowledgments

Thanks to the open-source computer vision community for resources and tools that made Sentinel possible.

---

Stay secure with **Sentinel** – your eyes when you can’t watch.
