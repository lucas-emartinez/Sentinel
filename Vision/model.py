from ultralytics import YOLO
import torch

def load_model():
    # Load a pretrained YOLO model
    device = torch.device("cpu")
    if torch.cuda.is_available():
        #Get the GPU device name
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using GPU Multi-Process Service (MPS)")
    model_path = "yolo11n.pt"
    model = YOLO(model_path, "v11")
    model.to(device)
    return model
