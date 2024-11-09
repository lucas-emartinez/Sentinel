from Memory.memory import MemoryData
from model import load_model
from cameraProcessor import CameraProcessor

def main():
    # Load memory and model
    memory = MemoryData()
    model = load_model()
    
    # Create and start camera processor
    processor = CameraProcessor(memory, model)
    processor.start()

if __name__ == "__main__":
    main()