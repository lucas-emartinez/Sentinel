import Memory.memory as M

class ModelInference:
    def __init__(self, model, memory_data: M.MemoryData):
        self.model = model
        self.infer_activated = memory_data.get_nested("inference.activated.status")
        self.infer_threshold = memory_data.get_nested("inference.threshold")

    def infer(self, frame):
        if self.infer_activated:
            results = self.model.predict(frame, classes=[0], verbose=False)
            return results
