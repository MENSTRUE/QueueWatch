from __future__ import annotations
from .ensemble import QueueWatchEnsemble
from .preprocess import build_inference_transform,prepare_image

class QueueWatchInference:
    def __init__(self,model_dir,device=None,image_size=320):
        self.transform=build_inference_transform(image_size)
        self.ensemble=QueueWatchEnsemble(model_dir=model_dir,device=device)

    def predict(self,image):
        tensor=prepare_image(image,self.transform)
        return self.ensemble.predict(tensor)
