import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import numpy as np
from .base_hand_gesture_recognizer import BaseHandGestureRecognizer


class HandGestureRecognizer(BaseHandGestureRecognizer):
    def __init__(self, device: str = 'cpu', confidence_threshold: float = 0.5):
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.model_name = "prithivMLmods/Hand-Gesture-19"
        self.model = SiglipForImageClassification.from_pretrained(self.model_name).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.labels = self.model.config.id2label
        self.confidence_threshold = confidence_threshold

    def predict(self, image: np.ndarray):
        if image is None:
            return None, None
        if isinstance(image, np.ndarray):
            if image.shape[-1] == 3:
                image = Image.fromarray(image[..., ::-1])
            else:
                image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("Input must be np.ndarray or PIL.Image")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            predicted_class_idx = int(logits.argmax(-1).item())
            score = float(probs[predicted_class_idx])
        probs_dict = {self.labels[i]: float(probs[i]) for i in range(len(probs))}
        if score < self.confidence_threshold:
            return None, probs_dict
        return self.labels[predicted_class_idx], probs_dict
