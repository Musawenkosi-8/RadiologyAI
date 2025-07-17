# streamlit_app/utils.py

import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import cv2
import numpy as np

def load_model(path):
    model = torch.jit.load(path, map_location=torch.device('cpu'))
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def load_labels(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def apply_gradcam(model, image_tensor):
    image_tensor.requires_grad_()
    output = model(image_tensor)
    class_idx = output.argmax(dim=1).item()
    output[0, class_idx].backward()

    gradients = image_tensor.grad[0].mean(dim=1).detach().numpy()
    grayscale_cam = np.maximum(gradients, 0)
    grayscale_cam = cv2.resize(grayscale_cam, (224, 224))
    grayscale_cam -= np.min(grayscale_cam)
    grayscale_cam /= np.max(grayscale_cam)
    return grayscale_cam, class_idx
