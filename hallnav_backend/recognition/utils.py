import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Load model
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model.load_state_dict(torch.load(
        "backend/hallnav_backend/recognition/hall_classifier_raw.pth",
        map_location=torch.device("cpu")
    ))
    model.eval()
    return model

# Initialize model once
model = load_model()

# Preprocessing same as training
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Recognition function
def recognize_hall(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)

    classes = ["LT1 & 2", "LT3 & 4"]
    return classes[pred.item()]
