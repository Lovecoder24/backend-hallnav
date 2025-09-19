<<<<<<< HEAD
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 1. Define the HallClassifier class correctly at the top level
class HallClassifier(nn.Module):
    def __init__(self):
        super(HallClassifier, self).__init__()
        # Define layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # Add a common activation function like ReLU
        self.relu = nn.ReLU()
        # Define a max pooling layer to reduce spatial dimensions, if needed
        self.pool = nn.MaxPool2d(2, 2)
        # Assuming the input image is 224x224, the output size would be smaller
        # You must calculate the correct input size for the linear layer
        self.fc1 = nn.Linear(16 * 112 * 112, 10) # Example: This calculation might be wrong.

    def forward(self, x):
        # Define the forward pass
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x) # Example: This might be needed
        # Flatten the tensor
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

def load_model():
    # Construct the absolute path to the model file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "hall_classifier_raw.pth")
    
    try:
        model = HallClassifier()
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval() 
        return model
    except FileNotFoundError:
        print(f"Error: The model file was not found at {model_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None
=======
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
>>>>>>> 438a38c6f51f4eb2942ae6d74ada2d1f1b1cb28f

# Initialize model once
model = load_model()

<<<<<<< HEAD
# Preprocessing
=======
# Preprocessing same as training
>>>>>>> 438a38c6f51f4eb2942ae6d74ada2d1f1b1cb28f
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
<<<<<<< HEAD
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
=======
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
>>>>>>> 438a38c6f51f4eb2942ae6d74ada2d1f1b1cb28f
])

# Recognition function
def recognize_hall(image_path):
<<<<<<< HEAD
    if model is None:
        print("Model is not loaded. Cannot perform recognition.")
        return None

=======
>>>>>>> 438a38c6f51f4eb2942ae6d74ada2d1f1b1cb28f
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)

    classes = ["LT1 & 2", "LT3 & 4"]
<<<<<<< HEAD
    return classes[pred.item()]
=======
    return classes[pred.item()]
>>>>>>> 438a38c6f51f4eb2942ae6d74ada2d1f1b1cb28f
