from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from django.shortcuts import render # You may need to add this import
from django.views.generic import TemplateView
from .models import Hall, Schedule

# --- PyTorch Model Integration ---
# Model file colocated in the recognition app directory
MODEL_PATH = Path(__file__).resolve().parent / 'hall_classifier_raw.pth'

# IMPORTANT: Update CLASS_NAMES with your actual hall names in the order your model expects them
# Model was trained with 2 classes, so we need exactly 2 class names
CLASS_NAMES = ['LT1 & 2', 'LT3 & 4']

# Configuration for edge case handling
MIN_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to accept a prediction
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB max file size
ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
MIN_IMAGE_DIMENSIONS = (50, 50)  # Minimum image size
MAX_IMAGE_DIMENSIONS = (5000, 5000)  # Maximum image size

# Load model (cached for performance)
_model = None

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        model = models.mobilenet_v2(pretrained=False)
        num_classes = len(CLASS_NAMES)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        state = torch.load(str(MODEL_PATH), map_location=torch.device('cpu'))
        model.load_state_dict(state)
        # Sanity check: classifier output size must match class names
        out_features = model.classifier[1].out_features
        if out_features != num_classes:
            raise ValueError(f"Model classifier out_features={out_features} does not match len(CLASS_NAMES)={num_classes}")
        model.eval()
        _model = model
    return _model

# Validation functions for edge cases

def validate_file_size(file_size):
    """Validate file size is within acceptable limits"""
    if file_size > MAX_FILE_SIZE:
        raise ValueError(f"File too large. Maximum size allowed: {MAX_FILE_SIZE // (1024*1024)}MB")
    if file_size < 1024:  # Less than 1KB
        raise ValueError("File too small. Please upload a valid image file.")
    return True

def validate_file_extension(filename):
    """Validate file extension is supported"""
    if not filename:
        raise ValueError("No filename provided")
    # Handle cases where filename might be empty or just whitespace
    filename = filename.strip()
    if not filename:
        raise ValueError("Empty filename provided")
    # Check if filename has an extension
    if '.' not in filename:
        raise ValueError("No file extension found. Please ensure your file has a proper extension (e.g., .jpg, .png)")
    extension = filename.lower().split('.')[-1]
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file format '{extension}'. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}")
    return True

def validate_file_content_type(image_file):
    """Validate file content type as a fallback for missing extensions"""
    try:
        image_file.seek(0)  # Reset file pointer
        test_image = Image.open(image_file)
        image_file.seek(0)  # Reset file pointer again
        if test_image.format and test_image.format.lower() in ['jpeg', 'jpg', 'png', 'bmp', 'tiff']:
            return True
        else:
            raise ValueError(f"Unsupported image format: {test_image.format}")
    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")

def validate_image_content(image):
    """Validate image content and dimensions"""
    if not image:
        raise ValueError("Invalid image file")
    width, height = image.size
    # Check minimum dimensions
    if width < MIN_IMAGE_DIMENSIONS[0] or height < MIN_IMAGE_DIMENSIONS[1]:
        raise ValueError(f"Image too small. Minimum size: {MIN_IMAGE_DIMENSIONS[0]}x{MIN_IMAGE_DIMENSIONS[1]} pixels")
    # Check maximum dimensions
    if width > MAX_IMAGE_DIMENSIONS[0] or height > MAX_IMAGE_DIMENSIONS[1]:
        raise ValueError(f"Image too large. Maximum size: {MAX_IMAGE_DIMENSIONS[0]}x{MAX_IMAGE_DIMENSIONS[1]} pixels")
    # Check if image is mostly one color (might be corrupted or invalid)
    image_array = np.array(image)
    if len(image_array.shape) != 3:
        raise ValueError("Invalid image format. Expected RGB image.")
    mean_color = np.mean(image_array)
    if mean_color < 10 or mean_color > 245:
        raise ValueError("Image appears to be corrupted or invalid (too dark or too bright)")
    return True

def validate_confidence(confidence, predicted_class):
    """Validate prediction confidence meets threshold"""
    if confidence < MIN_CONFIDENCE_THRESHOLD:
        raise ValueError(f"Low confidence prediction ({confidence:.2%}). This doesn't appear to be a lecture hall image. Please upload a clear image of a lecture hall.")
    return True

# Preprocess and predict returns (index, confidence)

def predict_image(image, model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    model = model.to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)
    return predicted.item(), float(conf.item())

class HomePageView(TemplateView):
    template_name = 'index.html'


@csrf_exempt
def recognize_hall(request):
    # Require POST with a file
    if request.method != "POST" or not request.FILES.get("file"):
        return JsonResponse({"error": "Invalid request. Please upload an image file.", "status": "invalid_request"}, status=400)

    try:
        image_file = request.FILES["file"]
        # Edge case validations
        validate_file_size(image_file.size)
        # Try to validate file extension first, with fallback to content validation
        try:
            validate_file_extension(image_file.name)
        except ValueError as e:
            if "No file extension found" in str(e) or "Empty filename" in str(e):
                validate_file_content_type(image_file)
            else:
                raise e
        # Try to open and validate image
        try:
            pil_image = Image.open(image_file).convert('RGB')
        except Exception as e:
            raise ValueError(f"Invalid image file: {str(e)}")
        validate_image_content(pil_image)
        # Get model and make prediction
        model = get_model()
        predicted_class_idx, confidence = predict_image(pil_image, model)
        # Validate prediction index
        if predicted_class_idx < 0 or predicted_class_idx >= len(CLASS_NAMES):
            raise ValueError(f"Predicted class index {predicted_class_idx} out of range for CLASS_NAMES of length {len(CLASS_NAMES)}")
        hall_id = CLASS_NAMES[predicted_class_idx]
        # Validate confidence threshold
        validate_confidence(confidence, hall_id)
        # Get schedule data
        try:
            hall = Hall.objects.get(name=hall_id)
            schedules = Schedule.objects.filter(hall=hall).order_by('start_time')
            schedule_str = "; ".join([
                f"{s.start_time.strftime('%H:%M')}-{s.end_time.strftime('%H:%M')} {s.course_name}"
                for s in schedules
            ]) or "No schedule found"
        except Hall.DoesNotExist:
            schedule_str = "No schedule found"
        return JsonResponse({
            "hall_id": hall_id,
            "confidence": round(confidence, 4),
            "schedule": schedule_str,
            "status": "success"
        })
    except ValueError as e:
        # User-friendly validation errors
        return JsonResponse({"error": str(e), "status": "validation_error"}, status=400)
    except Exception as e:
        # System errors
        return JsonResponse({"error": f"Recognition processing failed: {str(e)}", "status": "system_error"}, status=500)
