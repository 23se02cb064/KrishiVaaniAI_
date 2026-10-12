import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import base64
from torchvision import transforms

# Use the SAME preprocessing as the main inference pipeline (Resize 256 -> CenterCrop 224)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []

        self.handles.append(target_layer.register_forward_hook(self.save_activation))
        self.handles.append(target_layer.register_full_backward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __call__(self, x, class_idx=None):
        self.model.eval()

        # Forward pass
        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        if gradients is None or activations is None:
            self.remove_hooks()
            raise RuntimeError("Gradients or activations not captured. Check target layer.")

        # Compute Grad-CAM
        weights = gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * activations).sum(dim=1)
        cam = torch.clamp(cam, min=0)

        # Normalize
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min < 1e-8:
            # Nearly flat activation - return uniform low attention
            cam = torch.zeros_like(cam)
        else:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        self.remove_hooks()
        return cam.squeeze().cpu().numpy(), class_idx


def find_target_layer(model):
    """
    Find the correct target layer for MobileNetV2.
    For MobileNetV2, the last convolutional layer is in model.features[-1]
    which is a Conv2d layer with 1280 output channels.
    """
    # For MobileNetV2, the last conv layer is the last layer in features
    # It's a Sequential containing Conv2d + BatchNorm + ReLU6
    last_conv = None

    # First try: model.features[-1] (last block in features)
    if hasattr(model, 'features'):
        for module in reversed(list(model.features.modules())):
            if isinstance(module, nn.Conv2d):
                last_conv = module
                break

    # Fallback: search all modules for the last Conv2d
    if last_conv is None:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module

    if last_conv is None:
        raise ValueError("No Conv2d layer found in model")

    print(f"Grad-CAM target layer: {last_conv}")
    return last_conv


def jet_colormap(cam):
    """Generate a jet colormap heatmap."""
    heatmap = np.zeros((cam.shape[0], cam.shape[1], 3), dtype=np.float32)
    heatmap[:, :, 0] = np.clip(1.5 - np.abs(4 * cam - 3), 0, 1)
    heatmap[:, :, 1] = np.clip(1.5 - np.abs(4 * cam - 2), 0, 1)
    heatmap[:, :, 2] = np.clip(1.5 - np.abs(4 * cam - 1), 0, 1)
    return heatmap


def generate_gradcam_overlay(model, image_bytes, target_layer=None):
    """
    Generate Grad-CAM overlay for the predicted class.

    Returns:
        overlay_b64: Base64 encoded overlay image (original + heatmap)
        heatmap_b64: Base64 encoded heatmap only
        class_idx: Predicted class index
        error: Error message if any, None otherwise
    """
    error = None

    try:
        if target_layer is None:
            target_layer = find_target_layer(model)

        gradcam = GradCAM(model, target_layer)

        # Load and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        original_size = image.size

        # Use same preprocessing as inference
        img_resized = image.resize((224, 224), Image.Resampling.LANCZOS)
        device = next(model.parameters()).device
        img_tensor = preprocess(img_resized).unsqueeze(0).to(device)

        # Generate CAM
        cam, class_idx = gradcam(img_tensor)

        # Resize CAM to original image size
        cam_img = Image.fromarray(np.uint8(255 * cam))
        cam_img = cam_img.resize(original_size, Image.Resampling.LANCZOS)
        cam_resized = np.array(cam_img) / 255.0

        # Generate heatmap
        heatmap = jet_colormap(cam_resized)

        # Prepare original image for overlay
        original = np.array(image.resize(original_size, Image.Resampling.LANCZOS)) / 255.0

        # Alpha blending: heatmap * alpha + original * (1 - alpha)
        alpha = 0.4
        overlay = (heatmap * alpha + original * (1 - alpha))
        overlay = np.clip(overlay, 0, 1)
        overlay = np.uint8(255 * overlay)

        # Convert to base64
        overlay_img = Image.fromarray(overlay)
        buffered = io.BytesIO()
        overlay_img.save(buffered, format='PNG')
        overlay_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
        heatmap_buffered = io.BytesIO()
        heatmap_img.save(heatmap_buffered, format='PNG')
        heatmap_b64 = base64.b64encode(heatmap_buffered.getvalue()).decode('utf-8')

        return overlay_b64, heatmap_b64, class_idx, error

    except Exception as e:
        error = f"Grad-CAM generation failed: {str(e)}"
        print(f"Grad-CAM Error: {error}")
        # Return placeholder empty images
        placeholder = Image.new('RGB', (224, 224), color='gray')
        buffered = io.BytesIO()
        placeholder.save(buffered, format='PNG')
        placeholder_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return placeholder_b64, placeholder_b64, -1, error