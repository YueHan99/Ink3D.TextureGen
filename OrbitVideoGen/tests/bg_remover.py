"""Background removal using RMBG-2.0, output RGB with black background."""

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


_model = None

import os as _os
_LOCAL = os.environ.get("RMBG_PATH", "./RMBG-2.0")
_REMOTE = os.environ.get("RMBG_PATH", "./RMBG-2.0")
MODEL_PATH = _REMOTE if _os.path.exists(_REMOTE) else _LOCAL
IMAGE_SIZE = (1024, 1024)

_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def get_model():
    global _model
    if _model is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _model = AutoModelForImageSegmentation.from_pretrained(
            MODEL_PATH, trust_remote_code=True
        ).eval().to(device)
    return _model


def remove_background(image: Image.Image) -> Image.Image:
    """Remove background and return RGBA image."""
    model = get_model()
    device = next(model.parameters()).device

    orig_size = image.size
    input_tensor = _transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    mask = transforms.ToPILImage()(pred).resize(orig_size)

    image = image.convert("RGBA")
    image.putalpha(mask)
    return image


def rgba_to_rgb_black(rgba_image: Image.Image) -> Image.Image:
    """Convert RGBA to RGB with black background (matching training conditions)."""
    if rgba_image.mode == 'RGBA':
        bg = Image.new('RGB', rgba_image.size, (0, 0, 0))
        bg.paste(rgba_image.convert('RGB'), mask=rgba_image.split()[-1])
        return bg
    return rgba_image.convert('RGB')
