from __future__ import annotations
from PIL import Image
from torchvision import transforms

IMAGE_SIZE=320
IMAGENET_MEAN=[0.485,0.456,0.406]
IMAGENET_STD=[0.229,0.224,0.225]

def build_inference_transform(image_size:int=IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=transforms.InterpolationMode.BILINEAR,antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD),
    ])

def prepare_image(image:Image.Image,transform=None):
    if transform is None:
        transform=build_inference_transform()
    if image.mode!="RGB":
        image=image.convert("RGB")
    return transform(image)
