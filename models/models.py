import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor
from utils.utonia_backbone import UtoniaBackbone

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_sam(ckpt):
    print("🎯 Initializing SAM...")
    sam = sam_model_registry["vit_h"](checkpoint=ckpt).to(DEVICE)
    return SamPredictor(sam)

def init_dino():
    print("🦖 Initializing DINOv2...")
    return torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEVICE).eval()

def init_utonia(ckpt_path):
    print("🧠 Initializing Utonia...")
    backbone = UtoniaBackbone(use_mock=False).to(DEVICE).eval()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    backbone.load_state_dict(
        {k.replace('backbone.', ''): v for k, v in sd.items() if k.startswith('backbone.')},
        strict=False
    )
    return backbone