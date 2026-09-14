import os
import numpy as np
import torch
import sys
from scipy.spatial import cKDTree

from configs.run_config import get_run_dir
import trimesh
import matplotlib.pyplot as plt
import pandas as pd
import time

from utils.evaluation import evaluate_test_set, extract_implicit_shape
from utils.implicit_dataset2 import ScanNetppImplicitDataset
from models.encoders import AblationEncoder
from models.decoders import ImplicitDecoderBasic
from configs.ablation_decoders import (
    DECODER_ABLATIONS,
    DECODER_ABLATION_DESCRIPTIONS,
)
from configs.ablation_encoders import (
    ENCODER_ABLATIONS,
    ENCODER_ABLATION_DESCRIPTIONS,
)

# Run the evaluation if this script is executed directly
# must match the model 
# and dataset used for training and testing
if __name__ == "__main__":
    import glob
    from torch.utils.data import random_split

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ENCODER_ABLATION = "coord_dino"
    # ENCODER_ABLATION = "utonia_dino"
    ENCODER_ABLATION = "utonia"

    # DECODER_ABLATION = "baseline"
    # DECODER_ABLATION = "fourier"
    DECODER_ABLATION = "fourier_residual"

    if ENCODER_ABLATION not in ENCODER_ABLATIONS:
        raise ValueError(
            f"Unknown encoder ablation '{ENCODER_ABLATION}'. "
            f"Available: {list(ENCODER_ABLATIONS.keys())}"
        )
    if DECODER_ABLATION not in DECODER_ABLATIONS:
        raise ValueError(
            f"Unknown decoder ablation '{DECODER_ABLATION}'. "
            f"Available: {list(DECODER_ABLATIONS.keys())}"
        )
    encoder_config = ENCODER_ABLATIONS[ENCODER_ABLATION]
    decoder_config = DECODER_ABLATIONS[DECODER_ABLATION]
                                       
    import json

    with open(
        "data/pairs_scannet/test_split.json",
        "r"
    ) as f:
        test_files = json.load(f)

    print(
        f"Loaded fixed test set: {len(test_files)} samples"
    )

    test_dataset = ScanNetppImplicitDataset(
        test_files,
        num_input_pts=2048,
        num_queries=8192,
    )

    encoder = AblationEncoder(
        encoder_type=encoder_config["type"],
    ).to(device)

    decoder = ImplicitDecoderBasic(
        latent_dim=1024,
        **decoder_config,
    ).to(device)
    
    checkpoint_path = (
        "runs/scannet_finetune_ours_utonia_fourier_residual/"
        "checkpoints/best_model.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    encoder.eval()
    decoder.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Device: {device}")

    evaluate_test_set(
        encoder=encoder,
        decoder=decoder,
        test_dataset=test_dataset,
        device=device,
        RUN_DIR="runs/scannet_finetune_ours_utonia_fourier_residual/",
    )