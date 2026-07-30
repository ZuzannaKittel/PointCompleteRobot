import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import trimesh
from torch.utils.data import DataLoader, Dataset, random_split

from configs.run_config import get_run_dir, get_run_name
from models.implicit_network import ImplicitDecoderBasic, MultiModalFeatureEncoder
from models.dinocomplete import DinoCompleteBaselineEncoder
from utils.evaluation import evaluate_test_set
from utils.implicit_dataset import ScanNetppImplicitDataset
from utils.training import train_model

RUN_NAME = get_run_name()
RUN_DIR = get_run_dir()


# ==========================================
# ENGINE RUNTIME PIPELINE
# ==========================================
if __name__ == "__main__":
    print(f"🚀 Initializing experiment: {RUN_NAME}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================================
    # TRAINING MODE
    # ==========================================================
    
    # TODO: Select the ARCHITECTURE
    #ARCHITECTURE = "dinocomplete"
    ARCHITECTURE = "ours"

    MODE = "scannet_train"

    DATA_DIR = "data/pairs_scannet/train/*.pt"

    LR = 1e-3
    WD = 1e-4

    RUN_NAME = "scannet_train"

    if ARCHITECTURE == "ours":
        RUN_NAME += "_ours"
    else:
        RUN_NAME += "_dinocomplete"

    RUN_DIR = os.path.join("runs", RUN_NAME)

    # Determine which checkpoint (if any) should be loaded.
    resume_ckpt = os.path.join(
        RUN_DIR,
        "checkpoints",
        "latest_model.pth",
    )

    LOAD_MODEL = resume_ckpt if os.path.exists(resume_ckpt) else None

    if LOAD_MODEL is None:
        print("No previous checkpoint found. Starting from scratch.")
    else:
        print(f"Resuming from {LOAD_MODEL}")

    print(f"Training mode : {MODE}")
    print(f"Architecture  : {ARCHITECTURE}")
    print(f"Checkpoint    : {LOAD_MODEL}")
    print(f"Dataset       : {DATA_DIR}")
    print(f"Learning rate : {LR}")

    data_files = glob.glob(DATA_DIR)
    if not data_files:
        raise FileNotFoundError(f"Missing data packages inside {data_files}.")
    print(f"📂 Discovered {len(data_files)} partial-complete training samples.")
    torch.manual_seed(42)
    train_size = int(0.8 * len(data_files))
    val_size = int(0.1 * len(data_files))
    test_size = len(data_files) - train_size - val_size

    train_files, val_files, test_files = random_split(
        data_files,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset = ScanNetppImplicitDataset(train_files)
    val_dataset = ScanNetppImplicitDataset(val_files)
    test_dataset = ScanNetppImplicitDataset(test_files)

    print(
        f"📊 Train: {len(train_dataset)} | "
        f"Val: {len(val_dataset)} | "
        f"Test: {len(test_dataset)}"
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    sample_data = torch.load(data_files[0], map_location="cpu", weights_only=False)
    detected_dim = sample_data['partial_feats'].shape[-1]
    print(f"🧬 Automatically detected feature dimension from factory output: {detected_dim}")

    if detected_dim not in (1024, 1408):
        raise ValueError(
            f"Unsupported feature dimension {detected_dim}. Expected 1024 or 1408."
        )

    INPUT_DIM = detected_dim
    print(f"Using feature dimension: {INPUT_DIM}")

    # Encoder selection based on the architecture
    if ARCHITECTURE == "ours":
        encoder = MultiModalFeatureEncoder(
            input_feat_dim=INPUT_DIM,
            latent_dim=512,
        ).to(device)

    elif ARCHITECTURE == "dinocomplete":
        encoder = DinoCompleteBaselineEncoder().to(device)

    decoder = ImplicitDecoderBasic(
        latent_dim=1024,
        hidden_dim=256
    ).to(device)

    # Calculate total number of parameters in the model
    num_params = (
        sum(p.numel() for p in encoder.parameters())
        +
        sum(p.numel() for p in decoder.parameters())
    )

    print(f"Total parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) +
        list(decoder.parameters()),
        lr=LR,
        weight_decay=WD,
    )
    criterion = nn.BCEWithLogitsLoss()

    # ==========================================
    # RESUME FROM LATEST CHECKPOINT (IF EXISTS)
    # ==========================================
    start_epoch = 0

    if LOAD_MODEL is not None:

        print(f"\nLoading weights from:\n{LOAD_MODEL}")

        checkpoint = torch.load(
            LOAD_MODEL,
            map_location=device,
        )

        state = checkpoint["encoder_state_dict"]

        if ARCHITECTURE == "ours":
            # Rename ShapeNet projection to the new Utonia branch
            renamed_state = {}
            for k, v in state.items():
                if k.startswith("input_projection"):
                    new_key = k.replace(
                        "input_projection",
                        "utonia_projection"
                    )
                else:
                    new_key = k

                renamed_state[new_key] = v

            state = renamed_state

        current = encoder.state_dict()
        compatible = {}
        loaded = []
        skipped = []

        for k, v in state.items():
            if k in current and current[k].shape == v.shape:
                compatible[k] = v
                loaded.append(k)
            else:
                skipped.append(k)

        print("\nLoaded:")
        for k in loaded:
            print("  ", k)

        print("\nSkipped:")
        for k in skipped:
            print("  ", k)

        current.update(compatible)

        encoder.load_state_dict(current)

        # Verify how many weight tensors were loaded successfully
        loaded = len(compatible)
        total = len(current)

        print(f"Loaded {loaded}/{total} encoder tensors")

        decoder.load_state_dict(
            checkpoint["decoder_state_dict"]
        )

        # Resume optimizer and epoch when continuing the same ScanNet run
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = checkpoint["epoch"]

        print(f"Resuming from epoch {start_epoch}")

    os.makedirs(f"{RUN_DIR}/snapshots", exist_ok=True)
    os.makedirs(f"{RUN_DIR}/checkpoints", exist_ok=True)

    epochs = 150
    
    # ==========================================
    # TRAINING CONFIGURATION
    # ==========================================
    with open(f"{RUN_DIR}/model_summary.txt", "w") as f:
        f.write(f"Architecture: {ARCHITECTURE}\n")

        if ARCHITECTURE == "ours":
            f.write(f"Input feature dimension: {detected_dim}\n")
            f.write("Input projection: input_dim -> 1024\n")
            f.write("Encoder latent: 512\n")
            f.write("Global latent (max+mean): 1024\n")
            f.write("Decoder latent: 1024\n")
            f.write("Hidden dim: 256\n")
            f.write("Batch size: 32\n")
            f.write(f"Epochs: {epochs}\n")
            f.write("Optimizer: AdamW\n")
            f.write(f"Learning rate: {LR}\n")
            f.write(f"Parameters: {num_params:,}\n")
        else:
            f.write("Geometry encoder: XYZ\n")
            f.write("Semantic encoder: DINO\n")
            f.write("Context: Shared Transformer\n")
            f.write("Pooling: Attention\n")
            f.write(f"Epochs: {epochs}\n")
            f.write("Optimizer: AdamW\n")
            f.write(f"Learning rate: {LR}\n")
            f.write(f"Parameters: {num_params:,}\n")

    first_batch = next(iter(train_loader))

    print(first_batch['partial_feats'].shape)

    print("Feature dim:", detected_dim)
    print("Encoder input:", INPUT_DIM)

    print("🔥 Starting training...")

    train_model(
        encoder,
        decoder,
        train_loader,
        val_loader,
        val_dataset,
        optimizer,
        criterion,
        device,
        epochs,
        start_epoch=start_epoch,
        RUN_NAME=RUN_NAME,
        RUN_DIR=RUN_DIR,
        resume_training=(LOAD_MODEL is not None),
    )

    print("\n🏁 Framework routine finished. Run your evaluation snapshots through CloudCompare to see the improvements.")

    # Load best model for evaluation
    best_ckpt = torch.load(
        os.path.join(
            RUN_DIR,
            "checkpoints",
            "best_model.pth"
        ),
        map_location=device
    )

    encoder.load_state_dict(best_ckpt["encoder_state_dict"])
    decoder.load_state_dict(best_ckpt["decoder_state_dict"])

    # --- FINAL TEST SET EVALUATION ---
    evaluate_test_set(encoder, decoder, test_dataset, device, RUN_DIR=RUN_DIR)