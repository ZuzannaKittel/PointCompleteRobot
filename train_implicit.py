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

    MODE = "scannet_finetune"
    # MODE = "shapenet_pretrain"
    # MODE = "shapenet_resume"

    if MODE == "shapenet_pretrain":
        DATA_DIR = "data/pairs_shapenet/train/*.pt"
        INPUT_DIM = 1024
        LR = 5e-4
        LOAD_MODEL = None
        WD = 1e-4
        RUN_NAME = "shapenet_pretraining"
    elif MODE == "shapenet_resume":
        DATA_DIR = "data/pairs_shapenet/train/*.pt"
        INPUT_DIM = 1024
        LR = 5e-4
        LOAD_MODEL = "latest"
        WD = 1e-4
        RUN_NAME = "shapenet_pretraining"
    elif MODE == "scannet_finetune":
        DATA_DIR = "data/geometric_pairs_dataset2/train/*.pt"
        INPUT_DIM = 1408
        LR = 1e-4
        WD = 5e-5
        RUN_NAME = "scannet_finetuning"
        LOAD_MODEL = "runs/shapenet_pretraining/checkpoints/best_model.pth"

    RUN_DIR = os.path.join("runs", RUN_NAME)

    print(f"Training mode : {MODE}")
    print(f"Dataset       : {DATA_DIR}")
    print(f"Learning rate : {LR}")
    print(f"Input dim     : {INPUT_DIM}")

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

    # Assert that the detected feature dimension matches the expected input dimension
    assert detected_dim == INPUT_DIM, (
        f"Dataset features ({detected_dim}) "
        f"do not match encoder input ({INPUT_DIM})"
    )

    encoder = MultiModalFeatureEncoder(
        input_feat_dim=INPUT_DIM,
        latent_dim=512,
    ).to(device)

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

        if LOAD_MODEL == "latest":

            checkpoint_dir = f"{RUN_DIR}/checkpoints"

            checkpoint_files = glob.glob(
                os.path.join(checkpoint_dir, "epoch_*.pth")
            )

            if checkpoint_files:

                LOAD_MODEL = max(
                    checkpoint_files,
                    key=lambda x: int(
                        re.search(r"epoch_(\d+)", x).group(1)
                    )
                )

        print(f"\nLoading weights from:\n{LOAD_MODEL}")

        checkpoint = torch.load(
            LOAD_MODEL,
            map_location=device,
        )

        state = checkpoint["encoder_state_dict"]

        # Rename ShapeNet projection to the new Utonia branch
        renamed_state = {}

        for k, v in state.items():
            if k.startswith("input_projection"):
                new_key = k.replace("input_projection", "utonia_projection")
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

        # Verify how many weighttensors were loaded successfully
        loaded = len(compatible)
        total = len(current)

        print(f"Loaded {loaded}/{total} encoder tensors")

        decoder.load_state_dict(
            checkpoint["decoder_state_dict"]
        )

        # only resume optimizer if continuing SAME experiment
        if MODE == "shapenet_resume":

            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

            start_epoch = checkpoint["epoch"]

    os.makedirs(f"{RUN_DIR}/snapshots", exist_ok=True)
    os.makedirs(f"{RUN_DIR}/checkpoints", exist_ok=True)

    if MODE == "scannet_finetune":
        epochs = 60
    else:
        epochs = 100
    
    # ==========================================
    # TRAINING CONFIGURATION
    # ==========================================
    with open(f"{RUN_DIR}/model_summary.txt", "w") as f:
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
        resume_training=(MODE == "shapenet_resume"),
    )

    print("\n🏁 Framework routine finished. Run your evaluation snapshots through CloudCompare to see the improvements.")

    # --- FINAL TEST SET EVALUATION ---
    evaluate_test_set(encoder, decoder, test_dataset, device, RUN_DIR=RUN_DIR)