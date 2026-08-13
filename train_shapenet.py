import glob
import os
import re

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models.implicit_network import (
    MultiModalFeatureEncoder,
    ImplicitDecoderBasic,
)

from configs.run_config import get_run_name, get_run_dir

from utils.training import train_model
from utils.implicit_dataset import ScanNetppImplicitDataset
from utils.evaluation import evaluate_test_set


if __name__ == "__main__":

    # ============================================================
    # DEVICE
    # ============================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ============================================================
    # EXPERIMENT
    # ============================================================

    RUN_NAME = get_run_name()
    RUN_DIR = get_run_dir()

    print(
        f"🚀 Initializing ShapeNet pretraining: "
        f"{RUN_NAME}..."
    )

    # ============================================================
    # DATA
    # ============================================================

    DATA_DIR = "data/pairs_shapenet/train/*.pt"

    data_files = glob.glob(DATA_DIR)

    if not data_files:
        raise FileNotFoundError(
            f"Missing data packages inside {DATA_DIR}."
        )

    print(
        f"📂 Discovered {len(data_files)} "
        f"ShapeNet training samples."
    )

    torch.manual_seed(42)

    train_size = int(
        0.8 * len(data_files)
    )

    val_size = int(
        0.1 * len(data_files)
    )

    test_size = (
        len(data_files)
        - train_size
        - val_size
    )

    train_files, val_files, test_files = random_split(
        data_files,
        [
            train_size,
            val_size,
            test_size,
        ],
        generator=torch.Generator().manual_seed(42),
    )

    train_dataset = ScanNetppImplicitDataset(
        train_files
    )

    val_dataset = ScanNetppImplicitDataset(
        val_files
    )

    test_dataset = ScanNetppImplicitDataset(
        test_files
    )

    print(
        f"📊 Train: {len(train_dataset)} | "
        f"Val: {len(val_dataset)} | "
        f"Test: {len(test_dataset)}"
    )

    # ============================================================
    # DATA LOADERS
    # ============================================================

    BATCH_SIZE = 32

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    # ============================================================
    # FEATURE DIMENSION
    # ============================================================

    sample_data = torch.load(
        data_files[0],
        map_location="cpu",
        weights_only=False,
    )

    detected_dim = (
        sample_data["partial_feats"]
        .shape[-1]
    )

    print(
        f"\n🧬 Detected feature dimension: "
        f"{detected_dim}"
    )

    if detected_dim != 1024:
        raise ValueError(
            "ShapeNet pretraining expects "
            f"1024-D features, but found "
            f"{detected_dim}."
        )

    # ============================================================
    # MODEL
    # ============================================================

    encoder = MultiModalFeatureEncoder(
        input_feat_dim=1024,
        latent_dim=512,
    ).to(device)

    decoder = ImplicitDecoderBasic(
        latent_dim=1024,
        hidden_dim=256,
    ).to(device)

    # ============================================================
    # MODEL INFORMATION
    # ============================================================

    num_params = (
        sum(
            p.numel()
            for p in encoder.parameters()
        )
        +
        sum(
            p.numel()
            for p in decoder.parameters()
        )
    )

    print(
        f"Total parameters: "
        f"{num_params:,}"
    )

    # ============================================================
    # OPTIMIZER
    # ============================================================

    LR = 5e-4
    WD = 1e-4

    optimizer = torch.optim.AdamW(
        list(encoder.parameters())
        +
        list(decoder.parameters()),
        lr=LR,
        weight_decay=WD,
    )

    criterion = nn.BCEWithLogitsLoss()

    # ============================================================
    # DIRECTORIES
    # ============================================================

    os.makedirs(
        f"{RUN_DIR}/snapshots",
        exist_ok=True,
    )

    os.makedirs(
        f"{RUN_DIR}/checkpoints",
        exist_ok=True,
    )

    # ============================================================
    # RESUME
    # ============================================================

    start_epoch = 0

    checkpoint_dir = (
        f"{RUN_DIR}/checkpoints"
    )

    checkpoint_files = glob.glob(
        os.path.join(
            checkpoint_dir,
            "epoch_*.pth",
        )
    )

    if checkpoint_files:

        latest_checkpoint = max(
            checkpoint_files,
            key=lambda x: int(
                re.search(
                    r"epoch_(\d+)",
                    x,
                ).group(1)
            ),
        )

        print(
            f"\n📂 Loading checkpoint:\n"
            f"{latest_checkpoint}"
        )

        checkpoint = torch.load(
            latest_checkpoint,
            map_location=device,
            weights_only=False,
        )

        encoder.load_state_dict(
            checkpoint[
                "encoder_state_dict"
            ]
        )

        decoder.load_state_dict(
            checkpoint[
                "decoder_state_dict"
            ]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        start_epoch = checkpoint[
            "epoch"
        ]

        print(
            f"✅ Resuming from epoch "
            f"{start_epoch}"
        )

    else:

        print(
            "\n🆕 No checkpoint found."
            "\nStarting ShapeNet pretraining "
            "from scratch."
        )

    # ============================================================
    # TRAINING CONFIGURATION
    # ============================================================

    EPOCHS = 150

    with open(
        f"{RUN_DIR}/model_summary.txt",
        "w",
    ) as f:

        f.write(
            "Dataset: ShapeNet\n"
        )

        f.write(
            "Training purpose: "
            "ShapeNet pretraining\n"
        )

        f.write(
            f"Input feature dimension: "
            f"{detected_dim}\n"
        )

        f.write(
            "Input projection: "
            "1024 -> 1024\n"
        )

        f.write(
            "Encoder latent: 512\n"
        )

        f.write(
            "Global latent: "
            "max + mean -> 1024\n"
        )

        f.write(
            "Decoder latent: 1024\n"
        )

        f.write(
            "Decoder hidden dim: 256\n"
        )

        f.write(
            f"Batch size: "
            f"{BATCH_SIZE}\n"
        )

        f.write(
            f"Epochs: "
            f"{EPOCHS}\n"
        )

        f.write(
            "Optimizer: AdamW\n"
        )

        f.write(
            f"Learning rate: "
            f"{LR}\n"
        )

        f.write(
            f"Weight decay: "
            f"{WD}\n"
        )

        f.write(
            f"Parameters: "
            f"{num_params:,}\n"
        )

    # ============================================================
    # FIRST BATCH CHECK
    # ============================================================

    first_batch = next(
        iter(train_loader)
    )

    print(
        "\nFirst training batch:"
    )

    print(
        "partial_pts:",
        first_batch["partial_pts"].shape,
    )

    print(
        "partial_feats:",
        first_batch["partial_feats"].shape,
    )

    print(
        "query_coords:",
        first_batch["query_coords"].shape,
    )

    print(
        "target_surface:",
        first_batch["target_surface"].shape,
    )

    print(
        "positive ratio:",
        first_batch[
            "target_surface"
        ].float().mean().item(),
    )

    print(
        "Feature dimension:",
        detected_dim,
    )

    # ============================================================
    # TRAIN
    # ============================================================

    print(
        "\n🔥 Starting ShapeNet pretraining..."
    )

    train_model(
        encoder,
        decoder,
        train_loader,
        val_loader,
        val_dataset,
        optimizer,
        criterion,
        device,
        EPOCHS,
        start_epoch=start_epoch,
        RUN_NAME=RUN_NAME,
        RUN_DIR=RUN_DIR,
    )

    print(
        "\n🏁 ShapeNet pretraining finished."
    )

    # ============================================================
    # LOAD BEST MODEL
    # ============================================================

    best_ckpt_path = os.path.join(
        RUN_DIR,
        "checkpoints",
        "best_model.pth",
    )

    if os.path.exists(
        best_ckpt_path
    ):

        best_ckpt = torch.load(
            best_ckpt_path,
            map_location=device,
            weights_only=False,
        )

        encoder.load_state_dict(
            best_ckpt[
                "encoder_state_dict"
            ]
        )

        decoder.load_state_dict(
            best_ckpt[
                "decoder_state_dict"
            ]
        )

        print(
            "\n✅ Best ShapeNet model loaded."
        )

    else:

        print(
            "\n⚠️ best_model.pth not found."
            "\nUsing final model weights."
        )

    # ============================================================
    # FINAL TEST EVALUATION
    # ============================================================

    print(
        "\n📊 Evaluating ShapeNet test set..."
    )

    evaluate_test_set(
        encoder,
        decoder,
        test_dataset,
        device,
        RUN_DIR=RUN_DIR,
    )

    print(
        "\n✅ ShapeNet pretraining experiment "
        "completed."
    )