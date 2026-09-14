import glob
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from configs.run_config import get_run_dir, get_run_name
from configs.ablation_decoders import (
    DECODER_ABLATIONS,
    DECODER_ABLATION_DESCRIPTIONS,
)
from configs.ablation_encoders import (
    ENCODER_ABLATIONS,
    ENCODER_ABLATION_DESCRIPTIONS,
)
from models.decoders import ImplicitDecoderBasic

from models.dinocomplete import (
    DinoCompleteBaselineEncoder,
    DinoCompleteProxyDecoder,
)

from models.encoders import AblationEncoder
from utils.evaluation import evaluate_test_set
from utils.implicit_dataset2 import ScanNetppImplicitDataset
from utils.training import train_model

from utils.evaluation import THRESHOLD

if __name__ == "__main__":


    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ============================================================
    # EXPERIMENT SELECTION
    # ============================================================

    # Keep DinoComplete as the fixed external baseline.
    # For our method, independently select encoder and decoder.
    ARCHITECTURE = "ours"
    # ARCHITECTURE = "dinocomplete"

    # ENCODER_ABLATION = "coord_dino"
    ENCODER_ABLATION = "utonia_dino"
    # ENCODER_ABLATION = "utonia"

    # DECODER_ABLATION = "baseline"
    # DECODER_ABLATION = "fourier"
    DECODER_ABLATION = "fourier_residual"

    MODE = "scannet_train"
    DATA_DIR = "data/pairs_scannet/train/*.pt"

    LR = 1e-4
    WD = 1e-4
    BATCH_SIZE = 32
    EPOCHS = 100

    # ============================================================
    # VALIDATE EXPERIMENT CONFIGURATION
    # ============================================================

    if ARCHITECTURE == "ours":
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

    else:
        encoder_config = None
        decoder_config = None

    # ============================================================
    # RUN DIRECTORY
    # ============================================================

    RUN_NAME = "scannet_train"

    if ARCHITECTURE == "ours":
        RUN_NAME += (
            f"_ours_{ENCODER_ABLATION}_{DECODER_ABLATION}"
        )
    else:
        RUN_NAME += "_dinocomplete"

    RUN_DIR = os.path.join(
        "runs",
        RUN_NAME,
    )

    print(f"🚀 Initializing experiment: {RUN_NAME}...")

    resume_ckpt = os.path.join(
        RUN_DIR,
        "checkpoints",
        "latest_model.pth",
    )

    LOAD_MODEL = (
        resume_ckpt
        if os.path.exists(resume_ckpt)
        else None
    )

    if LOAD_MODEL is None:
        print("No previous checkpoint found. Starting from scratch.")
    else:
        print(f"Resuming from {LOAD_MODEL}")

    # ============================================================
    # EXPERIMENT INFORMATION
    # ============================================================

    print(f"Training mode : {MODE}")
    print(f"Architecture  : {ARCHITECTURE}")

    if ARCHITECTURE == "ours":
        print(f"Encoder       : {ENCODER_ABLATION}")
        print(
            f"Encoder desc  : "
            f"{ENCODER_ABLATION_DESCRIPTIONS[ENCODER_ABLATION]}"
        )
        print(f"Decoder       : {DECODER_ABLATION}")
        print(
            f"Decoder desc  : "
            f"{DECODER_ABLATION_DESCRIPTIONS[DECODER_ABLATION]}"
        )
        print(f"Encoder config: {encoder_config}")
        print(f"Decoder config: {decoder_config}")

    print(f"Checkpoint    : {LOAD_MODEL}")
    print(f"Dataset       : {DATA_DIR}")
    print(f"Learning rate : {LR}")
    print(f"Weight decay  : {WD}")

    # ============================================================
    # DATASET
    # ============================================================

    data_files = sorted(glob.glob(DATA_DIR))

    if not data_files:
        raise FileNotFoundError(
            f"Missing data packages inside {DATA_DIR}."
        )

    print(
        f"📂 Discovered {len(data_files)} "
        f"partial-complete training samples."
    )

    torch.manual_seed(42)

    train_size = int(0.8 * len(data_files))
    val_size = int(0.1 * len(data_files))
    test_size = len(data_files) - train_size - val_size

    train_files, val_files, test_files = random_split(
        data_files,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_dataset = ScanNetppImplicitDataset(
        train_files,
        num_input_pts=2048,
        num_queries=8192,
        occupancy_threshold=0.01,
        boundary_sigma=0.005,
        #boundary_fraction=0.3,
        deterministic=False,
    )

    val_dataset = ScanNetppImplicitDataset(
        val_files,
        num_input_pts=2048,
        num_queries=8192,
        occupancy_threshold=0.01,
        boundary_sigma=0.005,
        #boundary_fraction=0.3,
        deterministic=True,
    )

    test_dataset = ScanNetppImplicitDataset(
        test_files,
        num_input_pts=2048,
        num_queries=8192,
        occupancy_threshold=0.01,
        boundary_sigma=0.005,
        #boundary_fraction=0.3,
        deterministic=True,
    )

    print("\nChecking training samples...")

    for i in range(5):
        sample = train_dataset[i]
        targets = sample["target_surface"]

        print(
            f"Sample {i}: "
            f"queries={sample['query_coords'].shape[0]} "
            f"positive_ratio={targets.float().mean().item():.4f}"
        )

    print(
        f"📊 Train: {len(train_dataset)} | "
        f"Val: {len(val_dataset)} | "
        f"Test: {len(test_dataset)}"
    )

    # ============================================================
    # DATA LOADERS
    # ============================================================

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

    detected_dim = sample_data["partial_feats"].shape[-1]

    print(
        "🧬 Automatically detected feature dimension: "
        f"{detected_dim}"
    )

    if detected_dim != 1408:
        raise ValueError(
            "The current ScanNet++ ablation encoders expect "
            f"1408-D features (1024 Utonia + 384 DINO), "
            f"got {detected_dim}."
        )

    INPUT_DIM = detected_dim

    # ============================================================
    # ENCODER
    # ============================================================

    if ARCHITECTURE == "ours":
        encoder = AblationEncoder(
            encoder_type=encoder_config["type"],
        ).to(device)

    elif ARCHITECTURE == "dinocomplete":
        encoder = DinoCompleteBaselineEncoder().to(device)

    # ============================================================
    # DECODER
    # ============================================================

    if ARCHITECTURE == "ours":
        decoder = ImplicitDecoderBasic(
            latent_dim=1024,
            **decoder_config,
        ).to(device)

    elif ARCHITECTURE == "dinocomplete":
        decoder = DinoCompleteProxyDecoder().to(device)

    # ============================================================
    # MODEL INFORMATION
    # ============================================================

    num_params = (
        sum(p.numel() for p in encoder.parameters())
        + sum(p.numel() for p in decoder.parameters())
    )

    print(f"Total parameters: {num_params:,}")

    # ============================================================
    # OPTIMIZER
    # ============================================================

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=LR,
        weight_decay=WD,
    )

    criterion = nn.BCEWithLogitsLoss()

    # ============================================================
    # RESUME FROM CHECKPOINT
    # ============================================================

    start_epoch = 0

    if LOAD_MODEL is not None:
        print(f"\nLoading weights from:\n{LOAD_MODEL}")

        checkpoint = torch.load(
            LOAD_MODEL,
            map_location=device,
        )

        encoder.load_state_dict(
            checkpoint["encoder_state_dict"]
        )

        decoder.load_state_dict(
            checkpoint["decoder_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = checkpoint["epoch"]

        print(
            f"Successfully restored encoder, decoder, "
            f"optimizer and epoch {start_epoch}."
        )

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
    # MODEL SUMMARY
    # ============================================================

    with open(
        f"{RUN_DIR}/model_summary.txt",
        "w",
    ) as f:
        f.write(
            f"Architecture: {ARCHITECTURE}\n"
        )

        if ARCHITECTURE == "ours":
            f.write(
                f"Encoder ablation: {ENCODER_ABLATION}\n"
            )
            f.write(
                "Encoder description: "
                f"{ENCODER_ABLATION_DESCRIPTIONS[ENCODER_ABLATION]}\n"
            )
            f.write(
                f"Decoder ablation: {DECODER_ABLATION}\n"
            )
            f.write(
                "Decoder description: "
                f"{DECODER_ABLATION_DESCRIPTIONS[DECODER_ABLATION]}\n"
            )
            f.write(
                f"Input feature dimension: {detected_dim}\n"
            )
            f.write(
                "Encoder output: 1024-D "
                "(mean + max pooling)\n"
            )

            f.write(
                f"Decoder hidden dim: "
                f"{decoder_config['hidden_dim']}\n"
            )

            f.write(
                "Fourier encoding: "
                f"{decoder_config.get('use_fourier', False)}\n"
            )

            f.write(
                "Residual decoder: "
                f"{decoder_config.get('use_residual', False)}\n"
            )

            if decoder_config.get("use_fourier", False):
                f.write(
                    "Fourier bands: "
                    f"{decoder_config.get('num_fourier_bands', 6)}\n"
                )

            if decoder_config.get("use_residual", False):
                f.write(
                    "Residual blocks: "
                    f"{decoder_config.get('num_residual_blocks', 2)}\n"
                )

        else:
            f.write(
                "Encoder: DinoCompleteBaselineEncoder\n"
            )
            f.write(
                "Decoder: DinoInspiredDecoder\n"
            )

        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write("Optimizer: AdamW\n")
        f.write(f"Learning rate: {LR}\n")
        f.write(f"Weight decay: {WD}\n")
        f.write(f"Parameters: {num_params:,}\n")

    # ============================================================
    # FIRST BATCH CHECK
    # ============================================================

    first_batch = next(iter(train_loader))

    print("\nFirst training batch:")
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
        first_batch["target_surface"]
        .float()
        .mean()
        .item(),
    )

    print(
        "Feature dim:",
        detected_dim,
    )

    print(
        "Encoder:",
        ENCODER_ABLATION
        if ARCHITECTURE == "ours"
        else "dinocomplete",
    )

    print(
        "Decoder:",
        DECODER_ABLATION
        if ARCHITECTURE == "ours"
        else "dinocomplete",
    )

    # Build config dictionary for logging and reproducibility
    experiment_config = {
        "architecture": ARCHITECTURE,
        "encoder_ablation": (
            ENCODER_ABLATION
            if ARCHITECTURE == "ours"
            else "dinocomplete"
        ),
        "decoder_ablation": (
            DECODER_ABLATION
            if ARCHITECTURE == "ours"
            else "dinocomplete"
        ),
        "encoder_description": (
            ENCODER_ABLATION_DESCRIPTIONS[ENCODER_ABLATION]
            if ARCHITECTURE == "ours"
            else "Original DINOComplete encoder"
        ),
        "decoder_description": (
            DECODER_ABLATION_DESCRIPTIONS[DECODER_ABLATION]
            if ARCHITECTURE == "ours"
            else "Original DINOComplete decoder"
        ),

        # Dataset
        "num_input_points": 2048,
        "num_queries": 8192,
        "occupancy_distance_threshold": 0.01,
        "boundary_sigma": 0.005,
        "boundary_fraction": 0.3,

        # Training
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "weight_decay": WD,

        # Inference
        "reconstruction_probability_threshold": THRESHOLD,

        # Optimization
        "scheduler": "ReduceLROnPlateau",
        "scheduler_factor": 0.5,
        "scheduler_patience": 5,
        "scheduler_min_lr": 1e-6,
        "gradient_clip_norm": 1.0,
        "optimizer": "AdamW",
    }

    # ============================================================
    # TRAIN
    # ============================================================

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
        EPOCHS,
        start_epoch=start_epoch,
        RUN_NAME=RUN_NAME,
        RUN_DIR=RUN_DIR,
        resume_training=(LOAD_MODEL is not None),
        config=experiment_config,
    )

    print(
        "\n🏁 Framework routine finished. "
        "Run your evaluation snapshots through CloudCompare "
        "to inspect the reconstructions."
    )

    # ============================================================
    # LOAD BEST MODEL
    # ============================================================

    best_ckpt = torch.load(
        os.path.join(
            RUN_DIR,
            "checkpoints",
            "best_model.pth",
        ),
        map_location=device,
    )

    encoder.load_state_dict(
        best_ckpt["encoder_state_dict"]
    )

    decoder.load_state_dict(
        best_ckpt["decoder_state_dict"]
    )

    # ============================================================
    # FINAL TEST EVALUATION
    # ============================================================

    evaluate_test_set(
        encoder,
        decoder,
        test_dataset,
        device,
        RUN_DIR=RUN_DIR,
    )