import glob
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from configs.ablation_decoders import (
    DECODER_ABLATIONS,
    DECODER_ABLATION_DESCRIPTIONS,
)
from configs.ablation_encoders import (
    ENCODER_ABLATIONS,
    ENCODER_ABLATION_DESCRIPTIONS,
)

from models.decoders import ImplicitDecoderBasic
from models.encoders import AblationEncoder

from models.shapenet_pretrain import (
    ShapeNetBranchPretrainEncoder,
)

from models.dinocomplete import (
    DinoCompleteBaselineEncoder,
    DinoCompleteProxyDecoder,
)

from utils.evaluation import (
    evaluate_test_set,
    THRESHOLD,
)

from utils.implicit_dataset2 import (
    ScanNetppImplicitDataset,
)

from utils.training import train_model


def build_dataloaders(
    data_dir,
    num_input_pts=2048,
    num_queries=8192,
):
    data_files = glob.glob(data_dir)

    if not data_files:
        raise FileNotFoundError(
            f"Missing data packages inside {data_dir}."
        )

    print(
        f"\n📂 Discovered {len(data_files)} "
        f"training samples."
    )

    train_size = int(0.8 * len(data_files))
    val_size = int(0.1 * len(data_files))
    test_size = (
        len(data_files)
        - train_size
        - val_size
    )

    train_files, val_files, test_files = random_split(
        data_files,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_dataset = ScanNetppImplicitDataset(
        train_files,
        num_input_pts=num_input_pts,
        num_queries=num_queries,
        occupancy_threshold=0.01,
        boundary_sigma=0.005,
        deterministic=False,
    )

    val_dataset = ScanNetppImplicitDataset(
        val_files,
        num_input_pts=num_input_pts,
        num_queries=num_queries,
        occupancy_threshold=0.01,
        boundary_sigma=0.005,
        deterministic=True,
    )

    test_dataset = ScanNetppImplicitDataset(
        test_files,
        num_input_pts=num_input_pts,
        num_queries=num_queries,
        occupancy_threshold=0.01,
        boundary_sigma=0.005,
        deterministic=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )

    print(
        f"📊 Train: {len(train_dataset)} | "
        f"Val: {len(val_dataset)} | "
        f"Test: {len(test_dataset)}"
    )

    return (
        data_files,
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    )


def detect_feature_dim(data_files):
    sample_data = torch.load(
        data_files[0],
        map_location="cpu",
        weights_only=False,
    )

    detected_dim = (
        sample_data["partial_feats"].shape[-1]
    )

    print(
        "🧬 Automatically detected feature "
        f"dimension: {detected_dim}"
    )

    return detected_dim


def transfer_compatible_weights(
    source_model,
    target_model,
    model_name,
):
    """
    Transfer parameters when both name and tensor shape match.

    This is intentionally generic because the ShapeNet encoder
    and ScanNet encoder are different classes.
    """

    source_state = source_model.state_dict()
    target_state = target_model.state_dict()

    compatible = {}
    skipped = []

    for key, value in source_state.items():

        if (
            key in target_state
            and target_state[key].shape == value.shape
        ):
            compatible[key] = value
        else:
            skipped.append(key)

    target_state.update(compatible)
    target_model.load_state_dict(target_state)

    print(
        f"\n{model_name} transfer:"
    )

    print(
        f"  Loaded : {len(compatible)} tensors"
    )

    print(
        f"  Skipped: {len(skipped)} tensors"
    )

    if skipped:
        print(
            f"\nSkipped {model_name} parameters:"
        )

        for key in skipped:
            print(
                f"  {key}"
            )

    return len(compatible), len(skipped)

def transfer_shapenet_encoder(
    shapenet_encoder,
    scannet_encoder,
    encoder_type,
):
    print("\nShapeNet → ScanNet encoder transfer:")

    if encoder_type == "utonia_dino":
        scannet_encoder.utonia_input_norm.load_state_dict(
            shapenet_encoder.utonia_input_norm.state_dict()
        )
        scannet_encoder.utonia_encoder.load_state_dict(
            shapenet_encoder.utonia_encoder.state_dict()
        )
        scannet_encoder.utonia_context.load_state_dict(
            shapenet_encoder.utonia_context.state_dict()
        )

        print("  Utonia input norm      : transferred")
        print("  Utonia encoder         : transferred")
        print("  Utonia context         : transferred")
        print("  DINO branch            : random initialization")
        print("  Multimodal fusion      : random initialization")
        print("  Global projection      : random initialization")

    elif encoder_type == "coord_dino":
        scannet_encoder.geometry_encoder.load_state_dict(
            shapenet_encoder.geometry_encoder.state_dict()
        )
        scannet_encoder.geo_context.load_state_dict(
            shapenet_encoder.geo_context.state_dict()
        )

        print("  Geometry encoder       : transferred")
        print("  Geometry context       : transferred")
        print("  DINO branch            : random initialization")
        print("  Multimodal fusion      : random initialization")
        print("  Global projection      : random initialization")

    elif encoder_type == "utonia":
        scannet_encoder.utonia_input_norm.load_state_dict(
            shapenet_encoder.utonia_input_norm.state_dict()
        )

        scannet_encoder.utonia_encoder.load_state_dict(
            shapenet_encoder.utonia_encoder.state_dict()
        )

        scannet_encoder.context.load_state_dict(
            shapenet_encoder.utonia_context.state_dict()
        )

        # For the Utonia-only model, the post-branch fusion is also
        # dimensionally identical to the ShapeNet branch projection.
        scannet_encoder.fusion.load_state_dict(
            shapenet_encoder.branch_projection.state_dict()
        )

        scannet_encoder.global_projection.load_state_dict(
            shapenet_encoder.global_projection.state_dict()
        )

        print("  Utonia input norm      : transferred")
        print("  Utonia encoder         : transferred")
        print("  Utonia context         : transferred")
        print("  Fusion                 : transferred")
        print("  Global projection      : transferred")

    else:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}"
        )


def build_scanet_model(
    architecture,
    encoder_config,
    decoder_config,
    device,
):
    if architecture == "ours":

        encoder = AblationEncoder(
            encoder_type=encoder_config["type"],
        ).to(device)

        decoder = ImplicitDecoderBasic(
            latent_dim=1024,
            **decoder_config,
        ).to(device)

    elif architecture == "dinocomplete":

        encoder = DinoCompleteBaselineEncoder().to(
            device
        )

        decoder = DinoCompleteProxyDecoder().to(device)

    else:
        raise ValueError(
            f"Unknown architecture: {architecture}"
        )

    return encoder, decoder


def build_shapenet_model(
    pretrain_mode,
    decoder_config,
    device,
):
    encoder = ShapeNetBranchPretrainEncoder(
        mode=pretrain_mode,
    ).to(device)

    decoder = ImplicitDecoderBasic(
        latent_dim=1024,
        **decoder_config,
    ).to(device)

    return encoder, decoder


def train_shapenet_stage(
    device,
    pretrain_mode,
    decoder_config,
    run_name,
    run_dir,
    epochs,
):
    print(
        "\n"
        "============================================================"
    )
    print("STAGE 1: SHAPENET PRETRAINING")
    print(
        "============================================================"
    )

    (
        data_files,
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = build_dataloaders(
        "data/pairs_shapenet/train/*.pt"
    )

    detected_dim = detect_feature_dim(
        data_files
    )

    if detected_dim != 1024:
        raise ValueError(
            "ShapeNet pretraining expects 1024-D "
            f"Utonia features, got {detected_dim}."
        )

    encoder, decoder = build_shapenet_model(
        pretrain_mode=pretrain_mode,
        decoder_config=decoder_config,
        device=device,
    )

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
        f"ShapeNet model parameters: "
        f"{num_params:,}"
    )

    optimizer = torch.optim.AdamW(
        list(encoder.parameters())
        +
        list(decoder.parameters()),
        lr=5e-4,
        weight_decay=1e-4,
    )

    criterion = nn.BCEWithLogitsLoss()

    os.makedirs(
        f"{run_dir}/snapshots",
        exist_ok=True,
    )

    os.makedirs(
        f"{run_dir}/checkpoints",
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Resume ShapeNet pretraining if a checkpoint exists
    # ------------------------------------------------------------

    start_epoch = 0

    resume_ckpt = os.path.join(
        run_dir,
        "checkpoints",
        "latest_model.pth",
    )

    if os.path.exists(resume_ckpt):

        print(
            f"\n📂 Existing ShapeNet checkpoint found:"
            f"\n{resume_ckpt}"
        )

        checkpoint = torch.load(
            resume_ckpt,
            map_location=device,
            weights_only=False,
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
            f"✅ Resuming ShapeNet from "
            f"epoch {start_epoch}"
        )

    else:

        print(
            "\n🆕 No ShapeNet checkpoint found."
        )

        print(
            "Starting ShapeNet pretraining "
            "from scratch."
        )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    with open(
        f"{run_dir}/model_summary.txt",
        "w",
    ) as f:

        f.write("Dataset: ShapeNet\n")
        f.write("Training stage: ShapeNet branch pretraining\n")
        f.write(
            f"Pretraining mode: {pretrain_mode}\n"
        )
        f.write(
            "Input feature dimension: 1024\n"
        )

        if pretrain_mode == "utonia":
            f.write(
                "Pretrained branch: Utonia\n"
            )
        elif pretrain_mode == "coords":
            f.write(
                "Pretrained branch: Coordinates\n"
            )

        f.write(
            "Encoder output: 1024-D\n"
        )
        f.write(
            "Decoder: ImplicitDecoderBasic\n"
        )
        f.write(
            f"Decoder hidden dimension: "
            f"{decoder_config['hidden_dim']}\n"
        )
        f.write(
            f"Fourier encoding: "
            f"{decoder_config.get('use_fourier', False)}\n"
        )
        f.write(
            f"Residual decoder: "
            f"{decoder_config.get('use_residual', False)}\n"
        )
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write("Optimizer: AdamW\n")
        f.write("Learning rate: 5e-4\n")
        f.write("Weight decay: 1e-4\n")
        f.write(f"Parameters: {num_params:,}\n")

    first_batch = next(
        iter(train_loader)
    )

    print(
        "\nShapeNet first batch:"
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

    experiment_config = {
        "dataset": "shapenet",
        "training_stage": "pretraining",
        "input_feature_dim": detected_dim,
        "num_input_points": 2048,
        "num_queries": 8192,
        "occupancy_distance_threshold": 0.01,
        "boundary_sigma": 0.005,
        "batch_size": 32,
        "epochs": epochs,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "optimizer": "AdamW",
    }

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
        epochs,
        start_epoch=start_epoch,
        RUN_NAME=run_name,
        RUN_DIR=run_dir,
        resume_training=(
            start_epoch > 0
        ),
        config=experiment_config,
    )

    # ------------------------------------------------------------
    # Load BEST ShapeNet model
    # ------------------------------------------------------------

    best_path = os.path.join(
        run_dir,
        "checkpoints",
        "best_model.pth",
    )

    if not os.path.exists(best_path):
        raise FileNotFoundError(
            "ShapeNet training finished but "
            "best_model.pth was not found:\n"
            f"{best_path}"
        )

    best_ckpt = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )

    encoder.load_state_dict(
        best_ckpt["encoder_state_dict"]
    )

    decoder.load_state_dict(
        best_ckpt["decoder_state_dict"]
    )

    print(
        "\n✅ Best ShapeNet model loaded."
    )

    return (
        encoder,
        decoder,
        best_path,
    )

def load_shapenet_best_model(
    pretrain_mode,
    decoder_config,
    device,
    run_dir,
):
    print("\n============================================================")
    print("LOADING BEST SHAPENET PRETRAINED MODEL")
    print("============================================================")

    encoder, decoder = build_shapenet_model(
        pretrain_mode=pretrain_mode,
        decoder_config=decoder_config,
        device=device,
    )

    best_path = os.path.join(
        run_dir,
        "checkpoints",
        "best_model.pth",
    )

    if not os.path.exists(best_path):
        raise FileNotFoundError(
            f"ShapeNet best checkpoint not found:\n{best_path}"
        )

    checkpoint = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )

    encoder.load_state_dict(
        checkpoint["encoder_state_dict"]
    )

    decoder.load_state_dict(
        checkpoint["decoder_state_dict"]
    )

    print(f"✅ Loaded ShapeNet best model:\n{best_path}")

    return encoder, decoder, best_path

if __name__ == "__main__":

    # ============================================================
    # DEVICE
    # ============================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ============================================================
    # MAIN EXPERIMENT SWITCH
    # ============================================================

    # True:
    #   1. Pretrain on ShapeNet
    #   2. Automatically fine-tune on ScanNet
    #
    # False:
    #   Train ScanNet directly from scratch.

    USE_SHAPENET_PRETRAIN = True

    # ============================================================
    # ARCHITECTURE
    # ============================================================

    ARCHITECTURE = "ours"
    # ARCHITECTURE = "dinocomplete"

    # ENCODER_ABLATION = "coord_dino"
    ENCODER_ABLATION = "utonia_dino"
    # ENCODER_ABLATION = "utonia"

    # DECODER_ABLATION = "baseline"
    # DECODER_ABLATION = "fourier"
    DECODER_ABLATION = "fourier_residual"

    # ============================================================
    # TRAINING
    # ============================================================

    BATCH_SIZE = 32

    SHAPENET_EPOCHS = 150
    SCANNET_EPOCHS = 100

    SHAPENET_LR = 5e-4
    SHAPENET_WD = 1e-4

    SCANNET_LR = 1e-4
    SCANNET_WD = 1e-4

    if ENCODER_ABLATION == "utonia_dino":
        SHAPENET_PRETRAIN_MODE = "utonia"
    elif ENCODER_ABLATION == "coord_dino":
        SHAPENET_PRETRAIN_MODE = "coords"
    elif ENCODER_ABLATION == "utonia":
        SHAPENET_PRETRAIN_MODE = "utonia"
    else:
        raise ValueError(
            f"No ShapeNet pretraining mode defined for "
            f"{ENCODER_ABLATION}"
        )

    # ============================================================
    # VALIDATE CONFIG
    # ============================================================

    if ARCHITECTURE == "ours":

        if (
            ENCODER_ABLATION
            not in ENCODER_ABLATIONS
        ):
            raise ValueError(
                f"Unknown encoder ablation "
                f"'{ENCODER_ABLATION}'. "
                f"Available: "
                f"{list(ENCODER_ABLATIONS.keys())}"
            )

        if (
            DECODER_ABLATION
            not in DECODER_ABLATIONS
        ):
            raise ValueError(
                f"Unknown decoder ablation "
                f"'{DECODER_ABLATION}'. "
                f"Available: "
                f"{list(DECODER_ABLATIONS.keys())}"
            )

        encoder_config = (
            ENCODER_ABLATIONS[
                ENCODER_ABLATION
            ]
        )

        decoder_config = (
            DECODER_ABLATIONS[
                DECODER_ABLATION
            ]
        )

    else:

        encoder_config = None
        decoder_config = None

    # ============================================================
    # RUN DIRECTORIES
    # ============================================================

    if ARCHITECTURE == "ours":

        shapenet_run_name = (
            "shapenet_pretrain_ours_"
            f"{ENCODER_ABLATION}_"
            f"{DECODER_ABLATION}"
        )

        scannet_run_name = (
            "scannet_finetune_ours_"
            f"{ENCODER_ABLATION}_"
            f"{DECODER_ABLATION}"
        )

    else:

        shapenet_run_name = (
            "shapenet_pretrain_dinocomplete"
        )

        scannet_run_name = (
            "scannet_train_dinocomplete"
        )

    shapenet_run_dir = os.path.join(
        "runs",
        shapenet_run_name,
    )

    scannet_run_dir = os.path.join(
        "runs",
        scannet_run_name,
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "EXPERIMENT CONFIGURATION"
    )

    print(
        "============================================================"
    )

    print(
        f"Architecture          : "
        f"{ARCHITECTURE}"
    )

    print(
        f"Encoder               : "
        f"{ENCODER_ABLATION}"
    )

    print(
        f"Decoder               : "
        f"{DECODER_ABLATION}"
    )

    print(
        f"ShapeNet pretraining  : "
        f"{USE_SHAPENET_PRETRAIN}"
    )

    print(
        f"ShapeNet epochs       : "
        f"{SHAPENET_EPOCHS}"
    )

    print(
        f"ScanNet epochs        : "
        f"{SCANNET_EPOCHS}"
    )

    # ============================================================
    # STAGE 1: SHAPENET PRETRAINING
    # ============================================================

    if USE_SHAPENET_PRETRAIN:

        (
            shapenet_encoder,
            shapenet_decoder,
            shapenet_best_path,
        ) = train_shapenet_stage(
            device=device,
            pretrain_mode=SHAPENET_PRETRAIN_MODE,
            decoder_config=decoder_config,
            run_name=shapenet_run_name,
            run_dir=shapenet_run_dir,
            epochs=SHAPENET_EPOCHS,
        )

        print("\n============================================================")
        print("STAGE 1 COMPLETE")
        print("============================================================")

        print(
            f"ShapeNet best model:\n"
            f"{shapenet_best_path}"
        )

    else:
        shapenet_encoder = None
        shapenet_decoder = None

        print("\n⏭️ ShapeNet pretraining disabled.")
        
    # ============================================================
    # STAGE 2: SCANNET
    # ============================================================

    print(
        "\n"
        "============================================================"
    )

    print(
        "STAGE 2: SCANNET TRAINING"
    )

    print(
        "============================================================"
    )

    # ------------------------------------------------------------
    # Load ScanNet data
    # ------------------------------------------------------------

    (
        data_files,
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = build_dataloaders(
        "data/pairs_scannet/train/*.pt"
    )

    detected_dim = detect_feature_dim(
        data_files
    )

    if detected_dim != 1408:
        raise ValueError(
            "ScanNet++ expects 1408-D features "
            "(1024 Utonia + 384 DINO), "
            f"got {detected_dim}."
        )

    # ------------------------------------------------------------
    # Build ScanNet model
    # ------------------------------------------------------------

    encoder, decoder = build_scanet_model(
        ARCHITECTURE,
        encoder_config,
        decoder_config,
        device,
    )

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
        f"\nScanNet model parameters: "
        f"{num_params:,}"
    )

    # ============================================================
    # SCANNET OPTIMIZER
    # ============================================================

    optimizer = torch.optim.AdamW(
        list(encoder.parameters())
        +
        list(decoder.parameters()),
        lr=SCANNET_LR,
        weight_decay=SCANNET_WD,
    )

    criterion = nn.BCEWithLogitsLoss()

    # ============================================================
    # SCANNET CHECKPOINT RESUME / SHAPENET INITIALIZATION
    # ============================================================

    os.makedirs(
        f"{scannet_run_dir}/snapshots",
        exist_ok=True,
    )

    os.makedirs(
        f"{scannet_run_dir}/checkpoints",
        exist_ok=True,
    )

    start_epoch = 0

    scanet_resume_ckpt = os.path.join(
        scannet_run_dir,
        "checkpoints",
        "latest_model.pth",
    )

    if os.path.exists(scanet_resume_ckpt):

        print(
            f"\n📂 Existing ScanNet checkpoint found:"
            f"\n{scanet_resume_ckpt}"
        )

        checkpoint = torch.load(
            scanet_resume_ckpt,
            map_location=device,
            weights_only=False,
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
            f"✅ Resuming ScanNet from epoch "
            f"{start_epoch}"
        )

    else:

        if USE_SHAPENET_PRETRAIN:

            print(
                "\n"
                "============================================================"
            )

            print(
                "TRANSFERRING SHAPENET → SCANNET"
            )

            print(
                "============================================================"
            )

            transfer_shapenet_encoder(
                shapenet_encoder,
                encoder,
                ENCODER_ABLATION,
            )

            transfer_compatible_weights(
                shapenet_decoder,
                decoder,
                "Decoder",
            )

            print(
                "\n✅ ShapeNet weights transferred "
                "where architectures/shapes matched."
            )

            print(
                "⚠️ ScanNet-specific input layers "
                "remain randomly initialized when "
                "their shapes do not match."
            )

        else:

            print(
                "\n🆕 No ScanNet checkpoint found."
            )

            print(
                "Initializing ScanNet model from scratch."
            )

    # ============================================================
    # SCANNET MODEL SUMMARY
    # ============================================================

    with open(
        f"{scannet_run_dir}/model_summary.txt",
        "w",
    ) as f:

        f.write(
            "Dataset: ScanNet++\n"
        )

        f.write(
            "Training stage: "
            "fine-tuning\n"
            if USE_SHAPENET_PRETRAIN
            else
            "Training stage: "
            "training from scratch\n"
        )

        f.write(
            f"Architecture: "
            f"{ARCHITECTURE}\n"
        )

        if ARCHITECTURE == "ours":

            f.write(
                f"Encoder ablation: "
                f"{ENCODER_ABLATION}\n"
            )

            f.write(
                "Encoder description: "
                f"{ENCODER_ABLATION_DESCRIPTIONS[ENCODER_ABLATION]}\n"
            )

            f.write(
                f"Decoder ablation: "
                f"{DECODER_ABLATION}\n"
            )

            f.write(
                "Decoder description: "
                f"{DECODER_ABLATION_DESCRIPTIONS[DECODER_ABLATION]}\n"
            )

        f.write(
            "Input feature dimension: 1408\n"
        )

        if ARCHITECTURE == "ours":
            f.write(
                "Encoder output: 1024-D "
                "(mean + max pooling)\n"
            )
        else:
            f.write(
                "Encoder output: 1024-D "
                "(attention pooling + projection)\n"
            )

        if ARCHITECTURE == "ours":

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

        f.write(
            f"Batch size: "
            f"{BATCH_SIZE}\n"
        )

        f.write(
            f"Epochs: "
            f"{SCANNET_EPOCHS}\n"
        )

        f.write(
            "Optimizer: AdamW\n"
        )

        f.write(
            f"Learning rate: "
            f"{SCANNET_LR}\n"
        )

        f.write(
            f"Weight decay: "
            f"{SCANNET_WD}\n"
        )

        f.write(
            f"ShapeNet pretrained: "
            f"{USE_SHAPENET_PRETRAIN}\n"
        )

        f.write(
            f"Parameters: "
            f"{num_params:,}\n"
        )

    # ============================================================
    # FIRST SCANNET BATCH
    # ============================================================

    first_batch = next(
        iter(train_loader)
    )

    print(
        "\nFirst ScanNet training batch:"
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
    # SCANNET CONFIG
    # ============================================================

    experiment_config = {
        "dataset": "scannet",
        "training_mode": (
            "shapenet_pretrained"
            if USE_SHAPENET_PRETRAIN
            else "from_scratch"
        ),
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
        "num_input_points": 2048,
        "num_queries": 8192,
        "occupancy_distance_threshold": 0.01,
        "boundary_sigma": 0.005,
        "batch_size": BATCH_SIZE,
        "epochs": SCANNET_EPOCHS,
        "learning_rate": SCANNET_LR,
        "weight_decay": SCANNET_WD,
        "input_feature_dim": 1408,
        "shapenet_pretrained": USE_SHAPENET_PRETRAIN,
        "shapenet_checkpoint": (
            shapenet_best_path
            if USE_SHAPENET_PRETRAIN
            else None
        ),
        "reconstruction_probability_threshold": THRESHOLD,
        "optimizer": "AdamW",
    }

    # ============================================================
    # TRAIN SCANNET
    # ============================================================

    print(
        "\n🔥 Starting ScanNet training..."
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
        SCANNET_EPOCHS,
        start_epoch=start_epoch,
        RUN_NAME=scannet_run_name,
        RUN_DIR=scannet_run_dir,
        resume_training=(
            start_epoch > 0
        ),
        config=experiment_config,
    )

    print(
        "\n🏁 ScanNet training finished."
    )

    # ============================================================
    # LOAD BEST SCANNET MODEL
    # ============================================================

    best_scannet_path = os.path.join(
        scannet_run_dir,
        "checkpoints",
        "best_model.pth",
    )

    if not os.path.exists(
        best_scannet_path
    ):
        raise FileNotFoundError(
            "ScanNet best_model.pth was not found:\n"
            f"{best_scannet_path}"
        )

    best_ckpt = torch.load(
        best_scannet_path,
        map_location=device,
        weights_only=False,
    )

    encoder.load_state_dict(
        best_ckpt["encoder_state_dict"]
    )

    decoder.load_state_dict(
        best_ckpt["decoder_state_dict"]
    )

    print(
        "\n✅ Best ScanNet model loaded."
    )

    # ============================================================
    # FINAL TEST EVALUATION
    # ============================================================

    print(
        "\n📊 Running final ScanNet test evaluation..."
    )

    evaluate_test_set(
        encoder,
        decoder,
        test_dataset,
        device,
        RUN_DIR=scannet_run_dir,
    )

    print(
        "\n✅ Complete experiment finished."
    )