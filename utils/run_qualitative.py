import os
import glob
import torch

from torch.utils.data import random_split

from models.implicit_network import (
    ImplicitDecoderBasic,
    MultiModalFeatureEncoder
)

from models.dinocomplete import (
    DinoCompleteBaselineEncoder
)

from utils.implicit_dataset import (
    ScanNetppImplicitDataset
)

from utils.qualitative_evaluation import (
    run_qualitative_evaluation
)



# ==========================================================
# CONFIGURATION
# ==========================================================

ARCHITECTURE = "ours_residual_block"  # Options: "ours" or "dinocomplete"
# ARCHITECTURE = "dinocomplete"


RUN_NAME = "scannet_train"

if ARCHITECTURE == "ours_residual_block":
    RUN_NAME += "_ours_residual_block"
else:
    RUN_NAME += "_dinocomplete"



RUN_DIR = os.path.join(
    "runs",
    RUN_NAME
)



CHECKPOINT = os.path.join(
    RUN_DIR,
    "checkpoints",
    "best_model.pth"
)



DATA_DIR = "data/pairs_scannet/train/*.pt"



NUM_SAMPLES = 10



# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":


    print("\n================================")
    print("QUALITATIVE EVALUATION")
    print("================================")


    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )


    print(
        f"Device: {device}"
    )


    print(
        f"Architecture: {ARCHITECTURE}"
    )


    print(
        f"Loading checkpoint:\n{CHECKPOINT}"
    )



    if not os.path.exists(CHECKPOINT):

        raise FileNotFoundError(
            f"Checkpoint does not exist:\n{CHECKPOINT}"
        )



    # ======================================================
    # DATASET SPLIT
    # MUST MATCH TRAINING EXACTLY
    # ======================================================


    data_files = glob.glob(
        DATA_DIR
    )


    if len(data_files) == 0:

        raise RuntimeError(
            "No dataset files found."
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
        -
        train_size
        -
        val_size
    )


    _, _, test_files = random_split(
        data_files,
        [
            train_size,
            val_size,
            test_size
        ],
        generator=torch.Generator().manual_seed(42)
    )



    test_dataset = ScanNetppImplicitDataset(
        test_files
    )



    print(
        f"Test samples: {len(test_dataset)}"
    )



    # ======================================================
    # FEATURE DIMENSION
    # ======================================================


    sample = torch.load(
        data_files[0],
        map_location="cpu",
        weights_only=False
    )


    INPUT_DIM = sample[
        "partial_feats"
    ].shape[-1]


    print(
        f"Feature dimension: {INPUT_DIM}"
    )



    # ======================================================
    # MODEL INITIALIZATION
    # SAME AS TRAINING
    # ======================================================


    if ARCHITECTURE == "ours_residual_block":

        encoder = MultiModalFeatureEncoder(
            input_feat_dim=INPUT_DIM,
            latent_dim=512
        ).to(device)


    elif ARCHITECTURE == "dinocomplete":

        encoder = DinoCompleteBaselineEncoder().to(device)



    decoder = ImplicitDecoderBasic(
        latent_dim=1024,
        hidden_dim=256
    ).to(device)



    # ======================================================
    # LOAD WEIGHTS
    # ======================================================


    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
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



    encoder.eval()
    decoder.eval()



    print(
        "Model loaded successfully."
    )



    # ======================================================
    # QUALITATIVE EXPORT
    # ======================================================


    run_qualitative_evaluation(
        encoder,
        decoder,
        test_dataset,
        device,
        RUN_DIR,
        n_samples=NUM_SAMPLES
    )


    print(
        "\nFinished."
    )