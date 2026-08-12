import os
import glob
import numpy as np
import torch
import torch.nn as nn
import trimesh
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from torch.utils.data import random_split

from models.dinocomplete import DinoCompleteBaselineEncoder
from models.implicit_network import ImplicitDecoderBasic
from implicit_dataset2 import ScanNetppImplicitDataset


def save_ply(path, points):
    points = np.asarray(points, dtype=np.float32)
    trimesh.points.PointCloud(points).export(path)


def extract_prediction(
    encoder,
    decoder,
    partial_pts,
    partial_feats,
    device,
    output_dir,
    resolution=128,
):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        latent = encoder(
            partial_pts,
            partial_feats,
        )

        linear = torch.linspace(-0.5, 0.5, resolution, device=device)

        grid_x, grid_y, grid_z = torch.meshgrid(
            linear,
            linear,
            linear,
            indexing="ij",
        )

        coords = torch.stack(
            [grid_x, grid_y, grid_z],
            dim=-1,
        ).reshape(1, -1, 3)

        logits = []

        chunk_size = 50000

        for start in range(0, coords.shape[1], chunk_size):
            chunk = coords[:, start:start + chunk_size]
            logits.append(decoder(chunk, latent))

        logits = torch.cat(logits, dim=1)
        probabilities = torch.sigmoid(logits).squeeze(0)

        print(
            f"Grid occupancy | "
            f"min={probabilities.min().item():.6f} "
            f"mean={probabilities.mean().item():.6f} "
            f"max={probabilities.max().item():.6f}"
        )

        for threshold in [0.5, 0.6, 0.7, 0.8]:
            mask = probabilities > threshold
            points = coords.squeeze(0)[mask].cpu().numpy()

            save_ply(
                f"{output_dir}/pred_{threshold:.1f}.ply",
                points,
            )

            print(
                f"Threshold {threshold:.1f}: "
                f"{len(points)} points"
            )


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    torch.manual_seed(42)
    np.random.seed(42)

    output_dir = "overfit_eight_samples"
    os.makedirs(output_dir, exist_ok=True)

    data_files = glob.glob(
        "data/pairs_scannet/train/*.pt"
    )

    if not data_files:
        raise FileNotFoundError(
            "No training samples found."
        )

    train_size = int(0.8 * len(data_files))
    val_size = int(0.1 * len(data_files))
    test_size = (
        len(data_files)
        - train_size
        - val_size
    )

    train_files, _, _ = random_split(
        data_files,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    dataset = ScanNetppImplicitDataset(
        train_files,
        num_input_pts=2048,
        num_queries=16384,
    )

    # Freeze exactly eight samples once.
    fixed_samples = [
        dataset[i]
        for i in range(8)
    ]

    partial_pts = torch.stack([
        sample["partial_pts"]
        for sample in fixed_samples
    ]).to(device)

    partial_feats = torch.stack([
        sample["partial_feats"]
        for sample in fixed_samples
    ]).to(device)

    query_coords = torch.stack([
        sample["query_coords"]
        for sample in fixed_samples
    ]).to(device)

    target = torch.stack([
        sample["target_surface"]
        for sample in fixed_samples
    ]).to(device)

    print("\nFixed batch:")
    print("partial_pts:", partial_pts.shape)
    print("partial_feats:", partial_feats.shape)
    print("query_coords:", query_coords.shape)
    print("target:", target.shape)
    print(
        "positive ratio:",
        target.mean().item(),
    )

    # Save the normalized geometry for sample 0.
    sample = fixed_samples[0]

    save_ply(
        f"{output_dir}/partial.ply",
        sample["partial_pts"].numpy(),
    )

    gt_normalized = (
        sample["gt_pts_world"]
        - sample["centroid"]
    ) * sample["scale_factor"]

    save_ply(
        f"{output_dir}/gt_normalized.ply",
        gt_normalized.numpy(),
    )

    positive_queries = (
        sample["query_coords"][
            sample["target_surface"] > 0.5
        ]
    )

    save_ply(
        f"{output_dir}/positive_queries.ply",
        positive_queries.numpy(),
    )

    # Fresh model.
    encoder = DinoCompleteBaselineEncoder().to(device)

    decoder = ImplicitDecoderBasic(
        latent_dim=1024,
        hidden_dim=256,
        use_fourier=True,
        use_residual=False,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        list(encoder.parameters())
        + list(decoder.parameters()),
        lr=1e-3,
    )

    print("\nStarting eight-sample overfit...\n")

    best_loss = float("inf")

    for iteration in range(3001):

        encoder.train()
        decoder.train()

        latent = encoder(
            partial_pts,
            partial_feats,
        )

        logits = decoder(
            query_coords,
            latent,
        )

        loss = criterion(
            logits,
            target,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 100 == 0:

            with torch.no_grad():
                probabilities = torch.sigmoid(logits)

                predictions = (
                    probabilities > 0.5
                ).float()

                accuracy = (
                    predictions == target
                ).float().mean()

                positive_mask = target > 0.5
                negative_mask = target < 0.5

                positive_accuracy = (
                    predictions[positive_mask].mean()
                    if positive_mask.any()
                    else torch.tensor(0.0, device=device)
                )

                negative_accuracy = (
                    (predictions[negative_mask] == 0)
                    .float()
                    .mean()
                    if negative_mask.any()
                    else torch.tensor(0.0, device=device)
                )

            print(
                f"{iteration:5d} "
                f"loss={loss.item():.6f} "
                f"acc={accuracy.item():.4f} "
                f"pos_acc={positive_accuracy.item():.4f} "
                f"neg_acc={negative_accuracy.item():.4f}"
            )

            if loss.item() < best_loss:
                best_loss = loss.item()

                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict(),
                        "iteration": iteration,
                        "loss": loss.item(),
                    },
                    f"{output_dir}/best_overfit.pth",
                )

    print("\nGenerating dense reconstruction...")

    checkpoint = torch.load(
        f"{output_dir}/best_overfit.pth",
        map_location=device,
        weights_only=False,
    )

    encoder.load_state_dict(
        checkpoint["encoder"]
    )

    decoder.load_state_dict(
        checkpoint["decoder"]
    )

    # Reconstruct sample 0 only.
    extract_prediction(
        encoder,
        decoder,
        partial_pts[0:1],
        partial_feats[0:1],
        device,
        output_dir,
        resolution=128,
    )

    print("\nDone.")
    print("Best loss:", checkpoint["loss"])
    print("Best iteration:", checkpoint["iteration"])


if __name__ == "__main__":
    main()