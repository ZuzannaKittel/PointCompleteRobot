import os
import json
import numpy as np
import torch
import trimesh

from pathlib import Path

from configs.run_config import get_run_dir

from models.dinocomplete import DinoCompleteBaselineEncoder


# ==========================================================
# Configuration
# ==========================================================

NUM_SAMPLES = 10
RESOLUTION = 128
THRESHOLD = 0.6

SEED = 42


# ==========================================================
# Reconstruction inference
# ==========================================================

def reconstruct_shape(
        encoder,
        decoder,
        partial_pts,
        partial_feats,
        device,
        resolution=128,
        threshold=0.6):


    encoder.eval()
    decoder.eval()


    with torch.no_grad():

        # --------------------------------------
        # Encode partial observation
        # --------------------------------------

        if isinstance(
            encoder,
            DinoCompleteBaselineEncoder
        ):

            latent = encoder(
                partial_pts.unsqueeze(0),
                partial_feats.unsqueeze(0)
            )

        else:

            latent = encoder(
                partial_feats.unsqueeze(0)
            )


        # --------------------------------------
        # Query occupancy field
        # --------------------------------------

        lin = torch.linspace(
            -0.5,
            0.5,
            resolution,
            device=device
        )


        gx, gy, gz = torch.meshgrid(
            lin,
            lin,
            lin,
            indexing="ij"
        )


        queries = torch.stack(
            [
                gx,
                gy,
                gz
            ],
            dim=-1
        )


        queries = queries.reshape(
            1,
            -1,
            3
        )


        chunk_size = 50000

        logits = []


        for i in range(
            0,
            queries.shape[1],
            chunk_size
        ):

            q = queries[
                :,
                i:i+chunk_size,
                :
            ]


            out = decoder(
                q,
                latent
            )


            logits.append(out)


        logits = torch.cat(
            logits,
            dim=1
        )


        probs = torch.sigmoid(
            logits
        ).squeeze(0)



        occupied = probs > threshold


        reconstructed = queries.squeeze(0)[occupied]


        reconstructed = reconstructed.cpu().numpy()


    return reconstructed



# ==========================================================
# Category extraction helper
# ==========================================================

def get_category(dataset, idx):

    path = dataset.file_paths[idx]

    name = os.path.basename(path)

    parts = name.split("_")

    if len(parts) > 1:
        return parts[-2]

    return "unknown"



# ==========================================================
# Balanced sample selection
# ==========================================================

def select_samples_by_category(
        dataset,
        n_samples=10,
        seed=42):


    rng = np.random.default_rng(seed)


    categories = {}


    for idx in range(len(dataset)):

        cat = get_category(
            dataset,
            idx
        )


        if cat not in categories:
            categories[cat] = []


        categories[cat].append(idx)



    selected = []


    cats = list(categories.keys())


    rng.shuffle(cats)



    while len(selected) < n_samples:

        progress = False


        for cat in cats:

            if len(selected) >= n_samples:
                break


            available = categories[cat]


            if len(available) == 0:
                continue


            idx = rng.choice(
                available
            )


            selected.append(idx)

            available.remove(idx)

            progress = True



        if not progress:
            break



    return selected[:n_samples]



# ==========================================================
# Main qualitative evaluation
# ==========================================================

def run_qualitative_evaluation(
        encoder,
        decoder,
        test_dataset,
        device,
        run_dir,
        n_samples=10):


    output_dir = Path(run_dir) / "qualitative_results"

    output_dir.mkdir(
        exist_ok=True,
        parents=True
    )


    indices = select_samples_by_category(
        test_dataset,
        n_samples=n_samples,
        seed=SEED
    )


    print("\nSelected samples:")

    for idx in indices:
        print(
            idx,
            get_category(
                test_dataset,
                idx
            )
        )



    for counter, idx in enumerate(indices):


        print(
            f"\nProcessing {counter+1}/{len(indices)} "
            f"(dataset idx={idx})"
        )


        sample = test_dataset[idx]


        partial_pts = sample[
            "partial_pts"
        ].to(device)


        partial_feats = sample[
            "partial_feats"
        ].to(device)



        reconstruction = reconstruct_shape(
            encoder,
            decoder,
            partial_pts,
            partial_feats,
            device,
            resolution=RESOLUTION,
            threshold=THRESHOLD
        )



        if len(reconstruction) == 0:

            print(
                "Empty reconstruction, skipping"
            )

            continue



        # --------------------------------------
        # Convert back to world coordinates
        # --------------------------------------

        centroid = sample[
            "centroid"
        ].numpy()


        scale = sample[
            "scale_factor"
        ].item()



        reconstruction_world = (
            reconstruction / scale
            +
            centroid
        )



        sample_dir = (
            output_dir /
            f"{counter:02d}_{get_category(test_dataset,idx)}_{idx}"
        )


        sample_dir.mkdir(
            exist_ok=True
        )



        # --------------------------------------
        # Save point clouds
        # --------------------------------------

        trimesh.points.PointCloud(
            sample[
                "partial_pts_world"
            ].numpy()
        ).export(
            sample_dir /
            "partial.ply"
        )


        trimesh.points.PointCloud(
            sample[
                "gt_pts_world"
            ].numpy()
        ).export(
            sample_dir /
            "ground_truth.ply"
        )


        trimesh.points.PointCloud(
            reconstruction_world.astype(
                np.float32
            )
        ).export(
            sample_dir /
            "prediction.ply"
        )



        # --------------------------------------
        # Metadata
        # --------------------------------------

        metadata = {

            "dataset_index": int(idx),

            "category":
                get_category(
                    test_dataset,
                    idx
                ),

            "num_prediction_points":
                int(len(reconstruction)),

            "threshold":
                THRESHOLD,

            "resolution":
                RESOLUTION

        }


        with open(
            sample_dir / "metadata.json",
            "w"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4
            )



        print(
            f"Saved {sample_dir}"
        )



    print(
        "\nQualitative evaluation finished."
    )