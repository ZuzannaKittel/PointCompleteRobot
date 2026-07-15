import os
import glob
import sys
import random
rng = random.Random(42)

import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.models import init_utonia
from data_processing.data_engine_shapenet import ShapeNetDataEngine

SHAPENET_ROOT = os.path.join(PROJECT_ROOT, "data/ShapeNet/ShapeNet_preprocessed")
OUTPUT = os.path.join(PROJECT_ROOT, "data/pairs_shapenet/train")
CKPT_UTONIA = os.path.join(PROJECT_ROOT, "checkpoints/utoniadreamer/latest.pth")

MAX_MODELS_PER_CATEGORY = 500  # Limit the number of models per category for faster data generation and balancing the dataset

from shapenet_categories import SHAPENET_CATEGORIES

def generate():
    os.makedirs(OUTPUT, exist_ok=True)

    print("Loading Utonia...")
    uto = init_utonia(ckpt_path=CKPT_UTONIA)

    engine = ShapeNetDataEngine(uto)
    counter = 0

    for category, category_id in SHAPENET_CATEGORIES.items():
        cat_dir = os.path.join(SHAPENET_ROOT, category_id)
        print(f"⚙️ Processing category: {category} ({cat_dir})")

        if not os.path.exists(cat_dir):
            print(f"Warning: Category directory {cat_dir} does not exist. Skipping.")
            continue

        all_models = glob.glob(
            os.path.join(cat_dir, "**", "model_normalized.obj"),
            recursive=True,
        )

        rng.shuffle(all_models)

        models = all_models[:MAX_MODELS_PER_CATEGORY]

        print(f"{category}: {len(models)} / {len(all_models)} models selected")

        for mesh_path in tqdm(models):
            print(f"    -> 🪑Processing mesh: {mesh_path}")
            try:
                sample = engine.generate_pair(mesh_path)

                name = os.path.basename(
                    os.path.dirname(os.path.dirname(mesh_path))
                )

                data = {
                    "dataset": "ShapeNet",
                    "object_id": name,
                    "category": category,
                    "feature_dim": sample["partial_feats"].shape[-1],
                    "partial_pts": torch.from_numpy(sample["partial_pts"]).half(),
                    "partial_feats": torch.from_numpy(sample["partial_feats"]).half(),
                    "gt_pts": torch.from_numpy(sample["gt_pts"]).half(),
                    "anchor_world": torch.zeros(3).float(),
                }

                print("SAVE CHECK:")
                print(data["partial_pts"].shape)
                print(data["partial_feats"].shape)
                print(data["gt_pts"].shape)

                torch.save(
                    data,
                    os.path.join(OUTPUT, f"pair_{counter:05d}_{category}_{name}.pt"),
                )

                counter += 1

            except Exception as e:
                print("FAILED:", mesh_path, e)

    print("Generated:", counter)


if __name__ == "__main__":
    generate()