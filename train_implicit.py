import os
import glob
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import trimesh
import matplotlib.pyplot as plt

from models.implicit_network import MultiModalFeatureEncoder, ImplicitDecoder

from utils.training import train_model

from utils.implicit_dataset import ScanNetppImplicitDataset

from utils.evaluation import evaluate_test_set

# ==========================================
# ENGINE RUNTIME PIPELINE
# ==========================================
if __name__ == "__main__":
    print("🚀 Initializing Multi-Modal Multi-Class Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_files = glob.glob("data/geometric_pairs_dataset/train/*.pt")
    if not data_files:
        raise FileNotFoundError("Missing data packages inside 'data/geometric_pairs_dataset/train/'")
    print(f"📂 Discovered {len(data_files)} multi-modal data matrices.")

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

    #print(f"📊 Training Sets: {len(train_dataset)} shapes | Validation Sets: {len(val_dataset)} shapes")

    sample_data = torch.load(data_files[6], weights_only=False)
    detected_dim = sample_data['partial_feats'].shape[-1]
    print(f"🧬 Automatically detected feature dimension from factory output: {detected_dim}")

    encoder = MultiModalFeatureEncoder(
        input_feat_dim=detected_dim,
        latent_dim=512
    ).to(device)
    
    decoder = ImplicitDecoder(
        latent_dim=1024,
        hidden_dim=256
    ).to(device)
    
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs("runs/multimodal_baseline/snapshots", exist_ok=True)
    epochs = 50

    first_batch = next(iter(train_loader))

    print(first_batch['partial_feats'].shape)

    print("Feature dim:", detected_dim)
    print("Encoder dim:", detected_dim + 3)

    print("🔥 Starting training...")

    # --- TRAINING LOOP ---
    train_model(encoder, decoder, train_loader, val_dataset, optimizer, criterion, device, epochs)

    print("\n🏁 Framework routine finished. Run your evaluation snapshots through CloudCompare to see the improvements.")

    # --- FINAL TEST SET EVALUATION ---
    evaluate_test_set(encoder, decoder, test_dataset, device)