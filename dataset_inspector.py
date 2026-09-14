import glob
import torch
import numpy as np


FILES = sorted(glob.glob("data/geometric_pairs_dataset2/train/*.pt"))

print(f"Found {len(FILES)} files")

all_partial = []
all_gt = []

extent_partial = []
extent_gt = []

center_partial = []
center_gt = []

for i, f in enumerate(FILES):

    d = torch.load(f)

    partial = d["partial_pts"].float().numpy()
    gt = d["gt_pts"].float().numpy()

    if i % 250 == 0:

        print("="*60)
        print(f"Processed {i}/{len(FILES)}")

        print("Partial extent:",
            np.round((partial.max(0)-partial.min(0)),3))

        print("GT extent:",
            np.round((gt.max(0)-gt.min(0)),3))

        print("Partial center:",
            np.round(partial.mean(0),3))

        print("GT center:",
            np.round(gt.mean(0),3))

    all_partial.append(partial)
    all_gt.append(gt)

    pmin = partial.min(axis=0)
    pmax = partial.max(axis=0)

    gmin = gt.min(axis=0)
    gmax = gt.max(axis=0)

    extent_partial.append(pmax - pmin)
    extent_gt.append(gmax - gmin)

    center_partial.append(partial.mean(axis=0))
    center_gt.append(gt.mean(axis=0))

all_partial = np.concatenate(all_partial, axis=0)
all_gt = np.concatenate(all_gt, axis=0)

extent_partial = np.stack(extent_partial)
extent_gt = np.stack(extent_gt)

center_partial = np.stack(center_partial)
center_gt = np.stack(center_gt)

print("\n============================")
print("POINT STATISTICS")
print("============================")

print("\nPartial")

print("min :", all_partial.min(axis=0))
print("max :", all_partial.max(axis=0))
print("mean:", all_partial.mean(axis=0))
print("std :", all_partial.std(axis=0))

print("\nGT")

print("min :", all_gt.min(axis=0))
print("max :", all_gt.max(axis=0))
print("mean:", all_gt.mean(axis=0))
print("std :", all_gt.std(axis=0))

print("\n============================")
print("OBJECT EXTENTS")
print("============================")

print("\nPartial extent")

print("mean:", extent_partial.mean(axis=0))
print("median:", np.median(extent_partial, axis=0))
print("max:", extent_partial.max(axis=0))

print("\nGT extent")

print("mean:", extent_gt.mean(axis=0))
print("median:", np.median(extent_gt, axis=0))
print("max:", extent_gt.max(axis=0))

print("\n============================")
print("OBJECT CENTERS")
print("============================")

print("\nPartial")

print("mean:", center_partial.mean(axis=0))
print("std :", center_partial.std(axis=0))

print("\nGT")

print("mean:", center_gt.mean(axis=0))
print("std :", center_gt.std(axis=0))