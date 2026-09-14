# 3D Shape Completion for Robotic Perception

**Master Thesis — MICS, University of Luxembourg**

This repository contains the code and experimental framework developed for my Master thesis, **“3D Shape Completion for Robotic Perception.”**

The project investigates **3D point-cloud completion from partial RGB-D observations of objects in realistic indoor scenes**. The main objective is to study whether features extracted from realistic RGB-D data can support reliable shape completion without relying on privileged ground-truth information at inference time.

---

## Overview

3D shape completion aims to reconstruct the complete geometry of an object from an incomplete observation.

This is particularly relevant for robotic perception: an object may be partially occluded, observed from only one side, or represented by a sparse and noisy depth measurement.

Many existing point-cloud completion approaches are evaluated primarily on clean synthetic CAD data, where partial observations are generated under controlled conditions. Real RGB-D observations introduce additional challenges:

* sensor noise and irregular sampling,
* occlusions and clutter,
* incomplete object visibility,
* imperfect segmentation,
* variations in viewpoint and scene context.

This project therefore studies point-cloud completion in a more realistic setting using **ScanNet++ RGB-D scenes** and object-level annotations aligned with **ShapeNet** models.

### Main questions

The experiments investigate:

1. How well can implicit point-cloud completion models generalize to realistic RGB-D observations?
2. How do different visual/geometric feature representations affect completion quality?
3. Does Fourier-based coordinate encoding improve implicit shape reconstruction?
4. Does residual decoding further improve the reconstruction?
5. Can pretraining on clean synthetic ShapeNet data improve performance on realistic observations?

---

## Method

The proposed pipeline consists of three main stages:

```text
ScanNet++ RGB-D scenes
        │
        ▼
Object extraction & partial point cloud generation
        │
        ▼
Multi-modal feature extraction
        │
        ▼
Implicit shape completion
        │
        ▼
Completed 3D object
```

### 1. Realistic partial observations

Object instances are extracted from ScanNet++ RGB-D scenes using available scene annotations, camera calibration and object geometry.

The extraction pipeline selects suitable RGB-D observations, projects the object into the camera views, performs object segmentation and reconstructs a partial 3D observation.

The resulting partial point cloud contains **2048 input points**.

The complete ShapeNet geometry associated with the object instance is used as the reconstruction target during training and evaluation.

### 2. Multi-modal features

The proposed model combines geometric and visual information extracted from the observed object.

The investigated encoders are:

| Encoder               | Features                                    |
| --------------------- | ------------------------------------------- |
| **Coordinate + DINO** | Point coordinates + DINOv2 visual features  |
| **Utonia + DINO**     | Utonia 3D features + DINOv2 visual features |
| **Utonia**            | Utonia 3D features                          |

The resulting features are aggregated into a **1024-dimensional global latent representation**.

The visual features are extracted using **DINOv2 ViT-S/14**, while Utonia provides learned 3D features.

### 3. Implicit decoder

The decoder represents the completed object as an implicit occupancy function:

```text
(x, z) → occupancy
```

where `x` is a queried 3D coordinate and `z` is the global feature representation of the partial observation.

Three decoder variants are evaluated:

* **Baseline**
* **Fourier**
* **Fourier + Residual**

The Fourier variant uses **six frequency bands**, while the residual architecture introduces **two residual blocks**.

---

## Experimental Design

The main experiment uses a **3 × 3 factorial ablation study**:

| Encoder ↓ / Decoder → | Baseline | Fourier | Fourier + Residual |
| --------------------- | -------: | ------: | -----------------: |
| Coordinate + DINO     |        ✓ |       ✓ |                  ✓ |
| Utonia + DINO         |        ✓ |       ✓ |                  ✓ |
| Utonia                |        ✓ |       ✓ |                  ✓ |

This gives **nine proposed configurations**.

An additional experiment evaluates the **DinoComplete** model as an external baseline.

A separate experiment investigates whether **ShapeNet pretraining** can improve performance when transferring the model to realistic ScanNet++ observations.

---

## Dataset

The main experiments use object instances extracted from **ScanNet++**.

The final dataset contains:

* **1,817 object samples**
* **80 / 10 / 10 train / validation / test split**
* fixed random seed: `42`
* `2,048` input points
* `8,192` implicit query coordinates per training sample
* stochastic sampling during training
* deterministic sampling during validation and testing

The complete target shape is represented using **16,384 points**.

At inference time, the implicit occupancy function is evaluated on a **64³ canonical grid**, followed by thresholding at `0.5`.

> The raw ScanNet++ and ShapeNet datasets are not redistributed in this repository. Please obtain them from their respective official sources and follow their licensing conditions.

### Data structure

```
data/
├── ScanNetpp/
│   ├── annotations/         # SCANnotate++ .pkl files go here
│   │   └── 30966f4c6e/
│   └── data/                # Raw ScanNet++ .ply and images go here
│       └── 30966f4c6e/
└── ShapeNet/
    ├── ShapeNetCore.v2/     # The 40GB raw download
    └── ShapeNet_preprocessed/ # The result of the bash script
```

---

## Training

The implementation is based on **PyTorch** and CUDA.

Main training configuration:

```text
Optimizer:       AdamW
Learning rate:   1e-4
Weight decay:    1e-4
Batch size:      32
Epochs:          100
```

The models are trained using the realistic ScanNet++-derived dataset.

For the ShapeNet pretraining experiment, the corresponding model is trained separately on synthetic data and compatible weights are transferred to the realistic-data model.

---

## Evaluation

The main evaluation metrics are:

* **Chamfer Distance (CD)** — geometric reconstruction error
* **F-score** — reconstruction accuracy under a distance threshold

Both mean and median values are reported.

The **median metrics are particularly useful for this dataset**, since individual real-world observations can vary substantially in difficulty due to occlusion, visibility and reconstruction quality.

---

## Results

The experiments show that the choice of feature representation and implicit decoder architecture has a measurable impact on completion quality.

One representative result from the pretraining experiment is:

| Model                                   |   Median CD | Median F-score |
| --------------------------------------- | ----------: | -------------: |
| Coordinate + DINO — scratch             |     0.01160 |         0.3037 |
| Coordinate + DINO — ShapeNet pretrained | **0.00945** |     **0.3209** |

For this configuration, ShapeNet pretraining improves both reconstruction metrics on the realistic ScanNet++ test set.

The complete set of experimental results, including all encoder/decoder ablations and the DinoComplete baseline, is available in the repository's result files.

---

## Qualitative Results

The repository also contains qualitative comparisons between:

* the partial RGB-D observation,
* the ground-truth complete geometry,
* and the predicted completed shape.

Examples include successful, median and challenging test cases to illustrate the behaviour of the model across different object instances.

---

## Repository Structure

```text
.
├── data/
│   └── ...
├── models/
│   ├── encoders/
│   ├── decoders/
│   └── ...
├── datasets/
│   └── ...
├── scripts/
│   └── ...
├── experiments/
│   └── ...
├── results/
│   └── ...
├── requirements.txt
└── README.md
```

The exact contents may depend on which parts of the experimental pipeline are included in the public repository.

---

## Getting Started

### Requirements

The project requires:

* Python
* PyTorch
* CUDA-compatible GPU
* the required pretrained feature extractors
* ScanNet++ data
* ShapeNet data for the corresponding experiments

Create an environment and install the required dependencies:

```bash
git clone https://github.com/ZuzannaKittel/PointCompleteRobot
cd PointCompleteRobot

pip install -r requirements.txt
```

Dataset paths and model checkpoints should then be configured according to the paths expected by the scripts.

### Training

A typical experiment can be launched with:

```bash
python <training_script>.py
```

For example, the main ablation experiments follow the naming convention:

```text
scannet_train_ours_<encoder>_<decoder>
```

with configurations such as:

```text
scannet_train_ours_coord_dino_fourier
scannet_train_ours_utonia_dino_fourier_residual
scannet_train_ours_utonia_fourier_residual
```

### Evaluation

Evaluation uses the trained checkpoint and produces reconstruction metrics for the ScanNet++ test set.

```bash
python <evaluation_script>.py
```

---

## Pretraining

The repository also includes the ShapeNet pretraining experiments.

The purpose of this experiment is to evaluate whether learning from clean synthetic CAD geometry provides a useful initialization for the realistic RGB-D completion task.

Only architecture-compatible weights are transferred from the pretrained model to the ScanNet++ model.

---

## Technologies

**Deep Learning**

* PyTorch
* CUDA
* AdamW

**3D / Computer Vision**

* Point clouds
* RGB-D reconstruction
* Implicit occupancy representations
* ScanNet++
* ShapeNet
* Utonia
* DINOv2
* Segment Anything (SAM)

**Experimentation**

* Encoder/decoder ablation studies
* Synthetic-to-real transfer
* Quantitative and qualitative 3D evaluation

---

## Key Contributions

The thesis explores three aspects of realistic robotic shape completion:

**1. A realistic evaluation setting**

Instead of relying exclusively on clean synthetic partial point clouds, the study evaluates completion on partial observations derived from real RGB-D indoor scenes.

**2. A systematic encoder/decoder study**

A controlled 3 × 3 ablation evaluates the effect of different feature representations and implicit decoder architectures.

**3. Synthetic-to-real pretraining**

The study investigates whether pretraining on ShapeNet can provide useful representations for completion from realistic RGB-D observations.

---

## Thesis

This repository accompanies my Master thesis:

> **3D Shape Completion for Robotic Perception**
> MICS — Master in Computer Science
> University of Luxembourg

The thesis provides the full methodology, experimental setup, analysis and discussion of the results.

---

## Acknowledgements

This work builds upon several open-source datasets and research projects, including:

* ScanNet++
* ShapeNet
* DINOv2
* Segment Anything
* Utonia
* DinoComplete

Please refer to the corresponding projects and publications for their original implementations and licenses.

---

## License

The source code in this repository is provided under the license specified in [`LICENSE`](LICENSE).

Datasets, pretrained models and third-party components remain subject to their respective licenses and terms of use.
