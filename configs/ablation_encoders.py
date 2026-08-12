# ============================================================
# ENCODER ABLATION CONFIGURATIONS
# ============================================================

ENCODER_ABLATIONS = {
    # Experiment 1: Explicit geometry + visual semantics
    "coord_dino": {
        "type": "coord_dino",
    },

    # Experiment 2: Learned geometric features + visual semantics
    "utonia_dino": {
        "type": "utonia_dino",
    },

    # Experiment 3: Utonia only
    "utonia": {
        "type": "utonia",
    },
}

ENCODER_ABLATION_DESCRIPTIONS = {
    "coord_dino":
        "XYZ coordinates + DINO features",

    "utonia_dino":
        "Utonia features + DINO features",

    "utonia":
        "Utonia features only",
}