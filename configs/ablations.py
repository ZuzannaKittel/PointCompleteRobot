# ============================================================
# ABLATION CONFIGURATIONS
# ============================================================

ABLATIONS = {
    # Experiment 1: Original baseline
    "baseline": {
        "hidden_dim": 256,
        "use_fourier": False,
        "use_residual": False,
    },

    # Experiment 2: Increase decoder capacity
    "hidden384": {
        "hidden_dim": 384,
        "use_fourier": False,
        "use_residual": False,
    },

    # Experiment 3: Add Fourier coordinate encoding
    "hidden384_fourier": {
        "hidden_dim": 384,
        "use_fourier": True,
        "use_residual": False,
        "num_fourier_bands": 10,
    },

    # Experiment 4: Add residual block
    "hidden384_fourier_residual": {
        "hidden_dim": 384,
        "use_fourier": True,
        "use_residual": True,
        "num_fourier_bands": 10,
        "num_residual_blocks": 4,
    },
}

ABLATION_DESCRIPTIONS = {

    "baseline":
        "Baseline implicit decoder",

    "hidden384":
        "Baseline + hidden dimension 384",

    "hidden384_fourier":
        "Baseline + hidden dimension 384 + Fourier encoding",

    "hidden384_fourier_residual":
        "Baseline + hidden dimension 384 + Fourier encoding + residual blocks",
}