# ============================================================
# ABLATION DECODER CONFIGURATIONS
# ============================================================

DECODER_ABLATIONS = {
    "baseline": dict(
        hidden_dim=384,
        use_fourier=False,
        use_residual=False,
    ),
    "fourier": dict(
        hidden_dim=384,
        use_fourier=True,
        use_residual=False,
    ),
    "fourier_residual": dict(
        hidden_dim=384,
        use_fourier=True,
        use_residual=True,
    ),
}

DECODER_ABLATION_DESCRIPTIONS = {
    "baseline": "Coordinate MLP decoder, no positional encoding, no residual refinement.",
    "fourier": "Adds Fourier positional encoding to the coordinate input.",
    "fourier_residual": "Fourier positional encoding plus residual refinement blocks.",
}