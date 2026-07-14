import os
from pathlib import Path

DEFAULT_RUN_NAME = "fusion_basic_152550504096085"
RUN_NAME = os.getenv("RUN_NAME", DEFAULT_RUN_NAME)
RUN_DIR = os.getenv("RUN_DIR", str(Path("runs") / RUN_NAME))
RUN_DECODER = os.getenv("RUN_DECODER", "basic")  # Options: "basic" or "double_latent"
RUN_MODEL = os.getenv("RUN_MODEL", "fusion")  # Options: "fusion" or "DINO" or "Utonia"

def get_run_name() -> str:
    return RUN_NAME

def get_run_decoder() -> str:
    return RUN_DECODER

def get_run_dir() -> str:
    return RUN_DIR

def get_run_model() -> str:
    return RUN_MODEL