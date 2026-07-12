import os
from pathlib import Path

DEFAULT_RUN_NAME = "fusion_doublelatent"
RUN_NAME = os.getenv("RUN_NAME", DEFAULT_RUN_NAME)
RUN_DIR = os.getenv("RUN_DIR", str(Path("runs") / RUN_NAME))


def get_run_name() -> str:
    return RUN_NAME


def get_run_dir() -> str:
    return RUN_DIR
