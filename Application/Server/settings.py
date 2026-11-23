import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@dataclass
class Settings:
    models: Path = field(
        default_factory=lambda: Path(
            os.getenv("MRI_MODELS_DIR", ROOT / "Models")
        ).resolve()
    )
    samples: Path = field(
        default_factory=lambda: Path(
            os.getenv("MRI_SAMPLES_DIR", ROOT / "SampleCases")
        ).resolve()
    )
    storage: Path = field(
        default_factory=lambda: Path(
            os.getenv("MRI_STORAGE_DIR", ROOT / "Server/storage")
        ).resolve()
    )
    device: str = field(default_factory=lambda: os.getenv("MRI_DEVICE", "auto"))
    max_file_bytes: int = 128 * 1024 * 1024
    max_expanded_bytes: int = 256 * 1024 * 1024
    max_voxels: int = 32_000_000


MODALITIES = ("flair", "t1", "t1ce", "t2")
