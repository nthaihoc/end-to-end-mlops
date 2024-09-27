from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    data_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareModelConfig:
    root_dir: Path
    base_model_dir: Path
    image_size: list
    learning_rate: float
    include_top: bool
    weights: str 
    classes: int
    beta_1: float
    beta_2: float
    decay: float

@dataclass(frozen=True)
class TrainingModelConfig:
    root_dir: Path
    model_trained_dir: Path
    epochs: int
    list_folder_name: list
    list_label_name : list
    batch_size: int
