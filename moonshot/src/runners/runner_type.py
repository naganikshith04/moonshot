from enum import Enum


class RunnerType(Enum):
    BENCHMARK = "benchmark"
    REDTEAM = "redteam"
    DATASET_AUGMENTATION = "dataset_augmentation"
