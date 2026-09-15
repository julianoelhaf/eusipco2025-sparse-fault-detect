from copy import deepcopy
from dataclasses import dataclass
from typing import List


@dataclass
class ClusterConfig:
    port: int


@dataclass
class DatasetConfig:
    additional: int
    duration: float
    data_directory: str
    expected_name: str
    frequency: int
    multiple_of: int
    new_name: str
    preprocessed_data_directory: str
    raw_data_directory: str
    regex: str
    samples: int
    sampling_frequency: int
    step: int
    topology: str


@dataclass
class DataSparsityConfig:
    bus_failure_id: int
    current_loss: bool
    downsampling_factor: int
    phase_failure_id: str
    relay_failure_ids: List[int]
    voltage_loss: bool
    zeroing_duration_s: float


@dataclass
class ModelConfig:
    model: str
    model_directory: str


@dataclass
class TrainingConfig:
    fault_target: str
    n_splits: int
    random_state: int
    test_size: float


@dataclass
class WindowExtractionConfig:

    step_length_seconds: float
    period_of_interest: float
    time_diff: float
    window_length: float
    windows_directory: str


@dataclass
class MainConfig:
    batch_size: int
    cluster: ClusterConfig
    dataset: DatasetConfig
    data_sparsity: DataSparsityConfig
    full_dataset: bool
    model: ModelConfig
    n_samples: int
    overwrite: bool
    sample_ids: list[int]
    training: TrainingConfig
    window_extraction: WindowExtractionConfig

    def copy(self):
        return MainConfig(
            batch_size=self.batch_size,
            dataset=deepcopy(self.dataset),
            data_sparsity=deepcopy(self.data_sparsity),
            model=deepcopy(self.model),
            n_samples=self.n_samples,
            overwrite=self.overwrite,
            sample_ids=self.sample_ids,
            window_extraction=deepcopy(self.window_extraction),
        )
