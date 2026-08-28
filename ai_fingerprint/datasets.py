from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .dataset_manager import DatasetManager


class InputGenerator:
    """
    Compatibility wrapper retained for the client code.

    Unlike the earlier version, this class now loads the selected real dataset
    through DatasetManager. Synthetic data is used only when a synthetic
    dataset is explicitly selected.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.manager = DatasetManager(config)

    def sample(self) -> np.ndarray:
        return self.manager.sample()

    def training_batch(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.manager.sample_training_batch()

    def reset(self) -> None:
        self.manager.reset()
