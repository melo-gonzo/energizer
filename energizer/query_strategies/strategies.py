from typing import List, Tuple

import numpy as np
from torch import Tensor

from energizer.acquisition_functions import (
    bald,
    entropy,
    expected_entropy,
    expected_least_confidence,
    expected_margin_confidence,
    least_confidence,
    margin_confidence,
    predictive_entropy,
)
from energizer.query_strategies.base import AccumulatorStrategy, MCAccumulatorStrategy, NoAccumulatorStrategy
from energizer.utilities.types import MODEL_INPUT
from energizer.utils import time_it
"""
NoAccumulatorStrategy's
"""


class RandomStrategy(NoAccumulatorStrategy):
    def query(self) -> List[int]:
        pool_size = self.trainer.datamodule.pool_size
        size = np.min([pool_size, self.query_size])
        return np.random.choice(pool_size, size=size, replace=False).tolist()


"""
AccumulatorStrategy's
"""


class EntropyStrategy(AccumulatorStrategy):
    def pool_step(self, batch: MODEL_INPUT, batch_idx: int, *args, **kwargs) -> Tensor:
        logits = self(batch)
        batch_size = len(next(iter(logits.values())))
        if self.queries_made == 0:
            return Tensor(np.random.rand(batch_size)).to(device=self.trainer.lightning_module.device)
        else:
            return entropy(logits)


class LeastConfidenceStrategy(AccumulatorStrategy):
    def pool_step(self, batch: MODEL_INPUT, batch_idx: int, *args, **kwargs) -> Tensor:
        logits = self(batch)
        if self.queries_made == 0:
            return Tensor(np.random.rand(batch_size)).to(device=self.trainer.lightning_module.device)
        else:
            return least_confidence(logits)


class MarginStrategy(AccumulatorStrategy):
    def pool_step(self, batch: MODEL_INPUT, batch_idx: int, *args, **kwargs) -> Tensor:
        logits = self(batch)
        if self.queries_made == 0:
            return Tensor(np.random.rand(batch_size)).to(device=self.trainer.lightning_module.device)
        else:
            return margin_confidence(logits)


"""
MCAccumulatorStrategy's
"""


class ExpectedEntropyStrategy(MCAccumulatorStrategy):
    def pool_step(self, batch: MODEL_INPUT, batch_idx: int, *args, **kwargs) -> Tensor:
        logits = self(batch)
        values = logits
        if self.queries_made == 0:
            return Tensor(np.random.rand(batch_size)).to(device=self.trainer.lightning_module.device)
        else:
            return expected_entropy(logits)


class PredictiveEntropyStrategy(MCAccumulatorStrategy):
    def pool_step(self, batch: MODEL_INPUT, batch_idx: int, *args, **kwargs) -> Tensor:
        logits = self(batch)
        values = logits
        if self.queries_made == 0:
            return Tensor(np.random.rand(batch_size)).to(device=self.trainer.lightning_module.device)
        else:
            return predictive_entropy(logits)


class ExpectedLeastConfidenceStrategy(MCAccumulatorStrategy):
    def pool_step(self, batch: MODEL_INPUT, batch_idx: int, *args, **kwargs) -> Tensor:
        logits = self(batch)
        values = logits
        if self.queries_made == 0:
            return Tensor(np.random.rand(batch_size)).to(device=self.trainer.lightning_module.device)
        else:
            return expected_least_confidence(logits)


class ExpectedMarginStrategy(MCAccumulatorStrategy):
    def pool_step(self, batch: MODEL_INPUT, batch_idx: int, *args, **kwargs) -> Tensor:
        logits = self(batch)
        values = logits
        if self.queries_made == 0:
            return Tensor(np.random.rand(batch_size)).to(device=self.trainer.lightning_module.device)
        else:
            return expected_margin_confidence(logits)


class BALDStrategy(MCAccumulatorStrategy):
    def pool_step(self, batch: MODEL_INPUT, batch_idx: int, *args, **kwargs) -> Tensor:
        logits = self(batch)
        values = logits
        if self.queries_made == 0:
            return Tensor(np.random.rand(batch_size)).to(device=self.trainer.lightning_module.device)
        else:
            return bald(logits)
