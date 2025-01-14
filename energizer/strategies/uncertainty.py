from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from lightning.fabric.wrappers import _FabricDataLoader, _FabricModule

from energizer.datastores.base import Datastore
from energizer.enums import InputKeys, OutputKeys, RunningStage, SpecialKeys
from energizer.estimators.active_estimator import ActiveEstimator
from energizer.registries import SCORING_FUNCTIONS
from energizer.types import BATCH_OUTPUT, METRIC
from energizer.utilities import ld_to_dl


class UncertaintyBasedStrategy(ActiveEstimator):
    _scoring_fn_registry = SCORING_FUNCTIONS

    def __init__(self, *args, score_fn: Union[str, Callable], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.score_fn = score_fn if isinstance(score_fn, Callable) else self._scoring_fn_registry[score_fn]

    def run_query(self, model: _FabricModule, datastore: Datastore, query_size: int) -> List[int]:
        pool_loader = self.configure_dataloader(datastore.pool_loader())
        self.progress_tracker.pool_tracker.max = len(pool_loader)  # type: ignore
        return self.compute_most_uncertain(model, pool_loader, query_size)  # type: ignore

    def compute_most_uncertain(
        self, model: _FabricModule, pool_loader: _FabricDataLoader, query_size: int
    ) -> List[int]:

        # calls the pool_step and pool_epoch_end that we override
        out: List[Dict] = self.run_evaluation(model, pool_loader, RunningStage.POOL)  # type: ignore
        _out = ld_to_dl(out)
        scores = np.concatenate(_out[OutputKeys.SCORES])
        ids = np.concatenate(_out[SpecialKeys.ID])

        # compute topk
        topk_ids = scores.argsort()[-query_size:]
        return ids[topk_ids].tolist()

    def evaluation_step(
        self,
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        metrics: Optional[METRIC],
        stage: Union[str, RunningStage],
    ) -> Union[List[Dict], BATCH_OUTPUT]:
        if stage != RunningStage.POOL:
            return super().evaluation_step(model, batch, batch_idx, loss_fn, metrics, stage)

        # keep IDs here in case user messes up in the function definition
        ids = batch[InputKeys.ON_CPU][SpecialKeys.ID]
        pool_out = self.pool_step(model, batch, batch_idx, metrics)

        if isinstance(pool_out, torch.Tensor):
            pool_out = {OutputKeys.SCORES: pool_out}
        else:
            assert isinstance(pool_out, dict) and OutputKeys.SCORES in pool_out, (
                "In `pool_step` you must return a Tensor with the scores per each element in the batch "
                f"or a Dict with a '{OutputKeys.SCORES}' key and the Tensor of scores as the value."
            )

        pool_out[SpecialKeys.ID] = ids  # type: ignore

        return pool_out  # enforce that we always return a dict here

    def pool_step(
        self,
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        metrics: Optional[METRIC] = None,
    ) -> BATCH_OUTPUT:
        raise NotImplementedError

    def pool_epoch_end(self, output: List[Dict], metrics: Optional[METRIC]) -> List[Dict]:
        return output
