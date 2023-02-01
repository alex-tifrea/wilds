import numpy as np
import torch
from .strategy import Strategy

class LeastConfidence(Strategy):
    def __init__(self, dataset, config, logger, train_grouper):
        super(LeastConfidence, self).__init__(dataset, config, logger, train_grouper)

    def query(self, n, curr_datasets):
        probs = self.predict_prob()
        uncertainties = probs.max(1)[0]
        assert uncertainties.shape[0] == self.full_dataset.labeled_for_al_array[self.full_dataset.labeled_for_al_array == 0].shape[0], f"Wrong shapes {uncertainties.shape} and  {self.full_dataset.labeled_for_al_array[self.full_dataset.labeled_for_al_array == 0].shape}"
        output_idxs = uncertainties.sort()[1][:n]
        labeled_for_al_array_idxs = torch.where(self.full_dataset.labeled_for_al_array == 0)[0][output_idxs]

        return labeled_for_al_array_idxs
