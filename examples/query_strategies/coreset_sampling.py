import numpy as np
import torch
from .strategy import Strategy

class CoresetSampling(Strategy):
    def __init__(self, dataset, config, logger, train_grouper):
        super(CoresetSampling, self).__init__(dataset, config, logger, train_grouper)

    def query(self, n, curr_datasets):

        probs = self.predict_prob(
            model=self.algorithm_for_sampling.model,
            data_loader=curr_datasets["unlabeled_for_al"]["loader"])
        uncertainties = probs.max(1)[0]
        assert uncertainties.shape[0] == self.full_dataset.labeled_for_al_array[self.full_dataset.labeled_for_al_array == self.full_dataset.labeled_code["unlabeled"]].shape[0], f"Wrong shapes {uncertainties.shape} and {self.full_dataset.labeled_for_al_array[self.full_dataset.labeled_for_al_array == 0].shape}"
        output_idxs = uncertainties.sort()[1].cpu().numpy()[:n]
        labeled_for_al_array_idxs = np.where(self.full_dataset.labeled_for_al_array == self.full_dataset.labeled_code["unlabeled"])[0][output_idxs]

        return labeled_for_al_array_idxs
