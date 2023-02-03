import numpy as np
from .strategy import Strategy

class RandomSampling(Strategy):
    def __init__(self, dataset, config, logger, train_grouper):
        super(RandomSampling, self).__init__(dataset, config, logger, train_grouper)

    def query(self, n, curr_datasets):
        candidates = np.where((self.full_dataset.labeled_for_al_array == 0) & (self.full_dataset.split_array == self.full_dataset.split_dict["train"]))[0]
        return np.random.choice(candidates, n, replace=False)
