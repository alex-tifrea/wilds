import numpy as np
from .strategy import Strategy

class RandomSampling(Strategy):
    def __init__(self, dataset, config, logger, train_grouper):
        super(RandomSampling, self).__init__(dataset, config, logger, train_grouper)

    def query(self, n, curr_datasets):
        return np.random.choice(np.where(self.full_dataset.labeled_for_al_array == 0)[0], n, replace=False)
