import numpy as np
from .strategy import Strategy

class RandomSampling(Strategy):
    def __init__(self, dataset, config, logger, train_grouper):
        super(RandomSampling, self).__init__(dataset, config, logger, train_grouper)

    def query(self, n):
        return np.random.choice(np.where(self.dataset.labeled_idxs==0)[0], n, replace=False)
