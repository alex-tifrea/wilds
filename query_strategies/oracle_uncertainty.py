import numpy as np
from .strategy import Strategy

class OracleUncertainty(Strategy):
    def __init__(self, dataset, net):
        super(OracleUncertainty, self).__init__(dataset, net)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        uncertainties = self.dataset.confidences[unlabeled_idxs]
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
