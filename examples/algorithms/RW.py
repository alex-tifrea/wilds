import numpy as np
import torch
from algorithms.ERM import ERM
from models.initializer import initialize_model
from utils import move_to

class RW(ERM):
    def __init__(self, config, d_out, grouper, loss,
            metric, n_train_steps, group_counts_train):
        # initialize module
        super().__init__(
            config=config,
            d_out=d_out,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        self.group_counts_train = group_counts_train
        self.use_unlabeled_y = config.use_unlabeled_y  # Expect x,y,m from unlabeled loaders and train on the unlabeled y
        # TODO: change here to introduce InvRW; maybe have it as an exp temperature kind of thing
        # initialize weights
        if config.rw_min_group_weight is None:
            self.group_weights = 1. / self.group_counts_train
        else:
            threshold = np.median(self.group_weights.numpy())
            self.group_weights = torch.ones_like(self.group_counts_train)  # weight 1 for majority points
            self.group_weights[self.group_weights <= threshold] = config.rw_min_group_weight  # custom weight for minority
        self.group_weights = self.group_weights / self.group_weights.sum()
        self.group_weights = self.group_weights.to(self.device)

    def objective(self, results):
        group_losses, _, _ = self.loss.compute_group_wise(
            results['y_pred'],
            results['y_true'],
            results['g'],
            self.grouper.n_groups,
            return_dict=False)
        labeled_loss = group_losses @ self.group_weights

        if self.use_unlabeled_y and 'unlabeled_y_true' in results:
            raise RuntimeError("Unlabeled data is not supported for RW (reweighting) objective.")
        else:
            return labeled_loss
