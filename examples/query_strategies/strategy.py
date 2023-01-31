from collections import defaultdict
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from wilds.common.data_loaders import get_train_loader, get_eval_loader
from transforms import initialize_transform

class Strategy:
    def __init__(self, full_dataset, config, logger, train_grouper):
        self.full_dataset = full_dataset
        self.config = config
        self.logger = logger
        assert hasattr(self.full_dataset, "_labeled_for_al_array"), "Need to call init_for_al on the dataset first."
        self.train_grouper = train_grouper

    def query(self, n):
        pass

    def update(self, pos_idxs, neg_idxs=None):
        # Check that all the indices point to training samples (not validation).
        assert self.datasets.split_array[pos_idxs].sum() == len(pos_idxs), "Some indexes are not in the training set."

        self.dataset._labeled_for_al_array[pos_idxs] = 1
        if neg_idxs:
            self.dataset._labeled_for_al_array[neg_idxs] = 0

    def train(self, n_round=None):
        from train import train_round, evaluate

        datasets, algorithm = self.prepare_training(n_round)

        train_acc_avg, val_acc_avg = train_round(
            algorithm=algorithm,
            datasets=datasets,
            general_logger=self.logger,
            config=self.config,
            epoch_offset=0,
            best_val_metric=None,
            unlabeled_dataset=None,
        )

        for split in datasets:
            datasets[split]['eval_logger'].close()
            datasets[split]['algo_logger'].close()

        return train_acc_avg, val_acc_avg

    def prepare_training(self, n_round):
        from utils import BatchLogger, log_group_data
        from algorithms.initializer import initialize_algorithm

        # Transforms & data augmentations for labeled dataset
        # To modify data augmentation, modify the following code block.
        # If you want to use transforms that modify both `x` and `y`,
        # set `do_transform_y` to True when initializing the `WILDSSubset` below.
        train_transform = initialize_transform(
            transform_name=self.config.transform,
            config=self.config,
            dataset=self.full_dataset,
            additional_transform_name=self.config.additional_train_transform,
            is_training=True)
        eval_transform = initialize_transform(
            transform_name=self.config.transform,
            config=self.config,
            dataset=self.full_dataset,
            is_training=False)

        # Configure labeled torch datasets (WILDS dataset splits)
        datasets = defaultdict(dict)
        for split in self.full_dataset.split_dict.keys():
            if split == 'train':
                transform = train_transform
                verbose = True
            elif split == 'val':
                transform = eval_transform
                verbose = True
            else:
                transform = eval_transform
                verbose = False

            # Get subset
            if split == "train":
                datasets[split]['dataset'] = self.full_dataset.get_labeled_subset(
                    transform=transform
                )
            elif split == "unlabeled_for_al":
                datasets[split]['dataset'] = self.full_dataset.get_unlabeled_for_al_subset(
                    transform=transform
                )
            else:
                datasets[split]['dataset'] = self.full_dataset.get_subset(
                    split,
                    frac=self.config.frac,
                    transform=transform)

            if split == 'train':
                datasets[split]['loader'] = get_train_loader(
                    loader=self.config.train_loader,
                    dataset=datasets[split]['dataset'],
                    batch_size=self.config.batch_size,
                    uniform_over_groups=self.config.uniform_over_groups,
                    grouper=self.train_grouper,
                    distinct_groups=self.config.distinct_groups,
                    n_groups_per_batch=self.config.n_groups_per_batch,
                    **self.config.loader_kwargs)
            else:
                datasets[split]['loader'] = get_eval_loader(
                    loader=self.config.eval_loader,
                    dataset=datasets[split]['dataset'],
                    grouper=self.train_grouper,
                    batch_size=self.config.batch_size,
                    **self.config.loader_kwargs)

            # Set fields
            datasets[split]['split'] = split
            datasets[split]['name'] = self.full_dataset.split_names[split]
            datasets[split]['verbose'] = verbose

            # Loggers
            datasets[split]['eval_logger'] = BatchLogger(
                os.path.join(self.config.log_dir, f'{split}_{n_round}_eval.csv'), mode=self.logger.mode, use_wandb=self.config.use_wandb
            )
            datasets[split]['algo_logger'] = BatchLogger(
                os.path.join(self.config.log_dir, f'{split}_{n_round}_algo.csv'), mode=self.logger.mode, use_wandb=self.config.use_wandb
            )

        log_group_data(datasets, self.train_grouper, self.logger)

        # Initialize algorithm.
        algorithm = initialize_algorithm(
            config=self.config,
            datasets=datasets,
            train_grouper=self.train_grouper,
            unlabeled_dataset=None,
        )

        return datasets, algorithm

    def predict(self, data):
        # TODO: replace with run_epoch with train=False
        preds = self.net.predict(data)
        return preds

    def predict_prob(self, data):
        # TODO: replace with run_epoch with train=False
        probs = self.net.predict_prob(data)
        return probs

    # def predict_prob_dropout(self, data, n_drop=10):
    #     probs = self.net.predict_prob_dropout(data, n_drop=n_drop)
    #     return probs
    #
    # def predict_prob_dropout_split(self, data, n_drop=10):
    #     probs = self.net.predict_prob_dropout_split(data, n_drop=n_drop)
    #     return probs
    #
    # def get_embeddings(self, data):
    #     embeddings = self.net.get_embeddings(data)
    #     return embeddings

