from copy import deepcopy
from collections import defaultdict
import os

from wilds.common.data_loaders import get_train_loader, get_eval_loader
from transforms import initialize_transform
from utils import BatchLogger, log_group_data
from examples.configs.utils import populate_config
from examples.configs.algorithm import algorithm_defaults
from algorithms.initializer import initialize_algorithm


class Strategy:
    def __init__(self, full_dataset, config, logger, train_grouper):
        self.full_dataset = full_dataset
        self.config = config
        self.logger = logger
        assert hasattr(self.full_dataset, "labeled_for_al_array"), "Need to call init_for_al on the dataset first."
        self.train_grouper = train_grouper
        self.curr_datasets, self.algorithm_for_sampling = None, None

        self.logger_over_rounds = BatchLogger(
            os.path.join(self.config.log_dir, f'rounds_eval.csv'), mode=self.logger.mode,
            use_wandb=self.config.use_wandb
        )

    def query(self, n, curr_datasets):
        raise NotImplementedError

    def update(self, pos_idxs, neg_idxs=None):
        # Check that all the indices point to training samples (not validation).
        assert (self.full_dataset.split_array[pos_idxs] ==
        self.full_dataset.split_dict["train"]).sum() == len(pos_idxs), "Some indexes are not in the training set."

        self.full_dataset.labeled_for_al_array[pos_idxs] = self.full_dataset.labeled_code["labeled"]
        if neg_idxs:
            self.full_dataset.labeled_for_al_array[neg_idxs] = self.full_dataset.labeled_code["unlabeled"]

    def train(self, n_round=None):
        from train import train_round

        self.curr_datasets, curr_algorithm = self.prepare_training(self.config, n_round)

        train_acc_avg, val_acc_avg = train_round(
            algorithm=curr_algorithm,
            datasets=self.curr_datasets,
            general_logger=self.logger,
            logger_over_rounds=self.logger_over_rounds,
            config=self.config,
            n_round=n_round,
            epoch_offset=0,
            best_val_metric=None,
            unlabeled_dataset=None,
        )

        if self.config.algorithm == self.config.algorithm_for_sampling:
            self.algorithm_for_sampling = curr_algorithm
        else:
            config_for_sampling = deepcopy(self.config)
            # Set config appropriate for choice of algorithm for sampling
            config_for_sampling = populate_config(
                config_for_sampling,
                algorithm_defaults[self.config.algorithm_for_sampling],
                force_overwrite=True
            )
            config_for_sampling.algorithm = self.config.algorithm_for_sampling
            if (config_for_sampling.keep_only_sampling_model_at_epoch is not None and
                    config_for_sampling.keep_only_sampling_model_at_epoch.isnumeric()):
                config_for_sampling.n_epochs = int(config_for_sampling.keep_only_sampling_model_at_epoch)
            data_for_alg_for_sampling, self.algorithm_for_sampling = self.prepare_training(config_for_sampling, n_round)

            alg2_train_acc_avg, alg2_val_acc_avg = train_round(
                algorithm=self.algorithm_for_sampling,
                datasets=data_for_alg_for_sampling,
                general_logger=self.logger,
                logger_over_rounds=self.logger_over_rounds,
                config=config_for_sampling,
                n_round=n_round,
                epoch_offset=0,
                best_val_metric=None,
                unlabeled_dataset=None,
            )
            print(f"Finished training algorithm for sampling. Train acc {alg2_train_acc_avg}, Val acc {alg2_val_acc_avg}.")

        for split in self.curr_datasets:
            self.curr_datasets[split]['eval_logger'].close()
            self.curr_datasets[split]['algo_logger'].close()

        return train_acc_avg, val_acc_avg, self.curr_datasets

    def prepare_training(self, config, n_round):
        # Transforms & data augmentations for labeled dataset
        # To modify data augmentation, modify the following code block.
        # If you want to use transforms that modify both `x` and `y`,
        # set `do_transform_y` to True when initializing the `WILDSSubset` below.
        train_transform = initialize_transform(
            transform_name=config.transform,
            config=config,
            dataset=self.full_dataset,
            additional_transform_name=config.additional_train_transform,
            is_training=True)
        eval_transform = initialize_transform(
            transform_name=config.transform,
            config=config,
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
                    frac=config.frac,
                    transform=transform)

            if split == 'train':
                datasets[split]['loader'] = get_train_loader(
                    loader=config.train_loader,
                    dataset=datasets[split]['dataset'],
                    batch_size=config.batch_size,
                    uniform_over_groups=config.uniform_over_groups,
                    grouper=self.train_grouper,
                    distinct_groups=config.distinct_groups,
                    n_groups_per_batch=config.n_groups_per_batch,
                    **config.loader_kwargs)
            else:
                datasets[split]['loader'] = get_eval_loader(
                    loader=config.eval_loader,
                    dataset=datasets[split]['dataset'],
                    grouper=self.train_grouper,
                    batch_size=config.batch_size,
                    **config.loader_kwargs)

            # Set fields
            datasets[split]['split'] = split
            datasets[split]['name'] = self.full_dataset.split_names[split]
            datasets[split]['verbose'] = verbose

            # Loggers
            datasets[split]['eval_logger'] = BatchLogger(
                os.path.join(config.log_dir, f'{split}_{n_round}_eval.csv'), mode=self.logger.mode, use_wandb=config.use_wandb
            )
            datasets[split]['algo_logger'] = BatchLogger(
                os.path.join(config.log_dir, f'{split}_{n_round}_algo.csv'), mode=self.logger.mode, use_wandb=config.use_wandb
            )

        log_group_data(datasets, self.train_grouper, self.logger)

        # Initialize algorithm for prediction.
        algorithm = initialize_algorithm(
            algorithm_name=config.algorithm,
            config=config,
            datasets=datasets,
            train_grouper=self.train_grouper,
            unlabeled_dataset=None,
        )

        return datasets, algorithm

    def predict_prob(self, model, data_loader):
        from train import run_epoch, infer_predictions
        y_pred = infer_predictions(model, data_loader, self.config)
        return y_pred

    # def predict(self, data):
    #     results, y_pred = run_epoch(self.curr_algorithm, self.curr_datasets['unlabeled_for_al'], self.logger, "best", self.config, train=False)
    #     return y_pred
    #
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

