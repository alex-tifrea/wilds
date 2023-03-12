import os
import torch
import time
from torchvision import datasets
from torchvision import transforms
import numpy as np
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy

class INaturalist(WILDSDataset):
    _versions_dict = {'1.0': None}

    def __init__(self, dataset=None, version=None, root_dir='data', download=False, split_scheme='official', val_ratio=0.2):
        self._version = version
        self._data_dir = os.path.join(root_dir, dataset)
        self._dataset_name = dataset

        target_type = "phylum"
        self._data_train = datasets.INaturalist(self._data_dir, version="2021_train_mini", target_type=target_type, download=False)
        self._data_test = datasets.INaturalist(self._data_dir, version="2021_valid", target_type=target_type, download=False)

        # Get the y values
        self._y_array = torch.tensor(
            [self._data_train.categories_map[c][target_type] for c, _ in self._data_train.index] +
            [self._data_test.categories_map[c][target_type] for c, _ in self._data_test.index]
        )
        self._y_size = 1
        self._n_classes = len(np.unique(self._y_array))

        # TODO: remove the classes that have less than 1000 samples

        train_size = int((1 - val_ratio) * len(self._data_train))
        val_size = len(self._data_train) - train_size
        test_size = len(self._data_test)

        self._metadata_array = self._y_array.reshape((-1, 1))
        self._metadata_fields = ['y']

        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['y']))

        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')
        train_and_valid = np.zeros(len(self._data_train))
        np.random.seed(42) # need to have determinism in choosing the train/validation split
        train_and_valid[np.random.choice(range(len(self._data_train)), size=val_size, replace=False)] = 1
        self._split_array = np.concatenate((train_and_valid, 2*np.ones(test_size)))
        # np.random.seed(int(time.time()))

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        if idx < len(self._data_train):
            return self._data_train[idx][0]
        else:
            return self._data_test[idx - len(self._data_train)][0]
       # return transforms.ToPILImage()(self._input_array[idx])

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels 
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metric = Accuracy(prediction_fn=prediction_fn)
        return self.standard_group_eval(
            metric,
            self._eval_grouper,
            y_pred, y_true, metadata)
