import os
import torch
from torchvision import datasets
from torchvision import transforms
import numpy as np
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy

class LTDataset(WILDSDataset):
    _versions_dict = {'1.0': None}

    def __init__(self, dataset=None, version=None, root_dir='data', download=False, split_scheme='official', fraction_min_classes=0.5, maj_to_min_ratio=100, val_ratio=0.2):
        self._version = version
        self._data_dir = os.path.join(root_dir, dataset)
        self._dataset_name = dataset

        assert dataset in ["cifar10lt", "cifar100lt", "svhnlt"]
        dataset_fn = {
            "cifar10lt": datasets.CIFAR10,
            "cifar100lt": datasets.CIFAR100,
            "svhnlt": datasets.SVHN,
        }
        if "svhn" in dataset:
            data_train = dataset_fn[dataset](self._data_dir, split="train", download=download)
            data_test = dataset_fn[dataset](self._data_dir, split="test", download=download)
        else:
            data_train = dataset_fn[dataset](self._data_dir, train=True, download=download)
            data_test = dataset_fn[dataset](self._data_dir, train=False, download=download)

        data_train = self.subsample_min_classes(
            data=data_train,
            fraction_min_classes=fraction_min_classes,
            maj_to_min_ratio=maj_to_min_ratio,
        )

        train_size = int((1 - val_ratio) * len(data_train))
        val_size = len(data_train) - train_size
        test_size = len(data_test)

        self._input_array = torch.from_numpy(np.concatenate((data_train.data, data_test.data), axis=0)).permute(0, 3, 1, 2)

        # Get the y values
        self._y_array = torch.from_numpy(np.concatenate((data_train.targets, data_test.targets), axis=0))
        self._y_size = 1
        self._n_classes = len(np.unique(self._y_array))

        self._metadata_array = self._y_array.reshape((-1, 1))
        self._metadata_fields = ['y']

        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['y']))

        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')
        self._split_array = np.concatenate((np.zeros(train_size), np.ones(val_size), 2*np.ones(test_size)))

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
       return transforms.ToPILImage()(self._input_array[idx])

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

    def subsample_min_classes(self, data, fraction_min_classes, maj_to_min_ratio):
        np.random.seed(42)
        X, y = np.array(data.data), np.array(data.targets)

        n_classes = len(np.unique(y))
        n_min_classes = int(n_classes * fraction_min_classes)
        new_X, new_y = X[y == 0], y[y == 0]
        for curr_y in range(1, n_classes - n_min_classes):
            print(new_X.shape, new_y.shape)
            new_X = np.concatenate((new_X, X[y == curr_y]), axis=0)
            new_y = np.concatenate((new_y, y[y == curr_y]), axis=0)
        min_class_size = int(len(data) / n_classes / maj_to_min_ratio)
        for curr_y in range(n_classes - n_min_classes, n_classes):
            idxs = np.random.choice(len(y[y == curr_y]), min_class_size, replace=False)
            new_X = np.concatenate((new_X, X[y == curr_y][idxs]), axis=0)
            new_y = np.concatenate((new_y, y[y == curr_y][idxs]), axis=0)

        idxs = np.random.permutation(new_y.shape[0])
        data.data, data.targets = new_X[idxs], new_y[idxs]
        return data
