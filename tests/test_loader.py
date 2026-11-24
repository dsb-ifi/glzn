import unittest
from typing import NamedTuple

import torch
from torch import Tensor
from torch.utils.data import TensorDataset, default_collate

from glzn.data import build_dataloader, move_to_device


class ClassificationBatch(NamedTuple):
    images:Tensor
    labels:Tensor


class LoaderTests(unittest.TestCase):
    def test_move_to_device_preserves_named_batch_type(self):
        def collate_classification(samples:list[object]) -> ClassificationBatch:
            images, labels = default_collate(samples)
            return ClassificationBatch(images, labels)

        dataset = TensorDataset(torch.zeros(2, 3, 4, 4), torch.arange(2))
        loader = build_dataloader(
            dataset, batch_size=2, num_workers=0, drop_last=False,
            collate_fn=collate_classification,
        )
        batch = next(iter(loader))
        moved = move_to_device(batch, 'meta', non_blocking=False)

        assert isinstance(moved, ClassificationBatch)
        self.assertEqual(moved.images.shape, (2, 3, 4, 4))
        self.assertEqual(moved.labels.shape, (2,))
        self.assertEqual(moved.images.device, torch.device('meta'))
        self.assertEqual(moved.labels.device, torch.device('meta'))

        nested = move_to_device({'batches': [(batch, 'train')]}, 'meta', non_blocking=False)
        assert isinstance(nested, dict)
        self.assertIsInstance(nested['batches'], list)
        self.assertIs(type(nested['batches'][0]), tuple)
        nested_batch, fold = nested['batches'][0]
        self.assertIsInstance(nested_batch, ClassificationBatch)
        self.assertEqual(nested_batch.labels.device, torch.device('meta'))
        self.assertEqual(fold, 'train')
