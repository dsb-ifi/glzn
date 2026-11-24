import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms.functional import pil_to_tensor

from glzn.aug import Solarization, build_dino_pipeline
from glzn.data.dataset import iTarDataset
from glzn.data.itar.fold import iTarRetriever
from glzn.data.writer import iTarDatasetWriter


class ITarTests(unittest.TestCase):
    def test_repeated_fields_feed_dino_collation(self):
        pipeline = build_dino_pipeline(
            global_size=32, local_size=16, local_crops_number=1
        )
        with TemporaryDirectory() as directory:
            image = Image.new("RGB", (48, 48), "red")
            image.format = "JPEG"
            with iTarDatasetWriter("images", directory, folds=["train"]) as writer:
                writer.fold("train").write("sample", {"jpg": image})

            dataset = iTarDataset(
                "images", directory, "train",
                extensions=["jpg"] * pipeline.num_views,
                parse_if_missing=False,
                serialize=False,
                verbose=False,
            )
            self.assertEqual(dataset.logical_field_names, ("jpg.0", "jpg.1", "jpg.2"))
            first, second, third = dataset[0]
            self.assertIs(first, second)
            self.assertIs(first, third)

            dataset.map(lambda *fields: fields)
            dataset.map_tuple(*pipeline.field_transforms)
            batch = pipeline.collate([dataset[0]])
            self.assertEqual(
                [tuple(crop.shape) for crop in batch["global_crops"]],
                [(1, 3, 32, 32), (1, 3, 32, 32)],
            )
            self.assertEqual(
                [tuple(crop.shape) for crop in batch["local_crops"]],
                [(1, 3, 16, 16)],
            )

            dataset.map(lambda *fields: fields[:-1])
            with self.assertRaisesRegex(ValueError, "must preserve logical arity"):
                dataset[0]

    def test_tensor_jpeg_and_pseudoextensions(self):
        with TemporaryDirectory() as directory:
            with iTarDatasetWriter("images", directory, folds=["train"]) as writer:
                for image_format in ("JPEG", "PNG"):
                    image = Image.new("RGB", (12, 10), "red")
                    image.format = image_format
                    writer.fold("train").write(image_format, {"jpg": image, "cls": 3})

            extensions = ["jpg", "cls", "_idx", "_fid", "_stem", "_name"]
            pil_dataset = iTarDataset("images", directory, "train", extensions, verbose=False)
            tensor_dataset = iTarDataset(
                "images", directory, "train", extensions, jpeg_as_tensor=True, verbose=False
            )
            with patch.object(
                tensor_dataset.retriever, "hdrname", wraps=tensor_dataset.retriever.hdrname
            ) as hdrname:
                for index, stem in enumerate(("JPEG", "PNG")):
                    image, *metadata = tensor_dataset[index]
                    pil_image, *pil_metadata = pil_dataset[index]
                    self.assertIsInstance(image, tv_tensors.Image)
                    torch.testing.assert_close(image, pil_to_tensor(pil_image), rtol=0, atol=0)
                    solarize = Solarization()
                    torch.testing.assert_close(
                        solarize(image), pil_to_tensor(solarize(pil_image)), rtol=0, atol=0
                    )
                    self.assertEqual(metadata, pil_metadata)
                    self.assertEqual(metadata, [3, index, 0, stem, stem])
                self.assertEqual(hdrname.call_count, 2)

            by_stem = tensor_dataset.lookup_stems(["JPEG"], extensions=["jpg"])
            self.assertIsInstance(by_stem["JPEG"]["jpg"], tv_tensors.Image)
            tensor_dataset.filter_extensions(["jpg", "cls", "_idx", "_fid"])
            with patch.object(tensor_dataset.retriever, "hdrname") as hdrname:
                image, label, index, fid = tensor_dataset[0]
                self.assertIsInstance(image, tv_tensors.Image)
                self.assertEqual((label, index, fid), (3, 0, 0))
                hdrname.assert_not_called()

    def test_validation_reports_the_first_invalid_row_group(self):
        with TemporaryDirectory() as directory:
            with iTarDatasetWriter("images", directory, folds=["train"]) as writer:
                for index in range(3):
                    writer.fold("train").write(f"sample{index}", {"cls": 3, "txt": "text"})

            dataset = iTarDataset("images", directory, "train", verbose=False)
            state = dataset.fold.state
            rows = state.arr.copy()
            # Give the second sample reversed extension order and corrupt the last.
            rows[2:4] = rows[2:4][::-1]
            rows[5]["extid"] = rows[4]["extid"]
            dataset.fold.state = state._replace(arr=rows)
            with patch("glzn.data.dataset._VALIDATION_CHUNK_SIZE", 1):
                with self.assertRaisesRegex(ValueError, "row 4 .*expected exactly once each"):
                    dataset._validate_contiguous_row_groups("test")
                rows[3]["crashid"] = 1
                with self.assertRaisesRegex(ValueError, "row 2 mixes stems"):
                    dataset._validate_contiguous_row_groups("test")

    def test_pread_reuses_and_evicts_handles_and_can_be_pickled(self):
        with TemporaryDirectory() as directory:
            first = Path(directory) / "first.tar"
            second = Path(directory) / "second.tar"
            first.write_bytes(b"abcdef")
            second.write_bytes(b"123456")
            reader = iTarRetriever([first, second], prefer_mmap=False)

            files = []
            open_file = Path.open

            def track_open(path, *args, **kwargs):
                file = open_file(path, *args, **kwargs)
                files.append(file)
                return file

            with (
                patch.object(iTarRetriever, "_MAX_OPEN_SHARDS", 1),
                patch.object(Path, "open", track_open),
            ):
                self.assertEqual(reader.get(0, 1, 2).tobytes(), b"bc")
                self.assertEqual(reader.get(0, 2, 2).tobytes(), b"cd")
                self.assertEqual(len(files), 1)
                self.assertEqual(reader.get(1, 2, 2).tobytes(), b"34")
                self.assertTrue(files[0].closed)

                restored = pickle.loads(pickle.dumps(reader))
                self.assertEqual(restored.get(1, 1, 2).tobytes(), b"23")
                self.assertEqual(len(files), 3)
                self.assertFalse(files[1].closed)
