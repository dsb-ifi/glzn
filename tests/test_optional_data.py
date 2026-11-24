import unittest
from importlib.util import find_spec
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import tv_tensors

from glzn.data.dataset import _BrowserWrapper, iTarDataset
from glzn.data.encoders import JXL, MP4
from glzn.data.seismic import SgzDataset, SgzWindow
from glzn.data.writer import iTarDatasetWriter


class OptionalDataTests(unittest.TestCase):
    @unittest.skipUnless(find_spec("blosc2") is not None, "requires the blosc2 extra")
    def test_b2nd_preserves_float_array_shape_and_dtype(self):
        array = np.linspace(-1, 1, 12 * 18 * 32, dtype=">f4").reshape(12, 18, 32)[::2, :, ::-1]
        with TemporaryDirectory() as directory:
            with iTarDatasetWriter("arrays", directory, folds=["train"]) as writer:
                writer.fold("train").write("flow", {"b2nd": array})
            ds = iTarDataset(
                "arrays", directory, "train", extensions=["b2nd"], serialize=False, verbose=False,
            )
            (decoded,) = ds[0]
            self.assertEqual(decoded.dtype, array.dtype)
            np.testing.assert_array_equal(decoded, array)

    @unittest.skipUnless(find_spec("pyjpegxl") is not None, "requires the jxl extra")
    def test_jxl_round_trips_pil_pixels_and_recovers_original_jpeg(self):
        pixels = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
        image = Image.fromarray(pixels)
        jpeg = BytesIO()
        image.save(jpeg, format="JPEG")

        with TemporaryDirectory() as directory:
            with iTarDatasetWriter("photos", directory, folds=["train"]) as writer:
                writer.fold("train").write("pixels", {"jxl": image})
                writer.fold("train").write("original", {"jxl": jpeg.getvalue()})
            ds = iTarDataset(
                "photos", directory, "train", extensions=["jxl"], serialize=False, verbose=False,
            )
            preview, _, _ = _BrowserWrapper(ds, img_ext="jxl")[0]
            np.testing.assert_array_equal(np.asarray(preview), pixels)
            self.assertEqual(ds[1][0].size, image.size)
            raw = iTarDataset(
                "photos", directory, "train", extensions=["jxl"], serialize=False, verbose=False,
                override_decoders=[("jxl", JXL.restore_jpeg)],
            )
            self.assertEqual(raw[1][0], jpeg.getvalue())

    @unittest.skipUnless(find_spec("torchcodec") is not None, "requires the video extra")
    def test_mp4_selects_frames_in_requested_order(self):
        frames = torch.arange(6, dtype=torch.uint8)[:, None, None, None] * 40
        frames = frames.expand(-1, 3, 24, 32).contiguous()
        try:
            payload = MP4("encode", frame_rate=12)(frames)
        except RuntimeError as exc:
            if "libtorchcodec" in str(exc):
                self.skipTest(f"TorchCodec requires compatible FFmpeg shared libraries: {exc}")
            raise
        expected = MP4.decoder()(payload)
        self.assertEqual(tuple(expected.shape), (6, 3, 24, 32))
        self.assertEqual(expected.dtype, torch.uint8)
        with TemporaryDirectory() as directory:
            with iTarDatasetWriter("clips", directory, folds=["train"]) as writer:
                writer.fold("train").write("clip", {"mp4": payload})
            ds = iTarDataset(
                "clips", directory, "train", extensions=["mp4"], serialize=False, verbose=False,
                override_decoders=[("mp4", MP4("decode", frame_indices=[4, 0, 2]))],
            )
            (selected,) = ds[0]
            self.assertIsInstance(selected, tv_tensors.Video)
            torch.testing.assert_close(selected, expected[[4, 0, 2]])

    @unittest.skipUnless(find_spec("seismic_zfp") is not None, "requires the seismic extra")
    def test_sgz_windows_with_spawn_workers_after_parent_read(self):
        from seismic_zfp.conversion import NumpyConverter
        from seismic_zfp.read import SgzReader

        with TemporaryDirectory() as directory:
            path = Path(directory) / "cube.sgz"
            source = np.arange(8 * 12 * 64, dtype=np.float32).reshape(8, 12, 64)
            NumpyConverter(source).run(str(path), bits_per_voxel=8)
            ds = SgzDataset([
                SgzWindow(path, (0, 1, 8), (4, 4, 16)),
                SgzWindow(path, (2, 3, 16), (4, 4, 16)),
            ])
            with SgzReader(str(path)) as reader:
                expected = reader.read_volume()
            try:
                crop = ds[0]
                np.testing.assert_array_equal(crop, expected[:4, 1:5, 8:24])
                crop.fill(0)
                np.testing.assert_array_equal(ds[0], expected[:4, 1:5, 8:24])
                loader = DataLoader(ds, batch_size=1, num_workers=2, multiprocessing_context="spawn")
                samples = list(loader)
                np.testing.assert_array_equal(samples[0][0].numpy(), expected[:4, 1:5, 8:24])
                np.testing.assert_array_equal(samples[1][0].numpy(), expected[2:6, 3:7, 16:32])
            finally:
                ds.close()
