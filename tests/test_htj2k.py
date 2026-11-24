import unittest
from importlib.util import find_spec
from tempfile import TemporaryDirectory

import numpy as np

from glzn.data.dataset import iTarDataset
from glzn.data.encoders import HTJ2K
from glzn.data.writer import iTarDatasetWriter


class HTJ2KTests(unittest.TestCase):
    @unittest.skipUnless(find_spec("blosc2_htj2k") is not None, "requires the htj2k extra")
    def test_lossless_uint16_tile_and_uint8_rgb_shard_round_trip(self):
        tile = np.linspace(0, 65535, 64 * 96, dtype=np.uint16).reshape(64, 96)
        channel = (tile >> 8).astype(np.uint8)
        rgb = np.stack((channel, channel // 2, 255 - channel), axis=-1)[:, ::-1]
        self.assertLess(len(HTJ2K.encoder()(tile)), tile.nbytes)

        with TemporaryDirectory() as directory:
            with iTarDatasetWriter(
                "tiles", directory, folds=["train"],
                override_encoders=[("b2nd", HTJ2K.encoder())],
            ) as writer:
                writer.fold("train").write("tile", {"b2nd": tile.astype(">u2")})
                writer.fold("train").write("rgb", {"b2nd": rgb})
            ds = iTarDataset(
                "tiles", directory, "train", extensions=["b2nd"], serialize=False, verbose=False
            )
            for i, expected in enumerate((tile, rgb)):
                with self.subTest(dtype=expected.dtype):
                    (decoded,) = ds[i]
                    self.assertEqual(decoded.dtype, expected.dtype)
                    np.testing.assert_array_equal(decoded, expected)

    def test_rejects_floating_point_images(self):
        with self.assertRaisesRegex(TypeError, "uint8 or uint16"):
            HTJ2K.encoder()(np.zeros((16, 16), dtype=np.float32))
