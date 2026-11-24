import json
import unittest
from http import HTTPStatus
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.parse import urlsplit

import numpy as np
from PIL import Image
from torchvision.transforms import v2

from glzn.data._imgbrowser_web import _create_server
from glzn.data.dataset import _BrowserWrapper, iTarDataset
from glzn.data.imgbrowser import _Gallery
from glzn.data.writer import iTarDatasetWriter


class GalleryTests(unittest.TestCase):
    def test_mask_overlay_compares_stored_and_transformed_samples(self):
        with TemporaryDirectory() as directory:
            image = Image.new("RGB", (6, 4), "white")
            image.format = "JPEG"
            mask = np.zeros((4, 6), dtype=np.uint32)
            mask[:, :2] = 0xFEDCBA98
            with iTarDatasetWriter("masks", directory, folds=["train"]) as writer:
                writer.fold("train").write("pair", {"jpg": image, "seg32": mask, "cls": 3})
            dataset = iTarDataset(
                "masks", directory, "train",
                jpeg_as_tensor=True, mask_as_tensor=True, verbose=False,
            )
            view = _BrowserWrapper(dataset, mask_ext="seg32", width=14)
            overlay, _, _ = view[0]
            self.assertEqual(overlay.size, (6, 4))
            self.assertEqual(overlay.getpixel((5, 0)), (255, 255, 255))
            self.assertNotEqual(overlay.getpixel((0, 0)), (255, 255, 255))

            dataset.map_group(v2.RandomHorizontalFlip(p=1), indices=[0, 1])
            with patch.object(
                dataset.retriever, "from_row", wraps=dataset.retriever.from_row
            ) as read:
                preview, label, meta = view[0]
                self.assertEqual(read.call_count, 3)
            self.assertEqual((label, meta["stem"]), (3, "pair"))
            self.assertEqual(preview.size, (14, 4))
            np.testing.assert_array_equal(np.asarray(preview)[:, :6], np.asarray(overlay))
            np.testing.assert_array_equal(np.asarray(preview)[:, 8:], np.asarray(overlay)[:, ::-1])

    def test_thumbnail_disconnect_does_not_send_a_second_response(self):
        gallery = _Gallery([Image.new("RGB", (16, 16))], page_size=1, width=16)
        with patch("glzn.data._imgbrowser_web.ThreadingHTTPServer") as server:
            server.return_value.server_port = 8765
            _, url = _create_server(gallery, cols=1, port=0)
            handler_type = server.call_args.args[1]

        handler = handler_type.__new__(handler_type)
        handler.path = urlsplit(url).path + "thumbnail/0"
        with (
            patch.object(handler, "send_response") as send_response,
            patch.object(handler, "send_header"),
            patch.object(handler, "end_headers", side_effect=BrokenPipeError),
        ):
            handler.do_GET()

        send_response.assert_called_once_with(HTTPStatus.OK)
        self.assertTrue(handler.close_connection)

    def test_page_reads_only_visible_items_and_caches_thumbnails(self):
        class Source:
            def __init__(self):
                self.reads = []

            def __len__(self):
                return 10**9

            def __getitem__(self, index):
                self.reads.append(index)
                return Image.new("RGB", (32, 24)), index, f"sample{index}"

        source = Source()
        gallery = _Gallery(source, page_size=4, width=16)
        page = gallery.page(3)
        self.assertEqual(source.reads, [8, 9, 10, 11])
        self.assertEqual(page.total, len(source))
        gallery.select(9, True)
        selected = gallery.page(1, selected_only=True)
        self.assertEqual([item.index for item, _ in selected.items], [9])
        self.assertEqual(source.reads, [8, 9, 10, 11])
        self.assertEqual(json.loads(gallery.export_selection()[0]), ["sample9"])

    def test_itar_import_resolves_visible_positions_after_shuffling(self):
        with TemporaryDirectory() as directory:
            image = Image.new("RGB", (16, 16))
            image.format = "JPEG"
            with iTarDatasetWriter("images", directory, folds=["train"]) as writer:
                for i in range(8):
                    writer.fold("train").write(f"sample{i}", {"jpg": image, "cls": i})
            dataset = iTarDataset(
                "images", directory, "train", jpeg_as_tensor=True, verbose=False
            )
            dataset.set_shuffle().set_distributed(rank=1, world_size=2, drop_last=True)
            view = _BrowserWrapper(dataset)
            expected = {str(view[i][2]["stem"]): i for i in range(len(view))}
            stems = sorted(expected)
            gallery = _Gallery(view, page_size=8, width=16, stem_lookup=view.resolve_stems)
            with patch.object(
                dataset.retriever, "hdrname", wraps=dataset.retriever.hdrname
            ) as hdrname:
                payload = json.dumps(stems + ["missing", stems[0]]).encode()
                self.assertEqual(gallery.import_selection(payload), (4, 1))
                self.assertEqual(hdrname.call_count, 4)
            page = gallery.page(1, selected_only=True)
            self.assertEqual({item.stem: item.index for item, _ in page.items}, expected)
            self.assertTrue(all(selected for _, selected in page.items))
            with self.assertRaisesRegex(ValueError, "list of stem strings"):
                gallery.import_selection(b'{"stem": "sample0"}')
            self.assertEqual(json.loads(gallery.export_selection()[0]), stems)
