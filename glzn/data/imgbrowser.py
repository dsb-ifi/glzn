"""Paginated image galleries for indexable datasets."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from threading import Lock
from typing import Literal, Protocol

import numpy as np
from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import to_pil_image


class _GallerySource(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int, /) -> object: ...


def _parse_item(item):
    if not isinstance(item, tuple):
        return item, None, None
    if not item:
        raise ValueError("A gallery item tuple must contain an image.")

    image = item[0]
    label = item[1] if len(item) > 1 else None
    stem = item[2] if len(item) > 2 else None
    if isinstance(stem, Mapping):
        stem = stem.get("stem")
    return image, label, str(stem) if stem is not None and str(stem).strip() else None


def _mask_overlay(image: object, mask: object, width: int) -> Image.Image:
    if isinstance(image, Tensor):
        if image.is_floating_point():
            image = image.clamp(0, 1)
        image = to_pil_image(image)
    if not isinstance(image, Image.Image):
        raise TypeError("Mask overlays require a PIL image or an image tensor.")

    labels = mask.detach().cpu().numpy() if isinstance(mask, Tensor) else np.asarray(mask)
    if labels.ndim == 3 and labels.shape[0] == 1:
        labels = labels[0]
    if labels.shape != (image.height, image.width):
        raise ValueError(
            f"Mask shape {labels.shape} does not match image shape {(image.height, image.width)}."
        )
    if not np.issubdtype(labels.dtype, np.integer) and labels.dtype != np.bool_:
        raise ValueError("Mask overlays require integer class IDs.")

    base = image.convert("RGBA")
    base.thumbnail((width, width))
    if base.size != image.size:
        # Select the closest input pixel to each thumbnail pixel center.
        y = ((2 * np.arange(base.height) + 1) * image.height) // (2 * base.height)
        x = ((2 * np.arange(base.width) + 1) * image.width) // (2 * base.width)
        labels = labels[np.ix_(y, x)]

    # The same ID has the same color in every sample, including wide IDs.
    colors = labels.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
    rgba = np.empty((*labels.shape, 4), dtype=np.uint8)
    for channel, shift in enumerate((0, 16, 32)):
        rgba[..., channel] = 64 + (colors >> shift) % 192
    rgba[..., 3] = np.where(labels == 0, 0, 112)
    return Image.alpha_composite(base, Image.fromarray(rgba)).convert("RGB")


def _compare_mask_overlays(stored: Image.Image, transformed: Image.Image) -> Image.Image:
    x = stored.width + 2
    view = Image.new("RGB", (x + transformed.width, max(stored.height, transformed.height)), "#333")
    view.paste(stored, (0, 0))
    view.paste(transformed, (x, 0))
    return view


@dataclass(frozen=True)
class _GalleryItem:
    index: int
    label: str | None
    stem: str | None
    thumbnail: bytes

    @property
    def key(self) -> str | int:
        return self.stem if self.stem is not None else self.index


@dataclass(frozen=True)
class _GalleryPage:
    number: int
    pages: int
    start: int
    total: int
    items: list[tuple[_GalleryItem, bool]]
    selection_count: int


class _Gallery:
    def __init__(
        self,
        items: _GallerySource,
        page_size: int,
        width: int,
        stem_lookup: Callable[[set[str]], Mapping[str, int]] | None = None,
    ):
        if page_size < 1 or width < 1:
            raise ValueError("page_size and width must be positive.")
        self.items = items
        self.page_size = page_size
        self.width = width
        self._stem_lookup = stem_lookup
        self._selection: dict[str | int, int] = {}
        self._lock = Lock()
        self.item = lru_cache(maxsize=max(64, 2 * page_size))(self._load_item)

    def _load_item(self, index: int) -> _GalleryItem:
        if not 0 <= index < len(self.items):
            raise IndexError(f"Gallery index {index} is out of range.")
        image, label, stem = _parse_item(self.items[index])
        if isinstance(image, Tensor):
            image = to_pil_image(image)
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected a PIL image or tensor at index {index}.")
        thumbnail = image.copy()
        if thumbnail.mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "I;16"}:
            thumbnail = thumbnail.convert("RGB")
        thumbnail.thumbnail((self.width, self.width))
        with BytesIO() as buffer:
            thumbnail.save(buffer, format="PNG")
            return _GalleryItem(index, None if label is None else str(label), stem, buffer.getvalue())

    def page(self, number: int, selected_only: bool = False) -> _GalleryPage:
        with self._lock:
            visible = sorted(self._selection.values()) if selected_only else range(len(self.items))
        total = len(visible)
        pages = max(1, (total + self.page_size - 1) // self.page_size)
        number = min(max(1, number), pages)
        start = (number - 1) * self.page_size
        items = [self.item(index) for index in visible[start:start + self.page_size]]
        with self._lock:
            return _GalleryPage(
                number, pages, start, total,
                [(item, item.key in self._selection) for item in items], len(self._selection),
            )

    def select(self, index: int, value: bool) -> int:
        key = self.item(index).key
        with self._lock:
            if value:
                self._selection[key] = index
            else:
                self._selection.pop(key, None)
            return len(self._selection)

    def clear(self) -> None:
        with self._lock:
            self._selection.clear()

    def export_selection(self) -> tuple[bytes, int]:
        with self._lock:
            stems = sorted(key for key in self._selection if isinstance(key, str))
            skipped = len(self._selection) - len(stems)
        return json.dumps(stems, ensure_ascii=False, indent=2).encode("utf-8"), skipped

    def import_selection(self, data: bytes) -> tuple[int, int]:
        parsed = json.loads(data)
        if not isinstance(parsed, list) or not all(isinstance(stem, str) for stem in parsed):
            raise ValueError("Selection JSON must be a list of stem strings.")
        wanted = {stem.strip() for stem in parsed if stem.strip()}
        found: Mapping[str, int] = {}
        if wanted:
            if self._stem_lookup is not None:
                found = self._stem_lookup(wanted)
            else:
                matches = {}
                for index in range(len(self.items)):
                    _, _, stem = _parse_item(self.items[index])
                    if stem is not None and stem in wanted:
                        matches.setdefault(stem, index)
                        if len(matches) == len(wanted):
                            break
                found = matches
        with self._lock:
            self._selection = {stem: index for stem, index in found.items() if stem in wanted}
            imported = len(self._selection)
        return imported, len(wanted) - imported


def browse_dataset(
    items: _GallerySource,
    page_size: int = 24,
    cols: int = 6,
    width: int = 128,
    *,
    backend: Literal["notebook", "web"] = "notebook",
    port: int = 0,
    open_browser: bool = False,
    stem_lookup: Callable[[set[str]], Mapping[str, int]] | None = None,
) -> None:
    """Display a paginated image gallery with selection import and export.

    Parameters
    ----------
    items : indexable collection
        Each item is a PIL image, a tensor, or a tuple starting with an image
        and an optional label.
        The third element can be a stem string or a mapping with a ``stem`` entry.
    page_size : int, optional
        Number of samples per page, by default 24.
    cols : int, optional
        Number of columns in the grid, by default 6.
    width : int, optional
        Maximum width and height of each thumbnail in pixels, by default 128.
    backend : {"notebook", "web"}, optional
        Display widgets inline by default. The web backend prints a loopback
        URL and serves the gallery until interrupted.
    port : int, optional
        Web port. The default of zero lets the OS choose an available port.
    open_browser : bool, optional
        Open the web gallery in the default browser, by default False.
    stem_lookup : callable or None, optional
        Map a set of stems to their indices in ``items``. By default, imports
        find stems by scanning the item metadata.
    """
    if cols < 1:
        raise ValueError("cols must be positive.")
    gallery = _Gallery(items, page_size, width, stem_lookup)
    if backend == "notebook":
        try:
            from ._imgbrowser_notebook import show_gallery
        except ModuleNotFoundError as exc:
            if exc.name not in {"ipywidgets", "IPython"}:
                raise
            raise ImportError("Install GLZN with the notebook extra to use this backend.") from exc

        show_gallery(gallery, cols)
    elif backend == "web":
        from ._imgbrowser_web import serve_gallery

        serve_gallery(gallery, cols, port, open_browser=open_browser)
    else:
        raise ValueError(f"Unknown gallery backend {backend!r}.")
