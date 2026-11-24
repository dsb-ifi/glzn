"""Open an existing iTar dataset in the web gallery."""

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from .dataset import iTarDataset


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m glzn.data._imgbrowser_cli",
        description="Browse an indexed iTar dataset in a web browser.",
    )
    parser.add_argument("dataset", type=Path, help="Directory containing the shards and their indexes.")
    parser.add_argument("--fold", default="train", help="Fold to browse (default: train).")
    parser.add_argument("--image-ext", default="jpg", help="Image field (default: jpg).")
    parser.add_argument("--label-ext", help="Label field to display, such as cls.")
    parser.add_argument("--mask-ext", help="Mask field to overlay, such as seg8.")
    parser.add_argument("--port", type=int, default=0, help="Loopback port (default: automatically assigned).")
    parser.add_argument("--no-open", action="store_true", help="Print the URL without opening a browser.")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535.")

    extensions = [ext for ext in (args.image_ext, args.label_ext, args.mask_ext) if ext is not None]

    try:
        root = args.dataset.expanduser().resolve()
        dataset = iTarDataset(
            root.name, root.parent, args.fold, extensions=extensions,
            parse_if_missing=False, serialize=False, verbose=False,
        )
        dataset.browse(
            img_ext=args.image_ext,
            lab_ext=args.label_ext or "cls",
            backend="web",
            port=args.port,
            open_browser=not (args.no_open or os.environ.get("SSH_CONNECTION")),
            mask_ext=args.mask_ext,
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
