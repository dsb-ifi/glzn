import math
import ipywidgets as w

from PIL import Image
from io import BytesIO
from IPython.display import display


def _pil_to_bytes(img:Image.Image, fmt="png", width=None):
    buf = BytesIO()
    copy = img.copy()
    if width is None:
        width = copy.width
    copy.thumbnail((width, width))
    copy.save(buf, format=fmt)
    return buf.getvalue()


def _parse_item(item):
    if isinstance(item, tuple):
        if len(item) == 0:
            raise ValueError('Dataset item tuple must contain at least one element (image).')
        image = item[0]
        label = item[1] if len(item) > 1 else None
        return image, label
    return item, None


def browse_dataset(items, page_size=24, cols=6, width=128):
    '''Browses an indexable iterable in a paginated grid.

    Each item must be either:
    - a PIL image
    - a tuple where the first element is a PIL image and the second element is
      used as label text

    Parameters
    ----------
    items : sequence
        Sequence of samples. Each sample is either a PIL image or a tuple of
        `(image, label, ...)`.
    page_size : int, optional
        Number of images to show per page. Default is 24.
    cols : int, optional
        Number of columns in the grid. Default is 6.
    width : int, optional
        Maximum width of each image in pixels. Default is 128.
    '''
    n = len(items)

    total_pages = max(1, math.ceil(n / page_size))
    page = 1  # 1‐based page index

    # --- Widget elements ---
    out = w.Output()
    counter = w.HTML()
    btn_prev = w.Button(description="◀", layout=w.Layout(width="40px"))
    btn_next = w.Button(description="▶", layout=w.Layout(width="40px"))

    # BoundedIntText ensures the user can't type below 1 or above total_pages
    page_input = w.BoundedIntText(
        value=1,
        min=1,
        max=total_pages,
        step=1,
        layout=w.Layout(width="60px")
    )
    btn_go = w.Button(description="Go", layout=w.Layout(width="40px"))

    def render(p):
        """Redraw the grid for page p (1‐based)."""
        start = (p - 1) * page_size
        end = min(start + page_size, n)

        # Clear and build a new GridBox
        with out:
            out.clear_output(wait=True)
            cells = []
            for i in range(start, end):
                img, label = _parse_item(items[i])
                if not isinstance(img, Image.Image):
                    raise TypeError(
                        f'Expected PIL.Image.Image at index {i}, got {type(img).__name__}.'
                    )
                img_widget = w.Image(
                    value=_pil_to_bytes(img, width=width),
                    format="png",
                    width=width,
                )
                if label is not None:
                    cells.append(
                        w.VBox(
                            [img_widget, w.HTML(str(label))],
                            layout=w.Layout(align_items="center"),
                        )
                    )
                else:
                    cells.append(img_widget)

            grid = w.GridBox(
                cells,
                layout=w.Layout(
                    grid_template_columns=f"repeat({cols}, 150px)",
                    grid_gap="8px",
                ),
            )
            display(grid)

        # Update the counter and the page_input if needed
        if n == 0:
            counter.value = "<b>0 of 0</b>"
        else:
            display_start = start + 1
            display_end = end
            counter.value = f"<b>{display_start}–{display_end} of {n}</b>"
        if page_input.value != p:
            page_input.value = p

    def on_prev(_):
        nonlocal page
        if page > 1:
            page -= 1
            render(page)

    def on_next(_):
        nonlocal page
        if page < total_pages:
            page += 1
            render(page)

    def on_go(_):
        nonlocal page
        requested = page_input.value
        # BoundedIntText already enforces 1 ≤ requested ≤ total_pages
        if requested != page:
            page = requested
            render(page)

    btn_prev.on_click(on_prev)
    btn_next.on_click(on_next)
    btn_go.on_click(on_go)

    # Initial draw
    render(page)

    # Layout: Prev ◀  Next ▶   [page_input] Go   (counter)
    controls = w.HBox([
        btn_prev, btn_next,
        page_input, btn_go,
        counter
    ])
    display(w.VBox([controls, out]))

