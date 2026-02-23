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


def browse_dataset(images, labels=None, page_size=24, cols=6, width=128):
    '''Browses a list of PIL images in a paginated grid. Optionally, labels can be shown below each image.

    Parameters
    ----------
    images : list of PIL.Image
        The images to display.
    labels : list of str, optional
        Optional labels to show below each image. Must be the same length as `images`.
    page_size : int, optional
        Number of images to show per page. Default is 24.
    cols : int, optional
        Number of columns in the grid. Default is 6.
    width : int, optional
        Maximum width of each image in pixels. Default is 128.

    Raises
    ------
    ValueError
        If `labels` is provided and its length does not match the length of `images`.
    '''
    n = len(images)
    if labels is not None and len(labels) != n:
        raise ValueError("len(images) must equal len(labels)")

    total_pages = math.ceil(n / page_size)
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
                img_widget = w.Image(
                    value=_pil_to_bytes(images[i], width=width),
                    format="png",
                    width=width,
                )
                if labels is not None:
                    cells.append(
                        w.VBox(
                            [img_widget, w.HTML(str(labels[i]))],
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
        counter.value = f"<b>{p}/{total_pages}</b>"
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

