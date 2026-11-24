"""Notebook frontend for the dataset gallery."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from html import escape

import ipywidgets as w
from IPython.display import display

from .imgbrowser import _Gallery


def _upload_bytes(upload: w.FileUpload) -> bytes | None:
    value = upload.value
    files = value.values() if isinstance(value, Mapping) else value
    for file in files:
        content = file.get("content")
        if content is not None:
            return bytes(content)
    return None


def show_gallery(gallery: _Gallery, cols: int) -> None:
    page_memory = {False: 1, True: 1}
    out = w.Output()
    counter = w.HTML()
    status = w.HTML()
    selected_count = w.HTML()
    download = w.HTML()
    only_selected = w.Checkbox(value=False, description="Only Selected")
    page_input = w.BoundedIntText(value=1, min=1, max=1, layout=w.Layout(width="80px"))
    prev = w.Button(description="◀", layout=w.Layout(width="40px"))
    next_page = w.Button(description="▶", layout=w.Layout(width="40px"))
    go = w.Button(description="Go", layout=w.Layout(width="40px"))
    export = w.Button(description="Export Selection")
    upload = w.FileUpload(accept=".json", multiple=False)
    import_button = w.Button(description="Import Selection")
    clear = w.Button(description="Clear Selection")

    def render(number: int) -> None:
        mode = only_selected.value
        page = gallery.page(number, mode)
        page_memory[mode] = page.number
        page_input.max = page.pages
        page_input.value = page.number
        prev.disabled = page.number == 1
        next_page.disabled = page.number == page.pages
        start = page.start + 1 if page.items else 0
        counter.value = f"<b>{start}–{page.start + len(page.items)} of {page.total}</b>"
        selected_count.value = f"<b>Selected: {page.selection_count}</b>"
        download.value = ""

        cards = []
        for item, selected in page.items:
            image = w.Image(value=item.thumbnail, format="png", width=gallery.width)
            toggle = w.ToggleButton(value=selected, description="Selected" if selected else "Select")

            def change_selection(change, index=item.index, button=toggle):
                value = bool(change["new"])
                count = gallery.select(index, value)
                selected_count.value = f"<b>Selected: {count}</b>"
                download.value = ""
                button.description = "Selected" if value else "Select"
                if only_selected.value and not value:
                    render(page.number)

            toggle.observe(change_selection, names="value")
            children: list[w.Widget] = [image]
            if item.label is not None:
                children.append(w.HTML(escape(item.label)))
            children.append(toggle)
            cards.append(w.VBox(children, layout=w.Layout(align_items="center", padding="4px")))

        with out:
            out.clear_output(wait=True)
            if cards:
                display(w.GridBox(cards, layout=w.Layout(
                    grid_template_columns=f"repeat({cols}, {gallery.width + 24}px)",
                    grid_gap="8px",
                )))
            else:
                display(w.HTML("<i>No items to display.</i>"))

    def current_page() -> int:
        return page_memory[only_selected.value]

    def on_export(_):
        contents, skipped = gallery.export_selection()
        encoded = base64.b64encode(contents).decode("ascii")
        download.value = (
            '<a download="selection.json" '
            f'href="data:application/json;base64,{encoded}">Download selection.json</a>'
        )
        status.value = f"Skipped {skipped} selected items without stems." if skipped else ""

    def on_import(_):
        data = _upload_bytes(upload)
        if data is None:
            status.value = "Choose a selection JSON file."
            return
        try:
            imported, skipped = gallery.import_selection(data)
        except (ValueError, OSError) as exc:
            status.value = escape(str(exc))
            return
        status.value = f"Imported {imported}, skipped {skipped}."
        render(current_page())

    def on_clear(_):
        gallery.clear()
        status.value = ""
        render(current_page())

    prev.on_click(lambda _: render(current_page() - 1))
    next_page.on_click(lambda _: render(current_page() + 1))
    go.on_click(lambda _: render(page_input.value))
    only_selected.observe(lambda _: render(current_page()), names="value")
    export.on_click(on_export)
    import_button.on_click(on_import)
    clear.on_click(on_clear)
    render(1)
    display(w.VBox([
        w.HBox([prev, next_page, page_input, go, counter, selected_count, only_selected]),
        w.HBox([export, upload, import_button, clear, download]),
        status,
        out,
    ]))
