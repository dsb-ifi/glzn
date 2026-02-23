import math
import json
import base64
import ipywidgets as w

from PIL import Image
from io import BytesIO
from IPython.display import display
from collections.abc import Mapping


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
        meta = {}
        if len(item) > 2:
            if isinstance(item[2], Mapping):
                meta = dict(item[2])
            else:
                meta['stem'] = item[2]
        if len(item) > 3 and 'idx' not in meta:
            meta['idx'] = item[3]
        if len(item) > 4 and 'fid' not in meta:
            meta['fid'] = item[4]
        return image, label, meta
    return item, None, {}


def _coerce_int(value, fallback):
    try:
        return int(value)
    except Exception:
        return fallback


def _selection_entry(idx, label, meta):
    entry = {'idx': _coerce_int(meta.get('idx', idx), idx)}
    if 'stem' in meta and meta['stem'] is not None:
        entry['stem'] = str(meta['stem'])
    if 'fid' in meta and meta['fid'] is not None:
        entry['fid'] = _coerce_int(meta['fid'], meta['fid'])
    if label is not None:
        entry['label'] = str(label)
    return entry


def _key_from_meta(idx, meta):
    stem = meta.get('stem')
    if stem is not None and str(stem) != '':
        return f'stem:{stem}'
    return f'idx:{idx}'


def _extract_upload_bytes(value):
    if isinstance(value, dict):
        values = value.values()
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        values = []

    for obj in values:
        if isinstance(obj, Mapping):
            content = obj.get('content')
            if content is not None:
                return bytes(content)
        else:
            content = getattr(obj, 'content', None)
            if content is not None:
                return bytes(content)
    return None


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
    page_memory = {'all': 1, 'selected': 1}
    item_cache = {}
    selected = {}

    # --- Widget elements ---
    out = w.Output()
    counter = w.HTML()
    status = w.HTML()
    selection_count = w.HTML(value='<b>Selected: 0</b>')
    btn_prev = w.Button(description="◀", layout=w.Layout(width="40px"))
    btn_next = w.Button(description="▶", layout=w.Layout(width="40px"))
    only_selected = w.Checkbox(value=False, description='Only Selected')

    # BoundedIntText ensures the user can't type below 1 or above total_pages
    page_input = w.BoundedIntText(
        value=1,
        min=1,
        max=total_pages,
        step=1,
        layout=w.Layout(width="60px")
    )
    btn_go = w.Button(description="Go", layout=w.Layout(width="40px"))
    btn_export = w.Button(description="Export Selection")
    btn_import = w.Button(description="Import Selection")
    import_upload = w.FileUpload(accept='.json', multiple=False)
    download_link = w.HTML()

    def _item(i):
        if i not in item_cache:
            img, label, meta = _parse_item(items[i])
            if not isinstance(img, Image.Image):
                raise TypeError(
                    f'Expected PIL.Image.Image at index {i}, got {type(img).__name__}.'
                )
            key = _key_from_meta(i, meta)
            item_cache[i] = {
                'img': img,
                'label': label,
                'meta': meta,
                'key': key,
            }
        return item_cache[i]

    def _all_visible_indices():
        if not only_selected.value:
            return list(range(n))
        idxs = []
        seen = set()
        for rec in selected.values():
            idx = _coerce_int(rec.get('idx', -1), -1)
            if 0 <= idx < n and idx not in seen:
                seen.add(idx)
                idxs.append(idx)
        idxs.sort()
        return idxs

    def _update_selection_count():
        selection_count.value = f'<b>Selected: {len(selected)}</b>'

    def _set_selected(i, is_selected):
        it = _item(i)
        key = it['key']
        if is_selected:
            selected[key] = _selection_entry(i, it['label'], it['meta'])
        else:
            selected.pop(key, None)
        _update_selection_count()

    def render(p):
        """Redraw the grid for page p (1‐based)."""
        nonlocal page
        mode = 'selected' if only_selected.value else 'all'
        visible_indices = _all_visible_indices()
        n_visible = len(visible_indices)

        total_pages_local = max(1, math.ceil(max(1, n_visible) / page_size))
        page = min(max(1, p), total_pages_local)
        page_memory[mode] = page
        page_input.max = total_pages_local

        start = (page - 1) * page_size
        end = min(start + page_size, n_visible)
        page_indices = visible_indices[start:end]

        # Clear and build a new GridBox
        with out:
            out.clear_output(wait=True)
            cells = []
            for i in page_indices:
                it = _item(i)
                img = it['img']
                label = it['label']
                is_selected = it['key'] in selected

                img_widget = w.Image(
                    value=_pil_to_bytes(img, width=width),
                    format="png",
                    width=width,
                )

                toggle = w.ToggleButton(
                    value=is_selected,
                    description='Selected' if is_selected else 'Select',
                    icon='check' if is_selected else '',
                    layout=w.Layout(width='100px')
                )

                def _on_toggle(change, idx=i, button=toggle):
                    if change['name'] != 'value':
                        return
                    is_sel = bool(change['new'])
                    _set_selected(idx, is_sel)
                    button.description = 'Selected' if is_sel else 'Select'
                    button.icon = 'check' if is_sel else ''
                    if only_selected.value and not is_sel:
                        render(page)

                toggle.observe(_on_toggle, names='value')

                card_children = [img_widget]
                if label is not None:
                    card_children.append(w.HTML(str(label)))
                card_children.append(toggle)

                border = '2px solid #4a90e2' if is_selected else '1px solid #ddd'
                cells.append(
                    w.VBox(
                        card_children,
                        layout=w.Layout(align_items="center", border=border, padding='4px'),
                    )
                )

            if len(cells) == 0:
                display(w.HTML('<i>No items to display.</i>'))
            else:
                grid = w.GridBox(
                    cells,
                    layout=w.Layout(
                        grid_template_columns=f"repeat({cols}, 150px)",
                        grid_gap="8px",
                    ),
                )
                display(grid)

        # Update the counter and the page_input if needed
        if n_visible == 0:
            counter.value = "<b>0 of 0</b>"
        else:
            display_start = start + 1
            display_end = end
            counter.value = f"<b>{display_start}–{display_end} of {n_visible}</b>"
        if page_input.value != page:
            page_input.value = page

        _update_selection_count()

    def on_prev(_):
        nonlocal page
        if page > 1:
            page -= 1
            render(page)

    def on_next(_):
        nonlocal page
        max_page = max(1, _coerce_int(page_input.max, 1))
        if page < max_page:
            page += 1
            render(page)

    def on_go(_):
        nonlocal page
        requested = page_input.value
        # BoundedIntText already enforces 1 ≤ requested ≤ total_pages
        if requested != page:
            page = requested
            render(page)

    def on_only_selected(change):
        if change['name'] != 'value':
            return
        mode = 'selected' if bool(change['new']) else 'all'
        render(page_memory.get(mode, 1))

    def on_export(_):
        ds = getattr(items, 'dataset', None)
        fold_obj = getattr(ds, 'fold', None)
        payload = {
            'version': 1,
            'dataset': getattr(ds, 'dataset', None),
            'fold': getattr(fold_obj, 'fold', None),
            'extensions': list(getattr(ds, 'extensions', [])) if ds is not None else None,
            'selected': list(selected.values()),
        }
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        encoded = base64.b64encode(content.encode('utf8')).decode('ascii')
        download_link.value = (
            '<a download="selection.json" '
            f'href="data:application/json;base64,{encoded}">Download selection.json</a>'
        )
        status.value = f'<i>Prepared export with {len(selected)} selected entries.</i>'

    def on_import(_):
        raw = _extract_upload_bytes(import_upload.value)
        if raw is None:
            status.value = '<i>No JSON file uploaded.</i>'
            return

        try:
            parsed = json.loads(raw.decode('utf8'))
        except Exception as exc:
            status.value = f'<i>Import failed: invalid JSON ({exc}).</i>'
            return

        if isinstance(parsed, dict):
            records = parsed.get('selected', [])
        elif isinstance(parsed, list):
            records = parsed
        else:
            status.value = '<i>Import failed: JSON must be a list or an object with `selected`.</i>'
            return

        stem_to_index = {}
        for i in range(n):
            it = _item(i)
            stem = it['meta'].get('stem')
            if stem is not None and stem not in stem_to_index:
                stem_to_index[stem] = i

        imported = 0
        skipped = 0
        for rec in records:
            stem = None
            idx = None
            extra = {}

            if isinstance(rec, str):
                stem = rec
            elif isinstance(rec, Mapping):
                stem = rec.get('stem')
                idx = rec.get('idx')
                extra = dict(rec)
            else:
                skipped += 1
                continue

            target_idx = None
            if stem is not None:
                target_idx = stem_to_index.get(stem)
                if target_idx is None:
                    skipped += 1
                    continue
            elif idx is not None:
                idx = _coerce_int(idx, -1)
                if 0 <= idx < n:
                    target_idx = idx
                else:
                    skipped += 1
                    continue
            else:
                skipped += 1
                continue

            it = _item(target_idx)
            key = it['key']
            merged = _selection_entry(target_idx, it['label'], it['meta'])
            merged.update({k: v for k, v in extra.items() if k not in {'stem', 'idx', 'fid', 'label'}})
            selected[key] = merged
            imported += 1

        status.value = f'<i>Imported {imported} entries; skipped {skipped} missing/invalid entries.</i>'
        render(page)

    btn_prev.on_click(on_prev)
    btn_next.on_click(on_next)
    btn_go.on_click(on_go)
    btn_export.on_click(on_export)
    btn_import.on_click(on_import)
    only_selected.observe(on_only_selected, names='value')

    # Initial draw
    render(page)

    # Layout: Prev ◀  Next ▶   [page_input] Go   (counter)
    controls = w.HBox([
        btn_prev, btn_next,
        page_input, btn_go,
        counter,
        selection_count,
        only_selected,
    ])
    io_controls = w.HBox([btn_export, btn_import, import_upload, download_link])
    display(w.VBox([controls, io_controls, status, out]))

