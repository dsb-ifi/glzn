from __future__ import annotations
import os, re
from dataclasses import dataclass

def braceexpand(input:str, root:str) -> tuple[str,...]:
    """Performs brace expansion for a string given a root.

    Parameters
    ----------
    input_string : str
        Brace formatted string, e.g., '<pf><int>..<int><suf>'.
    root : str
        Root folder for the dataset.

    Returns
    -------
    Tuple[str, ...]
        Tuple of joined paths from brace expansion.
    """
    m = re.search(r'(.*)\{(\d+)\.\.(\d+)\}(.*)', input)
    if m is None:
        raise ValueError(f"Invalid brace expansion {input}")
    prefix, start, end, suffix = (
        m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    )
    num_digits = len(m.group(2))
    format_str = f"{prefix}{{:0{num_digits}d}}{suffix}"
    return tuple([
        os.path.join(root, format_str.format(i)) 
        for i in range(start, end + 1)
    ])

@dataclass(frozen=True, slots=True)
class StemHelper:
    rawstr: str

    @property
    def stem(self) -> str:
        base = self.rawstr.rsplit('/', 1)[-1]
        stem, _, _ = base.partition('.')
        return stem

    @property
    def suffix(self) -> str:
        base = self.rawstr.rsplit('/', 1)[-1]
        _, dot, ext = base.partition('.')
        if not dot:
            return ''
        return '.' + ext.lower()

    @property
    def suffix_no_dot(self) -> str:
        s = self.suffix
        return s[1:] if s else ''
    
    @classmethod
    def from_hdr(cls, buf:bytes) -> "StemHelper":
        s = buf[:100].split(b'\0', 1)[0].decode('utf-8', 'replace')
        return cls(s)

    def __str__(self) -> str:
        return self.rawstr

    def __repr__(self):
        return f"FileName({self.rawstr!r})"


def stripext(ext:str) -> str:
    return ext[1:] if ext.startswith('.') else ext