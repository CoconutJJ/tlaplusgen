"""
gpucode_adapter.py  --  Thin convenience wrapper around gpucode-analyzer's
SASS parser + slicer.

Loads a SASS file (cuobjdump-style; what gpucode-analyzer consumes natively
and what the regalloc-dataset ships with), auto-discovers a meta.json for
BRX indirect-branch targets if one sits next to the file, and exposes a
slice helper that returns a raw ``gpucodeanalyzer.generic_cfg.CFG``.

Usage
-----
    from sass.gpucode_adapter import parse_file, slice_cfg
"""

from __future__ import annotations

import os as _os
from typing import Dict

from gpucodeanalyzer.gpucodeanalyzer.generic_cfg import CFG as GCA_CFG
from gpucodeanalyzer.gpucodeanalyzer.isa.sass.sass import SASSFile
from gpucodeanalyzer.gpucodeanalyzer.slice import Slicer, get_labels


def _load_metadata(path: str, metadata: dict | None) -> dict | None:
    """Resolve the metadata dict for a SASS file.

    If ``metadata`` is explicitly supplied, return it as-is. Otherwise, look
    for a JSON metadata file adjacent to ``path`` using the conventions in
    the regalloc-dataset:

        foo.sass       → foo.sass.json, foo.json, meta.json
    """
    import json

    if metadata is not None:
        return metadata

    candidates = [
        path + ".json",
        _os.path.splitext(path)[0] + ".json",
        _os.path.join(_os.path.dirname(path), "meta.json"),
    ]
    for cand in candidates:
        if _os.path.isfile(cand):
            with open(cand, "r") as f:
                return json.load(f)
    return None


def parse_file(path: str, metadata: dict | None = None) -> Dict[str, GCA_CFG]:
    """Parse a SASS file and return function_name → built gpucode-analyzer CFG."""
    sass_file = SASSFile(path, metadata=_load_metadata(path, metadata))

    result: Dict[str, GCA_CFG] = {}
    if hasattr(sass_file, "codes") and sass_file.codes:
        for fn_name in sass_file.codes:
            cfg = GCA_CFG(sass_file, fn_name=fn_name)
            cfg.build()
            result[fn_name] = cfg
    else:
        cfg = GCA_CFG(sass_file)
        cfg.build()
        result["unknown_kernel"] = cfg
    return result


def slice_cfg(gca_cfg: GCA_CFG, pattern: str) -> GCA_CFG:
    """Slice ``gca_cfg`` on a regex of instruction opcodes and return the skeleton CFG."""
    labels = get_labels(gca_cfg, [f"re:{pattern}"])
    slicer = Slicer(gca_cfg)
    slicer.slice(labels)
    return slicer.get_skeleton_cfg()
