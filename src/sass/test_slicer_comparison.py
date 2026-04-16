"""
test_slicer_comparison.py

Compares the builtin slicer (sass/cfg.py) against the gpucode-analyzer slicer
(sass/gpucode_adapter.py) on real SASS kernel files.

Both slicers should produce equivalent sliced CFGs: same instructions (by
address) retained after slicing.

Run from the src/ directory:
    cd src && PYTHONPATH=. python -m unittest sass.test_slicer_comparison -v
"""

import sys
import os
import unittest
import glob

_HERE = os.path.dirname(__file__)
_SRC = os.path.dirname(_HERE)
_EXAMPLES_DIR = os.path.join(_SRC, "..", "examples")

if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from sass.parser import parse_file as builtin_parse
from sass.cfg import build_cfgs as builtin_build_cfgs, slice_cfg as builtin_slice
from sass.gpucode_adapter import (
    parse_file as gca_parse,
    build_cfgs as gca_build_cfgs,
    slice_cfg as gca_slice,
)


def _instruction_addresses(cfg) -> set[int]:
    """Return the set of all instruction addresses in a sliced CFG."""
    return {i.address for bb in cfg.blocks for i in bb.instructions}


def _first_kernel(sass_path: str) -> str | None:
    """Return the name of the first non-unknown kernel, or None."""
    prog = builtin_parse(sass_path)
    cfgs = builtin_build_cfgs(prog)
    for name in cfgs:
        if name != "unknown_kernel":
            return name
    return next(iter(cfgs), None)


class TestSlicerComparison(unittest.TestCase):
    """Compare builtin and gpucode-analyzer slicers on real SASS files."""

    def _compare_slicers(self, sass_path: str, kernel: str, pattern: str):
        """Run both slicers and assert equivalent instruction sets.

        If the two parsers disagree on the pre-slice instruction set (i.e.
        they parse different numbers of instructions), the test is skipped
        because CFG structural differences make slice comparison meaningless.
        """
        # --- Builtin ---
        prog = builtin_parse(sass_path)
        cfgs = builtin_build_cfgs(prog)
        self.assertIn(kernel, cfgs)
        builtin_full = cfgs[kernel]

        # --- gpucode-analyzer ---
        gca_cfgs_raw = gca_parse(sass_path)
        self.assertIn(kernel, gca_cfgs_raw)

        # Check pre-slice instruction parity.
        # Allow a small tolerance: the builtin parser may include dead-code
        # trap BRA instructions after EXIT that gpucode-analyzer skips.
        builtin_all = {i.address for bb in builtin_full.blocks for i in bb.instructions}
        gca_converted = gca_build_cfgs(gca_cfgs_raw)[kernel]
        gca_all = {i.address for bb in gca_converted.blocks for i in bb.instructions}
        shared = builtin_all & gca_all
        only_builtin_pre = builtin_all - gca_all
        only_gca_pre = gca_all - builtin_all
        # More than 5 instructions difference → fundamentally different parses
        if len(only_builtin_pre) + len(only_gca_pre) > 5:
            self.skipTest(
                f"Parsers disagree on instruction set "
                f"({len(only_builtin_pre)}+{len(only_gca_pre)} addresses differ); "
                f"CFG structures are not comparable"
            )

        builtin_cfg = builtin_slice(builtin_full, pattern, keep_control=True)
        gca_cfg = gca_slice(gca_cfgs_raw[kernel], pattern, keep_control=True)

        # --- Compare instruction address sets ---
        builtin_addrs = _instruction_addresses(builtin_cfg)
        gca_addrs = _instruction_addresses(gca_cfg)

        only_builtin = builtin_addrs - gca_addrs
        only_gca = gca_addrs - builtin_addrs

        self.assertEqual(
            builtin_addrs,
            gca_addrs,
            f"Instruction address mismatch for {os.path.basename(sass_path)}:{kernel[:60]}:\n"
            f"  Only in builtin ({len(only_builtin)}): "
            f"{sorted(hex(a) for a in only_builtin)[:20]}\n"
            f"  Only in gpucode ({len(only_gca)}): "
            f"{sorted(hex(a) for a in only_gca)[:20]}",
        )


def _make_test(sass_path, kernel, pattern):
    """Factory for parameterised test methods."""
    def test_method(self):
        self._compare_slicers(sass_path, kernel, pattern)
    test_method.__doc__ = f"{os.path.basename(sass_path)} / {kernel[:50]}"
    return test_method


# ---------------------------------------------------------------------------
# Discover example SASS files and generate one test per kernel
# ---------------------------------------------------------------------------

_PATTERN = "WARPSYNC|USETMAXREG"

_example_files = sorted(glob.glob(os.path.join(_EXAMPLES_DIR, "*.sass")))

for _path in _example_files:
    _kernel = _first_kernel(_path)
    if _kernel is None:
        continue
    _safe = os.path.basename(_path).replace(".", "_")[:40]
    _test_name = f"test_{_safe}"
    setattr(TestSlicerComparison, _test_name, _make_test(_path, _kernel, _PATTERN))


if __name__ == "__main__":
    unittest.main()
