"""
test_cfg_slicer.py

Tests for the CFG slicer in cfg.py:
  - defs_of / uses_of (def/use semantics)
  - compute_reaching_definitions
  - slice_cfg (data deps, control deps, structural integrity)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from parser import parse_text
from cfg import (
    build_cfg,
    build_cfgs,
    slice_cfg,
    defs_of,
    uses_of,
    compute_reaching_definitions,
    TerminatorKind,
    _write_count,
    _adjacent_regs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_cfg(sass: str):
    """Parse a single-kernel SASS snippet into a CFG."""
    prog = parse_text(sass)
    return build_cfg(prog)


def _all_mnemonics(cfg) -> list[str]:
    return [i.mnemonic for bb in cfg.blocks for i in bb.instructions]


def _instr_by_mnem(cfg, mnem: str):
    for bb in cfg.blocks:
        for i in bb.instructions:
            if i.mnemonic == mnem:
                return i
    return None


# ---------------------------------------------------------------------------
# 1. defs_of / uses_of
# ---------------------------------------------------------------------------

DEFS_USES_SASS = """\
Function : k
        /*0000*/  MOV R2, c[0x0][0x0] ;
        /*0010*/  IADD3 R4, R2, R3, RZ ;
        /*0020*/  STG.E [R4], R2 ;
        /*0030*/  EXIT ;
"""


def test_defs_of_mov():
    """MOV has 1 write (default): R2 is defined."""
    cfg = _parse_cfg(DEFS_USES_SASS)
    mov = _instr_by_mnem(cfg, "MOV")
    assert mov is not None
    d = defs_of(mov)
    assert d == {"R2"}, f"Expected {{R2}}, got {d}"
    print("PASS  test_defs_of_mov")


def test_uses_of_mov():
    """MOV R2, c[0x0][0x0]: no general-purpose register reads (const bank is not a reg)."""
    cfg = _parse_cfg(DEFS_USES_SASS)
    mov = _instr_by_mnem(cfg, "MOV")
    u = uses_of(mov)
    # c[0x0][0x0] has no register index, so no register uses
    assert "R2" not in u, f"R2 should not be a use of MOV: {u}"
    print("PASS  test_uses_of_mov")


def test_defs_of_stg():
    """STG.E writes 0 registers (it's a store)."""
    cfg = _parse_cfg(DEFS_USES_SASS)
    stg = _instr_by_mnem(cfg, "STG.E")
    assert stg is not None
    d = defs_of(stg)
    assert d == set(), f"STG.E should define nothing, got {d}"
    print("PASS  test_defs_of_stg")


def test_uses_of_stg():
    """STG.E [R4], R2: reads R4 (in mem addr) and R2 (value)."""
    cfg = _parse_cfg(DEFS_USES_SASS)
    stg = _instr_by_mnem(cfg, "STG.E")
    u = uses_of(stg)
    assert "R4" in u, f"R4 should be a use of STG.E: {u}"
    assert "R2" in u, f"R2 should be a use of STG.E: {u}"
    print("PASS  test_uses_of_stg")


def test_defs_of_exit():
    """EXIT defines nothing."""
    cfg = _parse_cfg(DEFS_USES_SASS)
    ex = _instr_by_mnem(cfg, "EXIT")
    assert defs_of(ex) == set()
    print("PASS  test_defs_of_exit")


def test_predicate_is_a_use():
    """A predicate guard (@P0) counts as a register read."""
    sass = """\
Function : k
        /*0000*/  MOV R0, c[0x0][0x0] ;
        /*0010*/  @P0 STG.E [R4], R0 ;
        /*0020*/  EXIT ;
"""
    cfg = _parse_cfg(sass)
    stg = _instr_by_mnem(cfg, "STG.E")
    u = uses_of(stg)
    assert "P0" in u, f"P0 guard should appear in uses: {u}"
    print("PASS  test_predicate_is_a_use")


# ---------------------------------------------------------------------------
# 2. _write_count and _adjacent_regs
# ---------------------------------------------------------------------------


def test_write_count_defaults():
    """Instructions not in _WRITE_COUNT default to 1 destination."""
    sass = """\
Function : k
        /*0000*/  MOV R0, c[0x0][0x0] ;
        /*0010*/  EXIT ;
"""
    cfg = _parse_cfg(sass)
    mov = _instr_by_mnem(cfg, "MOV")
    assert _write_count(mov) == 1
    print("PASS  test_write_count_defaults")


def test_adjacent_regs():
    """_adjacent_regs('R4', 3) should return ['R5','R6','R7']."""
    result = _adjacent_regs("R4", 3)
    assert result == ["R5", "R6", "R7"], f"Got {result}"
    print("PASS  test_adjacent_regs")


# ---------------------------------------------------------------------------
# 3. compute_reaching_definitions  (straight-line code)
# ---------------------------------------------------------------------------

REACHING_DEFS_SASS = """\
Function : k
        /*0000*/  MOV R0, c[0x0][0x0] ;
        /*0010*/  MOV R1, c[0x0][0x4] ;
        /*0020*/  IADD3 R2, R0, R1, RZ ;
        /*0030*/  EXIT ;
"""


def test_reaching_defs_sequential():
    """
    In straight-line code the IADD3 should see reaching defs for R0 and R1
    produced by the two MOVs.
    """
    cfg = _parse_cfg(REACHING_DEFS_SASS)
    rd_in = compute_reaching_definitions(cfg)

    iadd = _instr_by_mnem(cfg, "IADD3")
    assert iadd is not None

    reaching = rd_in[id(iadd)]
    reg_names = {r for r, _ in reaching}
    assert "R0" in reg_names, f"R0 not in reaching defs at IADD3: {reg_names}"
    assert "R1" in reg_names, f"R1 not in reaching defs at IADD3: {reg_names}"
    print("PASS  test_reaching_defs_sequential")


def test_reaching_defs_kill():
    """
    A later MOV to R0 should kill the earlier one: only the second
    definition of R0 reaches the IADD3.
    """
    sass = """\
Function : k
        /*0000*/  MOV R0, c[0x0][0x0] ;
        /*0010*/  MOV R0, c[0x0][0x8] ;
        /*0020*/  IADD3 R2, R0, R0, RZ ;
        /*0030*/  EXIT ;
"""
    cfg = _parse_cfg(sass)
    rd_in = compute_reaching_definitions(cfg)

    instrs = [i for bb in cfg.blocks for i in bb.instructions]
    first_mov, second_mov, iadd, _ = instrs

    reaching = rd_in[id(iadd)]
    def_ids = {did for _, did in reaching}

    assert id(second_mov) in def_ids, "second MOV R0 should reach IADD3"
    assert id(first_mov) not in def_ids, "first MOV R0 should be killed by second"
    print("PASS  test_reaching_defs_kill")


# ---------------------------------------------------------------------------
# 4. slice_cfg — data dependency only
# ---------------------------------------------------------------------------

SLICE_DATA_SASS = """\
Function : k
        /*0000*/  MOV R0, c[0x0][0x0] ;
        /*0010*/  MOV R1, c[0x0][0x4] ;
        /*0020*/  MOV R9, c[0x0][0x8] ;
        /*0030*/  IADD3 R2, R0, R1, RZ ;
        /*0040*/  WARPSYNC R9 ;
        /*0050*/  STG.E [R4], R2 ;
        /*0060*/  EXIT ;
"""


def test_slice_keeps_seeds():
    """The seed instruction (STG.E) must always appear in the slice."""
    cfg = _parse_cfg(SLICE_DATA_SASS)
    sliced = slice_cfg(cfg, "STG", keep_control=False)
    mnems = _all_mnemonics(sliced)
    assert "STG.E" in mnems, f"STG.E not in sliced: {mnems}"
    print("PASS  test_slice_keeps_seeds")


def test_slice_data_deps_included():
    """
    STG.E [R4], R2 reads R2.
    R2 is defined by IADD3 R2, R0, R1, RZ.
    R0 comes from first MOV, R1 from second MOV.
    All three must be in the slice.
    """
    cfg = _parse_cfg(SLICE_DATA_SASS)
    sliced = slice_cfg(cfg, "STG", keep_control=False)
    mnems = _all_mnemonics(sliced)
    assert "IADD3" in mnems, f"IADD3 (defines R2) should be in slice: {mnems}"
    # Both MOVs that feed IADD3
    mov_count = mnems.count("MOV")
    assert mov_count >= 2, f"Expected >=2 MOVs in slice, got {mov_count}: {mnems}"
    print("PASS  test_slice_data_deps_included")


def test_slice_unrelated_dropped():
    """
    WARPSYNC R9 and MOV R9 are unrelated to the STG.E → R2 chain.
    They must NOT appear in the slice (data-dep only, no control).
    """
    cfg = _parse_cfg(SLICE_DATA_SASS)
    sliced = slice_cfg(cfg, "STG", keep_control=False)
    mnems = _all_mnemonics(sliced)
    assert "WARPSYNC" not in mnems, f"WARPSYNC should be sliced out: {mnems}"
    # MOV R9 (third MOV, address 0x20) should be gone — but we check by count:
    # only 2 MOVs should remain (R0, R1); not the R9 one
    assert mnems.count("MOV") == 2, (
        f"Expected exactly 2 MOVs (R0 and R1), got {mnems.count('MOV')}: {mnems}"
    )
    print("PASS  test_slice_unrelated_dropped")


# ---------------------------------------------------------------------------
# 5. slice_cfg — pattern matching
# ---------------------------------------------------------------------------


def test_slice_regex_pattern():
    """Pattern 'WARPSYNC' matches WARPSYNC.ALL too (substring regex)."""
    sass = """\
Function : k
        /*0000*/  MOV R0, c[0x0][0x0] ;
        /*0010*/  WARPSYNC.ALL 0xffffffff ;
        /*0020*/  EXIT ;
"""
    cfg = _parse_cfg(sass)
    sliced = slice_cfg(cfg, "WARPSYNC", keep_control=False)
    mnems = _all_mnemonics(sliced)
    assert "WARPSYNC.ALL" in mnems, (
        f"WARPSYNC.ALL should match WARPSYNC pattern: {mnems}"
    )
    print("PASS  test_slice_regex_pattern")


def test_slice_no_match_empty(capsys=None):
    """A pattern that matches nothing produces a CFG with zero important instructions."""
    sass = """\
Function : k
        /*0000*/  MOV R0, c[0x0][0x0] ;
        /*0010*/  EXIT ;
"""
    cfg = _parse_cfg(sass)
    import io, contextlib

    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        sliced = slice_cfg(cfg, "NONEXISTENT_MNEM_XYZ", keep_control=False)
    total = sum(len(bb.instructions) for bb in sliced.blocks)
    assert total == 0, f"Expected 0 important instructions, got {total}"
    assert "WARNING" in buf.getvalue(), "Expected a warning message on stderr"
    print("PASS  test_slice_no_match_empty")


# ---------------------------------------------------------------------------
# 6. slice_cfg — control dependencies
# ---------------------------------------------------------------------------

SLICE_CTRL_SASS = """\
Function : k
.L_loop:
        /*0000*/  IADD3 R2, R0, R1, RZ ;
        /*0010*/  ISETP.LT.AND P0, PT, R2, R5, PT ;
        /*0020*/  @P0 BRA `(.L_loop) ;
        /*0030*/  STG.E [R4], R2 ;
        /*0040*/  EXIT ;
"""


def test_slice_control_deps_included():
    """
    STG.E is data-dependent on IADD3 (R2).
    STG.E is control-dependent on the @P0 BRA (which guards the loop).
    With keep_control=True the BRA must be pulled in.
    """
    cfg = _parse_cfg(SLICE_CTRL_SASS)
    sliced = slice_cfg(cfg, "STG", keep_control=True)
    mnems = _all_mnemonics(sliced)
    assert "BRA" in mnems, f"BRA (control dep) should be in slice: {mnems}"
    print("PASS  test_slice_control_deps_included")


def test_slice_no_control_deps_excludes_branch():
    """
    With keep_control=False the BRA is not a data dependency of STG.E,
    so it must NOT appear in the slice.
    """
    cfg = _parse_cfg(SLICE_CTRL_SASS)
    sliced = slice_cfg(cfg, "STG", keep_control=False)
    mnems = _all_mnemonics(sliced)
    assert "BRA" not in mnems, (
        f"BRA should be excluded with keep_control=False: {mnems}"
    )
    print("PASS  test_slice_no_control_deps_excludes_branch")


# ---------------------------------------------------------------------------
# 7. Structural integrity of sliced CFG
# ---------------------------------------------------------------------------


def test_sliced_cfg_block_count_preserved():
    """
    Slicing removes instructions from blocks but keeps the block structure
    (same number of blocks, same edges).
    """
    cfg = _parse_cfg(SLICE_DATA_SASS)
    sliced = slice_cfg(cfg, "STG", keep_control=False)
    assert len(sliced.blocks) == len(cfg.blocks), (
        f"Block count changed after slice: {len(cfg.blocks)} → {len(sliced.blocks)}"
    )
    print("PASS  test_sliced_cfg_block_count_preserved")


def test_sliced_cfg_edges_preserved():
    """Successor/predecessor edges are faithfully copied to the sliced CFG."""
    cfg = _parse_cfg(SLICE_CTRL_SASS)
    sliced = slice_cfg(cfg, "STG", keep_control=True)

    # Build edge set (src_id, dst_id) for original and sliced
    orig_edges = {(bb.id, s.id) for bb in cfg.blocks for s in bb.successors}
    sliced_edges = {(bb.id, s.id) for bb in sliced.blocks for s in bb.successors}
    assert orig_edges == sliced_edges, (
        f"Edge sets differ after slice.\n  orig:   {orig_edges}\n  sliced: {sliced_edges}"
    )
    print("PASS  test_sliced_cfg_edges_preserved")


def test_sliced_cfg_entry_is_first_block():
    """cfg.entry always points to the first block."""
    cfg = _parse_cfg(SLICE_DATA_SASS)
    sliced = slice_cfg(cfg, "STG", keep_control=False)
    assert sliced.entry is not None
    assert sliced.entry is sliced.blocks[0]
    print("PASS  test_sliced_cfg_entry_is_first_block")


def test_sliced_cfg_exit_blocks_tracked():
    """exit_blocks in the sliced CFG contains the EXIT-terminated block."""
    cfg = _parse_cfg(SLICE_DATA_SASS)
    sliced = slice_cfg(cfg, "STG", keep_control=False)
    assert len(sliced.exit_blocks) >= 1, (
        "Sliced CFG should have at least one exit block"
    )
    for eb in sliced.exit_blocks:
        assert eb.terminator_kind == TerminatorKind.EXIT
    print("PASS  test_sliced_cfg_exit_blocks_tracked")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_defs_of_mov,
        test_uses_of_mov,
        test_defs_of_stg,
        test_uses_of_stg,
        test_defs_of_exit,
        test_predicate_is_a_use,
        test_write_count_defaults,
        test_adjacent_regs,
        test_reaching_defs_sequential,
        test_reaching_defs_kill,
        test_slice_keeps_seeds,
        test_slice_data_deps_included,
        test_slice_unrelated_dropped,
        test_slice_regex_pattern,
        test_slice_no_match_empty,
        test_slice_control_deps_included,
        test_slice_no_control_deps_excludes_branch,
        test_sliced_cfg_block_count_preserved,
        test_sliced_cfg_edges_preserved,
        test_sliced_cfg_entry_is_first_block,
        test_sliced_cfg_exit_blocks_tracked,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            import traceback

            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")
    sys.exit(0 if failed == 0 else 1)
