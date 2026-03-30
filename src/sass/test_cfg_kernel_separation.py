"""
test_cfg_kernel_separation.py

Verifies that build_cfgs correctly separates instructions from different
SASS kernel functions and does NOT mix them into a single CFG.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from parser import parse_text, FunctionDef, Instruction, Label, Program
from cfg import build_cfgs

# ---------------------------------------------------------------------------
# Minimal SASS snippets for two distinct kernels
# ---------------------------------------------------------------------------

MULTI_KERNEL_SASS = """\
Function : kernel_alpha
        /*0000*/  MOV R0, c[0x0][0x0] ;
        /*0010*/  MOV R1, c[0x0][0x4] ;
        /*0020*/  EXIT ;

Function : kernel_beta
        /*0000*/  IADD3 R2, R0, R1, RZ ;
        /*0010*/  STG.E [R4], R2 ;
        /*0020*/  EXIT ;
"""

SINGLE_KERNEL_SASS = """\
Function : kernel_only
        /*0000*/  MOV R0, c[0x0][0x0] ;
        /*0010*/  EXIT ;
"""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _instr_mnemonics(cfg) -> list[str]:
    mnems = []
    for block in cfg.blocks:
        for instr in block.instructions:
            mnems.append(instr.mnemonic)
    return mnems


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_functiondef_parsed():
    """FunctionDef nodes are present in the parsed program."""
    prog = parse_text(MULTI_KERNEL_SASS)
    func_defs = [s for s in prog.statements if isinstance(s, FunctionDef)]
    assert len(func_defs) == 2, f"Expected 2 FunctionDef nodes, got {len(func_defs)}"
    names = [f.name for f in func_defs]
    assert "kernel_alpha" in names, f"kernel_alpha missing from {names}"
    assert "kernel_beta" in names, f"kernel_beta missing from {names}"
    print("PASS  test_functiondef_parsed")


def test_build_cfgs_produces_two_cfgs():
    """build_cfgs returns one CFG per kernel, not a merged one."""
    prog = parse_text(MULTI_KERNEL_SASS)
    cfgs = build_cfgs(prog)
    assert set(cfgs.keys()) == {"kernel_alpha", "kernel_beta"}, (
        f"Unexpected CFG keys: {set(cfgs.keys())}"
    )
    print("PASS  test_build_cfgs_produces_two_cfgs")


def test_kernels_not_mixed():
    """Instructions from kernel_alpha must not appear in kernel_beta's CFG and vice versa."""
    prog = parse_text(MULTI_KERNEL_SASS)
    cfgs = build_cfgs(prog)

    alpha_mnems = _instr_mnemonics(cfgs["kernel_alpha"])
    beta_mnems = _instr_mnemonics(cfgs["kernel_beta"])

    # kernel_alpha has MOV instructions; kernel_beta has IADD3 and STG
    assert "MOV" in alpha_mnems, f"MOV missing from kernel_alpha: {alpha_mnems}"
    assert "IADD3" not in alpha_mnems, f"IADD3 leaked into kernel_alpha: {alpha_mnems}"

    assert "IADD3" in beta_mnems, f"IADD3 missing from kernel_beta: {beta_mnems}"
    assert "MOV" not in beta_mnems, f"MOV leaked into kernel_beta: {beta_mnems}"

    print("PASS  test_kernels_not_mixed")


def test_instruction_counts():
    """Each kernel CFG contains the correct number of instructions."""
    prog = parse_text(MULTI_KERNEL_SASS)
    cfgs = build_cfgs(prog)

    alpha_instrs = sum(len(b.instructions) for b in cfgs["kernel_alpha"].blocks)
    beta_instrs = sum(len(b.instructions) for b in cfgs["kernel_beta"].blocks)

    assert alpha_instrs == 3, f"kernel_alpha: expected 3 instrs, got {alpha_instrs}"
    assert beta_instrs == 3, f"kernel_beta: expected 3 instrs, got {beta_instrs}"
    print("PASS  test_instruction_counts")


def test_single_kernel():
    """A file with one kernel produces exactly one CFG."""
    prog = parse_text(SINGLE_KERNEL_SASS)
    cfgs = build_cfgs(prog)
    assert list(cfgs.keys()) == ["kernel_only"], f"Unexpected keys: {list(cfgs.keys())}"
    instrs = sum(len(b.instructions) for b in cfgs["kernel_only"].blocks)
    assert instrs == 2, f"Expected 2 instrs, got {instrs}"
    print("PASS  test_single_kernel")


def test_no_functiondef_falls_back_to_unknown_kernel():
    """
    If the SASS text has no .function markers at all, build_cfgs
    collects everything under the 'unknown_kernel' fallback key.
    """
    bare_sass = """\
        /*0000*/  MOV R0, c[0x0][0x0] ;
        /*0010*/  EXIT ;
    """
    prog = parse_text(bare_sass)
    cfgs = build_cfgs(prog)
    assert "unknown_kernel" in cfgs, (
        f"Expected 'unknown_kernel', got {list(cfgs.keys())}"
    )
    print("PASS  test_no_functiondef_falls_back_to_unknown_kernel")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_functiondef_parsed,
        test_build_cfgs_produces_two_cfgs,
        test_kernels_not_mixed,
        test_instruction_counts,
        test_single_kernel,
        test_no_functiondef_falls_back_to_unknown_kernel,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")
    sys.exit(0 if failed == 0 else 1)
