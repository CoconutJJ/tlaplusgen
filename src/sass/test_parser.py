"""
test_parser.py — Tests for the SASS parser (parser.py).

Covers:
  - Operand parsing helpers (_parse_register, _parse_immediate)
  - Predicate parsing
  - Line-level parsing (instructions, labels, function defs)
  - Full parse_text public API
  - AST __str__ round-tripping
  - Program convenience helpers
  - dump pretty-printer

Run with:
    python3 -m sass.test_parser        (from tlaplusgen/src/)
"""

import sys
import os
import math
import importlib

# Ensure the parent of the 'sass' package is on sys.path so that
# `from .cleaner import clean` inside parser.py resolves correctly.
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Import through the package so relative imports work.
import sass.parser as _parser_mod

parse_text      = _parser_mod.parse_text
dump            = _parser_mod.dump
Predicate       = _parser_mod.Predicate
RegisterOp      = _parser_mod.RegisterOp
ImmediateOp     = _parser_mod.ImmediateOp
LabelRef        = _parser_mod.LabelRef
MemAddrOp       = _parser_mod.MemAddrOp
DescOp          = _parser_mod.DescOp
ConstBankOp     = _parser_mod.ConstBankOp
BranchTargetsOp = _parser_mod.BranchTargetsOp
Instruction     = _parser_mod.Instruction
Label           = _parser_mod.Label
FunctionDef     = _parser_mod.FunctionDef
Program         = _parser_mod.Program
_parse_register  = _parser_mod._parse_register
_parse_immediate = _parser_mod._parse_immediate
_parse_predicate = _parser_mod._parse_predicate
_parse_line      = _parser_mod._parse_line


# ===================================================================
# _parse_register
# ===================================================================


def test_parse_register_simple():
    """Bare register name with no modifiers."""
    r = _parse_register("R4")
    assert r == RegisterOp(name="R4", modifiers=()), f"got {r}"
    print("PASS  test_parse_register_simple")


def test_parse_register_rz():
    r = _parse_register("RZ")
    assert r == RegisterOp(name="RZ", modifiers=()), f"got {r}"
    print("PASS  test_parse_register_rz")


def test_parse_register_uniform():
    r = _parse_register("UR5")
    assert r == RegisterOp(name="UR5", modifiers=()), f"got {r}"
    print("PASS  test_parse_register_uniform")


def test_parse_register_predicate():
    r = _parse_register("P0")
    assert r == RegisterOp(name="P0", modifiers=()), f"got {r}"
    print("PASS  test_parse_register_predicate")


def test_parse_register_pt():
    r = _parse_register("PT")
    assert r == RegisterOp(name="PT", modifiers=()), f"got {r}"
    print("PASS  test_parse_register_pt")


def test_parse_register_upt():
    r = _parse_register("UPT")
    assert r == RegisterOp(name="UPT", modifiers=()), f"got {r}"
    print("PASS  test_parse_register_upt")


def test_parse_register_uniform_pred():
    r = _parse_register("UP0")
    assert r == RegisterOp(name="UP0", modifiers=()), f"got {r}"
    print("PASS  test_parse_register_uniform_pred")


def test_parse_register_with_reuse():
    r = _parse_register("R4.reuse")
    assert r == RegisterOp(name="R4", modifiers=("reuse",)), f"got {r}"
    print("PASS  test_parse_register_with_reuse")


def test_parse_register_multiple_modifiers():
    r = _parse_register("R100.F32x2.HI_LO")
    assert r == RegisterOp(name="R100", modifiers=("F32x2", "HI_LO")), f"got {r}"
    print("PASS  test_parse_register_multiple_modifiers")


def test_parse_register_sr_ctaid():
    """SR_ registers carry dimension as part of the name, not as modifier."""
    r = _parse_register("SR_CTAID.X")
    assert r == RegisterOp(name="SR_CTAID.X", modifiers=()), f"got {r}"
    print("PASS  test_parse_register_sr_ctaid")


def test_parse_register_sr_tid():
    r = _parse_register("SR_TID.Y")
    assert r == RegisterOp(name="SR_TID.Y", modifiers=()), f"got {r}"
    print("PASS  test_parse_register_sr_tid")


def test_parse_register_b0():
    r = _parse_register("B0")
    assert r == RegisterOp(name="B0", modifiers=()), f"got {r}"
    print("PASS  test_parse_register_b0")


def test_parse_register_with_64():
    r = _parse_register("R4.64")
    assert r == RegisterOp(name="R4", modifiers=("64",)), f"got {r}"
    print("PASS  test_parse_register_with_64")


# ===================================================================
# _parse_immediate
# ===================================================================


def test_parse_immediate_hex():
    i = _parse_immediate("0x3c")
    assert i.raw == "0x3c"
    assert i.value == 0x3C, f"got {i.value}"
    print("PASS  test_parse_immediate_hex")


def test_parse_immediate_neg_hex():
    i = _parse_immediate("-0x7f")
    assert i.raw == "-0x7f"
    assert i.value == -0x7F, f"got {i.value}"
    print("PASS  test_parse_immediate_neg_hex")


def test_parse_immediate_float():
    i = _parse_immediate("1.5")
    assert i.raw == "1.5"
    assert i.value == 1.5, f"got {i.value}"
    print("PASS  test_parse_immediate_float")


def test_parse_immediate_int():
    i = _parse_immediate("42")
    assert i.raw == "42"
    assert i.value == 42, f"got {i.value}"
    print("PASS  test_parse_immediate_int")


def test_parse_immediate_zero():
    i = _parse_immediate("0")
    assert i.raw == "0"
    assert i.value == 0, f"got {i.value}"
    print("PASS  test_parse_immediate_zero")


def test_parse_immediate_pos_inf():
    i = _parse_immediate("+INF")
    assert i.value == float("inf"), f"got {i.value}"
    print("PASS  test_parse_immediate_pos_inf")


def test_parse_immediate_neg_inf():
    i = _parse_immediate("-INF")
    assert i.value == float("-inf"), f"got {i.value}"
    print("PASS  test_parse_immediate_neg_inf")


def test_parse_immediate_qnan():
    i = _parse_immediate("-QNAN")
    assert math.isnan(i.value), f"got {i.value}"
    print("PASS  test_parse_immediate_qnan")


def test_parse_immediate_nan():
    i = _parse_immediate("NaN")
    assert math.isnan(i.value), f"got {i.value}"
    print("PASS  test_parse_immediate_nan")


def test_parse_immediate_neg_int():
    i = _parse_immediate("-5")
    assert i.value == -5, f"got {i.value}"
    print("PASS  test_parse_immediate_neg_int")


def test_parse_immediate_float_scientific():
    i = _parse_immediate("1.0e3")
    assert i.value == 1000.0, f"got {i.value}"
    print("PASS  test_parse_immediate_float_scientific")


# ===================================================================
# _parse_predicate
# ===================================================================


def test_parse_predicate_p0():
    p = _parse_predicate("@P0")
    assert p == Predicate(negated=False, is_uniform=False, name="P0"), f"got {p}"
    print("PASS  test_parse_predicate_p0")


def test_parse_predicate_negated():
    p = _parse_predicate("@!P1")
    assert p == Predicate(negated=True, is_uniform=False, name="P1"), f"got {p}"
    print("PASS  test_parse_predicate_negated")


def test_parse_predicate_uniform():
    p = _parse_predicate("@UP0")
    assert p == Predicate(negated=False, is_uniform=True, name="UP0"), f"got {p}"
    print("PASS  test_parse_predicate_uniform")


def test_parse_predicate_negated_uniform():
    p = _parse_predicate("@!UPT")
    assert p == Predicate(negated=True, is_uniform=True, name="UPT"), f"got {p}"
    print("PASS  test_parse_predicate_negated_uniform")


def test_parse_predicate_pt():
    p = _parse_predicate("@PT")
    assert p == Predicate(negated=False, is_uniform=False, name="PT"), f"got {p}"
    print("PASS  test_parse_predicate_pt")


def test_parse_predicate_b0():
    p = _parse_predicate("@B0")
    assert p == Predicate(negated=False, is_uniform=False, name="B0"), f"got {p}"
    print("PASS  test_parse_predicate_b0")


# ===================================================================
# _parse_line
# ===================================================================


def test_parse_line_instruction():
    stmt = _parse_line("  /*0a30*/  MOV  R0, c[0x0][0x0] ;")
    assert isinstance(stmt, Instruction)
    assert stmt.address == 0x0A30
    assert stmt.address_str == "0a30"
    assert stmt.mnemonic == "MOV"
    assert stmt.predicate is None
    assert len(stmt.operands) == 2
    print("PASS  test_parse_line_instruction")


def test_parse_line_instruction_with_predicate():
    stmt = _parse_line("  /*0010*/  @P0  IADD3  R2, R0, R1, RZ ;")
    assert isinstance(stmt, Instruction)
    assert stmt.predicate is not None
    assert stmt.predicate.name == "P0"
    assert stmt.predicate.negated is False
    assert stmt.mnemonic == "IADD3"
    assert len(stmt.operands) == 4
    print("PASS  test_parse_line_instruction_with_predicate")


def test_parse_line_instruction_negated_pred():
    stmt = _parse_line("  /*0020*/  @!P1  BRA  `(.L_x_0) ;")
    assert isinstance(stmt, Instruction)
    assert stmt.predicate.negated is True
    assert stmt.predicate.name == "P1"
    assert stmt.mnemonic == "BRA"
    assert len(stmt.operands) == 1
    assert isinstance(stmt.operands[0], LabelRef)
    print("PASS  test_parse_line_instruction_negated_pred")


def test_parse_line_label():
    stmt = _parse_line(".L_x_0:")
    assert isinstance(stmt, Label)
    assert stmt.name == ".L_x_0"
    print("PASS  test_parse_line_label")


def test_parse_line_function():
    stmt = _parse_line(".function kernel_foo")
    assert isinstance(stmt, FunctionDef)
    assert stmt.name == "kernel_foo"
    print("PASS  test_parse_line_function")


def test_parse_line_blank():
    assert _parse_line("") is None
    assert _parse_line("   ") is None
    print("PASS  test_parse_line_blank")


def test_parse_line_exit():
    stmt = _parse_line("  /*0020*/  EXIT ;")
    assert isinstance(stmt, Instruction)
    assert stmt.mnemonic == "EXIT"
    assert len(stmt.operands) == 0
    print("PASS  test_parse_line_exit")


def test_parse_line_desc():
    """Instruction with desc operand parses correctly."""
    stmt = _parse_line("  /*0740*/  LDG.E  R9, desc[UR6][R10.64+0xc] ;")
    assert isinstance(stmt, Instruction)
    assert stmt.mnemonic == "LDG.E"
    assert len(stmt.operands) == 2
    assert isinstance(stmt.operands[0], RegisterOp)
    assert isinstance(stmt.operands[1], DescOp)
    assert stmt.operands[1].kind == "desc"
    assert stmt.operands[1].indices == ("UR6", "R10.64+0xc")
    print("PASS  test_parse_line_desc")


def test_parse_line_const_bank():
    """Instruction with constant bank operand."""
    stmt = _parse_line("  /*0000*/  LDC  R1, c[0x0][0x28] ;")
    assert isinstance(stmt, Instruction)
    assert isinstance(stmt.operands[1], ConstBankOp)
    assert stmt.operands[1].bank == "0x0"
    assert stmt.operands[1].offset == "0x28"
    print("PASS  test_parse_line_const_bank")


def test_parse_line_mem_addr():
    """Instruction with memory address operand."""
    stmt = _parse_line("  /*0000*/  STG.E  [R4+0x10], R2 ;")
    assert isinstance(stmt, Instruction)
    assert isinstance(stmt.operands[0], MemAddrOp)
    assert stmt.operands[0].parts == ("R4", "0x10")
    print("PASS  test_parse_line_mem_addr")


def test_parse_line_label_ref():
    """Instruction with label ref operand."""
    stmt = _parse_line("  /*0020*/  BRA  `(.L_x_0) ;")
    assert isinstance(stmt, Instruction)
    assert isinstance(stmt.operands[0], LabelRef)
    assert stmt.operands[0].name == ".L_x_0"
    print("PASS  test_parse_line_label_ref")


def test_parse_line_abs_reg():
    """Instruction with absolute-value register operand."""
    stmt = _parse_line("  /*d250*/  FSETP.GTU.FTZ.AND  P1, PT, |R74|, +INF, PT ;")
    assert isinstance(stmt, Instruction)
    assert isinstance(stmt.operands[2], RegisterOp)
    assert stmt.operands[2].name == "R74"
    assert "abs" in stmt.operands[2].modifiers
    print("PASS  test_parse_line_abs_reg")


def test_parse_line_neg_pred():
    """Instruction with negated predicate in operand position."""
    stmt = _parse_line("  /*0060*/  UIADD3.X  UR11, URZ, UR7, URZ, UP1, !UPT ;")
    assert isinstance(stmt, Instruction)
    op = stmt.operands[5]
    assert isinstance(op, RegisterOp)
    assert op.name == "UPT"
    assert "neg" in op.modifiers
    print("PASS  test_parse_line_neg_pred")


def test_parse_line_neg_reg():
    """Instruction with negated register operand."""
    stmt = _parse_line("  /*0600*/  IMAD.MOV  R11, RZ, RZ, -R3 ;")
    assert isinstance(stmt, Instruction)
    op = stmt.operands[3]
    assert isinstance(op, RegisterOp)
    assert op.name == "R3"
    assert "neg" in op.modifiers
    print("PASS  test_parse_line_neg_reg")


def test_parse_line_special_imm():
    """Instruction with special immediate (+INF)."""
    stmt = _parse_line("  /*9cc0*/  @P3  FFMA  R4, R10, R6, +INF ;")
    assert isinstance(stmt, Instruction)
    op = stmt.operands[3]
    assert isinstance(op, ImmediateOp)
    assert op.value == float("inf")
    print("PASS  test_parse_line_special_imm")


# ===================================================================
# parse_text (public API)
# ===================================================================

SAMPLE_SASS = """\
Function : kernel_test
        /*0000*/  MOV R0, c[0x0][0x0] ;
        /*0010*/  MOV R1, c[0x0][0x4] ;
.L_x_0:
        /*0020*/  @P0  IADD3 R2, R0, R1, RZ ;
        /*0030*/  @!P1  BRA `(.L_x_0) ;
        /*0040*/  EXIT ;
"""


def test_parse_text_statement_count():
    prog = parse_text(SAMPLE_SASS)
    # Expect: 1 FunctionDef + 4 instructions + 1 label = 6 statements
    # (label .L_x_0 + instructions at 0000, 0010, 0020, 0030, 0040 + FunctionDef)
    func_defs = [s for s in prog.statements if isinstance(s, FunctionDef)]
    labels = prog.labels()
    instrs = prog.instructions()
    assert len(func_defs) == 1, f"Expected 1 FunctionDef, got {len(func_defs)}"
    assert len(labels) == 1, f"Expected 1 label, got {len(labels)}"
    assert len(instrs) == 5, f"Expected 5 instructions, got {len(instrs)}"
    print("PASS  test_parse_text_statement_count")


def test_parse_text_label_name():
    prog = parse_text(SAMPLE_SASS)
    labels = prog.labels()
    assert labels[0].name == ".L_x_0", f"got {labels[0].name}"
    print("PASS  test_parse_text_label_name")


def test_parse_text_instruction_addresses():
    prog = parse_text(SAMPLE_SASS)
    addrs = [i.address for i in prog.instructions()]
    assert addrs == [0x0, 0x10, 0x20, 0x30, 0x40], f"got {addrs}"
    print("PASS  test_parse_text_instruction_addresses")


def test_parse_text_predicated_instruction():
    prog = parse_text(SAMPLE_SASS)
    instrs = prog.instructions()
    # 3rd instruction: @P0 IADD3
    iadd = instrs[2]
    assert iadd.predicate is not None
    assert iadd.predicate.name == "P0"
    assert iadd.predicate.negated is False
    assert iadd.mnemonic == "IADD3"
    print("PASS  test_parse_text_predicated_instruction")


def test_parse_text_branch_with_label_ref():
    prog = parse_text(SAMPLE_SASS)
    bra = prog.by_mnemonic("BRA")
    assert len(bra) == 1
    assert isinstance(bra[0].operands[0], LabelRef)
    assert bra[0].operands[0].name == ".L_x_0"
    print("PASS  test_parse_text_branch_with_label_ref")


def test_parse_text_const_bank_operands():
    prog = parse_text(SAMPLE_SASS)
    movs = prog.by_mnemonic("MOV")
    assert len(movs) == 2
    for mov in movs:
        assert isinstance(mov.operands[1], ConstBankOp), (
            f"Expected ConstBankOp, got {type(mov.operands[1])}"
        )
    print("PASS  test_parse_text_const_bank_operands")


# ===================================================================
# Program convenience helpers
# ===================================================================


def test_program_instructions():
    prog = parse_text(SAMPLE_SASS)
    instrs = prog.instructions()
    assert all(isinstance(i, Instruction) for i in instrs)
    assert len(instrs) == 5
    print("PASS  test_program_instructions")


def test_program_labels():
    prog = parse_text(SAMPLE_SASS)
    labels = prog.labels()
    assert all(isinstance(l, Label) for l in labels)
    assert len(labels) == 1
    print("PASS  test_program_labels")


def test_program_by_mnemonic():
    prog = parse_text(SAMPLE_SASS)
    exits = prog.by_mnemonic("EXIT")
    assert len(exits) == 1
    assert exits[0].mnemonic == "EXIT"
    print("PASS  test_program_by_mnemonic")


def test_program_by_mnemonic_case_insensitive():
    prog = parse_text(SAMPLE_SASS)
    movs = prog.by_mnemonic("mov")
    assert len(movs) == 2
    print("PASS  test_program_by_mnemonic_case_insensitive")


def test_program_label_map():
    prog = parse_text(SAMPLE_SASS)
    lmap = prog.label_map()
    assert ".L_x_0" in lmap
    print("PASS  test_program_label_map")


# ===================================================================
# AST node __str__ methods (round-trip sanity)
# ===================================================================


def test_predicate_str():
    assert str(Predicate(negated=False, is_uniform=False, name="P0")) == "@P0"
    assert str(Predicate(negated=True, is_uniform=True, name="UPT")) == "@!UPT"
    print("PASS  test_predicate_str")


def test_register_str_simple():
    assert str(RegisterOp(name="R4", modifiers=())) == "R4"
    print("PASS  test_register_str_simple")


def test_register_str_reuse():
    assert str(RegisterOp(name="R4", modifiers=("reuse",))) == "R4.reuse"
    print("PASS  test_register_str_reuse")


def test_register_str_abs():
    assert str(RegisterOp(name="R74", modifiers=("abs",))) == "|R74|"
    print("PASS  test_register_str_abs")


def test_register_str_neg():
    assert str(RegisterOp(name="PT", modifiers=("neg",))) == "!PT"
    print("PASS  test_register_str_neg")


def test_immediate_str():
    assert str(ImmediateOp(raw="0x3c", value=0x3C)) == "0x3c"
    print("PASS  test_immediate_str")


def test_label_ref_str():
    assert str(LabelRef(name=".L_x_0")) == "`(.L_x_0)"
    print("PASS  test_label_ref_str")


def test_mem_addr_str():
    assert str(MemAddrOp(parts=("UR4", "0x8"))) == "[UR4+0x8]"
    print("PASS  test_mem_addr_str")


def test_desc_str():
    assert str(DescOp(kind="desc", indices=("UR4", "R4.64+0x8"))) == "desc[UR4][R4.64+0x8]"
    print("PASS  test_desc_str")


def test_const_bank_str():
    assert str(ConstBankOp(bank="0x0", offset="0x37c")) == "c[0x0][0x37c]"
    print("PASS  test_const_bank_str")


def test_branch_targets_str():
    bt = BranchTargetsOp(targets=(".L_x_0", ".L_x_1"))
    assert str(bt) == '(*"BRANCH_TARGETS .L_x_0, .L_x_1"*)'
    print("PASS  test_branch_targets_str")


def test_label_str():
    assert str(Label(name=".L_x_0")) == ".L_x_0:"
    print("PASS  test_label_str")


def test_functiondef_str():
    assert str(FunctionDef(name="kernel_foo")) == ".function kernel_foo"
    print("PASS  test_functiondef_str")


def test_instruction_str_no_pred():
    instr = Instruction(
        address=0x0,
        address_str="0000",
        predicate=None,
        mnemonic="MOV",
        operands=(RegisterOp(name="R0", modifiers=()), ConstBankOp(bank="0x0", offset="0x0")),
        raw="",
    )
    s = str(instr)
    assert "/*0000*/" in s
    assert "MOV" in s
    assert "R0" in s
    assert "c[0x0][0x0]" in s
    print("PASS  test_instruction_str_no_pred")


def test_instruction_str_with_pred():
    instr = Instruction(
        address=0x10,
        address_str="0010",
        predicate=Predicate(negated=True, is_uniform=False, name="P1"),
        mnemonic="BRA",
        operands=(LabelRef(name=".L_x_0"),),
        raw="",
    )
    s = str(instr)
    assert "@!P1" in s
    assert "BRA" in s
    assert "`(.L_x_0)" in s
    print("PASS  test_instruction_str_with_pred")


# ===================================================================
# dump pretty-printer
# ===================================================================


def test_dump_basic():
    prog = parse_text(SAMPLE_SASS)
    out = dump(prog)
    assert ".L_x_0:" in out
    assert "MOV" in out
    assert "EXIT" in out
    assert ".function kernel_test" in out
    print("PASS  test_dump_basic")


def test_dump_show_types():
    prog = parse_text(SAMPLE_SASS)
    out = dump(prog, show_operand_types=True)
    # With types, operands are printed via repr()
    assert "RegisterOp" in out or "ConstBankOp" in out
    print("PASS  test_dump_show_types")


# ===================================================================
# Edge cases & complex instructions
# ===================================================================


def test_multiple_labels():
    sass = """\
Function : kernel_labels
        /*0000*/  MOV R0, RZ ;
.L_a:
        /*0010*/  MOV R1, RZ ;
.L_b:
        /*0020*/  EXIT ;
"""
    prog = parse_text(sass)
    labels = prog.labels()
    assert len(labels) == 2
    names = {l.name for l in labels}
    assert ".L_a" in names
    assert ".L_b" in names
    print("PASS  test_multiple_labels")


def test_no_operand_instruction():
    """Instructions like EXIT have no operands."""
    sass = """\
Function : kernel_exit
        /*0000*/  EXIT ;
"""
    prog = parse_text(sass)
    instrs = prog.instructions()
    assert len(instrs) == 1
    assert instrs[0].mnemonic == "EXIT"
    assert instrs[0].operands == ()
    print("PASS  test_no_operand_instruction")


def test_dotted_mnemonic():
    """Mnemonics with dots like IADD3.X, STG.E."""
    sass = """\
Function : kernel_dots
        /*0000*/  IADD3.X R2, R0, R1, RZ ;
        /*0010*/  EXIT ;
"""
    prog = parse_text(sass)
    instrs = prog.instructions()
    assert instrs[0].mnemonic == "IADD3.X"
    print("PASS  test_dotted_mnemonic")


def test_instruction_hash_and_eq():
    """Instruction hash and equality based on address."""
    instr1 = Instruction(
        address=0x10, address_str="0010", predicate=None,
        mnemonic="MOV", operands=(), raw="",
    )
    instr2 = Instruction(
        address=0x10, address_str="0010", predicate=None,
        mnemonic="EXIT", operands=(), raw="different",
    )
    instr3 = Instruction(
        address=0x20, address_str="0020", predicate=None,
        mnemonic="MOV", operands=(), raw="",
    )
    assert instr1 == instr2, "Same address should be equal"
    assert instr1 != instr3, "Different address should not be equal"
    assert hash(instr1) == hash(instr2)
    print("PASS  test_instruction_hash_and_eq")


def test_empty_program():
    """Parsing empty text yields empty program."""
    prog = parse_text("")
    assert len(prog.statements) == 0
    assert prog.instructions() == []
    assert prog.labels() == []
    print("PASS  test_empty_program")


def test_register_with_abs_and_modifiers():
    """RegisterOp with abs modifier + extra modifiers roundtrips."""
    r = RegisterOp(name="R5", modifiers=("abs", "reuse"))
    assert str(r) == "|R5|.reuse"
    print("PASS  test_register_with_abs_and_modifiers")


def test_register_with_neg_and_modifiers():
    r = RegisterOp(name="P0", modifiers=("neg", "reuse"))
    assert str(r) == "!P0.reuse"
    print("PASS  test_register_with_neg_and_modifiers")


def test_brx_with_branch_targets():
    """BRX instruction with BRANCH_TARGETS annotation parses all three operands."""
    sass = '''\
Function : kernel_brx
        /*0b70*/  BRX R2 -0xb80 (*"BRANCH_TARGETS .L_x_183,.L_x_184,.L_x_185"*) ;
        /*0b80*/  EXIT ;
'''
    prog = parse_text(sass)
    brx = prog.by_mnemonic("BRX")
    assert len(brx) == 1, f"Expected 1 BRX, got {len(brx)}"
    ops = brx[0].operands
    assert len(ops) == 3, f"Expected 3 operands, got {len(ops)}: {ops}"
    assert isinstance(ops[0], RegisterOp) and ops[0].name == "R2"
    assert isinstance(ops[1], ImmediateOp) and ops[1].value == -0xb80
    assert isinstance(ops[2], BranchTargetsOp)
    assert ops[2].targets == (".L_x_183", ".L_x_184", ".L_x_185"), f"got {ops[2].targets}"
    print("PASS  test_brx_with_branch_targets")


def test_sr_cgactaid():
    """Mixed-case SR register SR_CgaCtaId parses correctly (not truncated to SR_C)."""
    stmt = _parse_line("  /*01d0*/  S2UR  UR8, SR_CgaCtaId ;")
    assert isinstance(stmt, Instruction)
    assert len(stmt.operands) == 2
    sr = stmt.operands[1]
    assert isinstance(sr, RegisterOp)
    assert sr.name == "SR_CgaCtaId", f"got {sr.name}"
    print("PASS  test_sr_cgactaid")


# ===================================================================
# Runner
# ===================================================================

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
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
