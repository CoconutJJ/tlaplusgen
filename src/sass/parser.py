"""
sass_parser.py  –  Parse the cleaned SASS output produced by clean_sass.py.

Public API
----------
    parse_text(text: str) -> Program
    parse_file(path: str) -> Program

AST nodes (all frozen dataclasses)
-----------------------------------
    Program          – top-level list of Statement
    Statement        – Label | Instruction
    Label            – dotted label definition  (.L_x_0:)
    Instruction      – one SASS instruction
    Predicate        – optional guard  (@[!]P0  @UP1  @!UPT …)
    Operand          – one of the concrete subtypes below:
    RegisterOp       – R4, RZ, PT, UR5, URZ, UPT, UP0, P0 …
    ImmediateOp      – 0x3c, -0x7f, 1.5, +INF, -INF, -QNAN …
    LabelRef         – `(.L_x_0)
    MemAddrOp        – [UR4+0x8]  [R43+URZ+0x70]  [R2+URZ]
    DescOp           – desc[UR4][R4.64+0x8]  gdesc[UR4]  tmem[UR27]
    TmemAddrOp       – idesc[UR29]
    ConstBankOp      – c[0x0][0x37c]   c[0x2][R2]

Usage example
-------------
    from sass_parser import parse_file, Instruction, Label
    prog = parse_file("kernel.sass")
    for stmt in prog.statements:
        if isinstance(stmt, Instruction):
            print(stmt.address, stmt.mnemonic, stmt.operands)
"""

from __future__ import annotations
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from .cleaner import clean

import pyparsing as pp
from pyparsing import (
    Char,
    CharsNotIn,
    Combine,
    Group,
    Literal,
    OneOrMore,
    Suppress,
    Word,
    ZeroOrMore,
    alphanums,
    alphas,
    hexnums,
    nums,
    oneOf,
    ParserElement,
    ParseException,
)

ParserElement.enablePackrat()


# ---------------------------------------------------------------------------
# AST node definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Predicate:
    negated: bool  # True if @!
    is_uniform: bool  # True if UP / UPT
    name: str  # "P0", "P1", "PT", "UPT", "UP0" …

    def __str__(self):
        bang = "!" if self.negated else ""
        return f"@{bang}{self.name}"


@dataclass(frozen=True)
class Op:
    pass


# -- Operand leaf types --


@dataclass(frozen=True)
class RegisterOp(Op):
    """Scalar or predicate register: R4, RZ, UR5, URZ, P0, PT, UP0, UPT, B0."""

    name: str  # bare name, e.g. "R4", "UR17", "P0"
    modifiers: Tuple[str, ...] = ()  # e.g. ("reuse",) or ("F32x2", "HI_LO") or ("64",)

    def __str__(self):
        mods = list(self.modifiers)
        prefix = ""
        if "neg" in mods:
            prefix = "!"
            mods = [m for m in mods if m != "neg"]
        name = self.name
        if "abs" in mods:
            name = f"|{name}|"
            mods = [m for m in mods if m != "abs"]
        suffix = ("." + ".".join(mods)) if mods else ""
        return f"{prefix}{name}{suffix}"


@dataclass(frozen=True)
class ImmediateOp(Op):
    """Integer or float literal: 0x3c, -0x7f, 1.5, +INF, -INF, -QNAN, 0."""

    raw: str  # the original text exactly as written
    value: object  # int or float, best-effort; None if unparseable

    def __str__(self):
        return self.raw


@dataclass(frozen=True)
class LabelRef(Op):
    """`(.L_x_0) — branch target."""

    name: str

    def __str__(self):
        return f"`({self.name})"


@dataclass(frozen=True)
class MemAddrOp(Op):
    """[base + offset?]  or  [base + index + offset?]
    Examples:  [UR4]   [UR4+0x8]   [R43+URZ+0x70]   [R2+URZ]
    """

    parts: Tuple[str, ...]  # raw tokens inside [ ], split on +

    def __str__(self):
        return "[" + "+".join(self.parts) + "]"


@dataclass(frozen=True)
class DescOp(Op):
    """desc[UR4][R4.64+0x8]   gdesc[UR4]   tmem[UR27]   idesc[UR29]
    kind   : "desc" | "gdesc" | "tmem" | "idesc"
    indices: list of raw bracketed strings (without outer brackets)
    """

    kind: str
    indices: Tuple[str, ...]

    def __str__(self):
        return self.kind + "".join(f"[{i}]" for i in self.indices)


@dataclass(frozen=True)
class ConstBankOp(Op):
    """c[0x0][0x37c]  or  c[0x2][R2]"""

    bank: str  # e.g. "0x0", "0x2"
    offset: str  # e.g. "0x37c", "R2"

    def __str__(self):
        return f"c[{self.bank}][{self.offset}]"


@dataclass(frozen=True)
class BranchTargetsOp(Op):
    """(*"BRANCH_TARGETS .L_x_0, .L_x_1"*)"""

    targets: Tuple[str, ...]

    def __str__(self):
        return f'(*"BRANCH_TARGETS {", ".join(self.targets)}"*)'


Operand = Union[
    RegisterOp, ImmediateOp, LabelRef, MemAddrOp, DescOp, ConstBankOp, BranchTargetsOp
]


@dataclass(frozen=True)
class Instruction:
    address: int  # numeric value of /*HEX*/ field
    address_str: str  # original hex string, e.g. "0a30"
    predicate: Optional[Predicate]
    mnemonic: str
    operands: Tuple[Operand, ...]
    raw: str  # original source line

    def __str__(self):
        parts = [f"/*{self.address_str}*/"]
        if self.predicate:
            parts.append(str(self.predicate))
        parts.append(self.mnemonic)
        if self.operands:
            parts.append(", ".join(str(o) for o in self.operands))
        return "  ".join(parts) + " ;"

    def __hash__(self) -> int:
        return self.address

    def __eq__(self, value: Instruction) -> bool:
        return self.address == value.address


@dataclass(frozen=True)
class Label:
    name: str  # including leading dot, e.g. ".L_x_0"

    def __str__(self):
        return f"{self.name}:"


@dataclass(frozen=True)
class FunctionDef:
    """Kernel/function boundary emitted by cleaner as '.function <name>'."""

    name: str  # e.g. "kernel_foo" or "_Z6kernelPf"

    def __str__(self):
        return f".function {self.name}"


Statement = Union[Label, Instruction, FunctionDef]


@dataclass
class Program:
    statements: List[Statement] = field(default_factory=list)

    # --- convenience helpers ---

    def instructions(self) -> List[Instruction]:
        return [s for s in self.statements if isinstance(s, Instruction)]

    def labels(self) -> List[Label]:
        return [s for s in self.statements if isinstance(s, Label)]

    def by_mnemonic(self, mnem: str) -> List[Instruction]:
        m = mnem.upper()
        return [i for i in self.instructions() if i.mnemonic == m]

    def label_map(self) -> dict:
        """Map label name → index into self.statements."""
        return {
            s.name: i for i, s in enumerate(self.statements) if isinstance(s, Label)
        }


# ---------------------------------------------------------------------------
# Operand-level parsing helpers
# ---------------------------------------------------------------------------


def _parse_register(raw: str) -> RegisterOp:
    """Parse a register token that may include dotted modifiers.

    Grammar (from docs/grammar.md)::

        register  ::= reg_base ("." MODIFIER)*
        sr_reg    ::= "SR_" SR_NAME
        SR_NAME   ::= ALNUM+ ("." LETTER)?

    For SR registers the optional single-letter dimension suffix (e.g. the
    ``X`` in ``SR_CTAID.X``) is part of the *name*, not a modifier, so that
    ``SR_TID.X`` and ``SR_TID.Y`` remain distinct.  Any dot-segments after
    the dimension are ordinary modifiers.

    Examples::

        'R4'                → RegisterOp('R4', ())
        'R4.reuse'          → RegisterOp('R4', ('reuse',))
        'R100.F32x2.HI_LO' → RegisterOp('R100', ('F32x2', 'HI_LO'))
        'SR_CTAID.X'        → RegisterOp('SR_CTAID.X', ())
        'SR_TID.Y'          → RegisterOp('SR_TID.Y', ())
        'SR_CgaCtaId'       → RegisterOp('SR_CgaCtaId', ())
        'SR_TID.X.reuse'    → RegisterOp('SR_TID.X', ('reuse',))
    """
    parts = raw.split(".")
    if not raw.startswith("SR_"):
        return RegisterOp(name=parts[0], modifiers=tuple(parts[1:]))
    # SR register: base name is parts[0] (e.g. "SR_CTAID").
    # If the next segment is a single letter it is the dimension component
    # of the name (SR_NAME ::= ALNUM+ ("." LETTER)?), not a modifier.
    name = parts[0]
    mod_start = 1
    if len(parts) > 1 and len(parts[1]) == 1 and parts[1].isalpha():
        name = f"{parts[0]}.{parts[1]}"
        mod_start = 2
    return RegisterOp(name=name, modifiers=tuple(parts[mod_start:]))


def _parse_immediate(raw: str) -> ImmediateOp:
    val: object = None
    r = raw.strip()
    upper = r.upper().lstrip("+-")
    if upper in ("INF", "+INF"):
        val = float("inf") * (1 if not r.startswith("-") else -1)
    elif upper in ("-INF",):
        val = float("-inf")
    elif "QNAN" in upper or "NAN" in upper:
        val = float("nan")
    else:
        try:
            val = (
                int(r, 16)
                if "0x" in r.lower()
                else (float(r) if ("." in r or "e" in r.lower()) else int(r, 0))
            )
        except (ValueError, OverflowError):
            pass
    return ImmediateOp(raw=raw, value=val)



# ---------------------------------------------------------------------------
# pyparsing grammar  (following docs/grammar.md)
# ---------------------------------------------------------------------------

# --- Predicate name (shared: instruction guards & !pred operands) ----------

_pp_pred_name = (
    Combine(Literal("UP") + Word(nums))  # UP0..UP6
    | Literal("UPT")  # uniform always-true
    | Literal("PT")  # always-true
    | Combine(Literal("P") + Word(nums))  # P0..P6
    | Combine(Literal("B") + Word(nums))  # barrier
)

# --- SR register: SR_TID.X  SR_CgaCtaId  SR_LANEID --------------------
# sr_reg  ::= "SR_" ALNUM+ ("." LETTER)?

_pp_sr_reg = Combine(
    Literal("SR_") + Word(alphanums + "_") + pp.Optional(Literal(".") + Char(alphas))
)

# --- Register base (ordered for correct PEG precedence) -----------------

_pp_reg_base = (
    _pp_sr_reg
    | Literal("URZ")
    | Combine(Literal("UR") + Word(nums))
    | Literal("RZ")
    | Combine(Literal("R") + Word(nums))
    | Literal("UPT")
    | Combine(Literal("UP") + Word(nums))
    | Literal("PT")
    | Combine(Literal("P") + Word(nums))
    | Combine(Literal("B") + Word(nums))
    | Literal("SRZ")
)

# --- Full register token: base + optional ".modifier" segments -----------
# Returned as a single combined string, then split by _parse_register().

_pp_register_token = Combine(
    _pp_reg_base + ZeroOrMore(Literal(".") + Word(alphanums + "_"))
)

# --- Operand types -------------------------------------------------------

# label_ref  ::= "`(" LABEL_NAME ")"
# LABEL_NAME may start with "." (.L_x_0), "$" ($__internal_...), or plain ident.
_pp_label_ref = Suppress("`(") + CharsNotIn(")") + Suppress(")")

# annotation  ::= '(*"' TEXT '"*)'
_pp_annotation = Suppress('(*"') + CharsNotIn('"') + Suppress('"*)')

# desc_op  ::= desc_kind bracket_idx+
_pp_bracket_inner = Suppress("[") + CharsNotIn("]") + Suppress("]")
_pp_desc = oneOf("gdesc desc tmem idesc") + Group(OneOrMore(_pp_bracket_inner))

# const_bank  ::= "c" "[" HEX "]" "[" OFFSET "]"
_pp_const_bank = (
    Suppress("c")
    + Suppress("[")
    + Combine(Literal("0x") + Word(hexnums))
    + Suppress("]")
    + Suppress("[")
    + CharsNotIn("]")
    + Suppress("]")
)

# mem_addr  ::= "[" PARTS "]"
_pp_mem_addr = Suppress("[") + CharsNotIn("]") + Suppress("]")

# abs_reg  ::= "|" register "|" ("." MODIFIER)*
_pp_abs_reg = (
    Suppress("|")
    + Combine(_pp_reg_base + ZeroOrMore(Literal(".") + Word(alphanums + "_")))
    + Suppress("|")
    + Group(ZeroOrMore(Suppress(".") + Word(alphanums + "_")))
)

# neg_pred  ::= "!" pred_name
_pp_neg_pred = Suppress("!") + _pp_pred_name

# special_imm  ::= "+INF" | "-INF" | "INF" | "-QNAN" | "QNAN" | "NaN"
_pp_special_imm = oneOf("+INF -INF INF -QNAN QNAN NaN")

# hex_imm  ::= "-"? "0x" HEX_DIGITS
_pp_hex_imm = Combine(pp.Optional(Literal("-")) + Literal("0x") + Word(hexnums))

# float_imm  ::= SIGN? (DIGITS "." DIGITS EXP? | "." DIGITS EXP?)
_pp_exp_suffix = Combine(oneOf("e E") + pp.Optional(oneOf("+ -")) + Word(nums))
_pp_float_imm = Combine(
    pp.Optional(oneOf("+ -"))
    + (
        (Word(nums) + Literal(".") + Word(nums) + pp.Optional(_pp_exp_suffix))
        | (Literal(".") + Word(nums) + pp.Optional(_pp_exp_suffix))
    )
)

# int_imm  ::= "-"? DIGITS
_pp_int_imm = Combine(pp.Optional(Literal("-")) + Word(nums))

# neg_reg  ::= "-" register
_pp_neg_reg = Suppress(Literal("-")) + _pp_register_token.copy()

# mnem_word  ::= UPPER_LETTER UPPER_ALNUM* ("." UPPER_ALNUM+)*
# Also matches lowercase identifiers (e.g. gsb0) as a final fallback.
_pp_mnem_word = Combine(
    Word(alphas, alphanums + "_") + ZeroOrMore(Literal(".") + Word(alphanums + "_"))
)

# --- Parse actions (convert matched tokens → AST nodes) ------------------

_pp_register_token.addParseAction(lambda t: _parse_register(t[0]))

_pp_label_ref.addParseAction(lambda t: LabelRef(name=t[0]))


def _annotation_action(tokens):
    text = tokens[0]
    if text.startswith("BRANCH_TARGETS"):
        content = text[len("BRANCH_TARGETS") :].strip()
        targets = tuple(t.strip() for t in content.split(",") if t.strip())
        return BranchTargetsOp(targets=targets)
    return []  # silently discard non-branch annotations


_pp_annotation.addParseAction(_annotation_action)

_pp_desc.addParseAction(lambda t: DescOp(kind=t[0], indices=tuple(t[1].asList())))

_pp_const_bank.addParseAction(lambda t: ConstBankOp(bank=t[0], offset=t[1]))


def _mem_addr_action(tokens):
    parts = tuple(p.strip() for p in tokens[0].split("+"))
    return MemAddrOp(parts=parts)


_pp_mem_addr.addParseAction(_mem_addr_action)


def _abs_reg_action(tokens):
    inner_reg = _parse_register(tokens[0])
    outer_mods = tuple(tokens[1].asList())
    return RegisterOp(
        name=inner_reg.name,
        modifiers=("abs",) + inner_reg.modifiers + outer_mods,
    )


_pp_abs_reg.addParseAction(_abs_reg_action)

_pp_neg_pred.addParseAction(lambda t: RegisterOp(name=t[0], modifiers=("neg",)))

_pp_special_imm.addParseAction(lambda t: _parse_immediate(t[0]))
_pp_hex_imm.addParseAction(lambda t: _parse_immediate(t[0]))
_pp_float_imm.addParseAction(lambda t: _parse_immediate(t[0]))
_pp_int_imm.addParseAction(lambda t: _parse_immediate(t[0]))


def _neg_reg_action(tokens):
    reg = tokens[0]
    if isinstance(reg, RegisterOp):
        return RegisterOp(name=reg.name, modifiers=("neg",) + reg.modifiers)
    # Fallback: token is still a raw string (copy didn't propagate action)
    parsed = _parse_register(reg)
    return RegisterOp(name=parsed.name, modifiers=("neg",) + parsed.modifiers)


_pp_neg_reg.addParseAction(_neg_reg_action)

_pp_mnem_word.addParseAction(lambda t: ImmediateOp(raw=t[0], value=None))

# --- Full operand (ordered as in grammar.md) -----------------------------

_pp_operand = (
    _pp_label_ref
    | _pp_annotation
    | _pp_desc
    | _pp_const_bank
    | _pp_mem_addr
    | _pp_abs_reg
    | _pp_neg_pred
    | _pp_special_imm
    | _pp_hex_imm
    | _pp_float_imm
    | _pp_int_imm
    | _pp_neg_reg
    | _pp_register_token
    | _pp_mnem_word
)

# Operand list: zero or more operands with optional comma / dot separators.
# Bare dots appear between a float immediate and a trailing modifier suffix
# (e.g. 4.789e-41.H0) and are silently discarded.
_pp_operand_list = ZeroOrMore(_pp_operand | Suppress(",") | Suppress("."))

# --- Instruction-line grammar --------------------------------------------

_pp_pred_token = Combine(Literal("@") + pp.Optional(Literal("!")) + _pp_pred_name)

_pp_mnemonic = Combine(
    Word(alphas.upper(), alphas.upper() + nums)
    + ZeroOrMore(Literal(".") + Word(alphas.upper() + nums + "_"))
)

_pp_instruction_line = (
    Suppress("/*")
    + Word(hexnums)("addr")
    + Suppress("*/")
    + pp.Optional(_pp_pred_token, default="")("pred")
    + _pp_mnemonic("mnem")
    + Group(_pp_operand_list)("operands")
    + Suppress(";")
)




# ---------------------------------------------------------------------------
# Predicate parser
# ---------------------------------------------------------------------------


def _parse_predicate(raw: str) -> Predicate:
    s = raw.lstrip("@")
    negated = s.startswith("!")
    if negated:
        s = s[1:]
    return Predicate(
        negated=negated,
        is_uniform=s.startswith("U"),
        name=s,
    )


# ---------------------------------------------------------------------------
# Line-level parser
# ---------------------------------------------------------------------------


def _parse_line(line: str) -> Optional[Statement]:
    stripped = line.strip()
    if not stripped:
        return None

    # Label?  .L_x_0:
    if stripped.endswith(":") and not stripped.startswith("/*"):
        name = stripped[:-1].rstrip()
        if name.startswith(".") and len(name) > 1:
            return Label(name=name)

    # Function definition?  .function kernel_foo
    if stripped.startswith(".function "):
        parts = stripped.split(None, 1)
        if len(parts) == 2:
            return FunctionDef(name=parts[1])
        return None

    # Instruction?  /*addr*/ [@pred] MNEMONIC operands ;
    if not stripped.startswith("/*"):
        return None

    try:
        result = _pp_instruction_line.parseString(stripped, parseAll=False)
    except ParseException:
        return None

    addr_str = result.addr
    pred_str = result.pred if result.pred else None
    mnem = result.mnem
    operands = tuple(result.operands.asList())

    predicate = _parse_predicate(pred_str) if pred_str else None

    return Instruction(
        address=int(addr_str, 16),
        address_str=addr_str,
        predicate=predicate,
        mnemonic=mnem,
        operands=operands,
        raw=line.rstrip(),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_text(text: str) -> Program:

    text = clean(text)

    prog = Program()
    for lineno, line in enumerate(text.splitlines(), 1):
        stmt = _parse_line(line)
        if stmt is not None:
            prog.statements.append(stmt)
        else:
            print(f"Failed to parse: {line}")
    return prog


def parse_file(path: str) -> Program:
    with open(path, "r", errors="replace") as fh:
        return parse_text(fh.read())


# ---------------------------------------------------------------------------
# Pretty-printer / dump
# ---------------------------------------------------------------------------


def dump(prog: Program, *, show_operand_types: bool = False) -> str:
    lines = []
    for stmt in prog.statements:
        if isinstance(stmt, Label) or isinstance(stmt, FunctionDef):
            lines.append(str(stmt))
        else:
            instr = stmt
            pred_s = f"  {instr.predicate}" if instr.predicate else ""
            if show_operand_types:
                ops_s = ", ".join(f"{o!r}" for o in instr.operands)
            else:
                ops_s = ", ".join(str(o) for o in instr.operands)
            ops_part = f"  {ops_s}" if ops_s else ""
            lines.append(f"{instr.address_str}: {pred_s}  {instr.mnemonic}{ops_part} ;")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI: python sass_parser.py <file.sass>  [--types]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Parse cleaned SASS dump and print AST summary."
    )
    ap.add_argument("file", nargs="?", help="Input file (default: stdin)")
    ap.add_argument("--types", action="store_true", help="Show operand types in output")
    ap.add_argument(
        "--stats",
        action="store_true",
        help="Print mnemonic frequency table instead of full dump",
    )
    ap.add_argument(
        "--roundtrip",
        action="store_true",
        help="Re-emit instructions; useful for sanity-checking the parser",
    )
    args = ap.parse_args()

    if args.file:
        prog = parse_file(args.file)
    else:
        prog = parse_text(sys.stdin.read())

    instrs = prog.instructions()
    print(
        f"# {len(prog.statements)} statements  "
        f"({len(instrs)} instructions, {len(prog.labels())} labels)",
        file=sys.stderr,
    )

    if args.stats:
        from collections import Counter

        freq = Counter(i.mnemonic for i in instrs)
        for mnem, count in freq.most_common():
            print(f"{count:6d}  {mnem}")
    elif args.roundtrip:
        print(dump(prog, show_operand_types=False))
    else:
        print(dump(prog, show_operand_types=args.types))
