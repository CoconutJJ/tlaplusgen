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
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from .cleaner import clean
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
    modifiers: Tuple[str, ...] = ()  # e.g. ("reuse",) or ("F32x2.HI_LO",) or ("64",)

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
# Tokeniser / operand parser
# ---------------------------------------------------------------------------

# --- token regexes (applied in order, LONGEST/MOST-SPECIFIC FIRST) ---
#
# Ordering rules that prevent mis-tokenisation:
#   • DESC before MEM_ADDR  (desc[…][…] has brackets too)
#   • REGISTER before MNEM_WORD  (avoid SR being caught as mnem)
#   • Within REGISTER: UP\d+ before U?PT? so UP4 beats UP|4
#                      P\d+  before U?PT? so P0 beats P|0
#   • Register modifier suffix (?:\.[\w]+)* absorbs .reuse .F32x2.HI_LO .64 etc.
#   • ABS_REG  before REGISTER  (|R74| wraps a register)
#   • NEG_PRED before REGISTER/MNEM_WORD  (!UPT !PT in operand position)
#   • ANNOTATION strips (*"…"*) blobs that nvdisasm emits after BRX targets
#   • SPECIAL_IMM before HEX/FLOAT/INT  (+INF -INF -QNAN …)

_TOK_PATTERNS = [
    # Structural / compound tokens
    ("LABEL_REF", r"`\(\.[A-Za-z_]\w*\)"),  # `(.L_x_0)
    ("ADDR_HEX", r"/\*([0-9a-fA-F]+)\*/"),  # /*0a30*/
    ("PRED", r"@!?(?:UP\d+|U?PT?|P\d+|B\d+)"),  # @P0 @!UP1 @UPT
    ("DESC", r"(?:g?desc|tmem|idesc)(?:\[[^\]]*\])+"),  # desc[UR4][R4.64+0x8]
    ("CONST_BANK", r"c\[0x[0-9a-fA-F]+\]\[[^\]]+\]"),  # c[0x0][0x37c]
    ("MEM_ADDR", r"\[[^\]]+\]"),  # [UR4+0x8]
    ("ANNOTATION", r'\(\*"[^"]*"\*\)'),  # (*"BRANCH_TARGETS…"*)
    # Absolute-value wrapper |R74|
    ("ABS_REG", r"\|[A-Za-z]\w*\|"),
    # Negated predicate in operand position: !UPT  !PT  !P0
    ("NEG_PRED_OP", r"!(?:UP\d+|U?PT?|P\d+)"),
    # Special float constants before numeric patterns
    ("SPECIAL_IMM", r"[+-]?(?:\+INF|-INF|INF|-?QNAN|NaN)"),
    ("HEX_IMM", r"-?0x[0-9a-fA-F]+"),
    ("FLOAT_IMM", r"[+-]?(?:\d+\.\d+(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)"),
    ("INT_IMM", r"-?\d+"),
    # Registers — include modifier suffix so R100.F32x2.HI_LO is ONE token.
    # SR_CTAID.X, SR_TID.X, etc. are matched by the SR_ branch.
    # Priority within the base name alternation:
    #   UP\d+ > U?PT? so UP4 is one token not UP|4
    #   P\d+  > U?PT? so P0  is one token not P|0
    (
        "REGISTER",
        r"(?:SR_[A-Z0-9_.]+|U?RZ?(?:\d+)?|UP\d+|P\d+|U?PT?|B\d+|SRZ?)"
        r"(?:\.(?:[A-Za-z0-9x_]+))*",
    ),
    # Bare upper-case words (mnemonics in operand slot, e.g. ALL in WARPSYNC.ALL)
    ("MNEM_WORD", r"[A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+)*"),
    ("COMMA", r","),
    ("SEMI", r";"),
    ("COLON", r":"),
    ("SKIP", r"\s+|//[^\n]*"),
]
_TOK_RE = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in _TOK_PATTERNS))

# Regex for a whole cleaned instruction line
_INSTR_LINE_RE = re.compile(
    r"^\s*/\*(?P<addr>[0-9a-fA-F]+)\*/\s*"
    r"(?P<pred>@[!]?(?:U?P(?:T|\d+)|B\d+))?\s*"
    r"(?P<mnem>[A-Z][A-Z0-9]*(?:\.[A-Z0-9_]+)*)\s*"
    r"(?P<body>[^;]*);?"
)

_LABEL_LINE_RE = re.compile(r"^\s*(?P<name>\.[A-Za-z_]\w*)\s*:")


# ---------------------------------------------------------------------------
# Operand-level parsing helpers
# ---------------------------------------------------------------------------


def _parse_register(raw: str) -> RegisterOp:
    """Parse a register token that may include dotted modifiers.
    'R4'               → RegisterOp('R4', ())
    'R4.reuse'         → RegisterOp('R4', ('reuse',))
    'R100.F32x2.HI_LO' → RegisterOp('R100', ('F32x2', 'HI_LO'))
    'SR_CTAID.X'       → RegisterOp('SR_CTAID.X', ())
    'SR_TID.Y'         → RegisterOp('SR_TID.Y', ())
    """
    idx = raw.find(".")
    if idx == -1:
        return RegisterOp(name=raw, modifiers=())
    # SR registers carry their dimension (X/Y/Z) as part of the name, not
    # as a modifier, so that SR_TID.X and SR_TID.Y remain distinct.
    if raw[:idx].startswith("SR_"):
        return RegisterOp(name=raw, modifiers=())
    return RegisterOp(name=raw[:idx], modifiers=tuple(raw[idx + 1 :].split(".")))


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


def _parse_desc(raw: str) -> DescOp | ImmediateOp:
    """desc[UR4][R4.64+0x8]  → DescOp('desc', ('UR4', 'R4.64+0x8'))"""
    m = re.match(r"(g?desc|tmem|idesc)((?:\[[^\]]*\])+)", raw)
    if not m:
        # fallback — treat as unknown immediate
        return ImmediateOp(raw=raw, value=None)
    kind = m.group(1)
    brackets = re.findall(r"\[([^\]]*)\]", m.group(2))
    return DescOp(kind=kind, indices=tuple(brackets))


def _parse_const_bank(raw: str) -> ConstBankOp:
    """c[0x0][0x37c]"""
    m = re.match(r"c\[(0x[0-9a-fA-F]+)\]\[([^\]]+)\]", raw)
    bank = m.group(1) if m else "?"
    offset = m.group(2) if m else raw
    return ConstBankOp(bank=bank, offset=offset)


def _parse_mem_addr(raw: str) -> MemAddrOp:
    """[UR4+0x8]  [R43+URZ+0x70]"""
    inner = raw[1:-1]  # strip [ ]
    parts = tuple(p.strip() for p in inner.split("+"))
    return MemAddrOp(parts=parts)


def _parse_label_ref(raw: str) -> LabelRef:
    """`(.L_x_0)"""
    m = re.match(r"`\((\.[A-Za-z_]\w*)\)", raw)
    name = m.group(1) if m else raw
    return LabelRef(name=name)


def _parse_operand_token(tok_type: str, tok_val: str) -> Optional[Operand]:
    if tok_type == "LABEL_REF":
        return _parse_label_ref(tok_val)
    if tok_type == "DESC":
        return _parse_desc(tok_val)
    if tok_type == "CONST_BANK":
        return _parse_const_bank(tok_val)
    if tok_type == "MEM_ADDR":
        return _parse_mem_addr(tok_val)
    if tok_type in ("HEX_IMM", "FLOAT_IMM", "INT_IMM", "SPECIAL_IMM"):
        return _parse_immediate(tok_val)
    if tok_type == "REGISTER":
        return _parse_register(tok_val)
    if tok_type == "ABS_REG":
        # |R74| → RegisterOp with "abs" modifier
        inner = tok_val[1:-1]  # strip |  |
        reg = _parse_register(inner)
        return RegisterOp(name=reg.name, modifiers=("abs",) + reg.modifiers)
    if tok_type == "NEG_PRED_OP":
        # !UPT  !PT  !P0 → RegisterOp with "neg" modifier
        inner = tok_val[1:]  # strip leading !
        reg = _parse_register(inner)
        return RegisterOp(name=reg.name, modifiers=("neg",) + reg.modifiers)
    if tok_type == "MNEM_WORD":
        # In operand position: ALL (WARPSYNC.ALL), or bare fallback keyword
        return ImmediateOp(raw=tok_val, value=None)
    if tok_type == "ANNOTATION":
        m = re.match(r'\(\*"BRANCH_TARGETS\s+([^"]*)"\*\)', tok_val)
        if m:
            parts = m.group(1).split(",")
            targets = tuple(p.strip() for p in parts if p.strip())
            return BranchTargetsOp(targets=targets)
        return None  # other annotations — silently discard
    return None  # COMMA, SEMI, SKIP, COLON → caller handles


# ---------------------------------------------------------------------------
# Operand list parser  (the full body string after the mnemonic)
# ---------------------------------------------------------------------------

# Collect register-like words that appear inside operand body and may have
# modifiers appended via dot: R4.reuse  R4.F32x2.HI_LO  UR4.F32  etc.
_REG_WITH_MODS_RE = re.compile(
    r"\b(U?RZ?(?:\d+)?|U?PT?|UP\d+|P\d+|B\d+|SRZ?)"  # register name
    r"((?:\.[A-Za-z0-9_]+)+)?"  # optional dotted mods
)


def _lex_operands(body: str) -> List[Operand]:
    """
    Tokenise the operand string and return a list of Operand objects.
    Commas are separators (ignored); semicolons stop lexing.
    The body may contain complex tokens like:
        desc[UR4][R4.64+0x8],  `(.L_x_0),  [R2+URZ+0xe0],  !PT,  -R5
    """
    ops: List[Operand] = []

    # We lex the body with the full token set but only consume operands
    for m in _TOK_RE.finditer(body):
        kind = m.lastgroup
        val = m.group()

        if kind in ("SKIP", "COMMA"):
            continue
        if kind == "SEMI":
            break

        op = _parse_operand_token(kind, val)
        if op is not None:
            ops.append(op)

    return ops


# ---------------------------------------------------------------------------
# Predicate parser
# ---------------------------------------------------------------------------

_PRED_RE = re.compile(r"@(?P<neg>!?)(?P<name>U?P(?:T|\d+)|B\d+)")


def _parse_predicate(raw: str) -> Predicate:
    m = _PRED_RE.match(raw)
    if not m:
        return Predicate(negated=False, is_uniform=False, name=raw.lstrip("@!"))
    name = m.group("name")
    return Predicate(
        negated=m.group("neg") == "!",
        is_uniform=name.startswith("U"),
        name=name,
    )


# ---------------------------------------------------------------------------
# Line-level parser
# ---------------------------------------------------------------------------


def _parse_line(line: str) -> Optional[Statement]:
    stripped = line.strip()
    if not stripped:
        return None

    # Label?
    lm = _LABEL_LINE_RE.match(stripped)
    if lm:
        # Make sure it's not an instruction line that starts with a weird token
        # (instruction lines always start with /*)
        if not stripped.startswith("/*"):
            return Label(name=lm.group("name"))

    # Function definition?
    fm = re.match(r"^\.function\s+(?P<name>\S+)$", stripped)
    if fm:
        return FunctionDef(name=fm.group("name"))

    # Instruction?
    im = _INSTR_LINE_RE.match(stripped)
    if im:
        addr_str = im.group("addr")
        pred_raw = im.group("pred")
        mnem = im.group("mnem")
        body = im.group("body") or ""

        predicate = _parse_predicate(pred_raw) if pred_raw else None
        operands = tuple(_lex_operands(body))

        return Instruction(
            address=int(addr_str, 16),
            address_str=addr_str,
            predicate=predicate,
            mnemonic=mnem,
            operands=operands,
            raw=line.rstrip(),
        )

    return None


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
        if isinstance(stmt, Label):
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
