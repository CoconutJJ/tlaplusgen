"""
sass_cfg.py  –  Build a Control Flow Graph from a parsed SASS program.

Public API
----------
    build_cfg(program: Program) -> CFG

Data types
----------
    TerminatorKind   – enum describing how a block exits
    BasicBlock       – a maximal straight-line sequence of instructions
    CFG              – the graph: blocks, edges, entry/exit sets

Design notes
------------
SASS control-flow peculiarities handled here:

  BRA / BRA.U (uniform)
    .U means the warp takes a *uniform* path (all lanes agree) – it is
    NOT the same as "unconditional".  Conditionality is determined solely
    by whether the instruction carries a predicate guard (@P0 / @!UP1 etc.).
    An unguarded BRA/BRA.U is unconditional; a guarded one is conditional.

  EXIT / RET.*
    Both are terminators with no successors.  RET.REL.NODEC is the form
    seen in inlined sub-routines (epilogue of the __internal_* helpers).

  CALL / CALL.REL.NOINC
    Treated as a fall-through instruction for intra-kernel CFG purposes.
    The callee is not modelled as a CFG edge.

  BRX
    Indirect branch.  The branch-target annotation (*"BRANCH_TARGETS …"*)
    is stripped by clean_sass.py, so targets are unknown at this stage.
    The block is marked INDIRECT; a TODO list is returned for callers that
    need to resolve them from the raw annotation.

  BSSY / BSYNC (convergence stack)
    Not control-flow edges for dataflow purposes; treated as ordinary
    instructions (fall-through).

  Predicated instructions (@P0 IADD3 …)
    Every SASS instruction can be predicated, but only branch instructions
    create actual CFG edges.  A predicated non-branch is a conditional
    *execution* within a single block – it stays in the block and produces
    no extra edge.

  Addresses are not densely sequential.
    nvdisasm leaves gaps (scheduling slots that are NOPs or are omitted).
    We connect blocks by label reference, not by address arithmetic.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

from .parser import (
    ConstBankOp,
    DescOp,
    FunctionDef,
    Instruction,
    Label,
    LabelRef,
    MemAddrOp,
    Program,
    RegisterOp,
    Statement,
    ImmediateOp,
)


class TerminatorKind(Enum):
    FALL_THROUGH = auto()  # last instr is not a branch; fall to next block
    UNCONDITIONAL = auto()  # BRA with no predicate guard
    CONDITIONAL = auto()  # BRA with predicate guard → taken + fall-through
    EXIT = auto()  # EXIT or RET – no successors
    INDIRECT = auto()  # BRX – targets unknown


# Mnemonic stems (before the first dot) that end a block with no successors.
_EXIT_MNEMONICS: Set[str] = {"EXIT", "RET"}

# Mnemonic stems that are branches (create CFG edges).
# CALL is deliberately excluded – treated as fall-through.
_BRANCH_MNEMONICS: Set[str] = {"BRA", "BRX"}

# Mnemonic stems that are *indirect* branches.
_INDIRECT_MNEMONICS: Set[str] = {"BRX"}


def _mnem_stem(mnemonic: str) -> str:
    """'BRA.U' → 'BRA',  'RET.REL.NODEC' → 'RET'"""
    return mnemonic.split(".")[0]


def _is_branch(instr: Instruction) -> bool:
    return _mnem_stem(instr.mnemonic) in _BRANCH_MNEMONICS


def _is_exit(instr: Instruction) -> bool:
    return _mnem_stem(instr.mnemonic) in _EXIT_MNEMONICS


def _is_indirect(instr: Instruction) -> bool:
    return _mnem_stem(instr.mnemonic) in _INDIRECT_MNEMONICS


# Uniform/non-uniform predicate register names that are NOT the always-true sentinel.
# UPT and PT mean "always execute" — branches using them are truly unconditional.
_ALWAYS_TRUE_PREDS: Set[str] = {"PT", "UPT"}


def _is_unconditional(instr: Instruction) -> bool:
    """
    A branch is unconditional iff it has no predicate guard AND carries no
    conditional predicate register as an operand.

    Two distinct patterns in SASS:

      @P0  BRA   `(.L_x_45)          -- guard syntax; predicate in instr.predicate
           BRA.U UP1, `(.L_x_0)      -- operand syntax; UP1 is the first operand
           BRA.U !UP0, `(.L_x_19)    -- negated form (NEG_PRED_OP token)

    A branch is unconditional only when BOTH hold:
      1. instr.predicate is None   (no @ guard)
      2. No RegisterOp operand before the LabelRef names a non-always-true pred
    """
    # Guard predicate present → conditional
    if instr.predicate is not None:
        return False

    # Check operands for a predicate register before the label ref
    for op in instr.operands:
        if isinstance(op, LabelRef):
            # Reached the target – no predicate reg found before it
            break
        if isinstance(op, RegisterOp):
            name = op.name
            # UP\d+ and P\d+ are genuine predicate registers
            is_pred = (name.startswith("UP") and name[2:].isdigit()) or (
                name.startswith("P") and name[1:].isdigit()
            )
            if is_pred and name not in _ALWAYS_TRUE_PREDS:
                return False  # conditional on this predicate

    return True


def _branch_target(instr: Instruction) -> LabelRef | ImmediateOp | None:
    """
    Return the label name that a BRA instruction jumps to, or None.
    The label ref is always a LabelRef operand; for BRA.U with a uniform
    predicate the operand order is  [UP1,  `(.L_x_0)]  or just  [`(.L_x_0)].
    """
    for op in instr.operands:
        if isinstance(op, LabelRef) or isinstance(op, ImmediateOp):
            return op

    return None


def _classify_terminator(instr: Instruction) -> TerminatorKind:
    if _is_exit(instr):
        return TerminatorKind.EXIT
    if _is_indirect(instr):
        return TerminatorKind.INDIRECT
    if _is_branch(instr):
        if _is_unconditional(instr):
            return TerminatorKind.UNCONDITIONAL
        return TerminatorKind.CONDITIONAL
    return TerminatorKind.FALL_THROUGH


# ---------------------------------------------------------------------------
# BasicBlock
# ---------------------------------------------------------------------------


@dataclass
class BasicBlock:
    id: int
    instructions: List[Instruction] = field(default_factory=list)

    # Label(s) that name the entry point of this block (usually 0 or 1).
    entry_labels: List[str] = field(default_factory=list)

    # Outgoing and incoming edges (by block id, resolved after all blocks built).
    successors: List["BasicBlock"] = field(default_factory=list)
    predecessors: List["BasicBlock"] = field(default_factory=list)

    # Describes the last instruction.
    terminator_kind: TerminatorKind = TerminatorKind.FALL_THROUGH

    # For INDIRECT blocks: raw BRX instruction for caller to inspect.
    indirect_branch: Optional[Instruction] = None

    # ------------------------------------------------------------------ #
    # Convenient accessors                                                 #
    # ------------------------------------------------------------------ #

    @property
    def first(self) -> Optional[Instruction]:
        return self.instructions[0] if self.instructions else None

    @property
    def last(self) -> Optional[Instruction]:
        return self.instructions[-1] if self.instructions else None

    @property
    def address(self) -> Optional[int]:
        """Address of the first instruction, or None if the block is empty."""
        return self.first.address if self.first else None

    @property
    def is_entry(self) -> bool:
        return self.id == 0

    @property
    def is_exit(self) -> bool:
        return self.terminator_kind in (TerminatorKind.EXIT, TerminatorKind.INDIRECT)

    @property
    def name(self) -> str:
        """Human-readable name: first entry label, or address, or id."""
        if self.entry_labels:
            return self.entry_labels[0]
        if self.address is not None:
            return f"bb_{self.address:#06x}"
        return f"bb_{self.id}"

    def __repr__(self) -> str:
        n_instrs = len(self.instructions)
        succs = [b.name for b in self.successors]
        return (
            f"BasicBlock({self.name!r}, "
            f"{n_instrs} instrs, "
            f"term={self.terminator_kind.name}, "
            f"succs={succs})"
        )

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, BasicBlock) and self.id == other.id


# ---------------------------------------------------------------------------
# CFG
# ---------------------------------------------------------------------------


@dataclass
class CFG:
    blocks: List[BasicBlock] = field(default_factory=list)
    entry: Optional[BasicBlock] = None

    # label name → block that starts with that label
    label_map: Dict[str, BasicBlock] = field(default_factory=dict)

    # Blocks whose terminator is EXIT or RET
    exit_blocks: List[BasicBlock] = field(default_factory=list)

    # Blocks with indirect branches (BRX) – caller must resolve
    indirect_blocks: List[BasicBlock] = field(default_factory=list)

    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.blocks)

    def __iter__(self):
        return iter(self.blocks)

    def block_at_label(self, label: str) -> Optional[BasicBlock]:
        return self.label_map.get(label)

    def successors_of(self, block: BasicBlock) -> List[BasicBlock]:
        return block.successors

    def predecessors_of(self, block: BasicBlock) -> List[BasicBlock]:
        return block.predecessors

    def postorder(self) -> List[BasicBlock]:
        """Blocks in reverse-postorder (standard for dataflow)."""
        visited: Set[int] = set()
        order: List[BasicBlock] = []

        def dfs(b: BasicBlock) -> None:
            if b.id in visited:
                return
            visited.add(b.id)
            for s in b.successors:
                dfs(s)
            order.append(b)

        if self.entry:
            dfs(self.entry)
        # append unreachable blocks at the end
        for b in self.blocks:
            if b.id not in visited:
                order.append(b)
        return order

    def reverse_postorder(self) -> List[BasicBlock]:
        return list(reversed(self.postorder()))


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_cfgs(program: Program) -> Dict[str, CFG]:
    """
    Build a dictionary of CFGs from a parsed SASS Program, one mapping per function name.
    """
    functions: Dict[str, CFG] = {}
    current_func = "unknown_kernel"
    current_stmts: List[Statement] = []

    for stmt in program.statements:
        if isinstance(stmt, FunctionDef):
            if current_stmts:
                functions[current_func] = _build_cfg_for_stmts(current_stmts)
            current_func = stmt.name
            current_stmts = []
        else:
            current_stmts.append(stmt)

    if current_stmts:
        functions[current_func] = _build_cfg_for_stmts(current_stmts)

    return functions


def build_cfg(program: Program) -> CFG:
    """Backward compatibility wrapper: return the first/only CFG found."""
    cfgs = build_cfgs(program)
    if not cfgs:
        return CFG()
    return list(cfgs.values())[0]


def _build_cfg_for_stmts(statements: List[Statement]) -> CFG:
    """
    Build a CFG from a linear sequence of statements (one function).

    The function performs four passes:

    Pass 1 – linearise
        Walk program.statements to extract a flat instruction list and a
        mapping from label name → the address of the instruction that
        immediately follows the label.

    Pass 2 – find leaders
        An instruction is a *leader* (first instruction of a new basic
        block) if:
          (a) it is the first instruction overall, or
          (b) the previous instruction was a terminator (branch / exit), or
          (c) it is the target of any branch (i.e. its address appears in
              the label→address map for some label that is referenced as a
              branch target in Pass 1).

    Pass 3 – build blocks
        Group consecutive instructions between leaders into BasicBlocks.
        Attach any labels whose associated address equals the block's first
        instruction address.

    Pass 4 – wire edges
        For each block, inspect the terminator instruction and add
        successor/predecessor edges.
    """

    # ------------------------------------------------------------------
    # Pass 1 – linearise instructions; build label → address map
    # ------------------------------------------------------------------

    instrs: List[Instruction] = []
    label_to_addr: Dict[str, int] = {}  # label name → address of next instr
    addr_to_labels: Dict[int, List[str]] = {}  # address → labels that precede it

    _pending_labels: List[str] = []

    for stmt in statements:
        if isinstance(stmt, Label):
            _pending_labels.append(stmt.name)
        elif isinstance(stmt, Instruction):
            for lbl in _pending_labels:
                label_to_addr[lbl] = stmt.address
                addr_to_labels.setdefault(stmt.address, []).append(lbl)
            _pending_labels.clear()
            instrs.append(stmt)

    if not instrs:
        return CFG()

    # addr → index in instrs
    addr_to_idx: Dict[int, int] = {instr.address: i for i, instr in enumerate(instrs)}

    # ------------------------------------------------------------------
    # Collect all branch target addresses (so we can mark leaders)
    # ------------------------------------------------------------------

    branch_target_addrs: Set[int] = set()
    for instr in instrs:
        if _is_branch(instr) and not _is_indirect(instr):
            tgt = _branch_target(instr)

            if isinstance(tgt, LabelRef) and tgt.name in label_to_addr:
                branch_target_addrs.add(label_to_addr[tgt.name])
            elif isinstance(tgt, ImmediateOp):
                branch_target_addrs.add(tgt.value)
    # ------------------------------------------------------------------
    # Pass 2 – identify leader indices
    # ------------------------------------------------------------------

    leaders: Set[int] = {0}  # first instruction is always a leader

    for i, instr in enumerate(instrs):
        kind = _classify_terminator(instr)
        if kind != TerminatorKind.FALL_THROUGH:
            # instruction after a terminator starts a new block
            if i + 1 < len(instrs):
                leaders.add(i + 1)

    # instruction whose address is a branch target starts a new block
    for addr in branch_target_addrs:
        if addr in addr_to_idx:
            leaders.add(addr_to_idx[addr])

    sorted_leaders = sorted(leaders)

    # ------------------------------------------------------------------
    # Pass 3 – partition instructions into blocks
    # ------------------------------------------------------------------

    cfg = CFG()
    leader_to_block: Dict[int, BasicBlock] = {}  # leader idx → block

    for block_idx, leader in enumerate(sorted_leaders):
        # Slice from this leader up to (but not including) the next leader
        if block_idx + 1 < len(sorted_leaders):
            end = sorted_leaders[block_idx + 1]
        else:
            end = len(instrs)

        block_instrs = instrs[leader:end]

        bb = BasicBlock(id=block_idx, instructions=block_instrs)

        # Attach labels whose address equals the first instruction's address
        first_addr = block_instrs[0].address
        for lbl in addr_to_labels.get(first_addr, []):
            bb.entry_labels.append(lbl)
            cfg.label_map[lbl] = bb

        # Classify terminator
        if block_instrs:
            last = block_instrs[-1]
            bb.terminator_kind = _classify_terminator(last)
            if bb.terminator_kind == TerminatorKind.INDIRECT:
                bb.indirect_branch = last

        cfg.blocks.append(bb)
        leader_to_block[leader] = bb

    # Convenience: reverse label_map – addr → block
    addr_to_block: Dict[int, BasicBlock] = {
        bb.first.address: bb for bb in cfg.blocks if bb.first
    }

    # ------------------------------------------------------------------
    # Pass 4 – wire edges
    # ------------------------------------------------------------------

    for i, bb in enumerate(cfg.blocks):
        if not bb.instructions:
            continue

        last = bb.last
        kind = bb.terminator_kind
        next_bb = cfg.blocks[i + 1] if i + 1 < len(cfg.blocks) else None
        if kind == TerminatorKind.EXIT:
            cfg.exit_blocks.append(bb)

        elif kind == TerminatorKind.INDIRECT:
            cfg.indirect_blocks.append(bb)

        elif kind == TerminatorKind.UNCONDITIONAL:
            tgt = _resolve_target(last, label_to_addr, addr_to_block)
            if tgt:
                _add_edge(bb, tgt)

        elif kind == TerminatorKind.CONDITIONAL:
            # taken edge
            tgt = _resolve_target(last, label_to_addr, addr_to_block)
            if tgt:
                _add_edge(bb, tgt)
            # fall-through edge
            if next_bb:
                _add_edge(bb, next_bb)

        else:  # FALL_THROUGH
            if next_bb:
                _add_edge(bb, next_bb)

    cfg.entry = cfg.blocks[0] if cfg.blocks else None
    return cfg


# ---------------------------------------------------------------------------
# SASS def/use semantics
# ---------------------------------------------------------------------------
#
# Ported from slicer.py's SASSInstruction.writes() / reads() logic, adapted
# for the new parser's operand types (RegisterOp, MemAddrOp, DescOp, etc.)
#
# The "first N operands are writes, rest are reads" convention is determined
# by the opcode (and sometimes arity).  MULTI_WRITER expands a single
# destination register into a contiguous range (e.g. LDG.E.128 writes 4).

# How many explicit write operands the instruction has.
# Key = opcode string  OR  (opcode, arity) tuple.
# Default (not listed) = 1.
_WRITE_COUNT: Dict[str | Tuple[str, int], int] = {
    "BSYNC": 0,
    ("IADD3", 5): 2,
    ("IADD3", 6): 3,
    ("UIADD3", 5): 2,
    ("LOP3.LUT", 7): 2,
    ("UIADD3", 6): 3,
    "RET.REL.NODEC": 0,
    ("BRA.U", 2): 0,
    "BRX": 0,
    "ELECT": 2,
    # instructions with no register write at all
    "EXIT": 0,
    "BRA": 0,
    "BRA.U": 0,
    "BRA.DIV": 0,
    "CALL.REL.NOINC": 0,
    "NOP": 0,
    "BAR.SYNC.DEFER_BLOCKING": 0,
    "BAR.SYNC": 0,
    "WARPSYNC": 0,
    "WARPSYNC.ALL": 0,
    "STG": 0,
    "STG.E": 0,
    "STG.E.128": 0,
    "STG.E.128.STRONG.GPU": 0,
    "STS": 0,
    "STS.64": 0,
    "STS.128": 0,
    "DEPBAR.WAIT": 0,
    "DEPBAR.WAIT.LE": 0,
    "DEPBAR": 0,
    "BSSY": 0,
    "MEMBAR.SC.GPU": 0,
    "MEMBAR.SC.CTA": 0,
    "MEMBAR.SC.SYS": 0,
}

# Some instructions write a contiguous range of registers starting at
# operand position `k`.  {operand_index: count}
_MULTI_WRITER: Dict[str, Dict[int, int]] = {
    "LDG.E.128.STRONG.GPU": {0: 4},
    "LDG.E.128.CONSTANT": {0: 4},
    "LDG.E.LTC128B.CONSTANT": {0: 4},
    "ULDC.64": {0: 2},
    "LDC.64": {0: 2},
    "HMMA.16816.F32": {0: 4},
    "IMAD.WIDE.U32": {0: 2},
    "UIMAD.WIDE.U32": {0: 2},
    "UIMAD.WIDE": {0: 2},
    "IMAD.WIDE": {0: 2},
    "LDSM.16.MT88.4": {0: 4},
    "LDS.128": {0: 4},
    "LDS.64": {0: 2},
    "LDG.E.128": {0: 4},
    "LDCU.64": {0: 2},
    "LDCU.128": {0: 4},
    "FMUL2.FTZ.RZ": {0: 2},
    "LDTM.x32": {0: 32},
    "FFMA2.FTZ.RZ": {0: 2},
    "LDTM.16dp256bit.x16": {0: 64},
    "LDTM.16dp256bit.x4": {0: 16},
    "LDTM.x128": {0: 128},
    "LDTM.x4": {0: 4},
    "HGMMA.64x256x16.F32": {0: 128},
}

# Opcodes where the first write operand is also read (read-modify-write).
_READ_WRITE: Dict[str, Set[int]] = {
    "IMAD.HI.U32": {0},
}

# Registers that are constant (always-available, never "defined").
_CONSTANT_REGS: Set[str] = {
    "RZ",
    "URZ",
    "SRZ",
    "PT",
    "UPT",
    "SR_TID.X",
    "SR_TID.Y",
    "SR_TID.Z",
    "SR_CTAID.X",
    "SR_CTAID.Y",
    "SR_CTAID.Z",
}


def _reg_name(op: RegisterOp) -> str:
    """Canonical register name for def/use tracking (strip modifiers)."""
    return op.name


def _is_constant_reg(name: str) -> bool:
    return name in _CONSTANT_REGS or name.startswith("SR_")


def _adjacent_regs(name: str, count: int) -> List[str]:
    """Given 'R4', return ['R5', 'R6', ...] for `count` adjacent registers."""
    import re as _re

    m = _re.match(r"([A-Za-z]+)(\d+)$", name)
    if not m:
        return [name] * count  # RZ etc. – repeat same
    pfx, num = m.group(1), int(m.group(2))
    return [f"{pfx}{num + i}" for i in range(1, count + 1)]


def _write_count(instr: Instruction) -> int:
    """How many leading operands are *writes*."""
    mnem = instr.mnemonic
    arity = len(instr.operands)
    # Check (opcode, arity) first, then plain opcode
    if (mnem, arity) in _WRITE_COUNT:
        return _WRITE_COUNT[(mnem, arity)]
    if mnem in _WRITE_COUNT:
        return _WRITE_COUNT[mnem]
    # For stems we haven't listed: try matching the stem
    stem = _mnem_stem(mnem)
    if stem in _WRITE_COUNT:
        return _WRITE_COUNT[stem]
    return 1  # default: first operand is the dest


def _extract_regs_from_operand(op) -> List[str]:
    """Extract register names from a compound operand (MemAddrOp, DescOp, etc.)."""
    import re as _re

    names: List[str] = []
    if isinstance(op, RegisterOp):
        n = _reg_name(op)
        if not _is_constant_reg(n):
            names.append(n)
    elif isinstance(op, MemAddrOp):
        # Parts are raw strings like "R4", "URZ", "0x70"
        for part in op.parts:
            p = part.strip()
            if _re.match(r"^[A-Za-z]", p):
                if not _is_constant_reg(p):
                    names.append(p)
    elif isinstance(op, DescOp):
        for idx in op.indices:
            # e.g. "UR4" or "R4.64+0x8"
            for token in _re.split(r"[+\[\]]", idx):
                token = token.strip()
                if _re.match(r"^[A-Za-z]", token):
                    # Strip trailing suffixes like .64
                    base = token.split(".")[0]
                    if not _is_constant_reg(base):
                        names.append(base)
    elif isinstance(op, ConstBankOp):
        # offset can be a register with an optional immediate: c[0x2][R4+0xc]
        import re as _re2

        if _re2.match(r"^[A-Za-z]", op.offset):
            # Strip any +/-offset suffix and dot-modifiers to get the bare name.
            base = _re2.split(r"[+\-]", op.offset)[0].split(".")[0].strip()
            if not _is_constant_reg(base):
                names.append(base)
    return names


def defs_of(instr: Instruction) -> Set[str]:
    """Set of register names written by `instr`."""
    wc = _write_count(instr)
    result: Set[str] = set()
    mnem = instr.mnemonic
    multi = _MULTI_WRITER.get(mnem, {})

    for k, op in enumerate(instr.operands[:wc]):
        if isinstance(op, RegisterOp):
            n = _reg_name(op)
            if not _is_constant_reg(n):
                result.add(n)
                # Expand multi-writer
                if k in multi:
                    for adj in _adjacent_regs(n, multi[k] - 1):
                        result.add(adj)
    return result


def uses_of(instr: Instruction) -> Set[str]:
    """Set of register names read by `instr`."""
    wc = _write_count(instr)
    result: Set[str] = set()
    mnem = instr.mnemonic

    # Read-modify-write: dest operand is also read
    rmw = _READ_WRITE.get(mnem, set())
    for k in rmw:
        if k < len(instr.operands):
            op = instr.operands[k]
            result.update(_extract_regs_from_operand(op))

    # All operands after the write-count are reads
    for op in instr.operands[wc:]:
        result.update(_extract_regs_from_operand(op))

    # Predicate guard is a read
    if instr.predicate is not None:
        pname = instr.predicate.name
        if not _is_constant_reg(pname):
            result.add(pname)

    return result


def _is_predicated(instr: Instruction) -> bool:
    """True if the instruction has a real (non-always-true) predicate guard."""
    if instr.predicate is None:
        return False
    return instr.predicate.name not in _ALWAYS_TRUE_PREDS


# ---------------------------------------------------------------------------
# Reaching definitions (iterative dataflow, instruction-level)
# ---------------------------------------------------------------------------

# A definition is identified by (register_name, id(instr)).
# We use id(instr) rather than instr.address because SASS files can contain
# multiple functions whose instruction addresses overlap.
Def = Tuple[str, int]


def compute_reaching_definitions(
    cfg: CFG,
) -> Dict[int, Set[Def]]:
    """
    Iterative reaching-definitions analysis at instruction granularity.

    Keys are ``id(instr)`` (Python object identity) so that duplicate
    instruction addresses across functions don't collide.

    Returns
    -------
    rd_in : dict[id(instr), set[(reg_name, id(def_instr))]]
    """

    # -- Phase 1: enumerate all definitions -----------------------------------
    all_defs_for_reg: Dict[str, Set[Def]] = {}
    gen_map: Dict[int, Set[Def]] = {}  # id(instr) → gen set

    for bb in cfg.blocks:
        for instr in bb.instructions:
            ds = defs_of(instr)
            gen: Set[Def] = set()
            for r in ds:
                d = (r, id(instr))
                gen.add(d)
                all_defs_for_reg.setdefault(r, set()).add(d)
            gen_map[id(instr)] = gen

    # -- Phase 2: build per-instruction kill sets -----------------------------
    kill_map: Dict[int, Set[Def]] = {}
    for bb in cfg.blocks:
        for instr in bb.instructions:
            k: Set[Def] = set()
            if not _is_predicated(instr):
                for r in defs_of(instr):
                    k |= all_defs_for_reg.get(r, set())
                    k -= gen_map[id(instr)]
            kill_map[id(instr)] = k

    # -- Phase 3: iterate to fixpoint ----------------------------------------
    rd: Dict[int, Set[Def]] = {
        id(instr): set() for bb in cfg.blocks for instr in bb.instructions
    }
    rd_in: Dict[int, Set[Def]] = {
        id(instr): set() for bb in cfg.blocks for instr in bb.instructions
    }

    changed = True
    while changed:
        changed = False
        for bb in cfg.blocks:
            for n, instr in enumerate(bb.instructions):
                if n == 0:
                    x: Set[Def] = set()
                    for pred in bb.predecessors:
                        if pred.instructions:
                            x |= rd[id(pred.last)]
                else:
                    x = rd[id(bb.instructions[n - 1])]

                rd_in[id(instr)] = x

                new_out = gen_map[id(instr)] | (x - kill_map[id(instr)])
                if new_out != rd[id(instr)]:
                    rd[id(instr)] = new_out
                    changed = True

    return rd_in


# ---------------------------------------------------------------------------
# Dominator tree & dominance frontiers  (for control-dependency slicing)
# ---------------------------------------------------------------------------


def _compute_dominators_for_blocks(
    blocks: List[BasicBlock],
    entry: BasicBlock,
    successors_fn,
    predecessors_fn,
) -> Tuple[Dict[int, Set[int]], Dict[int, Optional[int]]]:
    """
    Compute dominator sets and immediate dominators.
    """
    all_ids = {b.id for b in blocks}
    dom: Dict[int, Set[int]] = {}
    for b in blocks:
        dom[b.id] = {entry.id} if b.id == entry.id else set(all_ids)

    changed = True
    while changed:
        changed = False
        for b in blocks:
            if b.id == entry.id:
                continue
            preds = predecessors_fn(b)
            if not preds:
                new_dom = {b.id}
            else:
                new_dom = set.intersection(*(dom[p.id] for p in preds))
                new_dom = new_dom | {b.id}
            if new_dom != dom[b.id]:
                dom[b.id] = new_dom
                changed = True

    idom: Dict[int, Optional[int]] = {}
    for b in blocks:
        candidates = dom[b.id] - {b.id}
        if not candidates:
            idom[b.id] = None
            continue
        best = max(candidates, key=lambda c: len(dom[c]))
        idom[b.id] = best

    return dom, idom


def _compute_dominance_frontiers(
    blocks: List[BasicBlock],
    idom: Dict[int, Optional[int]],
    predecessors_fn,
) -> Dict[int, Set[int]]:
    """
    Dominance frontier: DF[b] = set of block ids where b's dominance ends.
    """
    df: Dict[int, Set[int]] = {b.id: set() for b in blocks}

    for b in blocks:
        preds = predecessors_fn(b)
        if len(preds) < 2:
            continue
        for p in preds:
            runner = p.id
            while runner is not None and runner != idom[b.id]:
                df[runner].add(b.id)
                runner = idom[runner]

    return df


# ---------------------------------------------------------------------------
# Program slicing
# ---------------------------------------------------------------------------


def slice_cfg(
    cfg: CFG,
    pattern: str,
    *,
    keep_control: bool = True,
) -> CFG:
    """
    Slice a CFG: keep only instructions necessary for instructions whose
    mnemonic matches ``pattern`` (a regex).

    Returns a **new** CFG with unneeded instructions removed; block
    structure and edges are preserved.
    """
    import re as _re

    pat = _re.compile(pattern)

    # ---- 1. Find seed instructions ----------------------------------------
    seed_ids: Set[int] = set()  # id(instr)
    id_to_instr: Dict[int, Instruction] = {}
    for bb in cfg.blocks:
        for instr in bb.instructions:
            id_to_instr[id(instr)] = instr
            if pat.search(instr.mnemonic) or _is_branch(instr):
                seed_ids.add(id(instr))

    if not seed_ids:
        print(
            f"WARNING: no instructions matched /{pattern}/. Slice will be empty.",
            file=sys.stderr,
        )

    # ---- 2. Reaching definitions ------------------------------------------
    rd_in = compute_reaching_definitions(cfg)

    # ---- 3. Backward walk: data dependencies ------------------------------
    important: Set[int] = set(seed_ids)  # id(instr) values
    worklist: List[int] = list(seed_ids)

    while worklist:
        iid = worklist.pop()
        instr = id_to_instr[iid]
        reads = uses_of(instr)
        reaching = rd_in.get(iid, set())
        for reg, def_id in reaching:
            if reg in reads and def_id not in important:
                important.add(def_id)
                worklist.append(def_id)

    # ---- 4. Control dependencies (optional) -------------------------------
    if keep_control and cfg.exit_blocks:
        virtual_exit_id = max(b.id for b in cfg.blocks) + 1
        virtual_exit = BasicBlock(id=virtual_exit_id)

        all_blocks_for_pdom = list(cfg.blocks) + [virtual_exit]

        rev_succs: Dict[int, List[BasicBlock]] = {
            b.id: list(b.predecessors) for b in cfg.blocks
        }
        rev_preds: Dict[int, List[BasicBlock]] = {
            b.id: list(b.successors) for b in cfg.blocks
        }
        rev_succs[virtual_exit_id] = list(cfg.exit_blocks)
        rev_preds[virtual_exit_id] = []
        for eb in cfg.exit_blocks:
            rev_preds[eb.id] = rev_preds.get(eb.id, []) + [virtual_exit]

        blk_by_id = {b.id: b for b in all_blocks_for_pdom}

        _, pdom_idom = _compute_dominators_for_blocks(
            all_blocks_for_pdom,
            virtual_exit,
            lambda b: rev_succs.get(b.id, []),
            lambda b: rev_preds.get(b.id, []),
        )

        pdf = _compute_dominance_frontiers(
            all_blocks_for_pdom,
            pdom_idom,
            lambda b: rev_preds.get(b.id, []),
        )

        def _blocks_with_important():
            result = set()
            for bb in cfg.blocks:
                for instr in bb.instructions:
                    if id(instr) in important:
                        result.add(bb.id)
                        break
            return result

        ctrl_changed = True
        while ctrl_changed:
            ctrl_changed = False
            blocks_with = _blocks_with_important()

            for bb in cfg.blocks:
                if bb.id not in blocks_with:
                    continue
                for cdep_id in pdf.get(bb.id, set()):
                    if cdep_id == virtual_exit_id:
                        continue
                    cdep_block = blk_by_id.get(cdep_id)
                    if cdep_block is None or not cdep_block.instructions:
                        continue
                    term = cdep_block.last
                    if term and _is_branch(term) and id(term) not in important:
                        important.add(id(term))
                        worklist.append(id(term))
                        ctrl_changed = True

            # Walk data deps of newly-added control branches
            while worklist:
                iid = worklist.pop()
                instr = id_to_instr[iid]
                reads = uses_of(instr)
                reaching = rd_in.get(iid, set())
                for reg, def_id in reaching:
                    if reg in reads and def_id not in important:
                        important.add(def_id)
                        worklist.append(def_id)

    # ---- 5. Build the sliced CFG ------------------------------------------
    new_cfg = CFG()

    for bb in cfg.blocks:
        new_bb = BasicBlock(
            id=bb.id,
            instructions=[i for i in bb.instructions if id(i) in important],
            entry_labels=list(bb.entry_labels),
            terminator_kind=bb.terminator_kind,
            indirect_branch=bb.indirect_branch,
        )
        new_cfg.blocks.append(new_bb)

    id_to_new = {b.id: b for b in new_cfg.blocks}
    new_cfg.label_map = {
        lbl: id_to_new[bb.id] for lbl, bb in cfg.label_map.items() if bb.id in id_to_new
    }

    for old_bb, new_bb in zip(cfg.blocks, new_cfg.blocks):
        for succ in old_bb.successors:
            new_succ = id_to_new[succ.id]
            _add_edge(new_bb, new_succ)

    new_cfg.entry = new_cfg.blocks[0] if new_cfg.blocks else None
    new_cfg.exit_blocks = [
        id_to_new[b.id] for b in cfg.exit_blocks if b.id in id_to_new
    ]
    new_cfg.indirect_blocks = [
        id_to_new[b.id] for b in cfg.indirect_blocks if b.id in id_to_new
    ]

    return new_cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_target(
    instr: Instruction,
    label_to_addr: Dict[str, int],
    addr_to_block: Dict[int, "BasicBlock"],
) -> Optional[BasicBlock]:
    """Look up the target block for a direct branch instruction."""
    label = _branch_target(instr)

    if isinstance(label, LabelRef):
        addr = label_to_addr[label.name]
    elif isinstance(label, ImmediateOp):
        addr = label.value

    return addr_to_block.get(addr)


def _add_edge(src: BasicBlock, dst: BasicBlock) -> None:
    """Add a directed edge src → dst (idempotent)."""
    if dst not in src.successors:
        src.successors.append(dst)
    if src not in dst.predecessors:
        dst.predecessors.append(src)


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------


def dump_cfg(cfg: CFG, *, show_instructions: bool = False) -> str:
    lines: List[str] = []
    lines.append(f"CFG: {len(cfg.blocks)} blocks")
    lines.append("")

    for bb in cfg.blocks:
        label_str = f"  [{', '.join(bb.entry_labels)}]" if bb.entry_labels else ""
        lines.append(f"  {bb.name}{label_str}")
        lines.append(f"    term : {bb.terminator_kind.name}")
        lines.append(f"    succs: {[b.name for b in bb.successors]}")
        lines.append(f"    preds: {[b.name for b in bb.predecessors]}")
        lines.append(f"    instrs: {len(bb.instructions)}")

        if show_instructions and bb.instructions:
            for instr in bb.instructions:
                pred_s = f"  {instr.predicate}" if instr.predicate else ""
                lines.append(f"      /*{instr.address_str}*/{pred_s}  {instr.mnemonic}")

        lines.append("")

    if cfg.exit_blocks:
        lines.append(f"  exit blocks : {[b.name for b in cfg.exit_blocks]}")
    if cfg.indirect_blocks:
        lines.append(f"  indirect BRX: {[b.name for b in cfg.indirect_blocks]}")

    return "\n".join(lines)


def _html(s: str) -> str:
    """Escape a plain string for use inside a Graphviz HTML-like label."""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def to_dot(
    cfg: CFG,
    *,
    title: str = "SASS CFG",
    show_instructions: bool = False,
    max_instrs: int = 40,
) -> str:
    """
    Emit a Graphviz DOT file.

    Parameters
    ----------
    show_instructions : bool
        If True, list every instruction inside each node.
        Defaults to False – the compact view just shows the block name
        and instruction count, which is much faster to render for large
        kernels.
    max_instrs : int
        When show_instructions=True, cap per-block instruction rows at
        this number and append a "… N more" footer.  Avoids huge SVGs.
    """
    lines: List[str] = []
    lines.append(f'digraph "{_html(title)}" {{')
    lines.append('  graph [fontname="Courier"];')
    lines.append('  node  [fontname="Courier" fontsize=10 shape=none margin=0];')
    lines.append('  edge  [fontname="Courier" fontsize=9];')
    lines.append("  rankdir=TB;")
    lines.append("")

    # ---- colour palette for terminator kinds ----
    _HDR_COLOUR = {
        TerminatorKind.FALL_THROUGH: "#D0E8FF",  # blue
        TerminatorKind.UNCONDITIONAL: "#D0FFD0",  # green
        TerminatorKind.CONDITIONAL: "#FFF5CC",  # yellow
        TerminatorKind.EXIT: "#FFD0D0",  # red
        TerminatorKind.INDIRECT: "#EED0FF",  # purple
    }
    _BORDER_COLOUR = {
        TerminatorKind.EXIT: "#AA0000",
        TerminatorKind.INDIRECT: "#7700AA",
    }

    for bb in cfg.blocks:
        hdr_bg = _HDR_COLOUR.get(bb.terminator_kind, "#EEEEEE")
        border_c = _BORDER_COLOUR.get(bb.terminator_kind, "#333333")
        n_instrs = len(bb.instructions)

        # --- header row: block name + labels + instruction count ---
        label_str = " ".join(_html(l) + ":" for l in bb.entry_labels)
        if label_str:
            hdr_text = (
                f"{label_str} &nbsp;<FONT COLOR='#555555'>({n_instrs} instrs)</FONT>"
            )
        else:
            hdr_text = (
                f"<FONT COLOR='#555555'>{_html(bb.name)} ({n_instrs} instrs)</FONT>"
            )

        rows: List[str] = [
            f'<TR><TD ALIGN="LEFT" BGCOLOR="{hdr_bg}" '
            f'BORDER="0"><B>{hdr_text}</B></TD></TR>'
        ]

        # --- instruction rows (optional) ---
        if show_instructions:
            shown = bb.instructions[:max_instrs]
            for instr in shown:
                pred_s = (
                    _html(str(instr.predicate)) + "&nbsp;" if instr.predicate else ""
                )
                ops_s = ",&nbsp;".join(_html(str(o)) for o in instr.operands)
                addr_s = f'<FONT COLOR="#888888">{instr.address_str}</FONT>'
                mnem_s = f"<B>{_html(instr.mnemonic)}</B>"
                cell = f"{addr_s}&nbsp;&nbsp;{pred_s}{mnem_s}"
                if ops_s:
                    cell += f"&nbsp;&nbsp;{ops_s}"
                rows.append(f'<TR><TD ALIGN="LEFT" BORDER="0">{cell}</TD></TR>')
            if n_instrs > max_instrs:
                omitted = n_instrs - max_instrs
                rows.append(
                    f'<TR><TD ALIGN="LEFT" BORDER="0">'
                    f'<FONT COLOR="#888888"><I>… {omitted} more</I></FONT>'
                    f"</TD></TR>"
                )

        # --- assemble HTML-like label ---
        table = (
            f'<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" '
            f'CELLPADDING="3" COLOR="{border_c}">' + "".join(rows) + "</TABLE>"
        )
        lines.append(f"  bb{bb.id} [label=<{table}>];")

    lines.append("")

    # ---- edges ----
    for bb in cfg.blocks:
        for i, succ in enumerate(bb.successors):
            attrs: List[str] = []
            if bb.terminator_kind == TerminatorKind.CONDITIONAL:
                taken = i == 0
                attrs.append(f'label="{"T" if taken else "F"}"')
                attrs.append(f'color="{"#007700" if taken else "#AA5500"}"')
            elif bb.terminator_kind == TerminatorKind.UNCONDITIONAL:
                attrs.append("style=bold")
            elif bb.terminator_kind == TerminatorKind.INDIRECT:
                attrs.append('style=dashed color="#7700AA"')

            attr_str = f" [{', '.join(attrs)}]" if attrs else ""
            lines.append(f"  bb{bb.id} -> bb{succ.id}{attr_str};")

    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from .parser import parse_file, parse_text

    ap = argparse.ArgumentParser(description="Build and display a SASS CFG.")
    ap.add_argument("file", nargs="?", help="Cleaned SASS file (default: stdin)")
    ap.add_argument("--dot", action="store_true", help="Emit Graphviz DOT instead")
    ap.add_argument("--instrs", action="store_true", help="Show instructions per block")
    ap.add_argument(
        "--slice",
        metavar="PATTERN",
        help="Slice CFG: keep only instructions whose mnemonic matches "
        "this regex, plus their data/control dependencies",
    )
    ap.add_argument(
        "--no-control-deps",
        action="store_true",
        help="With --slice, skip control-dependency tracking "
        "(keep only data dependencies)",
    )
    args = ap.parse_args()

    prog = parse_file(args.file) if args.file else parse_text(sys.stdin.read())
    cfgs = build_cfgs(prog)

    for func_name, cfg in cfgs.items():
        if args.slice:
            cfg = slice_cfg(cfg, args.slice, keep_control=not args.no_control_deps)
            n_kept = sum(len(bb.instructions) for bb in cfg.blocks)
            print(
                f"# Sliced {func_name} on /{args.slice}/ → {n_kept} instructions kept",
                file=sys.stderr,
            )

        if args.dot:
            print(f"\n// CFG for {func_name}")
            print(to_dot(cfg, show_instructions=args.instrs))
        else:
            print(f"\n# --- Function: {func_name} ---")
            print(dump_cfg(cfg, show_instructions=args.instrs))
