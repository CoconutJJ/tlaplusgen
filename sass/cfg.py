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

from parser import (
    Instruction,
    Label,
    LabelRef,
    Program,
    RegisterOp,
    Statement,
    ImmediateOp,
)


# ---------------------------------------------------------------------------
# Terminator classification
# ---------------------------------------------------------------------------


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


def build_cfg(program: Program) -> CFG:
    """
    Build a CFG from a parsed SASS Program.

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

    statements = program.statements

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


def compute_reaching_definitions(cfg: CFG):

    def transfer(in_set: set[Instruction], bb: BasicBlock):

        pass

    in_sets: dict[int, set[Instruction]] = dict()
    out_sets: dict[int, set[Instruction]] = dict()
    for bb in cfg.postorder():
        pass


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
    from parser import parse_file, parse_text

    ap = argparse.ArgumentParser(description="Build and display a SASS CFG.")
    ap.add_argument("file", nargs="?", help="Cleaned SASS file (default: stdin)")
    ap.add_argument("--dot", action="store_true", help="Emit Graphviz DOT instead")
    ap.add_argument("--instrs", action="store_true", help="Show instructions per block")
    args = ap.parse_args()

    prog = parse_file(args.file) if args.file else parse_text(sys.stdin.read())
    cfg = build_cfg(prog)

    if args.dot:
        print(to_dot(cfg, show_instructions=args.instrs))
    else:
        print(dump_cfg(cfg, show_instructions=args.instrs))
