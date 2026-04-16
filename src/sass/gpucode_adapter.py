"""
gpucode_adapter.py  --  Adapter that wraps gpucode-analyzer's parser, CFG
builder, and slicer so they produce the data structures expected by
tla_codegen.py (i.e. the types defined in sass/parser.py and sass/cfg.py).

Usage
-----
    from sass.gpucode_adapter import parse_file, build_cfgs, slice_cfg

These three functions have the same signatures and return types as their
counterparts in sass.parser / sass.cfg, but use gpucode-analyzer under the
hood.
"""

from __future__ import annotations

import sys
import re
from typing import Dict, List, Optional, Set, Tuple

# --- gpucode-analyzer imports (adjust path at runtime) ---------------------
import os as _os

_GCA_ROOT = _os.path.join(_os.path.dirname(__file__), "gpucode-analyzer")
if _GCA_ROOT not in sys.path:
    sys.path.insert(0, _GCA_ROOT)

from gpucodeanalyzer.generic_cfg import (
    CFG as GCA_CFG,
    BasicBlock as GCA_BasicBlock,
    Instruction as GCA_Instruction,
    ControlInsn as GCA_ControlInsn,
)
from gpucodeanalyzer.isa.sass.sass import (
    SASSFile,
    SASSInstruction,
    SASSControlInsn,
    SASSRegister,
    SASSAddress,
)
from gpucodeanalyzer.slice import Slicer, get_labels

# --- our native types (the ones tla_codegen.py consumes) -------------------
from .parser import (
    Instruction,
    Predicate,
    RegisterOp,
    ImmediateOp,
    MemAddrOp,
    ConstBankOp,

)
from .cfg import (
    CFG,
    BasicBlock,
    TerminatorKind,
)


# ---------------------------------------------------------------------------
# Operand conversion:  SASSRegister / SASSAddress / str  →  our Operand types
# ---------------------------------------------------------------------------

_HEX_IMM_RE = re.compile(r"^-?0x[0-9a-fA-F]+$")
_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+)([eE][+-]?\d+)?$")
_LABEL_RE = re.compile(r"^0x[0-9a-fA-F]+$")  # branch target addresses
_CBANK_RE = re.compile(r"^-?c\[(.+?)\]\[(.+?)\]$")
_CX_RE = re.compile(r"^-?cx\[(.+?)\]\[(.+?)\]$")
_SPECIAL_FLOAT = {"INF", "+INF", "-INF", "QNAN", "-QNAN", "NaN"}


def _convert_operand(op, opcode: str = "", arg_idx: int = 0):
    """Convert a single gpucode-analyzer operand to our parser.Operand type."""

    if isinstance(op, SASSRegister):
        mods: list[str] = []
        if op.is_inverted:
            mods.append("neg")  # ! prefix on predicates → "neg" modifier
        if op.is_negated:
            mods.append("neg")
        if op.is_reuse:
            mods.append("reuse")
        if op.suffix:
            # suffix is like ".B1" or ".H0_H0" (with leading dot)
            for part in op.suffix.lstrip(".").split("."):
                if part:
                    mods.append(part)
        return RegisterOp(name=op.n, modifiers=tuple(mods))

    if isinstance(op, SASSAddress):
        # SASSAddress: [R0.X16+0x1000] → MemAddrOp
        parts: list[str] = [op.reg1.n]
        if op.reg2:
            parts.append(op.reg2.n)
        if op.imm:
            parts.append(op.imm)
        return MemAddrOp(parts=tuple(parts))

    if isinstance(op, str):
        s = op.strip()

        # constant bank: c[0x0][0x37c] or cx[UR4][0x4]
        cm = _CBANK_RE.match(s) or _CX_RE.match(s)
        if cm:
            return ConstBankOp(bank=cm.group(1), offset=cm.group(2))

        # label ref (branch target as hex address)
        if _LABEL_RE.match(s) and _is_branch_target_context(opcode, arg_idx):
            return ImmediateOp(raw=s, value=int(s, 16))

        # hex immediate
        if _HEX_IMM_RE.match(s):
            return ImmediateOp(raw=s, value=int(s, 16))

        # special float constants
        if s.upper().rstrip() in _SPECIAL_FLOAT:
            return ImmediateOp(raw=s, value=None)

        # float
        if _FLOAT_RE.match(s):
            try:
                return ImmediateOp(raw=s, value=float(s))
            except ValueError:
                return ImmediateOp(raw=s, value=None)

        # plain integer
        if _INT_RE.match(s):
            return ImmediateOp(raw=s, value=int(s))

        # fallback: treat as immediate with unknown value
        return ImmediateOp(raw=s, value=None)

    # unknown type — wrap as immediate
    return ImmediateOp(raw=str(op), value=None)


def _is_branch_target_context(opcode: str, arg_idx: int) -> bool:
    """Heuristic: is this argument position a branch target address?"""
    stem = opcode.split(".")[0]
    if stem in ("BRA", "BRX"):
        return True
    return False


# ---------------------------------------------------------------------------
# Instruction conversion:  SASSInstruction  →  our Instruction
# ---------------------------------------------------------------------------

def _convert_predicate(pred_str: str | None) -> Predicate | None:
    """Convert gpucode-analyzer's predicate string ("!P0", "P1", None) to Predicate."""
    if pred_str is None:
        return None
    s = pred_str
    negated = s.startswith("!")
    if negated:
        s = s[1:]
    return Predicate(
        negated=negated,
        is_uniform=s.startswith("U"),
        name=s,
    )


def _convert_instruction(gca_instr: GCA_Instruction) -> Instruction:
    """Convert a gpucode-analyzer SASSInstruction to our Instruction type."""
    assert isinstance(gca_instr, SASSInstruction)

    # address: gpucode-analyzer stores as hex string in .label
    addr_str = gca_instr.label  # e.g. "0a30"
    addr_int = int(addr_str, 16)

    predicate = _convert_predicate(gca_instr.predicate)

    operands = tuple(
        _convert_operand(arg, gca_instr.opcode, idx)
        for idx, arg in enumerate(gca_instr.args)
    )

    return Instruction(
        address=addr_int,
        address_str=addr_str,
        predicate=predicate,
        mnemonic=gca_instr.opcode,
        operands=operands,
        raw=gca_instr.insn if hasattr(gca_instr, "insn") else "",
    )


# ---------------------------------------------------------------------------
# CFG conversion:  gpucode-analyzer CFG  →  our CFG
# ---------------------------------------------------------------------------

# Mnemonic stems that terminate with no successors
_EXIT_MNEMONICS = {"EXIT", "RET"}
_BRANCH_MNEMONICS = {"BRA", "BRX"}
_INDIRECT_MNEMONICS = {"BRX"}


def _mnem_stem(mnemonic: str) -> str:
    return mnemonic.split(".")[0]


def _classify_gca_block(gca_bb: GCA_BasicBlock) -> TerminatorKind:
    """Classify a gpucode-analyzer BasicBlock's terminator."""
    if not gca_bb.code:
        return TerminatorKind.FALL_THROUGH

    last = gca_bb.code[-1]
    if not last.is_control():
        return TerminatorKind.FALL_THROUGH

    stem = _mnem_stem(last.opcode)
    if stem in _EXIT_MNEMONICS:
        return TerminatorKind.EXIT
    if stem in _INDIRECT_MNEMONICS:
        return TerminatorKind.INDIRECT
    if stem in _BRANCH_MNEMONICS:
        if last.is_conditional():
            return TerminatorKind.CONDITIONAL
        return TerminatorKind.UNCONDITIONAL

    return TerminatorKind.FALL_THROUGH


def _convert_cfg(gca_cfg: GCA_CFG) -> CFG:
    """Convert a gpucode-analyzer CFG to our CFG type."""
    cfg = CFG()

    # Map gca block name → our BasicBlock
    name_to_bb: dict[str, BasicBlock] = {}

    for block_idx, gca_bb in enumerate(gca_cfg.blocks):
        # Skip _start and _exit sentinel blocks that have no instructions
        if gca_bb.name in ("_start", "_exit") and not gca_bb.code:
            continue

        instructions = [_convert_instruction(i) for i in gca_bb.code]

        # Entry labels: use the first instruction's address as a label
        entry_labels: list[str] = []
        if instructions:
            # Create a synthetic label from the hex address for label_map
            addr_label = f".L_{instructions[0].address_str}"
            entry_labels.append(addr_label)

        bb = BasicBlock(
            id=block_idx,
            instructions=instructions,
            entry_labels=entry_labels,
            terminator_kind=_classify_gca_block(gca_bb),
        )

        cfg.blocks.append(bb)
        name_to_bb[gca_bb.name] = bb

        for lbl in entry_labels:
            cfg.label_map[lbl] = bb

    # Renumber block ids sequentially
    for new_id, bb in enumerate(cfg.blocks):
        bb.id = new_id

    # Wire edges: translate gca successor dict to our successor list
    for gca_bb in gca_cfg.blocks:
        if gca_bb.name not in name_to_bb:
            continue
        our_bb = name_to_bb[gca_bb.name]

        # gpucode-analyzer successors are a dict with labels like
        # 'next', 'true', 'false', 'indirect1', etc.
        # For conditional branches, we need taken (true) first, then fall-through (false).
        # For unconditional, just 'next' or 'true'.
        succ_order = _ordered_successors(gca_bb)
        for gca_succ in succ_order:
            if gca_succ.name in name_to_bb:
                target = name_to_bb[gca_succ.name]
                if target not in our_bb.successors:
                    our_bb.successors.append(target)
                if our_bb not in target.predecessors:
                    target.predecessors.append(our_bb)

    cfg.entry = cfg.blocks[0] if cfg.blocks else None
    cfg.exit_blocks = [b for b in cfg.blocks if b.terminator_kind == TerminatorKind.EXIT]
    cfg.indirect_blocks = [
        b for b in cfg.blocks if b.terminator_kind == TerminatorKind.INDIRECT
    ]

    return cfg


def _ordered_successors(gca_bb: GCA_BasicBlock) -> list[GCA_BasicBlock]:
    """Return successor blocks in the order our codegen expects.

    For conditional branches: [taken, fall-through]
    For everything else: preserve insertion order.
    """
    succs = gca_bb.successors
    if not succs:
        return []

    # Conditional: 'true' first, 'false' second
    if "true" in succs and "false" in succs:
        result = [succs["true"]]
        result.append(succs["false"])
        # Add any extra targets (indirect, etc.)
        for label, bb in succs.items():
            if label not in ("true", "false") and bb not in result:
                result.append(bb)
        return result

    # Default: return in dict order (which preserves insertion order in Python 3.7+)
    return list(succs.values())


# ---------------------------------------------------------------------------
# Public API  (drop-in replacements for sass.parser / sass.cfg functions)
# ---------------------------------------------------------------------------


def _strip_liveness_comments(path: str) -> str:
    """Strip trailing register-liveness comments from nvdisasm output.

    nvdisasm emits lines like:
        /*0000*/   LDC R1, c[0x0][0x37c] ;   // |  1   ^ ...
    gpucode-analyzer expects:
        /*0000*/   LDC R1, c[0x0][0x37c] ;

    Also converts '.text.FUNC_NAME:' lines to 'Function : FUNC_NAME'
    so gpucode-analyzer recognises function boundaries.
    """
    _TEXT_SECTION_RE = re.compile(r"^\.text\.(\S+):")
    _FUNC_LABEL_RE = re.compile(r"^(kernel_\S+):")
    _LABEL_DEF_RE = re.compile(r"^([\.\$][\w\$]+):")
    _ADDR_RE = re.compile(r"/\*([0-9a-fA-F]+)\*/")
    _LABEL_REF_RE = re.compile(r"`\(([\.\$][\w\$]+)\)")
    # Labels inside BRANCH_TARGETS annotations: (*"BRANCH_TARGETS .L_x_1,.L_x_2"*)
    _BRANCH_TARGETS_RE = re.compile(r'\(\*"BRANCH_TARGETS ([^"]+)"\*\)')

    # --- Pass 1: collect raw lines and build label→address map ---
    raw_lines: list[str] = []
    label_to_addr: dict[str, str] = {}
    pending_labels: list[str] = []

    with open(path, "r", errors="replace") as f:
        for line in f:
            raw_lines.append(line.rstrip())

    for line in raw_lines:
        lm = _LABEL_DEF_RE.match(line)
        if lm:
            pending_labels.append(lm.group(1))
            continue
        if pending_labels:
            am = _ADDR_RE.search(line)
            if am:
                addr = am.group(1)
                # Pad to 4+ hex chars and prefix with 0x
                hex_addr = f"0x{addr}"
                for lbl in pending_labels:
                    label_to_addr[lbl] = hex_addr
                pending_labels.clear()

    # --- Pass 2: transform lines and extract BRX metadata ---
    lines: list[str] = []
    in_function = False
    current_fn: str | None = None
    # metadata: {fn_name: {"EIATTR_INDIRECT_BRANCH_TARGETS": {pc: [target_addrs]}}}
    metadata: dict[str, dict] = {}

    for line in raw_lines:
        # Convert nvdisasm function markers to cuobjdump format
        m = _TEXT_SECTION_RE.match(line)
        if m:
            if in_function:
                lines.append("\t..........")
            current_fn = m.group(1)
            lines.append(f"\tFunction : {current_fn}")
            in_function = True
            continue
        m = _FUNC_LABEL_RE.match(line)
        if m:
            continue
        # Skip label definition lines (gpucode-analyzer doesn't use them)
        if _LABEL_DEF_RE.match(line):
            continue

        # Extract BRX BRANCH_TARGETS metadata and strip annotation from line.
        # nvdisasm format:  BRX R4 -0xa40 ... (*"BRANCH_TARGETS .L_x_1,.L_x_2"*);
        # gpucode-analyzer needs: BRX R4 -0xa40 ;  (plus metadata dict)
        brx_m = _BRANCH_TARGETS_RE.search(line)
        if brx_m and current_fn:
            addr_m = _ADDR_RE.search(line)
            if addr_m:
                pc = addr_m.group(1)
                target_labels = [t.strip() for t in brx_m.group(1).split(",")]
                target_addrs = []
                for lbl in target_labels:
                    if lbl in label_to_addr:
                        resolved = label_to_addr[lbl]
                        if resolved.startswith("0x"):
                            resolved = resolved[2:]
                        target_addrs.append(resolved)
                if target_addrs:
                    fn_meta = metadata.setdefault(current_fn, {})
                    brx_targets = fn_meta.setdefault(
                        "EIATTR_INDIRECT_BRANCH_TARGETS", {}
                    )
                    brx_targets[pc] = target_addrs
            # Remove the annotation; gpucode-analyzer gets targets from metadata
            line = _BRANCH_TARGETS_RE.sub("", line)

        # Strip trailing // comments from instruction lines
        semi_idx = line.find(";")
        if semi_idx >= 0:
            after = line[semi_idx + 1:].lstrip()
            if after.startswith("//"):
                line = line[:semi_idx + 1]
        # Replace label references with hex addresses
        def _resolve_label(m):
            lbl = m.group(1)
            if lbl in label_to_addr:
                return label_to_addr[lbl]
            return m.group(0)
        line = _LABEL_REF_RE.sub(_resolve_label, line)
        lines.append(line)
    if in_function:
        lines.append("\t..........")
    return "\n".join(lines), metadata


def parse_file(path: str, metadata: dict | None = None) -> dict[str, GCA_CFG]:
    """Parse a SASS file using gpucode-analyzer and build CFGs.

    Returns a dict of function_name → gpucode-analyzer CFG (already built).
    This is an intermediate form; use build_cfgs() to get our CFG type.
    """
    import tempfile

    cleaned, brx_metadata = _strip_liveness_comments(path)

    # Merge extracted BRX metadata with any user-provided metadata
    if metadata is None:
        metadata = brx_metadata
    else:
        for fn, fn_meta in brx_metadata.items():
            metadata.setdefault(fn, {}).update(fn_meta)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sass", delete=False) as tmp:
        tmp.write(cleaned)
        tmp_path = tmp.name

    try:
        sass_file = SASSFile(tmp_path, metadata=metadata)
    finally:
        _os.unlink(tmp_path)

    result: dict[str, GCA_CFG] = {}

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


def build_cfgs(gca_cfgs: dict[str, GCA_CFG]) -> Dict[str, CFG]:
    """Convert gpucode-analyzer CFGs to our CFG type.

    Takes the output of parse_file() and returns a dict matching the
    signature of sass.cfg.build_cfgs().
    """
    return {name: _convert_cfg(gca_cfg) for name, gca_cfg in gca_cfgs.items()}


def slice_cfg(
    gca_cfg: GCA_CFG,
    pattern: str,
    *,
    keep_control: bool = True,
) -> CFG:
    """Slice a gpucode-analyzer CFG and return our CFG type.

    Parameters
    ----------
    gca_cfg   : A gpucode-analyzer CFG (from parse_file())
    pattern   : Regex for instruction opcodes to keep (e.g. "WARPSYNC|USETMAXREG")
    keep_control : Whether to include control-dependency branches
    """
    labels = get_labels(gca_cfg, [f"re:{pattern}"])

    slicer = Slicer(gca_cfg)
    slicer.slice(labels)
    sliced_gca = slicer.get_skeleton_cfg()

    return _convert_cfg(sliced_gca)
