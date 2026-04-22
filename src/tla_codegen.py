"""
tla_codegen.py  --  Lift a sliced SASS CFG into a TLASassProcess.

Implements exactly the instruction semantics defined in sass/sass_insns.h.

Uses gpucode-analyzer abstractions directly: the parser (SASSFile), CFG
(generic_cfg.CFG), slicer (Slicer) and instruction/operand types
(SASSInstruction, SASSRegister, SASSAddress) live in
src/sass/gpucode-analyzer.  This module exposes thin wrappers for parse
and slice so callers do not have to manage that import path themselves.
"""

from __future__ import annotations

import os as _os
import re
import sys as _sys
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# gpucode-analyzer import setup
# ---------------------------------------------------------------------------

# _GCA_ROOT = _os.path.join(
#     _os.path.dirname(_os.path.abspath(__file__)), "sass", "gpucode-analyzer"
# )
# if _GCA_ROOT not in _sys.path:
#     _sys.path.insert(0, _GCA_ROOT)

from sass.gpucodeanalyzer.gpucodeanalyzer.generic_cfg import (  # noqa: E402
    CFG as GCACFG,
    BasicBlock as GCABasicBlock,
)
from sass.gpucodeanalyzer.gpucodeanalyzer.isa.sass.sass import (  # noqa: E402
    SASSFile,
    SASSInstruction,
    SASSControlInsn,
    SASSRegister,
    SASSAddress,
)
from sass.gpucodeanalyzer.gpucodeanalyzer.slice import Slicer, get_labels  # noqa: E402

from tla_module import (  # noqa: E402
    Expr,
    Add,
    IfThenElse,
    Literal,
    Unchanged,
    Not,
    Neg,
    Abs,
    Or,
    Min,
    Max,
    FunnelShr,
    Constant,
)
from tla_sass import TLASassProcess, TLASassThread  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALWAYS_TRUE_PREDS: frozenset[str] = frozenset({"PT", "UPT"})
_ZERO_REGS: frozenset[str] = frozenset({"RZ", "URZ", "SRZ"})
_DISCARD_DSTS: frozenset[str] = _ZERO_REGS | _ALWAYS_TRUE_PREDS

_EXIT_MNEMONICS: frozenset[str] = frozenset({"EXIT", "RET"})
_BRANCH_MNEMONICS: frozenset[str] = frozenset({"BRA", "BRX"})
_INDIRECT_MNEMONICS: frozenset[str] = frozenset({"BRX"})

_PRED_NAME_RE = re.compile(r"^U?P\d+$")
_CBANK_RE = re.compile(r"^-?c\[(?P<bank>[^\]]+)\]\[(?P<offset>[^\]]+)\]$")
_CX_RE = re.compile(r"^-?cx\[(?P<bank>[^\]]+)\]\[(?P<offset>[^\]]+)\]$")
_HEX_IMM_RE = re.compile(r"^-?0x[0-9a-fA-F]+$")
_INT_IMM_RE = re.compile(r"^-?\d+$")
_FLOAT_IMM_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+)([eE][+-]?\d+)?$")


# ---------------------------------------------------------------------------
# Terminator classification (inferred from the last instruction of a block)
# ---------------------------------------------------------------------------


class TerminatorKind(Enum):
    FALL_THROUGH = auto()
    UNCONDITIONAL = auto()
    CONDITIONAL = auto()
    EXIT = auto()
    INDIRECT = auto()


def _mnem_stem(opcode: str) -> str:
    return opcode.split(".")[0]


def _pred_name(pred: Optional[str]) -> Optional[str]:
    if pred is None:
        return None
    return pred[1:] if pred.startswith("!") else pred


def _classify_terminator(bb: GCABasicBlock) -> TerminatorKind:
    if not bb.code:
        return TerminatorKind.FALL_THROUGH
    last = bb.code[-1]
    if not last.is_control():
        return TerminatorKind.FALL_THROUGH

    stem = _mnem_stem(last.opcode)
    if stem in _EXIT_MNEMONICS:
        name = _pred_name(last.predicate)
        if name is not None and name not in _ALWAYS_TRUE_PREDS:
            return TerminatorKind.CONDITIONAL
        return TerminatorKind.EXIT
    if stem in _INDIRECT_MNEMONICS:
        return TerminatorKind.INDIRECT
    if stem in _BRANCH_MNEMONICS:
        if last.is_conditional():
            return TerminatorKind.CONDITIONAL
        return TerminatorKind.UNCONDITIONAL
    return TerminatorKind.FALL_THROUGH


def _is_real_block(bb: GCABasicBlock) -> bool:
    """Skip the _start / _exit sentinel blocks that have no code."""
    return bool(bb.code)


def _ordered_successors(bb: GCABasicBlock) -> List[GCABasicBlock]:
    """Return successor blocks in [taken, fall-through, ...] order.

    gpucode-analyzer labels its successor edges:
      - 'true'/'false'            for conditional branches
      - 'next' (+ 'next1', ...)   for unconditional/indirect branches
      - 'next'                    for fall-through
    Conditional blocks want (taken, fall) in that order so the generator
    can emit appendBranchInstruction(pred, taken, fall).
    """
    succs = bb.successors
    if not succs:
        return []
    if "true" in succs:
        order = [succs["true"]]
        if "false" in succs:
            order.append(succs["false"])
        for lbl, s in succs.items():
            if lbl not in ("true", "false") and s not in order:
                order.append(s)
        return order
    return list(succs.values())


# ---------------------------------------------------------------------------
# Parsing + slicing entry points  (thin wrappers over gpucode-analyzer)
# ---------------------------------------------------------------------------


def _discover_metadata(path: str) -> Optional[dict]:
    """Look for a JSON metadata file adjacent to ``path``.

    Supports the regalloc-dataset conventions:
        foo.sass       → foo.sass.json, foo.json, meta.json (same directory)
    """
    import json

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


def parse_sass_file(
    path: str, extra_metadata: Optional[dict] = None
) -> Dict[str, GCACFG]:
    """Parse a SASS file and return a dict of function_name → gpucode-analyzer CFG.

    Expects cuobjdump-style input (``Function : NAME`` markers, ``..........``
    function terminators, raw SASS instruction lines) — the format
    gpucode-analyzer consumes natively and the regalloc-dataset ships with.
    Each returned CFG has already had ``cfg.build()`` called on it.

    If BRX indirect branches are present, the caller must supply
    ``extra_metadata`` (keyed by function name, containing
    ``EIATTR_INDIRECT_BRANCH_TARGETS``) or place a ``meta.json`` / ``<stem>.json``
    next to the SASS file.
    """
    metadata = extra_metadata if extra_metadata is not None else _discover_metadata(path)
    sass_file = SASSFile(path, metadata=metadata)

    result: Dict[str, GCACFG] = {}
    if hasattr(sass_file, "codes") and sass_file.codes:
        for fn_name in sass_file.codes:
            cfg = GCACFG(sass_file, fn_name=fn_name)
            cfg.build()
            result[fn_name] = cfg
    else:
        cfg = GCACFG(sass_file)
        cfg.build()
        result["unknown_kernel"] = cfg
    return result


def slice_cfg(cfg: GCACFG, pattern: str) -> GCACFG:
    """Slice a gpucode-analyzer CFG on a regex of instruction opcodes."""
    labels = get_labels(cfg, [f"re:{pattern}"])
    slicer = Slicer(cfg)
    slicer.slice(labels)
    return slicer.get_skeleton_cfg()


# ---------------------------------------------------------------------------
# Codegen
# ---------------------------------------------------------------------------


class SassCFGCodegen:
    def __init__(self) -> None:
        self.log: list[str] = []
        self._handlers: dict = {}
        self.process = None 
        self._build_handler_table()

    # ------------------------------------------------------------------
    # Handler table  (maps SASS mnemonics to handler methods)
    # ------------------------------------------------------------------

    def _build_handler_table(self) -> None:
        h = self._handlers

        for m in ("MOV", "UMOV", "S2R", "S2UR", "CS2R", "R2UR", "R2UR.OR"):
            h[m] = self._h_mov

        h["UP2UR"] = self._h_up2ur
        h["IABS"] = self._h_iabs

        h["I2F.RP"] = self._h_fp_passthrough
        h["MUFU.RCP"] = self._h_fp_passthrough
        h["F2I.FTZ.U32.TRUNC.NTZ"] = self._h_fp_passthrough

        for m in ("IADD3", "UIADD3"):
            h[m] = self._h_iadd3

        for m in (
            "IMAD",
            "IMAD.U32",
            "IMAD.MOV",
            "IMAD.MOV.U32",
            "IMAD.IADD",
            "IMAD.IADD.U32",
            "IMAD.SHL",
            "IMAD.SHL.U32",
            "UIMAD",
            "UIMAD.U32",
        ):
            h[m] = self._h_imad

        h["IMAD.HI.U32"] = self._h_imad_hi_u32

        for m in ("SHF.R.U32.HI", "USHF.R.U32.HI"):
            h[m] = self._h_shf_r_u32_hi
        for m in ("SHF.R.S32.HI", "USHF.R.S32.HI"):
            h[m] = self._h_shf_r_s32_hi
        for m in ("SHF.L.U32", "USHF.L.U32"):
            h[m] = self._h_shf_l_u32

        for m in ("LOP3.LUT", "ULOP3.LUT"):
            h[m] = self._h_lop3_lut
        h["PLOP3.LUT"] = self._h_plop3_lut
        for m in ("SEL", "USEL"):
            h[m] = self._h_sel

        h["LEA"] = self._h_lea
        for m in ("LEA.HI", "ULEA.HI"):
            h[m] = self._h_lea_hi
        for m in ("LEA.HI.SX32", "ULEA.HI.SX32"):
            h[m] = self._h_lea_hi_sx32

        h["IMNMX.U32"] = self._h_imnmx_u32

        h["VIADD"] = self._h_viadd
        h["VIMNMX"] = self._h_vimnmx
        for m in ("VIADDMNMX", "VIADDMNMX.U32"):
            h[m] = self._h_viaddmnmx

        for pfx in ("ISETP", "UISETP"):
            h[f"{pfx}.LT.AND"] = self._h_isetp_lt_and
            h[f"{pfx}.LT.U32.AND"] = self._h_isetp_lt_and
            h[f"{pfx}.GT.AND"] = self._h_isetp_gt_and
            h[f"{pfx}.GT.U32.AND"] = self._h_isetp_gt_and
            h[f"{pfx}.GE.AND"] = self._h_isetp_ge_and
            h[f"{pfx}.GE.U32.AND"] = self._h_isetp_ge_and
            h[f"{pfx}.NE.AND"] = self._h_isetp_ne_and
            h[f"{pfx}.NE.U32.AND"] = self._h_isetp_ne_and
            h[f"{pfx}.EQ.AND"] = self._h_isetp_eq_and
            h[f"{pfx}.EQ.U32.AND"] = self._h_isetp_eq_and
            h[f"{pfx}.LE.AND"] = self._h_isetp_le_and
            h[f"{pfx}.LE.U32.AND"] = self._h_isetp_le_and
            h[f"{pfx}.GE.OR"] = self._h_isetp_ge_or
            h[f"{pfx}.GE.U32.OR"] = self._h_isetp_ge_or
        h["P2R"] = self._h_p2r

        for m in ("ULDC", "LDC", "LDCU"):
            h[m] = self._h_uldc
        for m in ("ULDC.64", "LDC.64", "LDCU.64"):
            h[m] = self._h_uldc_64

        for m in ("SYNCS.PHASECHK.TRANS64.TRYWAIT", "SYNCS.PHASECHK.TRANS64"):
            h[m] = self._h_syncs_phasechk

        h["USETMAXREG.TRY_ALLOC.CTAPOOL"] = self._h_usetmaxnre_inc
        h["USETMAXREG.DEALLOC.CTAPOOL"] = self._h_usetmaxnre_dec

        for m in ("WARPSYNC", "WARPSYNC.ALL"):
            h[m] = self._h_stutter

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        cfg: GCACFG,
        name: str,
        regPerThread: int,
        n_warps: int = 1,
        gridDim: Tuple[int, int, int] = (1, 1, 1),
        blockDim: Tuple[int, int, int] = (1, 1, 1),
    ) -> TLASassProcess:
        """Lift ``cfg`` into a TLASassProcess named ``name``.

        ``cfg`` is a gpucode-analyzer CFG representing exactly one kernel
        (as returned by ``parse_sass_file`` or ``slice_cfg``).
        """
        regs_zero, regs_false, regs_true = self._collect_registers(cfg)
        used_regs = set(regs_zero) | set(regs_false) | set(regs_true)
        print(
            f"[codegen] registers in sliced CFG: {len(used_regs)} (was 410 hardcoded)",
            flush=True,
        )

        proc = TLASassProcess(name, gridDim, blockDim, regPerThread, apalache_compatible=True)
        proc.initialize()

        self._sr_constants: dict[str, Constant] = {}
        self._sr_proc = proc

        template = proc.template_thread
        assert template is not None

        self._encode_cfg(template, cfg)

        return proc

    # ------------------------------------------------------------------
    # Register discovery
    # ------------------------------------------------------------------

    def _collect_registers(
        self, cfg: GCACFG
    ) -> tuple[list[str], list[str], list[str]]:
        """Return (regs_zero, regs_false, regs_true)."""
        seen: set[str] = set()
        discard_written: set[str] = set()

        for bb in cfg.blocks:
            for instr in bb.code:
                if not isinstance(instr, SASSInstruction):
                    continue
                pname = _pred_name(instr.predicate)
                if pname is not None and pname not in _ALWAYS_TRUE_PREDS:
                    seen.add(pname)
                for r in instr.writes():
                    name = r.n if isinstance(r, SASSRegister) else str(r)
                    if name in _ALWAYS_TRUE_PREDS:
                        discard_written.add(name)
                    elif name not in _ZERO_REGS and not name.startswith("SR_"):
                        seen.add(name)
                for r in instr.reads():
                    name = r.n if isinstance(r, SASSRegister) else str(r)
                    if name not in _DISCARD_DSTS and not name.startswith("SR_"):
                        seen.add(name)

        def _key(name: str) -> tuple[int, int]:
            m = re.match(r"([A-Za-z]+)(\d+)$", name)
            order = {"R": 0, "UR": 1, "P": 2, "UP": 3}
            if m:
                return (order.get(m.group(1), 9), int(m.group(2)))
            return (9, 0)

        regs_zero = sorted([r for r in seen if not _PRED_NAME_RE.match(r)], key=_key)
        regs_false = sorted([r for r in seen if _PRED_NAME_RE.match(r)], key=_key)
        return regs_zero, regs_false, sorted(discard_written, key=_key)

    # ------------------------------------------------------------------
    # CFG -> TLA+ encoding
    # ------------------------------------------------------------------

    def _resolve_real_successor(
        self, cfg: GCACFG, bb: Optional[GCABasicBlock]
    ) -> Optional[GCABasicBlock]:
        """Follow single-successor empty (sliced-away) blocks until a block
        with code is reached. Returns None if the chain dead-ends.

        Successor lookups are normalised via ``cfg.names_to_blocks`` because
        ``CFG.copy()`` in gpucode-analyzer keeps the original successor-dict
        entries, so identity-based lookups against the sliced blocks can miss.
        """
        visited: set[str] = set()
        while bb is not None:
            # Always map back to the canonical block in this CFG.
            bb = cfg.names_to_blocks.get(bb.name, bb)

            assert bb is not None
            
            if bb.code:
                return bb
            if bb.name in visited:
                return None
            visited.add(bb.name)
            succs = list(bb.successors.values())
            if not succs:
                return None
            bb = succs[0]
        return None

    def _encode_cfg(self, thread: TLASassThread, cfg: GCACFG) -> None:
        self._cfg = cfg
        real_blocks = [b for b in cfg.blocks if _is_real_block(b)]
        if not real_blocks:
            return

        # Entry block: follow _start's successor chain through empty blocks.
        entry: Optional[GCABasicBlock] = None
        start = cfg.names_to_blocks.get("_start") if hasattr(cfg, "names_to_blocks") else None
        if start and start.successors:
            for s in start.successors.values():
                entry = self._resolve_real_successor(cfg, s)
                if entry is not None:
                    break
        if entry is None:
            entry = real_blocks[0]

        ordered: list[GCABasicBlock] = [entry] + [b for b in real_blocks if b is not entry]

        block_states: dict[str, str] = {}
        for idx, bb in enumerate(ordered):
            if idx == 0:
                block_states[bb.name] = thread._currentState()
            else:
                first_label = bb.code[0].label
                block_states[bb.name] = thread.allocateState(f"bb_{idx}_{first_label}")

        for bb in ordered:
            self._encode_block(thread, bb, block_states)

    def _encode_block(
        self,
        thread: TLASassThread,
        bb: GCABasicBlock,
        block_states: dict[str, str],
    ) -> None:
        thread.setState(block_states[bb.name])
        instrs = bb.code
        term_kind = _classify_terminator(bb)
        real_succs: list[GCABasicBlock] = []
        for s in _ordered_successors(bb):
            resolved = self._resolve_real_successor(self._cfg, s)
            if resolved is not None and resolved not in real_succs:
                real_succs.append(resolved)

        for instr in instrs[:-1]:
            self._emit_instruction(thread, instr)

        if not instrs:
            if real_succs:
                self._emit_goto(thread, block_states[real_succs[0].name])
            return

        last = instrs[-1]

        if term_kind == TerminatorKind.EXIT:
            thread.stopInstruction()

        elif term_kind == TerminatorKind.FALL_THROUGH:
            self._emit_instruction(thread, last)
            if real_succs:
                self._emit_goto(thread, block_states[real_succs[0].name])

        elif term_kind == TerminatorKind.UNCONDITIONAL:
            if real_succs:
                self._emit_goto(thread, block_states[real_succs[0].name])

        elif term_kind == TerminatorKind.CONDITIONAL:
            pred = self._branch_pred(thread, last)
            taken = block_states[real_succs[0].name] if len(real_succs) > 0 else None
            fall = block_states[real_succs[1].name] if len(real_succs) > 1 else None
            if pred is not None and taken and fall:
                thread.appendBranchInstruction(pred, taken, fall)
            elif taken:
                self._emit_goto(thread, taken)

        elif term_kind == TerminatorKind.INDIRECT:
            if real_succs:
                self._emit_indirect_branch(thread, bb, block_states, real_succs)
            else:
                self.log.append(
                    f"INDIRECT BRANCH at /*{last.label}*/ {last.opcode}: "
                    f"targets unknown -- treating block as EXIT"
                )
                thread.stopInstruction()

    def _emit_goto(self, thread: TLASassThread, target_state: str) -> None:
        current = thread._currentState()
        pc_trans = thread.pcTransition(current, target_state)
        unchanged_vars = thread._unchangedExcept([thread.process.getPcMap()])
        expr = (pc_trans & Unchanged(unchanged_vars)) if unchanged_vars else pc_trans
        safe = re.sub(r"[^a-zA-Z0-9]", "_", current)
        defn_name = f"goto_{safe}"
        defn = thread.process.createDefinition(defn_name, expr, params=[thread.process.thread_parameter])
        thread.thread_definitions.append(defn)

    def _emit_indirect_branch(
        self,
        thread: TLASassThread,
        bb: GCABasicBlock,
        block_states: dict[str, str],
        succs: List[GCABasicBlock],
    ) -> None:
        current = thread._currentState()
        unchanged_vars = thread._unchangedExcept([thread.process.getPcMap()])
        gotos = []
        for succ in succs:
            target = block_states[succ.name]
            gotos.append(thread.pcTransition(current, target))
        expr = Or(*gotos) if len(gotos) > 1 else gotos[0]
        if unchanged_vars:
            expr = expr & Unchanged(unchanged_vars)
        safe = re.sub(r"[^a-zA-Z0-9]", "_", current)
        defn_name = f"brx_{safe}"
        defn = thread.process.createDefinition(defn_name, expr, params=[thread.process.thread_parameter])
        thread.thread_definitions.append(defn)

    def _emit_instruction(self, thread: TLASassThread, instr: SASSInstruction) -> None:
        mnem = instr.opcode
        handler = self._handlers.get(mnem)
        if handler is None:
            handler = self._handlers.get(_mnem_stem(mnem))
        if handler is None:
            pred_str = instr.predicate if instr.predicate else ""
            self.log.append(
                f"UNSUPPORTED /*{instr.label}*/  {mnem}"
                + (f"  pred={pred_str}" if pred_str else "")
                + f"  args=[{', '.join(self._fmt_arg(a) for a in instr.args)}]"
            )
            return
        handler(thread, instr)

    @staticmethod
    def _fmt_arg(op) -> str:
        if isinstance(op, SASSRegister):
            return op.operand()
        return str(op)

    # ------------------------------------------------------------------
    # Operand helpers
    # ------------------------------------------------------------------

    def _op_expr(self, thread: TLASassThread, op) -> Expr:
        if isinstance(op, SASSRegister):
            name = op.n
            if name in _ZERO_REGS:
                return Literal(0)
            if name in _ALWAYS_TRUE_PREDS:
                return Literal(True)
            if name.startswith("SR_"):
                return self._sr_constant(name)
            expr = thread.getRegister(name)
            if op.is_inverted:
                expr = Not(expr)
            elif op.is_negated:
                if _PRED_NAME_RE.match(name):
                    expr = Not(expr)
                else:
                    expr = Neg(expr)
            return expr

        if isinstance(op, SASSAddress):
            return self._mem_addr_expr(thread, op)

        if isinstance(op, str):
            return self._str_op_expr(thread, op)

        return Literal(0)

    def _str_op_expr(self, thread: TLASassThread, raw: str) -> Expr:
        s = raw.strip()
        # Constant bank: c[...][...] or cx[...][...]
        cm = _CBANK_RE.match(s) or _CX_RE.match(s)
        if cm:
            offset_expr = self._const_bank_offset_expr(thread, cm.group("offset"))
            return thread.process.mem[offset_expr]

        # Hex / int / float immediates
        if _HEX_IMM_RE.match(s):
            return Literal(int(s, 16))
        if _INT_IMM_RE.match(s):
            return Literal(int(s))
        if _FLOAT_IMM_RE.match(s):
            try:
                return Literal(float(s))
            except ValueError:
                return Literal(0)
        # Unknown/label-like: keep raw as an opaque literal.
        return Literal(s)

    def _const_bank_offset_expr(self, thread: TLASassThread, offset: str) -> Expr:
        s = offset.strip()
        if re.match(r"^[A-Za-z]", s):
            m = re.match(r"^([A-Za-z]\w*)([+\-](?:0x[\da-fA-F]+|\d+))?", s)
            if m:
                base_expr = thread.getRegister(m.group(1))
                if m.group(2):
                    return base_expr + Literal(int(m.group(2), 0))
                return base_expr
            return Literal(0)
        try:
            return Literal(int(s, 0))
        except ValueError:
            return Literal(0)

    def _mem_addr_expr(self, thread: TLASassThread, op: SASSAddress) -> Expr:
        terms: list[Expr] = []
        if op.reg1 is not None and op.reg1.n not in _ZERO_REGS:
            terms.append(thread.getRegister(op.reg1.n.split(".")[0]))
        if op.reg2 is not None and op.reg2.n not in _ZERO_REGS:
            terms.append(thread.getRegister(op.reg2.n.split(".")[0]))
        if op.imm:
            try:
                terms.append(Literal(int(op.imm, 0)))
            except ValueError:
                pass
        if not terms:
            return Literal(0)
        if len(terms) == 1:
            return terms[0]
        return Add(*terms)

    def _dst(self, instr: SASSInstruction, idx: int = 0) -> str:
        op = instr.args[idx]
        return op.n if isinstance(op, SASSRegister) else "RZ"

    def _src(self, thread: TLASassThread, instr: SASSInstruction, idx: int) -> Expr:
        return self._op_expr(thread, instr.args[idx])

    def _coerce_bool(self, expr: Expr, operand) -> Expr:
        if isinstance(operand, SASSRegister) and _PRED_NAME_RE.match(operand.n):
            return expr
        if isinstance(expr, Literal) and isinstance(expr.value, bool):
            return expr
        if isinstance(expr, Literal) and isinstance(expr.value, int):
            return Literal(expr.value != 0)
        return expr != Literal(0)

    def _pred_expr(
        self, thread: TLASassThread, instr: SASSInstruction
    ) -> Optional[Expr]:
        p = instr.predicate
        if p is None:
            return None
        negated = p.startswith("!")
        name = p[1:] if negated else p
        if name in _ALWAYS_TRUE_PREDS:
            return None
        expr = thread.getRegister(name)
        if negated:
            expr = Not(expr)
        return expr

    def _branch_pred(
        self, thread: TLASassThread, instr: SASSInstruction
    ) -> Optional[Expr]:
        pname = _pred_name(instr.predicate)
        if pname is not None and pname not in _ALWAYS_TRUE_PREDS:
            return self._pred_expr(thread, instr)
        for op in instr.args:
            if isinstance(op, SASSRegister):
                name = op.n
                if _PRED_NAME_RE.match(name) and name not in _ALWAYS_TRUE_PREDS:
                    expr = thread.getRegister(name)
                    if op.is_inverted or op.is_negated:
                        expr = Not(expr)
                    return expr
            elif isinstance(op, str):
                # Reached the branch target address before any predicate.
                if _HEX_IMM_RE.match(op.strip()):
                    break
        return None

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def _instr_name(self, instr: SASSInstruction) -> str:
        return instr.opcode.lower().replace(".", "_")

    def _write_reg(
        self, thread: TLASassThread, instr: SASSInstruction, dst: str, value: Expr
    ) -> None:
        if dst in _DISCARD_DSTS:
            noop = thread._createUnchangedExceptExpr(
                Literal(True), [thread.process.getPcMap()]
            )
            thread.appendInstruction(self._instr_name(instr), noop)
            return
        pred = self._pred_expr(thread, instr)
        if pred is not None:
            value = IfThenElse(pred, value, thread.getRegister(dst))
        thread.appendRegisterInstruction(self._instr_name(instr), dst, value)

    def _emit_dual(
        self,
        thread: TLASassThread,
        instr: SASSInstruction,
        dst0: str,
        val0: Expr,
        dst1: str,
        val1: Expr,
    ) -> None:
        name = self._instr_name(instr)
        d0_ok = dst0 not in _DISCARD_DSTS
        d1_ok = dst1 not in _DISCARD_DSTS
        if d0_ok and d1_ok:
            thread._append_dual_reg_instr(name, dst0, val0, dst1, val1)
        elif d0_ok:
            thread.appendRegisterInstruction(name, dst0, val0)
        elif d1_ok:
            thread.appendRegisterInstruction(name, dst1, val1)
        else:
            thread._stutter(name)

    def _const_bank_raw_addr(
        self, thread: TLASassThread, instr: SASSInstruction, idx: int
    ) -> Expr:
        op = instr.args[idx]
        if isinstance(op, str):
            cm = _CBANK_RE.match(op.strip()) or _CX_RE.match(op.strip())
            if cm:
                return self._const_bank_offset_expr(thread, cm.group("offset"))
        return Literal(0)

    def _sr_constant(self, sr_name: str) -> Expr:
        tla_name = sr_name.replace(".", "_")

        # if tla_name not in self._sr_constants:
        #     # SR_* values feed arithmetic (e.g. SR_TID.X is an integer thread
        #     # id), so the constant has to be an Int rather than a symbolic
        #     # model value. Default to 0; model checks that don't care about
        #     # the specific id then collapse to a single concrete trace.
        #     self._sr_constants[tla_name] = self._sr_proc.createConstant(
        #         tla_name, Literal(0)
        #     )


        match tla_name:
            case "SR_TID_X":
                return self._sr_proc.threadIndiciesExpr("SR_TID_X")
            case "SR_TID_Y":
                return self._sr_proc.threadIndiciesExpr("SR_TID_Y")
            case "SR_TID_Z":
                return self._sr_proc.threadIndiciesExpr("SR_TID_Z")
            case "SR_CTAID_X":
                return self._sr_proc.blockIndicesExpr("SR_CTAID_X")
            case "SR_CTAID_Y":
                return self._sr_proc.blockIndicesExpr("SR_CTAID_Y")
            case "SR_CTAID_Z":
                return self._sr_proc.blockIndicesExpr("SR_CTAID_Z")
            case "SR_LANEID":
                return self._sr_proc.laneIndexExpr()
            case _:
                raise ValueError(f"Invalid special register name: {tla_name}")        
        

    # ==================================================================
    # Instruction handlers  (one per sass_insns.h macro)
    # ==================================================================

    def _h_mov(self, thread: TLASassThread, instr: SASSInstruction) -> None:
        self._write_reg(thread, instr, self._dst(instr, 0), self._src(thread, instr, 1))

    def _h_up2ur(self, thread: TLASassThread, instr: SASSInstruction) -> None:
        # UP2UR UR5, UPR, URZ, 0x2 — dst = imm (opaque predicate-pack).
        dst = self._dst(instr, 0)
        imm = self._src(thread, instr, 3)
        self._write_reg(thread, instr, dst, imm)

    def _h_iabs(self, thread: TLASassThread, instr: SASSInstruction) -> None:
        self._write_reg(
            thread, instr, self._dst(instr, 0), Abs(self._src(thread, instr, 1))
        )

    def _h_fp_passthrough(self, t: TLASassThread, i: SASSInstruction) -> None:
        self._write_reg(t, i, self._dst(i, 0), self._src(t, i, 1))

    def _h_iadd3(self, thread: TLASassThread, instr: SASSInstruction) -> None:
        dst = self._dst(instr, 0)
        val = (
            self._src(thread, instr, 1)
            + self._src(thread, instr, 2)
            + self._src(thread, instr, 3)
        )
        self._write_reg(thread, instr, dst, val)

    def _h_imad(self, thread: TLASassThread, instr: SASSInstruction) -> None:
        dst = self._dst(instr, 0)
        val = (
            self._src(thread, instr, 1) * self._src(thread, instr, 2)
            + self._src(thread, instr, 3)
        )
        self._write_reg(thread, instr, dst, val)

    def _h_imad_hi_u32(self, thread: TLASassThread, instr: SASSInstruction) -> None:
        dst = self._dst(instr, 0)
        s1 = self._src(thread, instr, 1)
        s2 = self._src(thread, instr, 2)
        s3 = self._src(thread, instr, 3)
        dst_cur = thread.getRegister(dst)
        full = s1 * s2 + ((dst_cur << Literal(32)) + s3)
        self._write_reg(thread, instr, dst, full >> Literal(32))

    def _h_imnmx_u32(self, thread: TLASassThread, instr: SASSInstruction) -> None:
        dst = self._dst(instr, 0)
        s1 = self._src(thread, instr, 1)
        s2 = self._src(thread, instr, 2)
        mnpred = self._src(thread, instr, 3)
        self._write_reg(
            thread, instr, dst, IfThenElse(mnpred, Min(s1, s2), Max(s1, s2))
        )

    def _h_viadd(self, thread: TLASassThread, instr: SASSInstruction) -> None:
        dst = self._dst(instr, 0)
        val = self._src(thread, instr, 1) + self._src(thread, instr, 2)
        self._write_reg(thread, instr, dst, val)

    def _h_viaddmnmx(self, thread: TLASassThread, instr: SASSInstruction) -> None:
        dst = self._dst(instr, 0)
        s1 = self._src(thread, instr, 1)
        s2 = self._src(thread, instr, 2)
        imm = self._src(thread, instr, 3)
        mnpred = self._src(thread, instr, 4)
        val = IfThenElse(mnpred, Min(s1 + imm, s2), Max(s1 + imm, s2))
        self._write_reg(thread, instr, dst, val)

    def _h_vimnmx(self, thread: TLASassThread, instr: SASSInstruction) -> None:
        dst = self._dst(instr, 0)
        pdst0 = self._dst(instr, 1)
        pdst1 = self._dst(instr, 2)
        s1 = self._src(thread, instr, 3)
        s2 = self._src(thread, instr, 4)
        mnpred = self._src(thread, instr, 5)
        result = IfThenElse(mnpred, Min(s1, s2), Max(s1, s2))
        cmp = s1 < s2
        not_cmp = Not(cmp)
        name = self._instr_name(instr)
        d0_ok = dst not in _DISCARD_DSTS
        p0_ok = pdst0 not in _DISCARD_DSTS
        p1_ok = pdst1 not in _DISCARD_DSTS
        writes: list[tuple[str, Expr]] = []
        if d0_ok:
            writes.append((dst, result))
        if p0_ok:
            writes.append((pdst0, cmp))
        if p1_ok:
            writes.append((pdst1, not_cmp))
        if len(writes) >= 2:
            thread._append_multi_reg_instr(name, writes)
        elif len(writes) == 1:
            thread.appendRegisterInstruction(name, writes[0][0], writes[0][1])
        else:
            thread._stutter(name)

    def _h_shf_r_u32_hi(self, thread: TLASassThread, instr: SASSInstruction) -> None:
        dst = self._dst(instr, 0)
        s1 = self._src(thread, instr, 1)  # lo
        rot = self._src(thread, instr, 2)
        s2 = self._src(thread, instr, 3)  # hi
        self._write_reg(thread, instr, dst, FunnelShr(s2, s1, rot))

    def _h_shf_r_s32_hi(self, thread: TLASassThread, instr: SASSInstruction) -> None:
        self._h_shf_r_u32_hi(thread, instr)

    def _h_shf_l_u32(self, thread: TLASassThread, instr: SASSInstruction) -> None:
        dst = self._dst(instr, 0)
        lo = self._src(thread, instr, 1)
        rot = self._src(thread, instr, 2)
        hi = self._src(thread, instr, 3)
        concat = (hi << Literal(32)) + lo
        self._write_reg(thread, instr, dst, (concat << rot) >> Literal(32))

    def _h_lop3_lut(self, thread: TLASassThread, instr: SASSInstruction) -> None:
        n = len(instr.args)
        if n >= 7:
            dst0 = self._dst(instr, 0)
            dst1 = self._dst(instr, 1)
            s1 = self._coerce_bool(self._src(thread, instr, 2), instr.args[2])
            s2 = self._coerce_bool(self._src(thread, instr, 3), instr.args[3])
            s3 = self._src(thread, instr, 4)
            lut = self._src(thread, instr, 5)
            result = IfThenElse(s1, IfThenElse(s2, s3, lut), IfThenElse(s2, lut, s3))
            self._emit_dual(thread, instr, dst0, result, dst1, result)
        else:
            dst = self._dst(instr, 0)
            s1 = self._coerce_bool(self._src(thread, instr, 1), instr.args[1])
            s2 = self._coerce_bool(self._src(thread, instr, 2), instr.args[2])
            s3 = self._src(thread, instr, 3)
            lut = self._src(thread, instr, 4)
            result = IfThenElse(s1, IfThenElse(s2, s3, lut), IfThenElse(s2, lut, s3))
            self._write_reg(thread, instr, dst, result)

    def _h_plop3_lut(self, thread: TLASassThread, instr: SASSInstruction) -> None:
        dst0 = self._dst(instr, 0)
        dst1 = self._dst(instr, 1)
        s1 = self._src(thread, instr, 2)
        s2 = self._src(thread, instr, 3)
        s3 = self._src(thread, instr, 4)
        lut = self._src(thread, instr, 5)
        s4 = self._src(thread, instr, 6)
        result = IfThenElse(s1, IfThenElse(s2, s3, lut), IfThenElse(s2, lut, s4))
        not_result = Not(result)
        self._emit_dual(thread, instr, dst0, result, dst1, not_result)

    def _h_sel(self, thread: TLASassThread, instr: SASSInstruction) -> None:
        dst = self._dst(instr, 0)
        s1 = self._src(thread, instr, 1)
        s2 = self._src(thread, instr, 2)
        pred = self._src(thread, instr, 3)
        self._write_reg(thread, instr, dst, IfThenElse(pred, s1, s2))

    def _h_lea(self, thread: TLASassThread, instr: SASSInstruction) -> None:
        dst = self._dst(instr, 0)
        src = self._src(thread, instr, 1)
        b = self._src(thread, instr, 2)
        imm_shift = self._src(thread, instr, 3)
        self._write_reg(thread, instr, dst, (src << imm_shift) + b)

    def _h_lea_hi(self, thread: TLASassThread, instr: SASSInstruction) -> None:
        dst = self._dst(instr, 0)
        alo = self._src(thread, instr, 1)
        b = self._src(thread, instr, 2)
        ahi = self._src(thread, instr, 3)
        imm_shift = self._src(thread, instr, 4)
        concat = (ahi << Literal(32)) + alo
        upper = (concat << imm_shift) >> Literal(32)
        self._write_reg(thread, instr, dst, upper + b)

    @staticmethod
    def _usetmaxreg_count_op(instr: SASSInstruction, idx: int):
        op = instr.args[idx]
        if isinstance(op, SASSRegister):
            return op.n
        if isinstance(op, str):
            s = op.strip()
            if _HEX_IMM_RE.match(s):
                return int(s, 16)
            if _INT_IMM_RE.match(s):
                return int(s)
        return None

    def _h_usetmaxnre_inc(self, thread: TLASassThread, instr: SASSInstruction) -> None:
        absReg = self._src(thread, instr, 1)
        thread.emit_usetmaxreg(absReg, self._usetmaxreg_count_op(instr, 1), is_inc=True)

    def _h_usetmaxnre_dec(self, thread: TLASassThread, instr: SASSInstruction) -> None:
        absReg = self._src(thread, instr, 0)
        thread.emit_usetmaxreg(absReg, self._usetmaxreg_count_op(instr, 0), is_inc=False)

    def _h_lea_hi_sx32(self, thread: TLASassThread, instr: SASSInstruction) -> None:
        dst = self._dst(instr)
        alo = self._src(thread, instr, 1)
        b = self._src(thread, instr, 2)
        imm_shift = self._src(thread, instr, 3)
        upper = (alo << imm_shift) >> Literal(32)
        self._write_reg(thread, instr, dst, upper + b)

    def _h_isetp_cond(
        self, t: TLASassThread, i: SASSInstruction, condition: Expr
    ) -> None:
        not_cond = Not(condition)
        self._emit_dual(t, i, self._dst(i, 0), condition, self._dst(i, 1), not_cond)

    def _h_isetp_lt_and(self, t: TLASassThread, i: SASSInstruction):
        self._h_isetp_cond(
            t, i, (self._src(t, i, 2) < self._src(t, i, 3)) & self._src(t, i, 4)
        )

    def _h_isetp_gt_and(self, t: TLASassThread, i: SASSInstruction):
        self._h_isetp_cond(
            t, i, (self._src(t, i, 2) > self._src(t, i, 3)) & self._src(t, i, 4)
        )

    def _h_isetp_ge_and(self, t: TLASassThread, i: SASSInstruction):
        self._h_isetp_cond(
            t, i, (self._src(t, i, 2) >= self._src(t, i, 3)) & self._src(t, i, 4)
        )

    def _h_isetp_ne_and(self, t: TLASassThread, i: SASSInstruction):
        self._h_isetp_cond(
            t, i, (self._src(t, i, 2) != self._src(t, i, 3)) & self._src(t, i, 4)
        )

    def _h_isetp_eq_and(self, t: TLASassThread, i: SASSInstruction):
        self._h_isetp_cond(
            t, i, (self._src(t, i, 2) == self._src(t, i, 3)) & self._src(t, i, 4)
        )

    def _h_isetp_le_and(self, t: TLASassThread, i: SASSInstruction):
        self._h_isetp_cond(
            t, i, (self._src(t, i, 2) <= self._src(t, i, 3)) & self._src(t, i, 4)
        )

    def _h_isetp_ge_or(self, t: TLASassThread, i: SASSInstruction):
        self._h_isetp_cond(
            t, i, (self._src(t, i, 2) >= self._src(t, i, 3)) | self._src(t, i, 4)
        )

    def _h_p2r(self, t: TLASassThread, i: SASSInstruction):
        self._write_reg(t, i, self._dst(i, 0), self._src(t, i, 1))

    def _h_uldc(self, t: TLASassThread, i: SASSInstruction):
        self._write_reg(t, i, self._dst(i, 0), self._src(t, i, 1))

    def _h_uldc_64(self, t: TLASassThread, i: SASSInstruction):
        t.emit_uldc_64(self._dst(i, 0), self._const_bank_raw_addr(t, i, 1))

    def _h_syncs_phasechk(self, t: TLASassThread, i: SASSInstruction):
        self._write_reg(t, i, self._dst(i, 0), Literal(True))

    def _h_stutter(self, t: TLASassThread, i: SASSInstruction):
        name = self._instr_name(i)
        noop = t._createUnchangedExceptExpr(Literal(True), [t.process.getPcMap()])
        t.appendInstruction(name, noop)
