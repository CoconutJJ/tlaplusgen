"""
tla_codegen.py  –  Lift a sliced SASS CFG into a TLASassProcess.

Usage
-----
    import sys
    sys.path.insert(0, "../sass")
    from cfg import build_cfgs, slice_cfg
    from parser import parse_file
    from tla_codegen import SassCFGCodegen

    prog  = parse_file("kernel.sass")
    cfgs  = build_cfgs(prog)
    sliced = slice_cfg(cfgs["my_kernel"], "WARPSYNC")

    codegen = SassCFGCodegen()
    proc    = codegen.generate(sliced, name="MyKernel")

    print(proc)
    print(proc.getConfiguration())
    for msg in codegen.log:
        print(msg, file=sys.stderr)

Extending
---------
To add support for a new instruction when TLASassThread gains a new emit_*
method, add one entry to _build_handler_table():

    h["NEW.MNEM"] = self._h_new_mnem

and define the handler:

    def _h_new_mnem(self, thread: TLASassThread, instr: Instruction) -> None:
        dst  = self._dst(instr, 0)
        src1 = self._src(thread, instr, 1)
        self._write_reg(thread, instr, dst, <value_expr>)
"""

from __future__ import annotations

import re

from typing import Callable, Optional

from sass.parser import (
    Instruction,
    RegisterOp,
    ImmediateOp,
    LabelRef,
    MemAddrOp,
    ConstBankOp,
)
from sass.cfg import (
    CFG,
    BasicBlock,
    TerminatorKind,
    defs_of,
    uses_of,
)

from tla_module import (
    Expr,
    Add,
    And,
    IfThenElse,
    Index,
    Literal,
    Unchanged,
    Not,
    Gt,
    Lt,
    GtE,
    Mul,
    Shl,
    Shr,
    Min,
    Max,
    FunnelShr,
    Equal,
    NotEqual,
    Or,
    Constant,
)
from tla_sass import TLASassProcess, TLASassThread

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALWAYS_TRUE_PREDS: frozenset[str] = frozenset({"PT", "UPT"})
_ZERO_REGS: frozenset[str] = frozenset({"RZ", "URZ", "SRZ"})
# Registers that are discard destinations (writes are no-ops architecturally).
_DISCARD_DSTS: frozenset[str] = _ZERO_REGS | _ALWAYS_TRUE_PREDS

_HandlerFn = Callable[["SassCFGCodegen", TLASassThread, Instruction], None]


# ---------------------------------------------------------------------------
# Codegen
# ---------------------------------------------------------------------------


class SassCFGCodegen:
    """
    Translates a single-kernel sliced SASS CFG into a TLASassProcess.

    After calling generate(), inspect self.log for any instructions that
    were skipped because no handler was registered.
    """

    def __init__(self) -> None:
        self.log: list[str] = []
        self._handlers: dict = {}
        self._build_handler_table()

    # ------------------------------------------------------------------
    # Handler table
    #
    # Each entry maps one (or more) exact mnemonic strings to a handler.
    # To add support for a new instruction:
    #   1. Define a _h_<name> method below.
    #   2. Add h["MNEMONIC"] = self._h_<name> here.
    # ------------------------------------------------------------------

    def _build_handler_table(self) -> None:
        h = self._handlers

        # ---- Data movement ----
        for m in ("MOV", "UMOV", "S2R", "CS2R"):
            h[m] = self._h_mov

        # ---- Integer arithmetic ----
        for m in ("IADD3", "UIADD3"):
            h[m] = self._h_iadd3
        for m in (
            "IMAD",
            "IMAD.U32",
            "IMAD.MOV",
            "IMAD.IADD",
            "IMAD.SHL",
            "UIMAD",
            "UIMAD.U32",
        ):
            h[m] = self._h_imad
        h["IMAD.HI.U32"] = self._h_imad_hi_u32
        for m in ("IMAD.WIDE", "IMAD.WIDE.U32", "UIMAD.WIDE", "UIMAD.WIDE.U32"):
            h[m] = self._h_imad_wide
        h["IABS"] = self._h_iabs
        h["IMNMX.U32"] = self._h_imnmx_u32

        # ---- Shift ----
        for m in ("SHF.R.U32.HI", "USHF.R.U32.HI"):
            h[m] = self._h_shf_r_u32_hi
        for m in ("SHF.R.S32.HI", "USHF.R.S32.HI"):
            h[m] = self._h_shf_r_s32_hi

        # ---- Logical ----
        for m in ("LOP3.LUT", "ULOP3.LUT"):
            h[m] = self._h_lop3_lut
        h["PLOP3.LUT"] = self._h_plop3_lut
        for m in ("SEL", "USEL"):
            h[m] = self._h_sel

        # ---- Address computation ----
        for m in ("LEA.HI", "ULEA.HI", "LEA.HI.SX32", "ULEA.HI.SX32"):
            h[m] = self._h_lea_hi

        # ---- Predicate ----
        for pfx in ("ISETP", "UISETP"):
            h[f"{pfx}.LT.AND"] = self._h_isetp_lt_and
            h[f"{pfx}.LT.U32.AND"] = self._h_isetp_lt_and
            h[f"{pfx}.GT.AND"] = self._h_isetp_gt_and
            h[f"{pfx}.GT.U32.AND"] = self._h_isetp_gt_u32_and
            h[f"{pfx}.GE.AND"] = self._h_isetp_ge_and
            h[f"{pfx}.GE.U32.AND"] = self._h_isetp_ge_and
            h[f"{pfx}.NE.AND"] = self._h_isetp_ne_and
            h[f"{pfx}.NE.U32.AND"] = self._h_isetp_ne_and
            h[f"{pfx}.EQ.AND"] = self._h_isetp_eq_u32_and
            h[f"{pfx}.EQ.U32.AND"] = self._h_isetp_eq_u32_and
            h[f"{pfx}.GE.OR"] = self._h_isetp_ge_or
            h[f"{pfx}.GE.U32.OR"] = self._h_isetp_ge_or
        h["P2R"] = self._h_p2r

        # ---- Memory loads ----
        for m in (
            "LDG.E",
            "LDG.E.CONSTANT",
            "LDG.E.LTC",
            "LDG.E.STRONG.GPU",
            "LDG.E.STRONG.SYS",
        ):
            h[m] = self._h_ldg
        for m in (
            "LDG.E.128",
            "LDG.E.128.STRONG.GPU",
            "LDG.E.128.CONSTANT",
            "LDG.E.LTC128B.CONSTANT",
        ):
            h[m] = self._h_ldg_128
        h["LDS"] = self._h_lds
        h["LDS.64"] = self._h_lds_64
        h["LDS.128"] = self._h_lds_128
        h["LDSM.16.MT88.4"] = self._h_ldsm
        for m in ("LDC", "LDCU"):
            h[m] = self._h_ldc
        for m in ("LDC.64", "LDCU.64"):
            h[m] = self._h_ldc_64
        h["LDCU.128"] = self._h_ldc_128
        h["ULDC.64"] = self._h_uldc_64
        for m in (
            "LDTM.x4",
            "LDTM.x32",
            "LDTM.x128",
            "LDTM.16dp256bit.x4",
            "LDTM.16dp256bit.x16",
        ):
            h[m] = self._h_ldtm

        # ---- Memory stores ----
        for m in ("STG", "STG.E", "STG.E.128", "STG.E.128.STRONG.GPU"):
            h[m] = self._h_stg
        for m in ("STS", "STS.64", "STS.128"):
            h[m] = self._h_sts

        # ---- Election ----
        h["ELECT"] = self._h_elect

        # ---- Synchronization / barriers ----
        for m in ("WARPSYNC", "WARPSYNC.ALL"):
            h[m] = self._h_warpsync
        for m in ("BAR.SYNC", "BAR.SYNC.DEFER_BLOCKING"):
            h[m] = self._h_bar_sync
        for m in ("MEMBAR.SC.GPU", "MEMBAR.SC.CTA", "MEMBAR.SC.SYS"):
            h[m] = self._h_membar
        h["BSSY"] = self._h_bssy
        h["BSYNC"] = self._h_bsync
        for m in ("DEPBAR", "DEPBAR.WAIT", "DEPBAR.WAIT.LE"):
            h[m] = self._h_depbar
        h["NOP"] = self._h_nop

        # ---- Register-pool management ----
        for m in ("USETMAXREG.TRY_ALLOC.CTAPOOL", "USETMAXREG.DEALLOC.CTAPOOL"):
            h[m] = self._h_usetmaxreg

        # EXIT / BRA / RET are terminators handled separately; do not add here.

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate(self, cfg: CFG, name: str, n_warps: int = 1) -> TLASassProcess:
        """
        Lift ``cfg`` into a TLASassProcess named ``name``.

        The CFG must represent exactly one kernel function (as returned by
        cfg.build_cfgs()).  For multi-kernel files, call generate() once per
        kernel.

        Parameters
        ----------
        cfg      : sliced or unsliced single-kernel CFG
        name     : TLA+ module name
        n_warps  : number of warp threads to model (default 1)
        """
        regs_zero, regs_false, regs_true = self._collect_registers(cfg)
        registers = regs_zero + regs_false + regs_true
        init_values = (
            [Literal(0)] * len(regs_zero)
            + [Literal(False)] * len(regs_false)
            + [Literal(True)] * len(regs_true)
        )

        proc = TLASassProcess(name)
        threads = proc.createThreads(registers, init_values, n_warps)
        proc.initialize()

        # SR_TID.X / SR_CTAID.Y etc. become TLA+ CONSTANTs (hardware-provided
        # per-thread values).  Reset the table so each generate() call starts
        # fresh, then lazily populate it as instructions are encoded.
        self._sr_constants: dict[str, Constant] = {}
        self._sr_proc = proc

        for thread in threads:
            self._encode_cfg(thread, cfg)

        return proc

    # ------------------------------------------------------------------
    # Register discovery
    # ------------------------------------------------------------------

    def _collect_registers(self, cfg: CFG) -> tuple[list[str], list[str], list[str]]:
        """
        Collect registers referenced in the CFG.

        Returns (regs_zero, regs_false, regs_true):
          regs_zero   – integer registers, initialized to 0
          regs_false  – predicate registers (P\d+, UP\d+), initialized to FALSE
          regs_true   – always-true predicate registers (PT/UPT) that appear
                        as explicit write destinations; initialized to TRUE so
                        that EXCEPT updates on them are valid TLA+.

        Uses defs_of / uses_of so multi-writer expansion is handled
        automatically (e.g. IMAD.WIDE writing R4 *and* R5).
        """
        seen: set[str] = set()
        discard_written: set[str] = set()
        _pred_re = re.compile(r"^U?P\d+$")
        for bb in cfg.blocks:
            for instr in bb.instructions:
                if instr.predicate and instr.predicate.name not in _ALWAYS_TRUE_PREDS:
                    seen.add(instr.predicate.name)
                for r in defs_of(instr):
                    if r in _ALWAYS_TRUE_PREDS:
                        discard_written.add(r)
                    elif r not in _ZERO_REGS and not r.startswith("SR_"):
                        seen.add(r)
                for r in uses_of(instr):
                    if r not in _DISCARD_DSTS and not r.startswith("SR_"):
                        seen.add(r)

        def _key(name: str) -> tuple[int, int]:
            m = re.match(r"([A-Za-z]+)(\d+)$", name)
            order = {"R": 0, "UR": 1, "P": 2, "UP": 3}
            if m:
                return (order.get(m.group(1), 9), int(m.group(2)))
            return (9, 0)

        regs_zero = sorted([r for r in seen if not _pred_re.match(r)], key=_key)
        regs_false = sorted([r for r in seen if _pred_re.match(r)], key=_key)
        return regs_zero, regs_false, sorted(discard_written, key=_key)

    # ------------------------------------------------------------------
    # CFG → TLA+ encoding
    # ------------------------------------------------------------------

    def _encode_cfg(self, thread: TLASassThread, cfg: CFG) -> None:
        if not cfg.blocks:
            return

        # Pre-allocate a TLA+ state name for every block's entry point.
        # Block 0 reuses the thread's current state ("start").
        # All others get an explicit name so branch targets are stable.
        block_states: dict[int, str] = {}
        for bb in cfg.blocks:
            if bb.id == 0:
                block_states[bb.id] = thread._currentState()
            else:
                label = (
                    "_" + bb.entry_labels[0].replace(".", "_")
                    if bb.entry_labels
                    else ""
                )
                block_states[bb.id] = thread.allocateState(name=f"bb_{bb.id}{label}")

        for bb in cfg.blocks:
            thread.setState(block_states[bb.id])
            self._encode_block(thread, bb, block_states)

    def _encode_block(
        self,
        thread: TLASassThread,
        bb: BasicBlock,
        block_states: dict[int, str],
    ) -> None:
        instrs = list(bb.instructions)

        if not instrs:
            # Empty block – wire directly to successor.
            if bb.successors:
                self._emit_goto(thread, block_states[bb.successors[0].id])
            return

        # Encode every instruction except the last.
        for instr in instrs[:-1]:
            self._emit_instruction(thread, instr)

        last = instrs[-1]

        if bb.terminator_kind == TerminatorKind.EXIT:
            # EXIT / RET – do not encode the instruction body, just stop.
            thread.stopInstruction()

        elif bb.terminator_kind == TerminatorKind.FALL_THROUGH:
            # The last instruction is a normal instruction, not a branch.
            self._emit_instruction(thread, last)
            if bb.successors:
                self._emit_goto(thread, block_states[bb.successors[0].id])

        elif bb.terminator_kind == TerminatorKind.UNCONDITIONAL:
            # BRA with no predicate – skip encoding the instruction itself,
            # emit a direct goto to the target block.
            if bb.successors:
                self._emit_goto(thread, block_states[bb.successors[0].id])

        elif bb.terminator_kind == TerminatorKind.CONDITIONAL:
            # @Px BRA or BRA.U with a uniform predicate operand.
            pred = self._branch_pred(thread, last)
            taken = (
                block_states[bb.successors[0].id] if len(bb.successors) > 0 else None
            )
            fall = block_states[bb.successors[1].id] if len(bb.successors) > 1 else None

            if pred is not None and taken and fall:
                thread.appendBranchInstruction(pred, taken, fall)
            elif taken:
                self._emit_goto(thread, taken)

        elif bb.terminator_kind == TerminatorKind.INDIRECT:
            self.log.append(
                f"INDIRECT BRANCH at /*{last.address_str}*/ {last.mnemonic}: "
                f"targets unknown – treating block as EXIT"
            )
            thread.stopInstruction()

    # ------------------------------------------------------------------
    # Unconditional goto helper
    # ------------------------------------------------------------------

    def _emit_goto(self, thread: TLASassThread, target_state: str) -> None:
        """
        Emit a definition that advances the PC from the current state to
        ``target_state`` while leaving all other variables unchanged.

        This is used to wire the auto-generated post-instruction state
        to the pre-allocated entry state of the successor block.
        """
        current = thread._currentState()
        pc_trans = thread.pcTransition(current, target_state)
        unchanged_vars = thread._unchangedExcept([thread.process.getPcMap()])
        expr = And(pc_trans, Unchanged(unchanged_vars)) if unchanged_vars else pc_trans
        # Use the current state name to guarantee a unique definition name.
        safe = re.sub(r"[^a-zA-Z0-9]", "_", current)
        defn_name = f"{thread.thread_name}_goto_{safe}"
        defn = thread.process.createDefinition(defn_name, expr)
        thread.thread_definitions.append(defn)

    # ------------------------------------------------------------------
    # Single-instruction dispatch
    # ------------------------------------------------------------------

    def _emit_instruction(self, thread: TLASassThread, instr: Instruction) -> None:
        mnem = instr.mnemonic
        handler = self._handlers.get(mnem)

        # Fallback: match on the mnemonic stem (everything before the first dot).
        if handler is None:
            stem = mnem.split(".")[0]
            handler = self._handlers.get(stem)

        if handler is None:
            self.log.append(
                f"UNSUPPORTED /*{instr.address_str}*/  {mnem}"
                + (f"  pred={instr.predicate}" if instr.predicate else "")
                + f"  ops=[{', '.join(str(o) for o in instr.operands)}]"
            )
            return

        handler(thread, instr)

    # ------------------------------------------------------------------
    # Operand helpers
    # ------------------------------------------------------------------

    def _op_expr(self, thread: TLASassThread, op) -> Expr:
        """Convert any parsed operand to a TLA+ Expr (read context)."""
        if isinstance(op, RegisterOp):
            if op.name in _ZERO_REGS:
                return Literal(0)
            if op.name in _ALWAYS_TRUE_PREDS:
                return Literal(True)
            if op.name.startswith("SR_"):
                # Hardware-provided constant (thread/block index component).
                return self._sr_constant(op.name)
            expr = thread.getRegister(op.name)
            if "neg" in op.modifiers:
                expr = Not(expr)
            return expr

        if isinstance(op, ImmediateOp):
            if isinstance(op.value, int):
                return Literal(op.value)
            # Float or unparseable – model as 0 and log nothing (float ignored).
            return Literal(0)

        if isinstance(op, ConstBankOp):
            # c[bank][offset] → mem[offset]
            # offset can be a plain register "R4", a register+imm "R4+0xc",
            # or a pure immediate "0x18".
            offset_str = op.offset.strip()
            if re.match(r"^[A-Za-z]", offset_str):
                m = re.match(r"^([A-Za-z]\w*)([+\-](?:0x[\da-fA-F]+|\d+))?", offset_str)
                if m:
                    base_expr = thread.getRegister(m.group(1))
                    if m.group(2):
                        addend = int(m.group(2), 0)
                        offset_expr = Add(base_expr, Literal(addend))
                    else:
                        offset_expr = base_expr
                else:
                    offset_expr = Literal(0)
            else:
                try:
                    offset_expr = Literal(int(offset_str, 0))
                except ValueError:
                    offset_expr = Literal(0)
            return Index(thread.process.mem, offset_expr)

        if isinstance(op, MemAddrOp):
            return self._mem_addr_expr(thread, op)

        if isinstance(op, LabelRef):
            return Literal(op.name)

        # DescOp and unknown – return 0.
        return Literal(0)

    def _mem_addr_expr(self, thread: TLASassThread, op: MemAddrOp) -> Expr:
        """Build an address Expr from MemAddrOp parts (sum of base + offsets)."""
        terms: list[Expr] = []
        for part in op.parts:
            part = part.strip()
            if not part:
                continue
            if re.match(r"^[A-Za-z]", part):
                base = part.split(".")[0]
                if base not in _ZERO_REGS:
                    terms.append(thread.getRegister(base))
            else:
                try:
                    terms.append(Literal(int(part, 0)))
                except ValueError:
                    pass
        if not terms:
            return Literal(0)
        if len(terms) == 1:
            return terms[0]
        return Add(*terms)

    def _dst(self, instr: Instruction, idx: int = 0) -> str:
        """Return the destination register name at operand position ``idx``."""
        op = instr.operands[idx]
        return op.name if isinstance(op, RegisterOp) else "RZ"

    def _src(self, thread: TLASassThread, instr: Instruction, idx: int) -> Expr:
        """Return the source Expr at operand position ``idx``."""
        return self._op_expr(thread, instr.operands[idx])

    _PRED_RE = re.compile(r"^U?P\d+$")

    def _coerce_bool(self, expr: Expr, operand) -> Expr:
        """Ensure expr is boolean for use as an IF condition.

        Predicate registers (P0, UP3, …) already hold booleans — return as-is.
        GPRs and immediates are integers — wrap with NotEqual(..., 0).
        """
        if isinstance(operand, RegisterOp) and self._PRED_RE.match(operand.name):
            return expr
        if isinstance(expr, Literal) and isinstance(expr.value, bool):
            return expr
        return NotEqual(expr, Literal(0))

    def _pred_expr(self, thread: TLASassThread, instr: Instruction) -> Optional[Expr]:
        """
        Return the guard predicate as a TLA+ boolean Expr, or None if
        the instruction is always-executed (no guard or PT/UPT guard).
        """
        p = instr.predicate
        if p is None or p.name in _ALWAYS_TRUE_PREDS:
            return None
        expr = thread.getRegister(p.name)
        if p.negated:
            expr = Not(expr)
        return expr

    def _branch_pred(self, thread: TLASassThread, instr: Instruction) -> Optional[Expr]:
        """
        Extract the branch predicate from a conditional branch instruction.

        Two forms appear in SASS:
          @P0  BRA  `(.L_x)          – predicate is the guard
          BRA.U !UP0, `(.L_x)        – predicate is the first RegisterOp operand
        """
        # Guard form
        if instr.predicate and instr.predicate.name not in _ALWAYS_TRUE_PREDS:
            return self._pred_expr(thread, instr)
        # Operand-register form (BRA.U UP0 / !UP0)
        for op in instr.operands:
            if isinstance(op, LabelRef):
                break
            if isinstance(op, RegisterOp):
                name = op.name
                is_pred = (name.startswith("UP") and name[2:].isdigit()) or (
                    name.startswith("P") and name[1:].isdigit()
                )
                if is_pred and name not in _ALWAYS_TRUE_PREDS:
                    expr = thread.getRegister(name)
                    if "neg" in op.modifiers:
                        expr = Not(expr)
                    return expr
        return None

    # ------------------------------------------------------------------
    # _write_reg  –  unified single-destination write with predication
    # ------------------------------------------------------------------

    def _write_reg(
        self,
        thread: TLASassThread,
        instr: Instruction,
        dst: str,
        value: Expr,
    ) -> None:
        """
        Emit a register write for ``dst = value``.

        If the instruction carries a predicate guard, wraps ``value`` as:
            IF pred THEN value ELSE current_dst
        so that a predicated instruction is a no-op when the guard is false.
        """
        if dst in _DISCARD_DSTS:
            # Destination is a discard register (RZ, PT, etc.) – advance PC only.
            name = instr.mnemonic.lower().replace(".", "_")
            noop = thread._createUnchangedExceptExpr(
                Literal(True), [thread.process.getPcMap()]
            )
            thread.appendInstruction(name, noop)
            return
        pred = self._pred_expr(thread, instr)
        if pred is not None:
            value = IfThenElse(pred, value, thread.getRegister(dst))
        thread.appendRegisterInstruction(
            instr.mnemonic.lower().replace(".", "_"), dst, value
        )

    # ------------------------------------------------------------------
    # ConstBankOp address helper
    # ------------------------------------------------------------------

    def _const_bank_raw_addr(
        self, thread: TLASassThread, instr: Instruction, idx: int
    ) -> Expr:
        """
        Return the raw offset expression for a ConstBankOp operand at
        position ``idx`` — suitable for passing to emit_ldc / emit_ldc_64
        etc., which will themselves wrap it in mem[...].

        Unlike _src, this does NOT add the outer mem[...] layer.
        """
        op = instr.operands[idx]
        if isinstance(op, ConstBankOp):
            offset_str = op.offset.strip()
            if re.match(r"^[A-Za-z]", offset_str):
                m = re.match(r"^([A-Za-z]\w*)([+\-](?:0x[\da-fA-F]+|\d+))?", offset_str)
                if m:
                    base_expr = thread.getRegister(m.group(1))
                    if m.group(2):
                        return Add(base_expr, Literal(int(m.group(2), 0)))
                    return base_expr
            else:
                try:
                    return Literal(int(offset_str, 0))
                except ValueError:
                    pass
        return Literal(0)

    # ------------------------------------------------------------------
    # SR register helpers  (SR_TID.X, SR_CTAID.Y, …)
    # ------------------------------------------------------------------

    def _sr_constant(self, sr_name: str) -> Constant:
        """
        Return the TLA+ Constant for an SR register (creating it on first use).
        Dots in the name are replaced with underscores so the identifier is
        valid TLA+: SR_TID.X → SR_TID_X.
        """
        tla_name = sr_name.replace(".", "_")
        if tla_name not in self._sr_constants:
            self._sr_constants[tla_name] = self._sr_proc.createConstant(tla_name)
        return self._sr_constants[tla_name]

    # ------------------------------------------------------------------
    # _emit_dual  –  dual-destination write with discard filtering
    # ------------------------------------------------------------------

    def _emit_dual(
        self,
        thread: TLASassThread,
        instr: Instruction,
        dst0: str,
        val0: Expr,
        dst1: str,
        val1: Expr,
    ) -> None:
        """
        Emit a dual-register write.  Any destination that is a discard
        register (RZ, PT, etc.) is silently dropped.  If both are discards,
        emits a PC-only stutter.  If exactly one is real, emits a single
        appendRegisterInstruction.
        """
        name = instr.mnemonic.lower().replace(".", "_")
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

    # ------------------------------------------------------------------
    # Instruction handlers
    #
    # Convention: _h_<stem>(self, thread, instr) -> None
    # Each handler extracts operands, computes the result expression, and
    # calls _write_reg (single dest) or the appropriate thread emit_* method
    # (multi-dest / side-effect-only).
    # ------------------------------------------------------------------

    # ---- Data movement ----

    def _h_mov(self, thread: TLASassThread, instr: Instruction) -> None:
        self._write_reg(thread, instr, self._dst(instr, 0), self._src(thread, instr, 1))

    # ---- Integer arithmetic ----

    def _h_iadd3(self, thread: TLASassThread, instr: Instruction) -> None:
        dst = self._dst(instr, 0)
        val = Add(
            self._src(thread, instr, 1),
            self._src(thread, instr, 2),
            self._src(thread, instr, 3),
        )
        self._write_reg(thread, instr, dst, val)

    def _h_imad(self, thread: TLASassThread, instr: Instruction) -> None:

        dst = self._dst(instr, 0)
        val = Add(
            Mul(self._src(thread, instr, 1), self._src(thread, instr, 2)),
            self._src(thread, instr, 3),
        )
        self._write_reg(thread, instr, dst, val)

    def _h_imad_hi_u32(self, thread: TLASassThread, instr: Instruction) -> None:

        dst = self._dst(instr, 0)
        s1 = self._src(thread, instr, 1)
        s2 = self._src(thread, instr, 2)
        s3 = self._src(thread, instr, 3)
        dst_cur = thread.getRegister(dst)
        full = Add(Mul(s1, s2), Add(Shl(dst_cur, Literal(32)), s3))
        self._write_reg(thread, instr, dst, Shr(full, Literal(32)))

    def _h_imad_wide(self, thread: TLASassThread, instr: Instruction) -> None:
        # Multi-register write; predication not applied (rare for wide ops).
        dst = self._dst(instr, 0)
        s1 = self._src(thread, instr, 1)
        s2 = self._src(thread, instr, 2)
        s3 = self._src(thread, instr, 3)
        thread.emit_imad_wide(dst, s1, s2, s3)

    def _h_iabs(self, thread: TLASassThread, instr: Instruction) -> None:
        dst = self._dst(instr, 0)
        self._write_reg(thread, instr, dst, self._src(thread, instr, 1))

    def _h_imnmx_u32(self, thread: TLASassThread, instr: Instruction) -> None:
        dst = self._dst(instr, 0)
        s1 = self._src(thread, instr, 1)
        s2 = self._src(thread, instr, 2)
        mnpred = self._src(thread, instr, 3)

        self._write_reg(
            thread, instr, dst, IfThenElse(mnpred, Min(s1, s2), Max(s1, s2))
        )

    # ---- Shift ----

    def _h_shf_r_u32_hi(self, thread: TLASassThread, instr: Instruction) -> None:

        dst = self._dst(instr, 0)
        s1 = self._src(thread, instr, 1)  # lo half
        rot = self._src(thread, instr, 2)  # shift amount
        s2 = self._src(thread, instr, 3)  # hi half
        self._write_reg(thread, instr, dst, FunnelShr(s2, s1, rot))

    def _h_shf_r_s32_hi(self, thread: TLASassThread, instr: Instruction) -> None:
        self._h_shf_r_u32_hi(thread, instr)  # same TLA+ semantics

    # ---- Logical ----

    def _h_lop3_lut(self, thread: TLASassThread, instr: Instruction) -> None:
        # 5-operand form: dst, src1, src2, src3, imm_lut  → single dest
        # 7-operand form: dst_pred, dst_reg, src1, src2, src3, imm_lut, src4 → dual dest
        n = len(instr.operands)
        if n >= 7:
            dst0 = self._dst(instr, 0)
            dst1 = self._dst(instr, 1)
            s1 = self._coerce_bool(self._src(thread, instr, 2), instr.operands[2])
            s2 = self._coerce_bool(self._src(thread, instr, 3), instr.operands[3])
            s3 = self._src(thread, instr, 4)
            lut = self._src(thread, instr, 5)
            result = IfThenElse(s1, IfThenElse(s2, s3, lut), IfThenElse(s2, lut, s3))
            self._emit_dual(thread, instr, dst0, result, dst1, result)
        else:
            dst = self._dst(instr, 0)
            s1 = self._coerce_bool(self._src(thread, instr, 1), instr.operands[1])
            s2 = self._coerce_bool(self._src(thread, instr, 2), instr.operands[2])
            s3 = self._src(thread, instr, 3)
            lut = self._src(thread, instr, 4)
            result = IfThenElse(s1, IfThenElse(s2, s3, lut), IfThenElse(s2, lut, s3))
            self._write_reg(thread, instr, dst, result)

    def _h_plop3_lut(self, thread: TLASassThread, instr: Instruction) -> None:
        # PLOP3.LUT dst0, dst1, src1, src2, src3, imm_lut, src4
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

    def _h_sel(self, thread: TLASassThread, instr: Instruction) -> None:
        dst = self._dst(instr, 0)
        s1 = self._src(thread, instr, 1)
        s2 = self._src(thread, instr, 2)
        pred = self._src(thread, instr, 3)
        self._write_reg(thread, instr, dst, IfThenElse(pred, s1, s2))

    # ---- Address computation ----

    def _h_lea_hi(self, thread: TLASassThread, instr: Instruction) -> None:

        # LEA.HI dst, alo, b, ahi, imm_shift
        dst = self._dst(instr, 0)
        alo = self._src(thread, instr, 1)
        b = self._src(thread, instr, 2)
        ahi = self._src(thread, instr, 3)
        imm_shift = self._src(thread, instr, 4)
        concat = Add(Shl(ahi, Literal(32)), alo)
        upper = Shr(Shl(concat, imm_shift), Literal(32))
        self._write_reg(thread, instr, dst, Add(upper, b))

    # ---- Predicate ----

    def _h_isetp_cond(self, t: TLASassThread, i, condition: Expr) -> None:
        """Common body for all ISETP variants: dst0=cond, dst1=~cond."""

        not_cond = Not(condition)
        self._emit_dual(t, i, self._dst(i, 0), condition, self._dst(i, 1), not_cond)

    def _h_isetp_lt_and(self, t: TLASassThread, i: Instruction):

        self._h_isetp_cond(
            t, i, And(Lt(self._src(t, i, 2), self._src(t, i, 3)), self._src(t, i, 4))
        )

    def _h_isetp_gt_and(self, t: TLASassThread, i: Instruction):

        self._h_isetp_cond(
            t, i, And(Gt(self._src(t, i, 2), self._src(t, i, 3)), self._src(t, i, 4))
        )

    def _h_isetp_ge_and(self, t: TLASassThread, i: Instruction):

        self._h_isetp_cond(
            t, i, And(GtE(self._src(t, i, 2), self._src(t, i, 3)), self._src(t, i, 4))
        )

    def _h_isetp_ne_and(self, t: TLASassThread, i: Instruction):

        self._h_isetp_cond(
            t,
            i,
            And(NotEqual(self._src(t, i, 2), self._src(t, i, 3)), self._src(t, i, 4)),
        )

    def _h_isetp_eq_u32_and(self, t: TLASassThread, i: Instruction):

        self._h_isetp_cond(
            t, i, And(Equal(self._src(t, i, 2), self._src(t, i, 3)), self._src(t, i, 4))
        )

    def _h_isetp_ge_or(self, t: TLASassThread, i: Instruction):

        self._h_isetp_cond(
            t, i, Or(GtE(self._src(t, i, 2), self._src(t, i, 3)), self._src(t, i, 4))
        )

    def _h_isetp_gt_u32_and(self, t: TLASassThread, i: Instruction):

        self._h_isetp_cond(
            t, i, And(Gt(self._src(t, i, 2), self._src(t, i, 3)), self._src(t, i, 4))
        )

    def _h_p2r(self, t: TLASassThread, i: Instruction):
        self._write_reg(t, i, self._dst(i, 0), self._src(t, i, 1))

    # ---- Memory loads ----

    def _h_ldg(self, t: TLASassThread, i: Instruction):
        t.emit_ldg(self._dst(i, 0), self._src(t, i, 1))

    def _h_ldg_128(self, t: TLASassThread, i: Instruction):
        t.emit_ldg_128(self._dst(i, 0), self._src(t, i, 1))

    def _h_lds(self, t: TLASassThread, i: Instruction):
        t.emit_lds(self._dst(i, 0), self._src(t, i, 1))

    def _h_lds_64(self, t: TLASassThread, i: Instruction):
        t.emit_lds_64(self._dst(i, 0), self._src(t, i, 1))

    def _h_lds_128(self, t: TLASassThread, i: Instruction):
        t.emit_lds_128(self._dst(i, 0), self._src(t, i, 1))

    def _h_ldsm(self, t: TLASassThread, i: Instruction):
        t.emit_ldsm(self._dst(i, 0), self._src(t, i, 1))

    def _h_ldc(self, t: TLASassThread, i: Instruction):
        # _src on a ConstBankOp already produces mem[offset]; use _write_reg
        # directly to avoid the double-dereference that emit_ldc would add.
        self._write_reg(t, i, self._dst(i, 0), self._src(t, i, 1))

    def _h_ldc_64(self, t: TLASassThread, i: Instruction):
        # emit_ldc_64 adds mem[...] itself, so pass the raw offset.
        t.emit_ldc_64(self._dst(i, 0), self._const_bank_raw_addr(t, i, 1))

    def _h_ldc_128(self, t: TLASassThread, i: Instruction):
        t.emit_ldc_128(self._dst(i, 0), self._const_bank_raw_addr(t, i, 1))

    def _h_uldc_64(self, t: TLASassThread, i: Instruction):
        t.emit_uldc_64(self._dst(i, 0), self._const_bank_raw_addr(t, i, 1))

    def _h_ldtm(self, t: TLASassThread, i: Instruction):
        # LDTM.xN – extract N from the mnemonic suffix
        m = re.search(r"x(\d+)$", i.mnemonic)
        count = int(m.group(1)) if m else 4
        t.emit_ldtm(self._dst(i, 0), count, self._src(t, i, 1))

    # ---- Memory stores ----

    def _h_stg(self, t: TLASassThread, i: Instruction):
        t.emit_stg(self._src(t, i, 0), self._src(t, i, 1))

    def _h_sts(self, t: TLASassThread, i: Instruction):
        t.emit_sts(self._src(t, i, 0), self._src(t, i, 1))

    # ---- Election ----

    def _h_elect(self, t: TLASassThread, i: Instruction):
        t.emit_elect(self._dst(i, 0), self._dst(i, 1))

    # ---- Synchronization ----

    def _h_warpsync(self, t: TLASassThread, i: Instruction):
        mask = self._src(t, i, 0) if i.operands else Literal(0xFFFFFFFF)
        t.emit_warpsync(mask)

    def _h_bar_sync(self, t: TLASassThread, i: Instruction):
        t.emit_bar_sync()

    def _h_membar(self, t: TLASassThread, i: Instruction):
        t.emit_membar()

    def _h_bssy(self, t: TLASassThread, i: Instruction):
        t.emit_bssy()

    def _h_bsync(self, t: TLASassThread, i: Instruction):
        t.emit_bsync()

    def _h_depbar(self, t: TLASassThread, i: Instruction):
        t.emit_depbar()

    def _h_nop(self, t: TLASassThread, i: Instruction):
        t.emit_nop()

    # ---- Register-pool management ----

    def _h_usetmaxreg(self, t: TLASassThread, i: Instruction) -> None:
        # TRY_ALLOC: USETMAXREG.TRY_ALLOC.CTAPOOL UP0, <size>
        #   operands[0] = UP0 (output pred), operands[1] = size immediate
        # DEALLOC:   USETMAXREG.DEALLOC.CTAPOOL <size>
        #   operands[0] = size immediate
        if "TRY_ALLOC" in i.mnemonic:
            size = self._src(t, i, 1)
        else:
            size = self._src(t, i, 0)
        t.emit_usetmaxreg(size)
