import re as _re

from tla_module import (
    Expr,
    Mul,
    Add,
    Sub,
    And,
    Or,
    Shl,
    Shr,
    IfThenElse,
    Equal,
    NotEqual,
    Gt,
    Lt,
    GtE,
    LtE,
    Literal,
    Max,
    Min,
    FunnelShr,
    Index,
    MappingIndex,
    MappingValue,
    Mapping,
    MappingUpdate,
)
from tla_thread import TLAProcess, TLAThread


class TLASassThread(TLAThread):
    def __init__(
        self,
        process: "TLASassProcess",
        thread_name: str,
        registers: list[str | int],
        initialRegisterValues: list[Expr],
    ) -> None:
        super().__init__(process, thread_name, registers, initialRegisterValues)
        self.seenRegInstr = process.createVariable(f"seenRegInstr_{thread_name}")
        process.addThreadInitialState(Equal(self.seenRegInstr, Literal(False)))
        self.errorState = self.allocateState(f"{thread_name}_error")

    def hasSeenRegInstrExpr(self):
        return Equal(self.seenRegInstr, Literal(True))

    def gotoErrorStateIfSeenRegInstr(self):
        nextState = self.allocateState()
        self.appendBranchInstruction(
            self.hasSeenRegInstrExpr(),
            self.errorState,
            nextState,
        )
        self.setState(nextState)

    def setSeenRegInstr(self, state: bool):
        instr = Equal(
            self.seenRegInstr.next(), (Literal(True) if state else Literal(False))
        )
        instr = self._createUnchangedExceptExpr(instr, [self.seenRegInstr])
        self.appendInstruction("setseenreginstr", instr)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _reg_name_plus(self, base: str, offset: int) -> str:
        """Adjacent register name: 'R4' + 1 → 'R5', 'UR6' + 2 → 'UR8'."""
        m = _re.match(r"([A-Za-z]+)(\d+)$", base)
        if not m:
            return base
        return f"{m.group(1)}{int(m.group(2)) + offset}"

    def _append_dual_reg_instr(
        self,
        instruction_name: str,
        dst0: str,
        val0: Expr,
        dst1: str,
        val1: Expr,
    ) -> str:
        """Emit an instruction that writes two registers atomically."""
        instr = Equal(
            self.regs.next(),
            MappingUpdate(self.regs, [(Literal(dst0), val0), (Literal(dst1), val1)]),
        )
        instr = self._createUnchangedExceptExpr(
            instr, [self.regs, self.process.getPcMap()]
        )
        return self.appendInstruction(instruction_name, instr)

    def _append_multi_reg_instr(
        self,
        instruction_name: str,
        updates: list[tuple[str, Expr]],
    ) -> str:
        """Emit an instruction that writes N registers atomically."""
        instr = Equal(
            self.regs.next(),
            MappingUpdate(self.regs, [(Literal(dst), val) for dst, val in updates]),
        )
        instr = self._createUnchangedExceptExpr(
            instr, [self.regs, self.process.getPcMap()]
        )
        return self.appendInstruction(instruction_name, instr)

    def _stutter(self, name: str) -> str:
        """PC-advancing no-op: advance PC, leave all other variables unchanged."""
        instr = self._createUnchangedExceptExpr(
            Literal(True), [self.process.getPcMap()]
        )
        return self.appendInstruction(name, instr)

    # -----------------------------------------------------------------------
    # Data movement
    # MOV / UMOV / S2R / CS2R / ULDC: dst = src
    # -----------------------------------------------------------------------

    def emit_mov(self, dst: str, src: Expr) -> str:
        """MOV / UMOV / S2R / CS2R / ULDC – dst = src"""
        return self.appendRegisterInstruction("mov", dst, src)

    # -----------------------------------------------------------------------
    # Integer arithmetic
    # -----------------------------------------------------------------------

    def emit_iadd3(self, dst: str, src1: Expr, src2: Expr, src3: Expr) -> str:
        """IADD3 / UIADD3 – dst = src1 + src2 + src3"""
        return self.appendRegisterInstruction("iadd3", dst, Add(src1, src2, src3))

    def emit_imad(self, dst: str, src1: Expr, src2: Expr, src3: Expr) -> str:
        """IMAD / IMAD.U32 / IMAD.MOV / IMAD.IADD / IMAD.SHL variants
        – dst = src1 * src2 + src3
        """
        return self.appendRegisterInstruction("imad", dst, Add(Mul(src1, src2), src3))

    def emit_imad_hi_u32(
        self, dst: str, src1: Expr, src2: Expr, src3: Expr
    ) -> str:
        """IMAD.HI.U32 – dst = ((src1*src2 + (dst<<32 | src3)) >> 32)

        The current value of dst participates as a read-modify-write source.
        Modelled as: (src1*src2 + dst*2^32 + src3) / 2^32
        """
        dst_expr = self.getRegister(dst)
        full = Add(Mul(src1, src2), Add(Shl(dst_expr, Literal(32)), src3))
        return self.appendRegisterInstruction(
            "imad_hi_u32", dst, Shr(full, Literal(32))
        )

    def emit_imad_wide(self, dst: str, src1: Expr, src2: Expr, src3: Expr) -> str:
        """IMAD.WIDE / IMAD.WIDE.U32 / UIMAD.WIDE / UIMAD.WIDE.U32
        – {dst+1, dst} = src1 * src2 + src3  (64-bit result in two registers)

        dst   = lower 32 bits of product
        dst+1 = upper 32 bits of product
        """
        product = Add(Mul(src1, src2), src3)
        return self._append_multi_reg_instr("imad_wide", [
            (dst,                            product),
            (self._reg_name_plus(dst, 1),    Shr(product, Literal(32))),
        ])

    def emit_iabs(self, dst: str, src: Expr) -> str:
        """IABS – dst = |src|"""
        return self.appendRegisterInstruction(
            "iabs", dst, Max(src, Sub(Literal(0), src))
        )

    def emit_imnmx_u32(
        self, dst: str, src1: Expr, src2: Expr, mnpred: Expr
    ) -> str:
        """IMNMX.U32 – dst = mnpred ? min(src1, src2) : max(src1, src2)"""
        return self.appendRegisterInstruction(
            "imnmx_u32", dst, IfThenElse(mnpred, Min(src1, src2), Max(src1, src2))
        )

    # -----------------------------------------------------------------------
    # Shift
    # SHF.R.*.HI: funnel-shift right, keep the upper 32 bits of the 64-bit result.
    # concat(src2, src1) is the 64-bit input; rot is the shift amount.
    # -----------------------------------------------------------------------

    def emit_shf_r_u32_hi(
        self, dst: str, src1: Expr, rot: Expr, src2: Expr
    ) -> str:
        """SHF.R.U32.HI / USHF.R.U32.HI
        – dst = upper32(concat(src2, src1) >> rot)
        """
        return self.appendRegisterInstruction(
            "shf_r_u32_hi", dst, FunnelShr(src2, src1, rot)
        )

    def emit_shf_r_s32_hi(
        self, dst: str, src1: Expr, rot: Expr, src2: Expr
    ) -> str:
        """SHF.R.S32.HI / USHF.R.S32.HI – signed funnel-shift right, upper 32 bits.
        Semantics are identical to the unsigned variant at the TLA+ level.
        """
        return self.appendRegisterInstruction(
            "shf_r_s32_hi", dst, FunnelShr(src2, src1, rot)
        )

    # -----------------------------------------------------------------------
    # Logical
    # -----------------------------------------------------------------------

    def emit_lop3_lut(
        self, dst: str, src1: Expr, src2: Expr, src3: Expr, imm_lut: Expr
    ) -> str:
        """LOP3.LUT / ULOP3.LUT – 3-input bitwise logic via 8-bit truth table.

        TLA+ has no bitwise operators; modelled as an opaque IF-THEN-ELSE that
        carries data dependencies on all three sources and the LUT constant.
        Sufficient for data-flow / reachability analysis.
        """
        result = IfThenElse(
            src1,
            IfThenElse(src2, src3, imm_lut),
            IfThenElse(src2, imm_lut, src3),
        )
        return self.appendRegisterInstruction("lop3_lut", dst, result)

    def emit_lop3_lut_dual(
        self,
        dst_pred: str,
        dst_reg: str,
        src1: Expr,
        src2: Expr,
        src3: Expr,
        imm_lut: Expr,
    ) -> str:
        """LOP3.LUT (7-operand / 2-write form) – writes a predicate and a GPR.
        # Not in sass_insns.h (7-operand variant with predicate destination)

        Both outputs carry the same logical result; the predicate output is the
        boolean coercion of the integer result.
        """
        result = IfThenElse(
            src1,
            IfThenElse(src2, src3, imm_lut),
            IfThenElse(src2, imm_lut, src3),
        )
        return self._append_dual_reg_instr(
            "lop3_lut_dual", dst_pred, result, dst_reg, result
        )

    def emit_plop3_lut(
        self,
        dst0: str,
        dst1: str,
        src1: Expr,
        src2: Expr,
        src3: Expr,
        imm_lut: Expr,
        src4: Expr,
    ) -> str:
        """PLOP3.LUT – 3-input predicate logic via 8-bit truth table; writes two
        predicate registers.
        # Not in sass_insns.h

        dst0 = logical_op3(src1, src2, src3, immLut)
        dst1 = ~dst0
        src4 is an additional predicate source feeding the LUT; modelled as
        an extra operand in the opaque IF-THEN-ELSE approximation.
        """
        result = IfThenElse(
            src1,
            IfThenElse(src2, src3, imm_lut),
            IfThenElse(src2, imm_lut, src4),
        )
        return self._append_dual_reg_instr(
            "plop3_lut", dst0, result, dst1, self._bool_not(result)
        )

    def emit_sel(self, dst: str, src1: Expr, src2: Expr, pred: Expr) -> str:
        """SEL / USEL – dst = pred ? src1 : src2"""
        return self.appendRegisterInstruction(
            "sel", dst, IfThenElse(pred, src1, src2)
        )

    # -----------------------------------------------------------------------
    # Address computation
    # -----------------------------------------------------------------------

    def emit_lea_hi(
        self, dst: str, alo: Expr, b: Expr, ahi: Expr, imm_shift: Expr
    ) -> str:
        """LEA.HI / ULEA.HI – dst = upper32(concat(ahi, alo) << imm_shift) + b

        Equivalent to: (ahi * 2^imm_shift + alo * 2^imm_shift / 2^32) + b
        Modelled as: Shr(Shl(ahi*2^32 + alo, imm_shift), 32) + b
        """
        concat = Add(Shl(ahi, Literal(32)), alo)
        upper = Shr(Shl(concat, imm_shift), Literal(32))
        return self.appendRegisterInstruction("lea_hi", dst, Add(upper, b))

    # -----------------------------------------------------------------------
    # Memory loads – modelled as dst = mem[addr]
    # The `mem` constant in TLAProcess represents an abstract address space.
    # Wide loads (64/128-bit) write contiguous register pairs/quads.
    # -----------------------------------------------------------------------

    def emit_ldg(self, dst: str, addr: Expr) -> str:
        """LDG.E – load 32 bits from global memory: dst = mem[addr]
        # Not in sass_insns.h
        """
        return self.appendRegisterInstruction(
            "ldg", dst, Index(self.process.mem, addr)
        )

    def emit_ldg_128(self, dst: str, addr: Expr) -> str:
        """LDG.E.128 / LDG.E.128.STRONG.GPU / LDG.E.128.CONSTANT /
        LDG.E.LTC128B.CONSTANT – load 128 bits (4 registers) from global memory.
        # Not in sass_insns.h
        """
        mem = self.process.mem
        return self._append_multi_reg_instr("ldg_128", [
            (self._reg_name_plus(dst, i), Index(mem, Add(addr, Literal(i))))
            for i in range(4)
        ])

    def emit_lds(self, dst: str, addr: Expr) -> str:
        """LDS – load 32 bits from shared memory: dst = mem[addr]
        # Not in sass_insns.h
        """
        return self.appendRegisterInstruction(
            "lds", dst, Index(self.process.mem, addr)
        )

    def emit_lds_64(self, dst: str, addr: Expr) -> str:
        """LDS.64 – load 64 bits (2 registers) from shared memory.
        # Not in sass_insns.h
        """
        mem = self.process.mem
        return self._append_multi_reg_instr("lds_64", [
            (self._reg_name_plus(dst, i), Index(mem, Add(addr, Literal(i))))
            for i in range(2)
        ])

    def emit_lds_128(self, dst: str, addr: Expr) -> str:
        """LDS.128 – load 128 bits (4 registers) from shared memory.
        # Not in sass_insns.h
        """
        mem = self.process.mem
        return self._append_multi_reg_instr("lds_128", [
            (self._reg_name_plus(dst, i), Index(mem, Add(addr, Literal(i))))
            for i in range(4)
        ])

    def emit_ldsm(self, dst: str, addr: Expr) -> str:
        """LDSM.16.MT88.4 – load shared-memory matrix tile (4 registers).
        # Not in sass_insns.h
        """
        mem = self.process.mem
        return self._append_multi_reg_instr("ldsm", [
            (self._reg_name_plus(dst, i), Index(mem, Add(addr, Literal(i))))
            for i in range(4)
        ])

    def emit_ldc(self, dst: str, addr: Expr) -> str:
        """LDC / LDCU – load 32 bits from constant memory: dst = mem[addr]
        # Not in sass_insns.h
        """
        return self.appendRegisterInstruction(
            "ldc", dst, Index(self.process.mem, addr)
        )

    def emit_ldc_64(self, dst: str, addr: Expr) -> str:
        """LDC.64 / LDCU.64 – load 64 bits (2 registers) from constant memory.
        # Not in sass_insns.h
        """
        mem = self.process.mem
        return self._append_multi_reg_instr("ldc_64", [
            (self._reg_name_plus(dst, i), Index(mem, Add(addr, Literal(i))))
            for i in range(2)
        ])

    def emit_ldc_128(self, dst: str, addr: Expr) -> str:
        """LDCU.128 – load 128 bits (4 registers) from constant memory.
        # Not in sass_insns.h
        """
        mem = self.process.mem
        return self._append_multi_reg_instr("ldc_128", [
            (self._reg_name_plus(dst, i), Index(mem, Add(addr, Literal(i))))
            for i in range(4)
        ])

    def emit_uldc_64(self, dst: str, addr: Expr) -> str:
        """ULDC.64 – load uniform 64-bit constant (2 registers).
        # Not in sass_insns.h
        """
        mem = self.process.mem
        return self._append_multi_reg_instr("uldc_64", [
            (self._reg_name_plus(dst, i), Index(mem, Add(addr, Literal(i))))
            for i in range(2)
        ])

    def emit_ldtm(self, dst: str, count: int, addr: Expr) -> str:
        """LDTM.x4 / LDTM.x32 / LDTM.x128 / LDTM.16dp256bit.x4 /
        LDTM.16dp256bit.x16 – tiling memory load; writes `count` registers.
        # Not in sass_insns.h
        """
        mem = self.process.mem
        return self._append_multi_reg_instr("ldtm", [
            (self._reg_name_plus(dst, i), Index(mem, Add(addr, Literal(i))))
            for i in range(count)
        ])

    # -----------------------------------------------------------------------
    # Store instructions – no register writes
    # Memory effects are not modelled; these advance the PC only.
    # -----------------------------------------------------------------------

    def emit_stg(self, addr: Expr, src: Expr) -> str:
        """STG / STG.E / STG.E.128 / STG.E.128.STRONG.GPU
        – store to global memory.
        Memory writes are not modelled; this is a PC-advancing no-op.
        # Not in sass_insns.h
        """
        return self._stutter("stg")

    def emit_sts(self, addr: Expr, src: Expr) -> str:
        """STS / STS.64 / STS.128 – store to shared memory.
        Memory writes are not modelled; this is a PC-advancing no-op.
        # Not in sass_insns.h
        """
        return self._stutter("sts")

    # -----------------------------------------------------------------------
    # Predicate instructions (ISETP / UISETP)
    #
    # ISETP always writes two predicate registers atomically:
    #   dst0 = condition
    #   dst1 = ~condition  (the complement output)
    #
    # Pass PT / UPT as dst1 when the complement is unused (it will be
    # overwritten by the always-true sentinel and discarded by the model).
    # -----------------------------------------------------------------------

    def _bool_not(self, cond: Expr) -> Expr:
        """Boolean negation expressed as IF cond THEN FALSE ELSE TRUE."""
        return IfThenElse(cond, Literal(False), Literal(True))

    def emit_isetp(self, dst0: str, dst1: str, condition: Expr) -> str:
        """Generic ISETP – dst0 = condition, dst1 = ~condition"""
        return self._append_dual_reg_instr(
            "isetp", dst0, condition, dst1, self._bool_not(condition)
        )

    def emit_isetp_lt_and(
        self, dst0: str, dst1: str, src1: Expr, src2: Expr, src3: Expr
    ) -> str:
        """ISETP.LT.AND / UISETP.LT.AND – dst0 = (src1 < src2) /\\ src3"""
        return self.emit_isetp(dst0, dst1, And(Lt(src1, src2), src3))

    def emit_isetp_gt_and(
        self, dst0: str, dst1: str, src1: Expr, src2: Expr, src3: Expr
    ) -> str:
        """ISETP.GT.AND / UISETP.GT.AND – dst0 = (src1 > src2) /\\ src3"""
        return self.emit_isetp(dst0, dst1, And(Gt(src1, src2), src3))

    def emit_isetp_ge_and(
        self, dst0: str, dst1: str, src1: Expr, src2: Expr, src3: Expr
    ) -> str:
        """ISETP.GE.AND / ISETP.GE.U32.AND / UISETP.GE.U32.AND
        – dst0 = (src1 >= src2) /\\ src3
        """
        return self.emit_isetp(dst0, dst1, And(GtE(src1, src2), src3))

    def emit_isetp_ne_and(
        self, dst0: str, dst1: str, src1: Expr, src2: Expr, src3: Expr
    ) -> str:
        """ISETP.NE.AND / ISETP.NE.U32.AND / UISETP.NE.U32.AND
        – dst0 = (src1 /= src2) /\\ src3
        """
        return self.emit_isetp(dst0, dst1, And(NotEqual(src1, src2), src3))

    def emit_isetp_eq_u32_and(
        self, dst0: str, dst1: str, src1: Expr, src2: Expr, src3: Expr
    ) -> str:
        """ISETP.EQ.U32.AND – dst0 = (src1 = src2) /\\ src3"""
        return self.emit_isetp(dst0, dst1, And(Equal(src1, src2), src3))

    def emit_isetp_ge_or(
        self, dst0: str, dst1: str, src1: Expr, src2: Expr, src3: Expr
    ) -> str:
        """ISETP.GE.OR / ISETP.GE.U32.OR – dst0 = (src1 >= src2) \\/ src3"""
        return self.emit_isetp(dst0, dst1, Or(GtE(src1, src2), src3))

    def emit_isetp_gt_u32_and(
        self, dst0: str, dst1: str, src1: Expr, src2: Expr, src3: Expr
    ) -> str:
        """ISETP.GT.U32.AND / UISETP.GT.U32.AND – dst0 = (src1 > src2) /\\ src3
        (unsigned comparison; at the TLA+ level identical to signed GT)
        """
        return self.emit_isetp(dst0, dst1, And(Gt(src1, src2), src3))

    # -----------------------------------------------------------------------
    # Warp-leader election
    # -----------------------------------------------------------------------

    def emit_elect(self, dst0: str, dst1: str) -> str:
        """ELECT – elect one thread in the warp as leader.
        # Not in sass_insns.h

        dst0 = TRUE  (this thread is modelled as the elected leader)
        dst1 = FALSE (complement)

        This is a deterministic approximation; the hardware result is
        non-deterministic across warps.
        """
        return self._append_dual_reg_instr(
            "elect", dst0, Literal(True), dst1, Literal(False)
        )

    # -----------------------------------------------------------------------
    # Predicate-to-register transfer
    # -----------------------------------------------------------------------

    def emit_p2r(self, dst: str, pred_set: Expr) -> str:
        """P2R – dst = pred_set (copy predicate register value into a GPR)"""
        return self.appendRegisterInstruction("p2r", dst, pred_set)

    # -----------------------------------------------------------------------
    # Synchronization / barriers – no register writes, advance PC only
    # -----------------------------------------------------------------------

    def emit_warpsync(self, mask: Expr) -> str:
        """WARPSYNC / WARPSYNC.ALL – synchronize threads within a warp.
        # Not in sass_insns.h
        """
        return self._stutter("warpsync")

    def emit_bar_sync(self) -> str:
        """BAR.SYNC / BAR.SYNC.DEFER_BLOCKING – CTA-wide barrier.
        # Not in sass_insns.h
        """
        return self._stutter("bar_sync")

    def emit_membar(self) -> str:
        """MEMBAR.SC.GPU / MEMBAR.SC.CTA / MEMBAR.SC.SYS – memory fence.
        # Not in sass_insns.h
        """
        return self._stutter("membar")

    def emit_bssy(self) -> str:
        """BSSY – push convergence-stack entry.
        # Not in sass_insns.h
        """
        return self._stutter("bssy")

    def emit_bsync(self) -> str:
        """BSYNC – synchronize at convergence barrier.
        # Not in sass_insns.h
        """
        return self._stutter("bsync")

    def emit_depbar(self) -> str:
        """DEPBAR / DEPBAR.WAIT / DEPBAR.WAIT.LE – scoreboard dependency barrier.
        # Not in sass_insns.h
        """
        return self._stutter("depbar")

    def emit_nop(self) -> str:
        """NOP – no operation.
        # Not in sass_insns.h
        """
        return self._stutter("nop")


class TLASassProcess(TLAProcess["TLASassThread"]):
    thread_factory = TLASassThread

    def __init__(self, name: str) -> None:
        super().__init__(name)
