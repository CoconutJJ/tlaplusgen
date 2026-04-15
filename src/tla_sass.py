from tla_module import Variable, Equal, Index
from typing import Tuple
import re as _re
import constants

from tla_module import (
    Expr,
    And,
    IfThenElse,
    Literal,
    Max,
    Min,
    FunnelShr,
    MappingIndex,
    MappingValue,
    MappingUpdate,
    Mapping,
    Variable,
    TLABool,
    TLAInt,
    TLAStr,
    TLAMap,
)
from tla_thread import TLAProcess, TLAThread
from itertools import product


class TLASassThread(TLAThread["TLASassProcess"]):
    def __init__(
        self,
        process: "TLASassProcess",
        global_thread_id: int = -1,
    ) -> None:
        super().__init__(process, global_thread_id)
        self.usetmaxreg_inc_pcs: list[str] = []

        # seenRegInstr is a shared map: thread_name -> bool
        self.seenRegInstr = process.createVariable(
            "seenRegInstr", tla_type=TLAMap(TLAStr(), TLABool())
        )
        process._shared_var_inits.append((self.seenRegInstr, Literal(False)))

        self.regular_regs = self.createRegisterSet(
            "regular", [f"R{i}" for i in range(0, 256)], [Literal(0)] * 256
        )

        self.predicate_regs = self.createRegisterSet(
            "predicate",
            list([f"P{i}" for i in range(0, 7)] + ["PT"]),
            list([Literal(False)] * 7 + [Literal(True)]),
        )

        self.uniform_regs = self.createRegisterSet(
            "uniform",
            list([f"UR{i}" for i in range(0, 80)] + ["URZ"]),
            [Literal(0)] * 81,
        )

        self.uniform_pred_regs = self.createRegisterSet(
            "uniform_pred",
            list([f"UP{i}" for i in range(0, 64)] + ["UPT"]),
            list([Literal(False)] * 64 + [Literal(True)]),
        )

    def hasSeenRegInstrExpr(self):
        # Explicit Equal to avoid Index.__eq__ update semantics
        return Equal(Index(self.seenRegInstr, self.t_param), Literal(True))

    def gotoErrorStateIfSeenRegInstr(self):
        nextState = self.allocateState()
        self.appendBranchInstruction(
            self.hasSeenRegInstrExpr(),
            self.process.errorState,
            nextState,
        )
        self.setState(nextState)

    def _emitSeenRegInstr(self, changeExpr: Expr):
        # seenRegInstr' = [seenRegInstr EXCEPT ![t] = TRUE]
        instr = Equal(
            self.seenRegInstr.next(),
            MappingUpdate(self.seenRegInstr, [(self.t_param, Literal(True))]),
        )
        instr = instr & changeExpr
        instr = self._createUnchangedExceptExpr(
            instr,
            [
                self.seenRegInstr,
                self.process.numRegThread,
                self.process.blockToRegPoolCount,
                self.process.getPcMap(),
            ],
        )
        return self.appendInstruction("usetmaxreg", instr)

    def disableSeenRegInstr(self):
        # seenRegInstr' = [seenRegInstr EXCEPT ![t] = FALSE]
        instr = Equal(
            self.seenRegInstr.next(),
            MappingUpdate(self.seenRegInstr, [(self.t_param, Literal(False))]),
        )
        instr = self._createUnchangedExceptExpr(
            instr,
            [
                self.seenRegInstr,
                self.process.getPcMap(),
            ],
        )
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
        return self._append_multi_reg_instr(
            instruction_name, [(dst0, val0), (dst1, val1)]
        )

    def _append_multi_reg_instr(
        self,
        instruction_name: str,
        updates: list[tuple[str, Expr]],
    ) -> str:
        """Emit an instruction that writes N registers atomically."""

        # Group updates by register set variable
        reg_maps = dict()
        reg_id_map = dict()
        for reg, value in updates:
            reg_set = self.register_set_map[reg]
            reg_var_id = id(reg_set)

            if reg_var_id not in reg_maps:
                reg_id_map[reg_var_id] = reg_set
                reg_maps[reg_var_id] = []

            reg_maps[reg_var_id].append((reg, value))

        mapping_updates = []
        unchangedExcept = [self.process.getPcMap()]
        for reg_var_id in reg_maps:
            reg_var = reg_id_map[reg_var_id]
            # Inner: [reg_var[t] EXCEPT !["R4"] = v1, !["R5"] = v2]
            inner = MappingUpdate(
                Index(reg_var, self.t_param),
                [(Literal(dst), val) for dst, val in reg_maps[reg_var_id]],
            )
            # Outer: reg_var' = [reg_var EXCEPT ![t] = inner]
            mapping_updates.append(
                Equal(
                    reg_var.next(),
                    MappingUpdate(reg_var, [(self.t_param, inner)]),
                )
            )
            unchangedExcept.append(reg_var)

        instr = And(*mapping_updates)
        instr = self._createUnchangedExceptExpr(instr, unchangedExcept)
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
        return self.appendRegisterInstruction("iadd3", dst, src1 + src2 + src3)

    def emit_imad(self, dst: str, src1: Expr, src2: Expr, src3: Expr) -> str:
        """IMAD / IMAD.U32 / IMAD.MOV / IMAD.IADD / IMAD.SHL variants
        – dst = src1 * src2 + src3
        """
        return self.appendRegisterInstruction("imad", dst, src1 * src2 + src3)

    def emit_imad_hi_u32(self, dst: str, src1: Expr, src2: Expr, src3: Expr) -> str:
        """IMAD.HI.U32 – dst = ((src1*src2 + (dst<<32 | src3)) >> 32)

        The current value of dst participates as a read-modify-write source.
        Modelled as: (src1*src2 + dst*2^32 + src3) / 2^32
        """
        dst_expr = self.getRegister(dst)
        full = src1 * src2 + ((dst_expr << Literal(32)) + src3)
        return self.appendRegisterInstruction("imad_hi_u32", dst, full >> Literal(32))

    def emit_imad_wide(self, dst: str, src1: Expr, src2: Expr, src3: Expr) -> str:
        """IMAD.WIDE / IMAD.WIDE.U32 / UIMAD.WIDE / UIMAD.WIDE.U32
        – {dst+1, dst} = src1 * src2 + src3  (64-bit result in two registers)

        dst   = lower 32 bits of product
        dst+1 = upper 32 bits of product
        """
        prod = src1 * src2 + src3
        return self._append_multi_reg_instr(
            "imad_wide",
            [
                (dst, prod),
                (self._reg_name_plus(dst, 1), prod >> Literal(32)),
            ],
        )

    def emit_iabs(self, dst: str, src: Expr) -> str:
        """IABS – dst = |src|"""
        return self.appendRegisterInstruction("iabs", dst, Max(src, Literal(0) - src))

    def emit_imnmx_u32(self, dst: str, src1: Expr, src2: Expr, mnpred: Expr) -> str:
        """IMNMX.U32 – dst = mnpred ? min(src1, src2) : max(src1, src2)"""
        return self.appendRegisterInstruction(
            "imnmx_u32", dst, IfThenElse(mnpred, Min(src1, src2), Max(src1, src2))
        )

    # -----------------------------------------------------------------------
    # Shift
    # SHF.R.*.HI: funnel-shift right, keep the upper 32 bits of the 64-bit result.
    # concat(src2, src1) is the 64-bit input; rot is the shift amount.
    # -----------------------------------------------------------------------

    def emit_shf_r_u32_hi(self, dst: str, src1: Expr, rot: Expr, src2: Expr) -> str:
        """SHF.R.U32.HI / USHF.R.U32.HI
        – dst = upper32(concat(src2, src1) >> rot)
        """
        return self.appendRegisterInstruction(
            "shf_r_u32_hi", dst, FunnelShr(src2, src1, rot)
        )

    def emit_shf_r_s32_hi(self, dst: str, src1: Expr, rot: Expr, src2: Expr) -> str:
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
        return self.appendRegisterInstruction("sel", dst, IfThenElse(pred, src1, src2))

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
        concat = (ahi << Literal(32)) + alo
        upper = (concat << imm_shift) >> Literal(32)
        return self.appendRegisterInstruction("lea_hi", dst, upper + b)

    # -----------------------------------------------------------------------
    # Memory loads – modelled as dst = mem[addr]
    # The `mem` constant in TLAProcess represents an abstract address space.
    # Wide loads (64/128-bit) write contiguous register pairs/quads.
    # -----------------------------------------------------------------------

    def emit_ldg(self, dst: str, addr: Expr) -> str:
        """LDG.E – load 32 bits from global memory: dst = mem[addr]
        # Not in sass_insns.h
        """
        return self.appendRegisterInstruction("ldg", dst, self.process.mem[addr])

    def emit_ldg_128(self, dst: str, addr: Expr) -> str:
        """LDG.E.128 / LDG.E.128.STRONG.GPU / LDG.E.128.CONSTANT /
        LDG.E.LTC128B.CONSTANT – load 128 bits (4 registers) from global memory.
        # Not in sass_insns.h
        """
        mem = self.process.mem
        return self._append_multi_reg_instr(
            "ldg_128",
            [(self._reg_name_plus(dst, i), mem[addr + Literal(i)]) for i in range(4)],
        )

    def emit_lds(self, dst: str, addr: Expr) -> str:
        """LDS – load 32 bits from shared memory: dst = mem[addr]
        # Not in sass_insns.h
        """
        return self.appendRegisterInstruction("lds", dst, self.process.mem[addr])

    def emit_lds_64(self, dst: str, addr: Expr) -> str:
        """LDS.64 – load 64 bits (2 registers) from shared memory.
        # Not in sass_insns.h
        """
        mem = self.process.mem
        return self._append_multi_reg_instr(
            "lds_64",
            [(self._reg_name_plus(dst, i), mem[addr + Literal(i)]) for i in range(2)],
        )

    def emit_lds_128(self, dst: str, addr: Expr) -> str:
        """LDS.128 – load 128 bits (4 registers) from shared memory.
        # Not in sass_insns.h
        """
        mem = self.process.mem
        return self._append_multi_reg_instr(
            "lds_128",
            [(self._reg_name_plus(dst, i), mem[addr + Literal(i)]) for i in range(4)],
        )

    def emit_ldsm(self, dst: str, addr: Expr) -> str:
        """LDSM.16.MT88.4 – load shared-memory matrix tile (4 registers).
        # Not in sass_insns.h
        """
        mem = self.process.mem
        return self._append_multi_reg_instr(
            "ldsm",
            [(self._reg_name_plus(dst, i), mem[addr + Literal(i)]) for i in range(4)],
        )

    def emit_ldc(self, dst: str, addr: Expr) -> str:
        """LDC / LDCU – load 32 bits from constant memory: dst = mem[addr]
        # Not in sass_insns.h
        """
        return self.appendRegisterInstruction("ldc", dst, self.process.mem[addr])

    def emit_ldc_64(self, dst: str, addr: Expr) -> str:
        """LDC.64 / LDCU.64 – load 64 bits (2 registers) from constant memory.
        # Not in sass_insns.h
        """
        mem = self.process.mem
        return self._append_multi_reg_instr(
            "ldc_64",
            [(self._reg_name_plus(dst, i), mem[addr + Literal(i)]) for i in range(2)],
        )

    def emit_ldc_128(self, dst: str, addr: Expr) -> str:
        """LDCU.128 – load 128 bits (4 registers) from constant memory.
        # Not in sass_insns.h
        """
        mem = self.process.mem
        return self._append_multi_reg_instr(
            "ldc_128",
            [(self._reg_name_plus(dst, i), mem[addr + Literal(i)]) for i in range(4)],
        )

    def emit_uldc_64(self, dst: str, addr: Expr) -> str:
        """ULDC.64 – load uniform 64-bit constant (2 registers).
        # Not in sass_insns.h
        """
        mem = self.process.mem
        return self._append_multi_reg_instr(
            "uldc_64",
            [(self._reg_name_plus(dst, i), mem[addr + Literal(i)]) for i in range(2)],
        )

    def emit_ldtm(self, dst: str, count: int, addr: Expr) -> str:
        """LDTM.x4 / LDTM.x32 / LDTM.x128 / LDTM.16dp256bit.x4 /
        LDTM.16dp256bit.x16 – tiling memory load; writes `count` registers.
        # Not in sass_insns.h
        """
        mem = self.process.mem
        return self._append_multi_reg_instr(
            "ldtm",
            [
                (self._reg_name_plus(dst, i), mem[addr + Literal(i)])
                for i in range(count)
            ],
        )

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
        return self.emit_isetp(dst0, dst1, (src1 < src2) & src3)

    def emit_isetp_gt_and(
        self, dst0: str, dst1: str, src1: Expr, src2: Expr, src3: Expr
    ) -> str:
        """ISETP.GT.AND / UISETP.GT.AND – dst0 = (src1 > src2) /\\ src3"""
        return self.emit_isetp(dst0, dst1, (src1 > src2) & src3)

    def emit_isetp_ge_and(
        self, dst0: str, dst1: str, src1: Expr, src2: Expr, src3: Expr
    ) -> str:
        """ISETP.GE.AND / ISETP.GE.U32.AND / UISETP.GE.U32.AND
        – dst0 = (src1 >= src2) /\\ src3
        """
        return self.emit_isetp(dst0, dst1, (src1 >= src2) & src3)

    def emit_isetp_ne_and(
        self, dst0: str, dst1: str, src1: Expr, src2: Expr, src3: Expr
    ) -> str:
        """ISETP.NE.AND / ISETP.NE.U32.AND / UISETP.NE.U32.AND
        – dst0 = (src1 /= src2) /\\ src3
        """
        return self.emit_isetp(dst0, dst1, (src1 != src2) & src3)

    def emit_isetp_eq_u32_and(
        self, dst0: str, dst1: str, src1: Expr, src2: Expr, src3: Expr
    ) -> str:
        """ISETP.EQ.U32.AND – dst0 = (src1 = src2) /\\ src3"""
        return self.emit_isetp(dst0, dst1, (src1 == src2) & src3)

    def emit_isetp_ge_or(
        self, dst0: str, dst1: str, src1: Expr, src2: Expr, src3: Expr
    ) -> str:
        """ISETP.GE.OR / ISETP.GE.U32.OR – dst0 = (src1 >= src2) \\/ src3"""
        return self.emit_isetp(dst0, dst1, (src1 >= src2) | src3)

    def emit_isetp_gt_u32_and(
        self, dst0: str, dst1: str, src1: Expr, src2: Expr, src3: Expr
    ) -> str:
        """ISETP.GT.U32.AND / UISETP.GT.U32.AND – dst0 = (src1 > src2) /\\ src3
        (unsigned comparison; at the TLA+ level identical to signed GT)
        """
        return self.emit_isetp(dst0, dst1, (src1 > src2) & src3)

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

    def emit_usetmaxreg(self, absReg: Expr, is_inc: bool):
        change = (self.process.incReg if is_inc else self.process.decReg)(self, absReg)
        pc_label = self._emitSeenRegInstr(change)
        if is_inc:
            self.usetmaxreg_inc_pcs.append(pc_label)


class TLASassProcess(TLAProcess["TLASassThread"]):
    thread_factory = TLASassThread

    def __init__(
        self,
        name: str,
        gridDim: Tuple[int, int, int],
        blockDim: Tuple[int, int, int],
        regPerThread: int,
        **kwargs,
    ) -> None:
        super().__init__(name, **kwargs)
        self.gridDims = gridDim
        self.blockDims = blockDim
        self.regPerCTA = self._getBlockSize() * regPerThread

        self.numRegThread = self.createVariable(
            "numRegThread", tla_type=TLAMap(TLAStr(), TLAInt())
        )
        self.blockToRegPoolCount = self.createVariable(
            "BlockToRegPoolCount", tla_type=TLAMap(TLAStr(), TLAInt())
        )
        self.threadToBlockIndex = self.createVariable("ThreadToBlockIndex", tla_type=TLAMap(TLAStr(), TLAInt()))
        self.errorState = "error"

        self.createThreads(self._getTotalThreadCount())

    def initialize(self):
        thread_names = [f"t{c}" for c in range(self._getTotalThreadCount())]

        self.addThreadInitialState(
            self.numRegThread
            == Mapping(
                list(thread_names),
                [Literal(constants.INITREG)] * self.current_thread_count,
            )
        )

        self.addThreadInitialState(
            self.blockToRegPoolCount
            == Mapping(
                [f"cta{i}" for i in range(self._getNumCTAs())],
                [Literal(self.regPerCTA)] * self._getNumCTAs(),
            )
        )

        self.addThreadInitialState(
            self.threadToBlockIndex
            == Mapping(
                list(thread_names),
                [
                    Literal(self._getBlockIndex(i))
                    for i in range(self._getTotalThreadCount())
                ],
            )
        )

        super().initialize()

    def getNumRegThread(self) -> Variable:
        return self.numRegThread

    def incReg(self, thread: "TLASassThread", absoluteReg: Expr):
        t = thread.t_param
        currentThreadReg = Index(self.numRegThread, t)
        ctaName = Index(self.threadToBlockIndex, t)
        currentCTAReg = Index(self.blockToRegPoolCount, ctaName)
        delta = absoluteReg - currentThreadReg
        return (
            (
                Equal(currentCTAReg + delta, Literal(self.regPerCTA))
                | (currentCTAReg + delta < Literal(self.regPerCTA))
            )
            & Equal(
                self.blockToRegPoolCount.next(),
                MappingUpdate(
                    self.blockToRegPoolCount, [(ctaName, currentCTAReg - delta)]
                ),
            )
            & Equal(
                self.numRegThread.next(),
                MappingUpdate(self.numRegThread, [(t, absoluteReg)]),
            )
        )

    def decReg(self, thread: "TLASassThread", absoluteReg: Expr):
        t = thread.t_param
        currentThreadReg = Index(self.numRegThread, t)
        ctaName = Index(self.threadToBlockIndex, t)
        currentCTAReg = Index(self.blockToRegPoolCount, ctaName)
        delta = currentThreadReg - absoluteReg
        return Equal(
            self.numRegThread.next(),
            MappingUpdate(self.numRegThread, [(t, absoluteReg)]),
        ) & Equal(
            self.blockToRegPoolCount.next(),
            MappingUpdate(self.blockToRegPoolCount, [(ctaName, currentCTAReg + delta)]),
        )

    def createPrivateThreadRegisters(
        self,
    ) -> tuple[list[MappingIndex], list[MappingValue]]:
        reg_names: list[MappingIndex] = [f"R{d}" for d in range(255)] + [
            f"P{d}" for d in range(7)
        ]  # type: ignore
        reg_values = [Literal(0)] * len(reg_names)

        return reg_names, reg_values  # type: ignore

    def _getBlockSize(self):
        blockDimX, blockDimY, blockDimZ = self.blockDims

        return blockDimX * blockDimY * blockDimZ

    def _getGlobalThreadId(
        self, gridCoord: tuple[int, int, int], blockCoord: tuple[int, int, int]
    ):
        gridDimX, gridDimY, gridDimZ = self.gridDims
        blockDimX, blockDimY, blockDimZ = self.blockDims
        gX, gY, gZ = gridCoord
        bX, bY, bZ = blockCoord

        blockThreadCount = self._getBlockSize()

        globalThreadId = (
            gZ * (gridDimY * gridDimX) + gY * (gridDimX) + gX
        ) * blockThreadCount + (bZ * (blockDimY * blockDimX) + bY * (blockDimX) + bX)

        return globalThreadId

    def _getNumCTAs(self):
        return self.gridDims[0] * self.gridDims[1] * self.gridDims[2]

    def _getTotalThreadCount(self):
        gridDimX, gridDimY, gridDimZ = self.gridDims
        blockDimX, blockDimY, blockDimZ = self.blockDims

        totalThreads = (
            gridDimX * gridDimY * gridDimZ * blockDimX * blockDimY * blockDimZ
        )

        return totalThreads

    def _getBlockIndex(self, globalThreadId: int):
        return globalThreadId // self._getBlockSize()
