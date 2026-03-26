from tla_module import Variable
from tla_module import (
    Expr,
    Mul,
    Add,
    Shl,
    IfThenElse,
    Equal,
    Literal,
    Max,
    Min,
    FunnelShr,
)
from tla_thread import TLAProcess, TLAThread


class TLASassProcess(TLAProcess):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def createThread(self, name: str, registers: list[str | int], initialRegisterValues: list[Expr]) -> "TLASassThread":
        thread = TLASassThread(self, name, registers, initialRegisterValues)
        self.threads.append(thread)
        return thread

class TLASassThread(TLAThread):
    def __init__(
        self,
        process: TLASassProcess,
        thread_name: str,
        registers: list[str | int],
        initialRegisterValues: list[Expr],
    ) -> None:
        self.seenRegInstr = process.createVariable(f"seenRegInstr_{thread_name}")
        process.addThreadInitialState(Equal(self.seenRegInstr, Literal(False)))
        super().__init__(process, thread_name, registers, initialRegisterValues)

    def setSeenRegInstr(self, state : bool):
        instr = Equal(self.seenRegInstr.next(), (Literal(True) if state else Literal(False)))
        instr = self._createUnchangedExceptExpr(instr, [self.seenRegInstr])
        self.appendInstruction("setseenreginstr", instr)

    def IMAD(self, dest_reg: str, l_reg: str, r_reg: str, c_reg: str):

        l = self.getRegister(l_reg)
        r = self.getRegister(r_reg)
        c = self.getRegister(c_reg)

        self.appendRegisterInstruction("imad", dest_reg, Add(Mul(l, r), c))

    def IMAD_MOV_U32(self, dest_reg: str, c_reg: str):

        c = self.getRegister(c_reg)
        self.appendRegisterInstruction("imad_mov_u32", dest_reg, c)

    def IMAD_SHL_U32(self, dest_reg: str, target_reg: str, amount: Expr):

        target = self.getRegister(target_reg)
        self.appendRegisterInstruction("imad_shl_u32", dest_reg, Shl(target, amount))

    def IADD3(self, dest_reg: str, r1: str, r2: str, r3: str):

        a = self.getRegister(r1)
        b = self.getRegister(r2)
        c = self.getRegister(r3)

        self.appendRegisterInstruction("iadd3", dest_reg, Add(a, b, c))

    def VIADD(self, dest_reg: str, r1: str, r2: str):
        a = self.getRegister(r1)
        b = self.getRegister(r2)

        self.appendRegisterInstruction("iadd3", dest_reg, Add(a, b))

    def VIADDMNMX_U32(self, predicate_reg: str, dest_reg: str, r1: str, r2: str):

        predicate = self.getRegister(predicate_reg)

        a = self.getRegister(r1)
        b = self.getRegister(r2)

        self.appendRegisterInstruction(
            "viaddmnmx_u32",
            dest_reg,
            IfThenElse(Equal(predicate, Literal(1)), Max(a, b), Min(a, b)),
        )

    def LEA(self, dest_reg: str, r1: str, r2: str, shift_amount: Expr):

        a = self.getRegister(r1)
        b = self.getRegister(r2)

        self.appendRegisterInstruction("lea", dest_reg, Add(a, Shl(b, shift_amount)))

    def SHF_R_U32_HI(self, dest_reg: str, r1: str, r2: str, shift_amount: Expr):

        a = self.getRegister(r1)
        b = self.getRegister(r2)

        self.appendRegisterInstruction(
            "shf_r_u32_hi", dest_reg, FunnelShr(a, b, shift_amount)
        )

    def USETMAXREGDEALLOC(self, reg_num: int):
        self.setSeenRegInstr(True)
    
    def USETMINREGDEALLOC(self, reg_num: int):
        self.setSeenRegInstr(False)
        
    def WARPSYNCALL(self):
        self.setSeenRegInstr(False)
        