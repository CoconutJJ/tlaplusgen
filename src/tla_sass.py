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
        super().__init__(process, thread_name, registers, initialRegisterValues)
        self.seenRegInstr = process.createVariable(f"seenRegInstr_{thread_name}")
        process.addThreadInitialState(Equal(self.seenRegInstr, Literal(False)))

    def setSeenRegInstr(self, state : bool):
        instr = Equal(self.seenRegInstr.next(), (Literal(True) if state else Literal(False)))
        instr = self._createUnchangedExceptExpr(instr, [self.seenRegInstr])
        self.appendInstruction("setseenreginstr", instr)

        