from tla_module import (
    Expr,
    Mul,
    Add,
    And,
    Shl,
    IfThenElse,
    Equal,
    Literal,
    Max,
    Min,
    FunnelShr,
    MappingIndex,
    MappingValue,
    Mapping,
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


class TLASassProcess(TLAProcess["TLASassThread"]):
    thread_factory = TLASassThread

    def __init__(self, name: str) -> None:
        super().__init__(name)
