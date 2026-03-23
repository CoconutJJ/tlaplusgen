from tla_module import (
    TLAModule,
    Mapping,
    MappingUpdate,
    MappingValue,
    MappingIndex,
    Definition,
    IfThenElse,
    Variable,
    And,
    Or,
    Add,
    Equal,
    Literal,
    Expr,
    Index,
    Unchanged,
)
from functools import reduce


class TLAProcess(TLAModule):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.mem = self.createConstant("mem")
        self.thread_initial_states = []
        self.thread_step_states = []
        self.threads: list[TLAThread] = []

    def createThread(
        self,
        name: str,
        registers: list[MappingIndex],
        initialRegisterValues: list[MappingValue],
    ) -> "TLAThread":
        thread = TLAThread(self, name, registers, initialRegisterValues)

        self.threads.append(thread)

        return thread

    def addThreadInitialState(self, state: Expr):
        self.thread_initial_states.append(state)

    def addThreadStepState(self, state: Definition):
        self.thread_step_states.append(state)

    def __str__(self):

        for t in self.threads:
            t._createStepState()

        self.setInitialState(
            reduce(
                lambda accum, x: And(accum, x),
                self.thread_initial_states[1:],
                self.thread_initial_states[0],
            )
        )

        self.setNextState(
            reduce(
                lambda accum, x: Or(accum, x),
                self.thread_step_states[1:],
                self.thread_step_states[0],
            )
        )

        return super().__str__()


class TLAThread:
    def __init__(
        self,
        process: TLAProcess,
        thread_name: str,
        registers: list[MappingIndex],
        initialRegisterValues: list[MappingValue],
    ) -> None:

        self.process = process
        self.pc_states = []
        self.registers = registers
        self.thread_name = thread_name
        self.pc = self.process.createVariable(f"pc_{thread_name}")
        self.reg_mapping = Mapping(registers, initialRegisterValues)
        self.regs = self.process.createVariable(f"regs_{thread_name}")
        start_state = self._createNewState("start")

        self.process.addThreadInitialState(
            And(
                Equal(self.pc, Literal(start_state)), Equal(self.regs, self.reg_mapping)
            )
        )

        self.thread_states: list[Definition] = []
        self.current_state = start_state

    def _uniqueName(self, suffix: str) -> str:

        return f"{self.thread_name}_step_{str(len(self.pc_states))}_{suffix}"

    def _currentState(self) -> str:
        return self.current_state

    def _getCurrentStep(self) -> int:

        return len(self.pc_states)

    def _createNewState(self, name: str = "") -> str:
        state_name = self._uniqueName("state") if len(name) == 0 else name
        self.pc_states.append(state_name)
        self.setState(state_name)
        return state_name

    def _pcTransition(self, current: str, next: str):
        return And(
            Equal(self.pc, Literal(current)), Equal(self.pc.next(), Literal(next))
        )

    def _createStepState(self):
        stepDef = self.process.createDefinition(
            self._uniqueName("step"),
            reduce(
                lambda accum, x: Or(accum, x),
                self.thread_states[1:],
                self.thread_states[0],
            ),
        )
        self.process.addThreadStepState(stepDef)

    def _goto(self, toState: str) -> Expr:

        return And(
            Equal(self.pc, Literal(self._currentState())),
            Equal(self.pc.next(), Literal(toState)),
        )

    def _unchangedExcept(self, variables: list[Variable]):

        variable_names: set[str] = set([str(v.name) for v in variables])

        unchanged = []
        for v in self.process.variables:
            if v.name not in variable_names:
                unchanged.append(v)

        return unchanged

    def setState(self, newState):
        self.current_state = newState

    def getRegister(self, name: str):

        return Index(self.regs, Literal(name))

    def appendInstruction(self, instruction_name: str, expr: Expr) -> str:

        currentState = self._currentState()

        pc_transition = self._pcTransition(currentState, self._createNewState())

        definition = self.process.createDefinition(
            self._uniqueName(instruction_name),
            And(pc_transition, expr),
        )
        self.thread_states.append(definition)

        return currentState

    def appendRegisterInstruction(
        self,
        instruction_name: str,
        destination_register: str,
        source: Expr,
    ) -> str:

        return self.appendInstruction(
            instruction_name,
            And(
                Equal(
                    self.regs.next(),
                    MappingUpdate(self.regs, [(Literal(destination_register), source)]),
                ),
                Unchanged(self._unchangedExcept([self.regs, self.pc])),
            ),
        )

    def appendBranchInstruction(
        self, condition: Expr, true_state: str, false_state: str
    ):

        definition = self.process.createDefinition(
            f"branch_{true_state}_{false_state}",
            And(
                IfThenElse(condition, self._goto(true_state), self._goto(false_state)),
                Unchanged(self._unchangedExcept([self.pc])),
            ),
        )

        self.thread_states.append(definition)


if __name__ == "__main__":
    tlaproc = TLAProcess("hello")
    tlathread = tlaproc.createThread(
        "t1", [f"r{c}" for c in range(0, 10)], [Literal(0)] * 10
    )
    s1 = tlathread.appendRegisterInstruction(
        "add_r0_r1", "r0", Add(tlathread.getRegister("r0"), tlathread.getRegister("r1"))
    )
    s2 = tlathread.appendRegisterInstruction(
        "add_r0_r1", "r0", Add(tlathread.getRegister("r0"), tlathread.getRegister("r1"))
    )

    tlathread.appendBranchInstruction(
        Add(tlathread.getRegister("r0"), tlathread.getRegister("r1")), s1, s2
    )

    print(tlaproc)
    # tlaasm.appendInstruction("add_r0_r1", ))
