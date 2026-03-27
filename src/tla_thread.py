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


class TLAProcess(TLAModule):
    def __init__(self, name: str, n_threads=1) -> None:
        super().__init__(name)
        self.mem = self.createConstant("mem")
        self.thread_initial_states = []
        self.thread_step_states = []
        self.threads: list[TLAThread] = []
        self.thread_pc_map = self.createVariable("pcs")
        self.start_state = "start"

    def _uniqueName(self, threadName: str, name: str):
        return f"{threadName}_{name}"

    def createThreads(
        self,
        registers: list[MappingIndex],
        initialRegisterValues: list[MappingValue],
        count: int,
        names: list[str] = [],
    ) -> list["TLAThread"]:

        if len(names) == 0:
            for c in range(count):
                self.threads.append(
                    TLAThread(self, f"t{c}", registers, initialRegisterValues)
                )
        else:
            assert len(names) == count

            for name in names:
                self.threads.append(
                    TLAThread(self, name, registers, initialRegisterValues)
                )

        self.thread_initial_states.append(
            Equal(
                self.thread_pc_map,
                Mapping([f"t{c}" for c in range(count)], [Literal("start")] * count),
            )
        )

        return self.threads

    def addThreadInitialState(self, state: Expr):
        self.thread_initial_states.append(state)

    def addThreadStepState(self, state: Definition):
        self.thread_step_states.append(state)

    def getPc(self, name: str):
        return Index(self.thread_pc_map, Literal(name))

    def getPcMap(self):
        return self.thread_pc_map

    def updatePcExpr(self, threadName: str, newState: str):
        return Equal(
            self.thread_pc_map.next(),
            MappingUpdate(
                self.thread_pc_map, [(Literal(threadName), Literal(newState))]
            ),
        )

    def __str__(self):

        for t in self.threads:
            t._createStepState()

        self.setInitialState(And(*self.thread_initial_states))
        self.setNextState(Or(*self.thread_step_states))

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
        self.pc = self.process.getPc(self.thread_name)
        self.reg_mapping = Mapping(registers, initialRegisterValues)
        self.regs = self.process.createVariable(f"regs_{thread_name}")
        self.current_state = self.process.start_state
        self.thread_states: list[Definition] = []
        self._pushNewState(self.current_state)
        self.process.addThreadInitialState(Equal(self.regs, self.reg_mapping))

    def _uniqueName(self, suffix: str) -> str:
        return (
            self.process._uniqueName(self.thread_name, suffix)
            + f"_{len(self.pc_states)}"
        )

    def _currentState(self) -> str:
        return self.current_state

    def _getCurrentStep(self) -> int:
        return len(self.pc_states)

    def _pushNewState(self, name: str = "") -> str:
        state_name = self.allocateState(name=name)
        self.setState(state_name)
        return state_name

    def pcTransition(self, current: str, next: str):
        return And(
            Equal(self.pc, Literal(current)),
            self.process.updatePcExpr(self.thread_name, next),
        )

    def _createStepState(self):
        stepDef = self.process.createDefinition(
            self._uniqueName("step"), Or(*self.thread_states)
        )
        self.process.addThreadStepState(stepDef)

    def _goto(self, toState: str) -> Expr:

        return And(
            Equal(self.pc, Literal(self._currentState())),
            self.process.updatePcExpr(self.thread_name, toState),
        )

    def _unchangedExcept(self, variables: list[Variable]):

        variable_names: set[str] = set([str(v.name) for v in variables])

        unchanged = []
        for v in self.process.variables:
            if v.name not in variable_names:
                unchanged.append(v)

        return unchanged

    def _createUnchangedExceptExpr(self, expr: Expr, variables: list[Variable]):

        unchanged = self._unchangedExcept(variables)

        if len(unchanged) > 0:
            return And(expr, Unchanged(unchanged))

        return expr

    def allocateState(self, name: str = "") -> str:
        state_name = self._uniqueName("state") if len(name) == 0 else name
        self.pc_states.append(state_name)
        return state_name

    def setState(self, newState):
        self.current_state = newState

    def getRegister(self, name: str):

        return Index(self.regs, Literal(name))

    def stopInstruction(self):

        instr = Equal(self.pc, Literal(self._currentState()))

        instr = self._createUnchangedExceptExpr(instr, [])

        definition = self.process.createDefinition(self._uniqueName("stop"), instr)
        self.thread_states.append(definition)

    def appendInstruction(self, instruction_name: str, expr: Expr, state=None) -> str:

        if state is not None:
            self.setState(state)

        currentState = self._currentState()

        pc_transition = self.pcTransition(currentState, self._pushNewState())

        definition = self.process.createDefinition(
            self._uniqueName(instruction_name),
            And(pc_transition, expr),
        )
        self.thread_states.append(definition)

        return currentState

    def appendRegisterInstruction(
        self, instruction_name: str, destination_register: str, source: Expr, state=None
    ) -> str:

        instr = Equal(
            self.regs.next(),
            MappingUpdate(self.regs, [(Literal(destination_register), source)]),
        )

        instr = self._createUnchangedExceptExpr(
            instr, [self.regs, self.process.getPcMap()]
        )

        return self.appendInstruction(
            instruction_name,
            instr,
            state=state,
        )

    def appendWaitInstruction(self, instruction_name: str, condition: Expr, state=None):
        self.appendInstruction(instruction_name, condition, state=state)

    def appendBranchInstruction(
        self, condition: Expr, true_state: str, false_state: str
    ):

        instr = IfThenElse(condition, self._goto(true_state), self._goto(false_state))
        instr = self._createUnchangedExceptExpr(instr, [self.process.getPcMap()])
        definition = self.process.createDefinition(
            f"branch_{true_state}_{false_state}",
            instr,
        )

        self.thread_states.append(definition)


if __name__ == "__main__":
    tlaproc = TLAProcess("hello")
    tlathread = tlaproc.createThreads(
        [f"r{c}" for c in range(0, 10)], [Literal(0)] * 10, 10
    )

    print(tlaproc)
