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
    Equal,
    Literal,
    Expr,
    Index,
    Unchanged,
    TLAMap,
    TLAInt,
    TLAStr,
    TLAType,
)
from typing import TypeVar, Generic, Type, cast
from functools import reduce
import re as _re

TProcess = TypeVar("TProcess", bound="TLAProcess")


class TLAThread(Generic[TProcess]):
    def __init__(
        self,
        process: TProcess,
        thread_name: str,
        global_thread_id: int = -1,
    ) -> None:

        self.process = process
        self.pc_states = []
        self.thread_name = thread_name
        self.global_thread_id = global_thread_id
        self.pc = self.process.getPc(self.thread_name)

        self.reg_set_mappings = []
        self.register_name_map: dict[str | int, Index] = dict()
        # Maps pc_label -> max regular register index accessed at that PC.
        self.pc_max_reg: dict[str, int] = {}

        self.current_state = self.process.start_state
        self.thread_definitions: list[Definition] = []
        self._pushNewState(self.current_state)

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

    def createRegisterSet(
        self,
        set_name: str,
        names: list[MappingIndex],
        initialValues: list[MappingValue],
    ) -> Variable:
        mapping = Mapping(names, initialValues)

        map_type = None
        if self.process.apalache_compatible:
            assert reduce(
                lambda accum, x: accum & x,
                [isinstance(r, type(names[0])) for r in names],
            )
            assert reduce(
                lambda accum, x: accum & x,
                [isinstance(r, type(initialValues[0])) for r in initialValues],
            )
            map_type = TLAMap(
                TLAType.fromNative(names[0]), TLAType.fromNative(initialValues[0])
            )

        reg_map = self.process.createVariable(
            f"regs_{self.thread_name}_{set_name}", tla_type=map_type
        )

        for r in names:
            assert r not in self.register_name_map
            self.register_name_map[r] = Index(reg_map, Literal(r))

        self.process.addThreadInitialState(Equal(reg_map, mapping))

        return reg_map

    def pcTransition(self, current: str, next: str):
        return And(
            Equal(self.pc, Literal(current)),
            self.process.updatePcExpr(self.thread_name, next),
        )

    def _createNextStepDefinition(self):
        if len(self.thread_definitions) > 0:
            stepDef = self.process.createDefinition(
                self._uniqueName("step"), Or(*self.thread_definitions)
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
        return self.register_name_map[name]

    def record_reg_access(self, reg_name: str) -> None:
        """Track the highest regular register index (R0-R255) accessed at the current PC."""
        m = _re.match(r'^R(\d+)$', reg_name)
        if m:
            idx = int(m.group(1))
            state = self._currentState()
            if idx > self.pc_max_reg.get(state, -1):
                self.pc_max_reg[state] = idx

    def stopInstruction(self):

        instr = Equal(self.pc, Literal(self._currentState()))

        instr = self._createUnchangedExceptExpr(instr, [])

        definition = self.process.createDefinition(self._uniqueName("stop"), instr)
        self.thread_definitions.append(definition)

    def appendInstruction(self, instruction_name: str, expr: Expr, state=None) -> str:

        if state is not None:
            self.setState(state)

        currentState = self._currentState()

        pc_transition = self.pcTransition(currentState, self._pushNewState())

        definition = self.process.createDefinition(
            self._uniqueName(instruction_name),
            And(pc_transition, expr),
        )
        self.thread_definitions.append(definition)

        return currentState

    def appendRegisterInstruction(
        self, instruction_name: str, destination_register: str, source: Expr, state=None
    ) -> str:

        self.record_reg_access(destination_register)
        dest_reg = self.getRegister(destination_register)

        assert isinstance(dest_reg.value, Variable)

        instr = Equal(
            dest_reg.value.next(),
            MappingUpdate(dest_reg.value, [(Literal(destination_register), source)]),
        )

        instr = self._createUnchangedExceptExpr(
            instr, [dest_reg.value, self.process.getPcMap()]
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

        self.thread_definitions.append(definition)


TThread = TypeVar("TThread", bound="TLAThread")


class TLAProcess(TLAModule, Generic[TThread]):
    thread_factory: Type[TThread] = cast(Type[TThread], TLAThread)

    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.mem = self.createVariable("mem", TLAMap(TLAInt(), TLAInt()))
        self.thread_initial_states = []
        self.thread_step_states = []
        self.threads: list[TThread] = []
        self.thread_pc_map = self.createVariable("pcs", TLAMap(TLAStr(), TLAStr()))
        self.start_state = "start"
        self.current_thread_count = 0

    def _uniqueName(self, threadName: str, name: str):
        return f"{threadName}_{name}"

    def initialize(self):
        self.thread_initial_states.append(
            Equal(
                self.thread_pc_map,
                Mapping(
                    [f"t{c}" for c in range(self.current_thread_count)],
                    [Literal("start")] * self.current_thread_count,
                ),
            )
        )

    def createThreads(
        self,
        count: int,
        names: list[str] = [],
    ) -> list["TThread"]:

        if len(names) == 0:
            for c in range(self.current_thread_count, count):
                self.threads.append(
                    self.thread_factory(self, f"t{c}", global_thread_id=c)
                )
        else:
            assert len(names) == count

            for c, name in enumerate(names):
                print(f"created thread {c}, {name}")
                self.threads.append(
                    self.thread_factory(
                        self, name, global_thread_id=self.current_thread_count + c
                    )
                )

        self.current_thread_count += count

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
            t._createNextStepDefinition()

        self.setInitialState(And(*self.thread_initial_states))
        self.setNextState(Or(*self.thread_step_states))

        return super().__str__()
