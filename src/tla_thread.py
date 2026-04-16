from tla_module import (
    TLAModule,
    Mapping,
    MappingUpdate,
    MappingValue,
    MappingIndex,
    MappingRange,
    Definition,
    Parameter,
    Concat,
    SetComprehension,
    MapComprehension,
    IfThenElse,
    Variable,
    And,
    Or,
    Equal,
    Literal,
    Expr,
    Index,
    Unchanged,
    Exists,
    Domain,
    TLAMap,
    TLAInt,
    TLAStr,
    TLAType,
    ToString
)
from typing import TypeVar, Generic, Type, cast
from functools import reduce

TProcess = TypeVar("TProcess", bound="TLAProcess")


class TLAThread(Generic[TProcess]):
    def __init__(
        self,
        process: TProcess,
        global_thread_id: int = -1,
    ) -> None:

        self.process = process
        self.pc_states = []
        self.global_thread_id = global_thread_id

        # The thread parameter used in quantified definitions
        self.t_param = Parameter("t")

        # PC is accessed via the shared pcs map with the thread parameter
        self.pc = Index(self.process.thread_pc_map, self.t_param)

        self.reg_set_mappings = []
        self.register_name_map: dict[str | int, Index] = dict()
        self.register_set_map: dict[str | int, Variable] = dict()

        self.current_state = self.process.start_state
        self.thread_definitions: list[Definition] = []
        self._pushNewState(self.current_state)

    def _uniqueName(self, suffix: str) -> str:
        return f"{suffix}_{len(self.pc_states)}"

    def _currentState(self) -> str:
        return self.current_state

    def _getCurrentStep(self) -> int:
        return len(self.pc_states)

    def _pushNewState(self, name: str = "") -> str:
        state_name = self.allocateState(name=name)
        self.setState(state_name)
        return state_name

    # def createRegisterSetRange(self, set_name:str, start: int, end: int, value: MappingValue):
    #     inner_mapping = MappingRange(start, end, value)

    #     map_type = None
    #     if self.process.apalache_compatible:
    #         map_type = TLAMap(
    #             TLAStr(),
    #             TLAMap(
    #                 TLAInt(), TLAType.fromNative(value)
    #             ),
    #         )

    #     # Shared variable: regs_{set_name} (not per-thread)
    #     reg_map = self.process.createVariable(f"regs_{set_name}", tla_type=map_type)

    #     # Store init mapping for process to replicate across threads
    #     self.process._register_inits.append((reg_map, inner_mapping))

    #     # Nested indexing: reg_map[t]["R4"]
    #     for r in names:
    #         assert r not in self.register_name_map
    #         self.register_name_map[r] = reg_map[self.t_param][r]
    #         self.register_set_map[r] = reg_map

    #     self.reg_set_mappings.append(reg_map)

    #     return reg_map

    def createRegisterSet(
        self,
        set_name: str,
        names: list[MappingIndex],
        initialValues: list[MappingValue],
    ) -> Variable:
        inner_mapping = Mapping(names, initialValues)

        map_type = None
        if self.process.apalache_compatible:
            assert reduce(
                lambda accum, x: accum and x,
                [isinstance(r, type(names[0])) for r in names],
            )
            assert reduce(
                lambda accum, x: accum and x,
                [isinstance(r, type(initialValues[0])) for r in initialValues],
            )
            map_type = TLAMap(
                TLAStr(),
                TLAMap(
                    TLAType.fromNative(names[0]), TLAType.fromNative(initialValues[0])
                ),
            )

        # Shared variable: regs_{set_name} (not per-thread)
        reg_map = self.process.createVariable(f"regs_{set_name}", tla_type=map_type)

        # Store init mapping for process to replicate across threads
        self.process._register_inits.append((reg_map, inner_mapping))

        # Nested indexing: reg_map[t]["R4"]
        for r in names:
            assert r not in self.register_name_map
            self.register_name_map[r] = reg_map[self.t_param][r]
            self.register_set_map[r] = reg_map

        self.reg_set_mappings.append(reg_map)

        return reg_map

    def pcTransition(self, current: str, next: str):
        return And(
            Equal(self.pc, Literal(current)),
            self.process.updatePcExpr(self.t_param, next),
        )

    def _createNextStepDefinition(self):
        if len(self.thread_definitions) > 0:
            t = self.t_param
            invocations = [d(t) for d in self.thread_definitions]
            stepDef = self.process.createDefinition(
                "step", Or(*invocations), params=[t]
            )
            self.process.addThreadStepState(stepDef)

    def _goto(self, toState: str) -> Expr:

        return And(
            Equal(self.pc, Literal(self._currentState())),
            self.process.updatePcExpr(self.t_param, toState),
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

    def stopInstruction(self):

        instr = Equal(self.pc, Literal(self._currentState()))

        instr = self._createUnchangedExceptExpr(instr, [])

        definition = self.process.createDefinition(
            self._uniqueName("stop"), instr, params=[self.t_param]
        )
        self.thread_definitions.append(definition)

    def appendInstruction(self, instruction_name: str, expr: Expr, state=None) -> str:

        if state is not None:
            self.setState(state)

        currentState = self._currentState()

        pc_transition = self.pcTransition(currentState, self._pushNewState())

        definition = self.process.createDefinition(
            self._uniqueName(instruction_name),
            And(pc_transition, expr),
            params=[self.t_param],
        )
        self.thread_definitions.append(definition)

        return currentState

    def appendRegisterInstruction(
        self, instruction_name: str, destination_register: str, source: Expr, state=None
    ) -> str:

        reg_set = self.register_set_map[destination_register]

        # Inner update: [reg_set[t] EXCEPT !["R4"] = source]
        inner = MappingUpdate(
            reg_set[self.t_param],
            [(Literal(destination_register), source)],
        )
        # Outer update: reg_set' = [reg_set EXCEPT ![t] = inner]
        instr = Equal(
            reg_set.next(),
            MappingUpdate(reg_set, [(self.t_param, inner)]),
        )

        instr = self._createUnchangedExceptExpr(
            instr, [reg_set, self.process.getPcMap()]
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
            params=[self.t_param],
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
        self.template_thread: TThread | None = None
        self.thread_pc_map = self.createVariable("pcs", TLAMap(TLAStr(), TLAStr()))
        self.start_state = "start"
        self.current_thread_count = 0

        # Storage for shared register set init mappings
        self._register_inits: list[tuple[Variable, Mapping | MappingRange]] = []
        # Storage for shared scalar-per-thread init values
        self._shared_var_inits: list[tuple[Variable, Expr]] = []

    def _uniqueName(self, threadName: str, name: str):
        return f"{threadName}_{name}"

    def initialize(self):
        thread_names = [f"t{c}" for c in range(self.current_thread_count)]

        t = Parameter("t")
        thread_name_set = SetComprehension(
            0, self.current_thread_count - 1, t, Concat(Literal("t"), ToString(t))
        )

        # PC init
        self.thread_initial_states.append(
            Equal(
                self.thread_pc_map,
                MapComprehension(t,
                    thread_name_set,
                    Literal("start")
                ),
            )
        )

        # Register set inits: each is a map from thread_name -> inner_mapping
        for reg_map, inner_mapping in self._register_inits:
            self.thread_initial_states.append(
                Equal(
                    reg_map,
                    Mapping(list(thread_names), [inner_mapping] * len(thread_names)),
                )
            )

        # Shared scalar-per-thread variable inits
        for var, init_val in self._shared_var_inits:
            self.thread_initial_states.append(
                Equal(
                    var,
                    Mapping(list(thread_names), [init_val] * len(thread_names)),
                )
            )

    def createThreads(
        self,
        count: int,
    ) -> "TThread":

        # Track total thread count for init state generation
        self.current_thread_count += count
        self.template_thread = self.thread_factory(self, global_thread_id=-1)
        return self.template_thread

    def addThreadInitialState(self, state: Expr):
        self.thread_initial_states.append(state)

    def addThreadStepState(self, state: Definition):
        self.thread_step_states.append(state)

    def getPc(self, name):
        if isinstance(name, Expr):
            return Index(self.thread_pc_map, name)
        return Index(self.thread_pc_map, Literal(name))

    def getPcMap(self):
        return self.thread_pc_map

    def updatePcExpr(self, threadParam: Expr, newState: str):
        return Equal(
            self.thread_pc_map.next(),
            MappingUpdate(self.thread_pc_map, [(threadParam, Literal(newState))]),
        )

    def __str__(self):

        assert self.template_thread is not None
        self.template_thread._createNextStepDefinition()

        self.setInitialState(And(*self.thread_initial_states))

        # Next == \E t \in DOMAIN pcs : step(t)
        t_var = Parameter("t")
        assert len(self.thread_step_states) == 1
        step_def = self.thread_step_states[0]
        self.setNextState(Exists(t_var, Domain(self.thread_pc_map), step_def(t_var)))

        return super().__str__()
