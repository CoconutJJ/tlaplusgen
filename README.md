# tla-gen

A Python library for programmatically generating TLA+ specifications. Build TLA+ modules, expressions, and concurrent process models from Python code — intended as the emission backend for symbolic executors, program analysis tools, or any system that needs to output verifiable TLA+ specs.

## Overview

TLA+ is a formal specification language used to model and verify concurrent and distributed systems. Writing TLA+ by hand is tedious and error-prone when the spec is generated from another tool (e.g. a symbolic executor analyzing assembly code). This library provides a Python AST for TLA+ expressions and a set of higher-level abstractions for building modules and concurrent thread models.

## Modules

### `tla_module.py` — Core Expression AST and Module Builder

Provides the expression tree and the `TLAModule` class that renders a complete TLA+ module.

#### Expressions

All expression types inherit from `Expr` and implement `__str__` to render valid TLA+ syntax.

| Class | TLA+ output | Description |
|---|---|---|
| `Literal(v)` | `"foo"` or `42` | String or integer constant |
| `Variable(name)` | `pc` | A state variable |
| `Variable.next()` | `pc'` | Primed (next-state) form of a variable |
| `Constant(name)` | `input_rax` | An uninterpreted symbolic input |
| `Index(expr, idx)` | `regs["rax"]` | Index into a mapping |
| `Mapping(keys, vals)` | `[r0 \|-> 0, r1 \|-> 0]` | Function literal |
| `MappingUpdate(var, updates)` | `[regs EXCEPT !["rax"] = ...]` | Functional update of a mapping |
| `Add(l, r)` | `(a + b)` | Integer addition |
| `Sub(l, r)` | `(a - b)` | Integer subtraction |
| `And(l, r)` | `(a /\\ b)` | Conjunction |
| `Or(l, r)` | `(a \\/ b)` | Disjunction |
| `Equal(l, r)` | `(a = b)` | Equality — lhs should be `Variable` or `Next` |
| `IfThenElse(c, t, e)` | `IF c THEN t ELSE e` | Conditional expression |
| `Definition(name, expr)` | `name == expr` | Named definition |

#### `TLAModule`

Collects variables, constants, definitions, and the `Init`/`Next` predicates, then renders them as a well-formed TLA+ module.

```python
from tla_module import TLAModule, And, Or, Equal, Literal, Add

module = TLAModule("MyProgram")

pc  = module.createVariable("pc")
reg = module.createVariable("reg")
inp = module.createConstant("input_rax")

module.setInitialState(And(
    Equal(pc, Literal("start")),
    Equal(reg, inp)
))

module.setNextState(Equal(pc.next(), Literal("end")))

print(module)
```

Output:
```tla
---------- MODULE MyProgram ----------
VARIABLES pc, reg
CONSTANTS input_rax
Init == ((pc = "start") /\ (reg = input_rax))
Next == (pc' = "end")
======================================
```

**Key methods:**

- `createVariable(name)` — declare a TLA+ state variable
- `createConstant(name)` — declare a TLA+ `CONSTANT` (symbolic input)
- `createDefinition(name, expr)` — add a named definition to the module
- `setInitialState(expr)` — set the `Init` predicate
- `setNextState(expr)` — set the `Next` predicate

---

### `tla_process.py` — Concurrent Process and Thread Model

Builds on `TLAModule` to model concurrent programs with multiple threads. Each thread has its own `pc` and register file. The module's `Init` is the conjunction of all thread initial states; `Next` is the disjunction of all thread step relations, modeling **interleaved concurrency** — at each step, exactly one thread advances.

#### `TLAProcess`

```python
from tla_process import TLAProcess
from tla_module import Literal, Add

proc = TLAProcess("MyProgram")
thread = proc.createThread(
    "t1",
    registers=["r0", "r1", "r2"],
    initialRegisterValues=[Literal(0), Literal(0), Literal(0)]
)
```

#### `TLAThread`

Represents a single thread. Instructions are appended sequentially — each one allocates a new `pc` state and emits a transition definition.

```python
# Straight-line register instruction: r0 = r0 + r1
s1 = thread.appendRegisterInstruction(
    "add_r0_r1",
    destination_register="r0",
    source=Add(thread.getRegister("r0"), thread.getRegister("r1"))
)

# Another instruction — s2 is the pc label of its entry point
s2 = thread.appendRegisterInstruction(
    "add_r0_r1_again",
    destination_register="r0",
    source=Add(thread.getRegister("r0"), thread.getRegister("r1"))
)

# Conditional branch — jumps to s1 or s2 depending on condition
thread.appendBranchInstruction(
    condition=Add(thread.getRegister("r0"), thread.getRegister("r1")),
    true_state=s1,
    false_state=s2
)

print(proc)
```

**Key methods:**

| Method | Description |
|---|---|
| `appendInstruction(name, expr)` | Append a generic instruction with an arbitrary transition expression |
| `appendRegisterInstruction(name, dst, src)` | Append a register write: `regs' = [regs EXCEPT ![dst] = src]` |
| `appendBranchInstruction(cond, true_state, false_state)` | Append a conditional branch to two existing pc labels |
| `getRegister(name)` | Return an `Index` expression reading a register from this thread's register file |

`appendInstruction` and `appendRegisterInstruction` both return the pc label of the instruction's **entry point**, which can be passed as a branch target to create loops or conditional jumps.

---

## Concurrency Model

`TLAProcess` models interleaved concurrency. The emitted `Next` relation is:

```tla
Next == thread1_step \/ thread2_step \/ ...
```

TLC explores all possible interleavings of thread steps, exposing race conditions and other concurrency bugs. Shared memory is modeled as a `CONSTANT` (`mem`) — it is symbolic and immutable across transitions. To model writable shared memory, promote it to a `VARIABLE` in a subclass.

---

## Symbolic Inputs

Program inputs are modeled as TLA+ `CONSTANTS` — uninterpreted values that are fixed for a given run but unconstrained by the spec. This enables symbolic reasoning:

- **TLC**: override constants with concrete values in the model config to check specific inputs
- **TLAPS**: reason about all inputs universally
- **Symbolic execution pipeline**: use Z3 to prune infeasible paths before generating the spec, so only reachable transitions are emitted

```python
input_rax = proc.createConstant("input_rax")
# input_rax is never primed — it doesn't change across transitions
```

---

## Design Notes

**`IF THEN ELSE` and symbolic constants**: TLC evaluates `IF THEN ELSE` conditions concretely at model-check time. Avoid using symbolic `CONSTANT` values in branch conditions — TLC cannot evaluate them. Instead, split branches into two separate guarded transitions and accumulate path conditions externally (e.g. with Z3).

**`Definition.__str__` vs `toDefString()`**: `str(d)` returns just the definition's name, for use as a reference inside other expressions. `d.toDefString()` returns the full `name == expr` declaration for emission into the module.

**`MappingUpdate`** constructs a new function via TLA+'s `EXCEPT` — it does not mutate the original mapping. All other keys are implicitly preserved.

---

## Requirements

- Python 3.12+ (uses `type` alias syntax)
- No external dependencies for core generation
- Z3 (`pip install z3-solver`) recommended for path condition feasibility checking in a symbolic execution pipeline