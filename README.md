# TLA+ SASS Process Modeler

A Python library for modeling NVIDIA SASS (Shader Assembly) GPU instructions as formal TLA+ specifications. Built on top of a general-purpose TLA+ expression and process modeling framework, this library lets you encode GPU thread behavior at the instruction level and verify it with the TLC model checker.

---

## Overview

The library has three layers:

| Module | Purpose |
|---|---|
| `tla_module.py` | Core TLA+ AST — expressions, operators, module/config generation |
| `tla_thread.py` | Process and thread abstractions over the TLA+ AST |
| `tla_sass.py` | SASS instruction set layer on top of `TLAThread` |

The end result is a `.tla` file and a `.cfg` configuration file you can feed directly into [TLC](https://lamport.azurewebsites.net/tla/tools.html).

---

## Architecture

```
tla_module.py          (Expr, BinOp, Mapping, TLAModule, ...)
      ↑
tla_thread.py          (TLAProcess, TLAThread)
      ↑
tla_sass.py            (TLASassProcess, TLASassThread)
```

**`tla_module.py`** provides the expression AST — literals, variables, mappings, logical/arithmetic operators, temporal operators, and the `TLAModule` class that serializes everything into a valid TLA+ module string plus a TLC configuration block.

**`tla_thread.py`** builds a process/thread model on top of `TLAModule`. Each `TLAThread` gets a program counter (`pc_<name>`) and a register file (`regs_<name>`), both modeled as TLA+ variables. Instructions are appended sequentially; each one generates a TLA+ definition that guards on the current `pc` value and advances it on completion.

**`tla_sass.py`** exposes individual SASS instructions as methods on `TLASassThread`, translating each opcode into the appropriate arithmetic expression and delegating to `appendRegisterInstruction`.

---

## SASS Instructions Supported

| Method | SASS Opcode | Semantics |
|---|---|---|
| `IMAD(dest, l, r, c)` | `IMAD` | `dest = (l * r) + c` |
| `IMAD_MOV_U32(dest, c)` | `IMAD.MOV.U32` | `dest = c` (move via IMAD) |
| `IMAD_SHL_U32(dest, target, amount)` | `IMAD.SHL.U32` | `dest = target << amount` |
| `IADD3(dest, r1, r2, r3)` | `IADD3` | `dest = r1 + r2 + r3` |
| `VIADD(dest, r1, r2)` | `IADD3` (2-operand) | `dest = r1 + r2` |
| `VIADDMNMX_U32(pred, dest, r1, r2)` | `VIADDMNMX.U32` | `dest = MAX(r1,r2)` if `pred=1`, else `MIN(r1,r2)` |
| `LEA(dest, r1, r2, shift)` | `LEA` | `dest = r1 + (r2 << shift)` |
| `SHF_R_U32_HI(dest, r1, r2, shift)` | `SHF.R.U32.HI` | `dest = funnel_shr(r1, r2, shift) >> 32` |

Shifts (`Shl`, `Shr`, `FunnelShr`) are encoded as integer arithmetic (`* 2^n`, `\div 2^n`) so TLC can reason about them without bit-vector support.

---

## Usage

```python
from tla_sass import TLASassProcess, TLASassThread
from tla_module import Literal

# 1. Create a process (one TLA+ module)
proc = TLASassProcess("MyKernel")

# 2. Create a thread with named registers and initial values
thread = TLASassThread(
    proc,
    thread_name="t0",
    registers=["r0", "r1", "r2", "r3"],
    initialRegisterValues=[Literal(0), Literal(1), Literal(2), Literal(0)],
)

# 3. Append instructions
thread.IMAD("r3", "r0", "r1", "r2")   # r3 = (r0 * r1) + r2
thread.IMAD_SHL_U32("r0", "r0", Literal(2))  # r0 = r0 << 2
thread.IADD3("r2", "r0", "r1", "r3")  # r2 = r0 + r1 + r3
thread.stopInstruction()

# 4. Emit TLA+ module and TLC config
print(proc)                    # → module .tla
print(proc.getConfiguration()) # → .cfg file
```

---

## Output

`print(proc)` produces a TLA+ module like:

```
---------- MODULE MyKernel ----------
VARIABLES pc_t0, regs_t0
CONSTANTS mem
Init == ...
t0_step_1_imad == ...
...
Next == t0_step_N_step
=====================================
```

`print(proc.getConfiguration())` produces the TLC config:

```
INIT Init
NEXT Next
CHECK_DEADLOCK TRUE
```

Save these as `MyKernel.tla` and `MyKernel.cfg` respectively, then run TLC to check invariants or liveness properties.

---

## Extending

To add a new SASS instruction, subclass `TLASassThread` (or add a method directly) and call `appendRegisterInstruction` with the appropriate expression built from the `tla_module` primitives:

```python
def ISETP_LT(self, dest_reg: str, r1: str, r2: str):
    a = self.getRegister(r1)
    b = self.getRegister(r2)
    self.appendRegisterInstruction("isetp_lt", dest_reg, Lt(a, b))
```

To add invariants or temporal properties, use the inherited `TLAModule` methods:

```python
proc.addInvariant(GtE(thread.getRegister("r0"), Literal(0)))
```

---

## Dependencies

- Python 3.12+ (uses `type` alias syntax)
- No external packages — pure stdlib
- [TLA+ Toolbox](https://lamport.azurewebsites.net/tla/toolbox.html) or the standalone `tlc2` CLI to run the generated specs