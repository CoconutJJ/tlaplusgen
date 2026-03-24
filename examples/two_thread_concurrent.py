"""
Example: Two-thread concurrent accumulation

Thread 1 (t1) computes a iterative sum:
    r0 = input_n        (loop counter, symbolic input)
    r1 = 0              (accumulator)
    loop:
        r1 = r1 + r0   (accumulate)
        r0 = r0 - 1    (decrement counter)
        if r0 == 0: goto done
        else: goto loop
    done:
        r2 = r1         (store result)

Thread 2 (t2) runs concurrently and computes a simple linear expression:
    r0 = input_a
    r1 = input_b
    r2 = r0 + r1        (r2 = a + b)
    r3 = r2 + r0        (r3 = 2a + b)
    r4 = r3 + r1        (r4 = 2a + 2b)
    branch: if r4 > 0 goto positive path, else goto zero path
    positive: r5 = r4
    zero:     r5 = r4 + r4  (double it)

NOTE: appendBranchInstruction uses IfThenElse, which requires the condition
to be concretely evaluable by TLC. In a real symbolic execution pipeline,
branches over symbolic inputs would instead be split into two separate
guarded transitions, with path conditions checked by Z3 before emission.
"""

from tla_thread import TLAProcess
from tla_module import (
    Literal,
    Add,
    Sub,
    Equal,
    Constant,
)

# ---------------------------------------------------------------------------
# Process setup
# ---------------------------------------------------------------------------

proc = TLAProcess("ConcurrentAccumulation")

# Symbolic inputs — unconstrained CONSTANTS in the emitted TLA+ spec.
# TLC treats these as opaque unless overridden in the model config.
input_n = proc.createConstant("input_n")  # loop bound for thread 1
input_a = proc.createConstant("input_a")  # first operand for thread 2
input_b = proc.createConstant("input_b")  # second operand for thread 2

# ---------------------------------------------------------------------------
# Thread 1: iterative sum  sum = n + (n-1) + ... + 1
# Registers:
#   r0 — loop counter (starts at input_n, decrements to 0)
#   r1 — accumulator  (starts at 0, accumulates r0 each iteration)
#   r2 — final result (copied from r1 on exit)
#   r3-r9 — unused, initialized to 0
# ---------------------------------------------------------------------------

t1 = proc.createThread(
    "t1",
    registers=[f"r{i}" for i in range(10)],
    initialRegisterValues=[
        input_n,  # r0 = input_n (loop counter)
        Literal(0),  # r1 = 0      (accumulator)
        Literal(0),  # r2 = 0      (result slot)
        *[Literal(0)] * 7,
    ],
)

t2 = proc.createThread(
    "t2",
    registers=[f"r{i}" for i in range(10)],
    initialRegisterValues=[
        input_a,  # r0 = input_a
        input_b,  # r1 = input_b
        *[Literal(0)] * 8,
    ],
)


# r1 = r1 + r0  (accumulate current counter value into sum)
# Returns the pc label of this instruction's entry — used as the loop target.
loop_top = t1.appendRegisterInstruction(
    "accumulate",
    destination_register="r1",
    source=Add(t1.getRegister("r1"), t1.getRegister("r0")),
)

# r0 = r0 - 1  (decrement loop counter)
t1.appendRegisterInstruction(
    "decrement_counter",
    destination_register="r0",
    source=Sub(t1.getRegister("r0"), Literal(1)),
)

# r2 = r1  (tentatively write result — will be overwritten if loop continues)
# In a real assembly encoding this would be after the loop, but we need
# to emit the done state before the branch so we have its pc label.
done = t1.appendRegisterInstruction(
    "store_result",
    destination_register="r2",
    source=t1.getRegister("r1"),
)

# branch: if r0 == 0 goto done (loop exit), else goto loop_top (back edge)
# The back edge is what creates the cycle in the TLA+ transition graph.
# TLC handles cycles naturally — it explores reachable states, not traces.
t1.appendBranchInstruction(
    condition=Equal(t1.getRegister("r0"), Literal(0)),
    true_state=done,
    false_state=loop_top,
)

# r2 = r0 + r1  (a + b)
t2.appendRegisterInstruction(
    "sum_a_b",
    destination_register="r2",
    source=Add(t2.getRegister("r0"), t2.getRegister("r1")),
)

# r3 = r2 + r0  (a + b + a = 2a + b)
t2.appendRegisterInstruction(
    "sum_2a_b",
    destination_register="r3",
    source=Add(t2.getRegister("r2"), t2.getRegister("r0")),
)

# r4 = r3 + r1  (2a + b + b = 2a + 2b)
t2.appendRegisterInstruction(
    "sum_2a_2b",
    destination_register="r4",
    source=Add(t2.getRegister("r3"), t2.getRegister("r1")),
)

zero = t2.allocateState()
positive = t2.allocateState()
# branch: if r4 == 0 goto zero path, else goto positive path
t2.appendBranchInstruction(
    condition=Equal(t2.getRegister("r4"), Literal(0)),
    true_state=zero,
    false_state=positive,
)

# positive path: r5 = r4  (identity)
t2.appendRegisterInstruction(
    "result_positive",
    destination_register="r5",
    source=t2.getRegister("r4"),
    state=positive,
)

t2.stopInstruction()

# zero/negative path: r5 = r4 + r4  (double it)
t2.appendRegisterInstruction(
    "result_zero",
    destination_register="r5",
    source=Add(t2.getRegister("r4"), t2.getRegister("r4")),
    state=zero,
)

t2.stopInstruction()

proc.allowDeadlock()

print(proc)
print(proc.getConfiguration())
