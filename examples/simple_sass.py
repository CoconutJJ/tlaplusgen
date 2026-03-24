from tla_sass import TLASassProcess, TLASassThread
from tla_module import Literal

# 1. Create a process (one TLA+ module)
proc = TLASassProcess("MyKernel")

# 2. Create a thread with named registers and initial values
thread = proc.createThread(
    name="t0",
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