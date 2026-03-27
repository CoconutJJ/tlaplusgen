from tla_sass import TLASassProcess
from tla_module import Literal

# 1. Create a process (one TLA+ module)
proc = TLASassProcess("MyKernel")

# 2. Create a thread with named registers and initial values
[thread] = proc.createThreads(
    ["r0", "r1", "r2", "r3"],
    [Literal(0), Literal(1), Literal(2), Literal(0)],
    1
)

thread.gotoErrorStateIfSeenRegInstr()
thread.setSeenRegInstr(True)
thread.stopInstruction()

# 4. Emit TLA+ module and TLC config
print(proc)                    # → module .tla
print(proc.getConfiguration()) # → .cfg file