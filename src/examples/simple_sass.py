from tla_sass import TLASassProcess
from tla_module import Literal, NotEqual
from argparse import ArgumentParser
from pathlib import Path

args = ArgumentParser()
args.add_argument("moduleName")
params = args.parse_args()

# 1. Create a process (one TLA+ module)
proc = TLASassProcess(params.moduleName)

# 2. Create a thread with named registers and initial values
[thread] = proc.createThreads(
    ["r0", "r1", "r2", "r3"], [Literal(0), Literal(1), Literal(2), Literal(0)], 1
)

thread.emit_usetmaxreg(100)
thread.stopInstruction()
proc.createInvariant("NoErrorState", NotEqual(thread.pc, Literal(thread.errorState)))

with open(f"{params.moduleName}.tla", "w") as f:
    f.write(str(proc))

with open(f"{params.moduleName}.cfg", "w") as f:
    f.write(proc.getConfiguration())
