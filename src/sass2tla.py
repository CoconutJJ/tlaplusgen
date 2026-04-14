import constants
import constants
from tla_module import (
    Variable,
    ForAll,
    NotEqual,
    Literal,
    Index,
    Domain,
    LtE,
    And,
    GtE,
    Mod,
    Equal,
    Eventually,
    Always,
    Add,
    LtE,
)
import re
import sys
from sass.cfg import build_cfgs, slice_cfg, to_dot
from sass.parser import parse_file
from tla_codegen import SassCFGCodegen
from argparse import ArgumentParser
from constants import *

args = ArgumentParser()

args.add_argument("sassfile")
args.add_argument("--module")
args.add_argument("--keep_control_edges", action="store_true")
args.add_argument("--instr_match", default="WARPSYNC")
args.add_argument("--export_dot", default="store_true")
args.add_argument("--kernel", default=None)
args.add_argument("--gridDim", type=int, nargs=3, default=(1, 1, 1))
args.add_argument("--blockDim", type=int, nargs=3, default=(1, 1, 1))
args.add_argument("--regs_per_thread", type=int, required=True)
params = args.parse_args()

prog = parse_file(params.sassfile)
cfgs = build_cfgs(prog)

if params.kernel is None:
    print("Please select a kernel using the --kernel option:")
    for k in cfgs:
        print(k)

    exit(0)

if params.kernel not in cfgs:
    print("Kernel name not found. Valid kernels are: ")
    for k in cfgs:
        print(k)

    exit(0)

cfg = cfgs[params.kernel]
sliced = slice_cfg(cfg, params.instr_match, keep_control=params.keep_control_edges)
codegen = SassCFGCodegen()
module_name = params.module or re.sub(r"[^A-Za-z0-9]", "_", params.kernel)[:64]
proc = codegen.generate(
    sliced,
    module_name,
    params.regs_per_thread,
    gridDim=tuple(params.gridDim),
    blockDim=tuple(params.blockDim),
)

t = Variable("t")
proc.createInvariant(
    "NoErrorState",
    ForAll(
        t,
        Domain(proc.getPcMap()),
        NotEqual(Index(proc.getPcMap(), t), Literal(proc.errorState)),
    ),
)

proc.createInvariant(
    "RegReqCheck",
    ForAll(
        t,
        Domain(proc.getNumReg()),
        And(
            LtE(t, constants.MAX_REG_REQ),
            GtE(t, Literal(constants.MIN_REG_REQ)),
            Equal(Mod(t, Literal(constants.DIVISIBLE_REG_REQ)), Literal(0)),
        ),
    ),
)

# Example: An invariant that holds for all thread blocks
block_invariants = []
total_blocks = proc._getTotalThreadCount() // proc._getBlockSize()

# for block_id in range(total_blocks):
#     threads_in_block = proc._iterBlockThreads(block_id)
#     # The threads in this block are TLAThread instances.
#     # You can get their string names (e.g. "t0", "t1") using t.thread_name
#     thread_names = [Literal(t.thread_name) for t in threads_in_block]

#     # Calculate the sum of numReg for all threads in this block
#     block_reg_sum = Add(*[Index(proc.getNumReg(), t_val) for t_val in thread_names])
#     proc.createInvariant(
#         "NoDLWaitingForReg",
#         Eventually(Always(LtE(block_reg_sum, Literal(self.reg_per_block)))),
#     )

# Example: Ensure at least one thread in the block has PC != "error"
# block_condition = Or(*[NotEqual(Index(proc.getPcMap(), t_val), Literal(proc.errorState)) for t_val in thread_names])
# block_invariants.append(block_condition)

# Now combine all the block conditions with a massive AND (unrolled ForAll)
# proc.createInvariant(
#     "AllBlocksInvariant",
#     And(*block_invariants)
# )

with open(f"{module_name}.tla", "w") as f:
    f.write(str(proc))

with open(f"{module_name}.cfg", "w") as f:
    f.write(proc.getConfiguration())

if params.export_dot:
    with open(f"{module_name}.dot", "w") as f:
        f.write(to_dot(sliced, show_instructions=True))

for msg in codegen.log:
    print(msg, file=sys.stderr)
