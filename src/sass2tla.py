from types import GetSetDescriptorType
import constants
from tla_module import (
    ForAll,
    NotEqual,
    Literal,
    Index,
    Domain,
    LtE,
    Lt,
    Gt,
    And,
    Or,
    GtE,
    Mod,
    Equal,
    LeadsTo,
    Implies,
    Parameter,
)
import re
import sys
from sass.cfg import build_cfgs, slice_cfg, collapse_try_alloc_retry, to_dot
from sass.parser import parse_file
from tla_codegen import SassCFGCodegen
from argparse import ArgumentParser
from sass.gpucode_adapter import (
    parse_file as gca_parse,
    build_cfgs as gca_build,
    slice_cfg as gca_slice,
)
from sass.cfg import build_cfgs, slice_cfg, to_dot
from sass.parser import parse_file

args = ArgumentParser()

args.add_argument("sassfile")
args.add_argument("--module")
args.add_argument("--keep_control_edges", action="store_true")
args.add_argument("--instr_match", default="WARPSYNC|USETMAXREG")
args.add_argument("--export_dot", default="store_true")
args.add_argument("--kernel", default=None)
args.add_argument("--gridDim", type=int, nargs=3, default=(1, 1, 1))
args.add_argument("--blockDim", type=int, nargs=3, default=(1, 1, 1))
args.add_argument("--regs_per_thread", type=int, required=True)
args.add_argument(
    "--backend",
    choices=["builtin", "gpucode"],
    default="builtin",
    help="Parser/slicer backend: 'builtin' (default) or 'gpucode' (gpucode-analyzer)",
)
params = args.parse_args()

if params.backend == "gpucode":
    gca_cfgs = gca_parse(params.sassfile)
    # kernel names come from the raw gpucode-analyzer CFGs
    kernel_names = list(gca_cfgs.keys())
else:
    prog = parse_file(params.sassfile)
    cfgs = build_cfgs(prog)
    kernel_names = list(cfgs.keys())

if params.kernel is None:
    print("Please select a kernel using the --kernel option:")
    for k in kernel_names:
        print(k)

    exit(0)

if params.kernel not in kernel_names:
    print("Kernel name not found. Valid kernels are: ")
    for k in kernel_names:
        print(k)

    exit(0)

cfg = cfgs[params.kernel]
sliced = slice_cfg(cfg, params.instr_match, keep_control=params.keep_control_edges)
for bb in sliced.blocks:
    print(f"  {bb}")
print("--- after collapse ---")
sliced = collapse_try_alloc_retry(sliced)
for bb in sliced.blocks:
    print(f"  {bb}")
# sliced = collapse_try_alloc_retry(sliced)

codegen = SassCFGCodegen()
module_name = params.module or re.sub(r"[^A-Za-z0-9]", "_", params.kernel)[:64]
proc = codegen.generate(
    sliced,
    module_name,
    params.regs_per_thread,
    gridDim=tuple(params.gridDim),
    blockDim=tuple(params.blockDim),
)

t = Parameter("t")
proc.createInvariant(
    "NoErrorState",
    ForAll(
        t,
        Domain(proc.getPcMap()),
        NotEqual(Index(proc.getPcMap(), t), Literal(proc.errorState)),
    ),
)

numRegThread = proc.getNumRegThread()

proc.createInvariant(
    "RegReqCheck",
    ForAll(
        t,
        Domain(numRegThread),
        And(
            LtE(numRegThread[t], Literal(constants.MAX_REG_REQ)),
            GtE(numRegThread[t], Literal(constants.MIN_REG_REQ)),
            Equal(
                Mod(numRegThread[t], Literal(constants.DIVISIBLE_REG_REQ)), Literal(0)
            ),
        ),
    ),
)

template = proc.template_thread

assert template is not None

for i, pc in enumerate(template.pc_inc):
    t = Parameter("t")
    proc.createProperty(
        f"UsetmaxregProgress_{i}",
        ForAll(
            t,
            Domain(proc.getPcMap()),
            LeadsTo(
                Equal(Index(proc.getPcMap(), t), Literal(pc)),
                NotEqual(Index(proc.getPcMap(), t), Literal(pc)),
            ),
        ),
    )

# Invariant: within a warp group, setmaxnreg is all-or-nothing.
# If any thread in the warp group is at a setmaxnreg PC, all must be.
for label, pcs_dict in [("inc", template.pc_inc), ("dec", template.pc_dec)]:
    for i, pc in enumerate(pcs_dict):
        t1 = Parameter("t1")
        t2 = Parameter("t2")
        proc.createInvariant(
            f"SetmaxnregUniform_{label}_{i}",
            ForAll(
                t1,
                Domain(proc.getPcMap()),
                ForAll(
                    t2,
                    Domain(proc.getPcMap()),
                    Implies(
                        Equal(
                            Index(proc.threadToWarpGroup, t1),
                            Index(proc.threadToWarpGroup, t2),
                        ),
                        Implies(
                            Equal(Index(proc.getPcMap(), t1), Literal(pc)),
                            Equal(Index(proc.getPcMap(), t2), Literal(pc)),
                        ),
                    ),
                ),
            ),
        )

# Invarient: any pc state that uses Reg Rn as a src a dest op => numRegThread[t] >= n+1
for instr, max_reg_idx in template.pc_max_reg.items():
    proc.createInvariant(
        f"RegInRange_{instr}",
        ForAll(
            t,
            Domain(proc.getPcMap()),
            Implies(
                Equal(Index(proc.getPcMap(), t), Literal(instr)),
                GtE(Index(proc.getNumRegThread, t), Literal(max_reg_idx + 1)),
            ),
        ),
    )
# Invariant: at a setmaxnreg PC, numRegThread[t] must cover the accessed register.
for pc_label, max_reg_idx in template.pc_inc.items():
    proc.createInvariant(
        f"IncLTCurr_{pc_label}",
        ForAll(
            t,
            Domain(proc.getPcMap()),
            Implies(
                Equal(Index(proc.getPcMap(), t), Literal(pc_label)),
                Lt(Index(proc.getNumRegThread(), t), Literal(max_reg_idx)),
            ),
        ),
    )

for pc_label, max_reg_idx in template.pc_dec.items():
    proc.createInvariant(
        f"DecGTCurr_{pc_label}",
        ForAll(
            t,
            Domain(proc.getPcMap()),
            Implies(
                Equal(Index(proc.getPcMap(), t), Literal(pc_label)),
                Gt(Index(proc.getNumRegThread(), t), Literal(max_reg_idx)),
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
#     block_reg_sum = Add(*[Index(proc.getNumRegThread(), t_val) for t_val in thread_names])
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

if params.export_dot and params.backend != "gpucode":
    with open(f"{module_name}.dot", "w") as f:
        f.write(to_dot(sliced, show_instructions=True))

for msg in codegen.log:
    print(msg, file=sys.stderr)
