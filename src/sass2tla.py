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
from tla_codegen import SassCFGCodegen, parse_sass_file, slice_cfg
from argparse import ArgumentParser

args = ArgumentParser()

args.add_argument("sassfile")
args.add_argument("--module")
args.add_argument("--keep_control_edges", action="store_true")
args.add_argument("--instr_match", default="BAR.SYNC|USETMAXREG|WARPSYNC")
args.add_argument("--export_dot", action="store_true")
args.add_argument(
    "--slice_only",
    action="store_true",
    help="Print the sliced CFG and exit without generating TLA+",
)
args.add_argument("--kernel", default=None)
args.add_argument("--grid_dim", type=int, nargs=3, default=(1, 1, 1))
args.add_argument("--block_dim", type=int, nargs=3, default=(1, 1, 1))
# args.add_argument("--regs_per_thread", type=int, required=True)
args.add_argument("--regs_per_thread", type=int, default=64)
params = args.parse_args()

cfgs = parse_sass_file(params.sassfile)
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

sliced = slice_cfg(cfgs[params.kernel], params.instr_match)

if params.slice_only:
    sliced.dump_code(sys.stdout)
    sys.exit(0)

codegen = SassCFGCodegen()
module_name = params.module or re.sub(r"[^A-Za-z0-9]", "_", params.kernel)[:64]
proc = codegen.generate(
    sliced,
    module_name,
    params.regs_per_thread,
    grid_dim=tuple(params.grid_dim),
    block_dim=tuple(params.block_dim),
)

template = proc.template_thread
assert template is not None

if not template.thread_definitions:
    print(
        f"[sass2tla] slice of {params.kernel!r} on {params.instr_match!r} "
        f"is empty -- no TLA+ module emitted",
        file=sys.stderr,
    )
    sys.exit(0)

t = Parameter("t")
proc.createInvariant(
    "NoErrorState",
    ForAll(
        t,
        Domain(proc.getPcMap()),
        NotEqual(Index(proc.getPcMap(), t), Literal(proc.error_state)),
    ),
)

num_reg_thread = proc.getNumRegThread()

proc.createInvariant(
    "RegReqCheck",
    ForAll(
        t,
        Domain(num_reg_thread),
        And(
            LtE(num_reg_thread[t], Literal(constants.MAX_REG_REQ)),
            GtE(num_reg_thread[t], Literal(constants.MIN_REG_REQ)),
            Equal(
                Mod(num_reg_thread[t], Literal(constants.DIVISIBLE_REG_REQ)), Literal(0)
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
            proc.getThreadNameSet(),
            LeadsTo(
                Equal(Index(proc.getPcMap(), t), Literal(pc)),
                NotEqual(Index(proc.getPcMap(), t), Literal(pc)),
            ),
        ),
    )

# Invariant: within a warp group, setmaxnreg is all-or-nothing.
# If any thread in the warp group is at a setmaxnreg PC, all must be.
for i, pc in enumerate(template.pc_inc):
    t1 = Parameter("t1")
    t2 = Parameter("t2")
    proc.createInvariant(
        f"SetmaxnregUniform_{i}",
        ForAll(
            t1,
            Domain(proc.getPcMap()),
            ForAll(
                t2,
                Domain(proc.getPcMap()),
                Implies(
                    Equal(
                        Index(proc.thread_to_warp_group, t1),
                        Index(proc.thread_to_warp_group, t2),
                    ),
                    Implies(
                        Equal(Index(proc.getPcMap(), t1), Literal(pc)),
                        Equal(Index(proc.getPcMap(), t2), Literal(pc)),
                    ),
                ),
            ),
        ),
    )

# Invarient: any pc state that uses Reg Rn as a src a dest op => num_reg_thread[t] >= n+1
for instr, max_reg_idx in template.pc_max_reg.items():
    proc.createInvariant(
        f"RegInRange_{instr}",
        ForAll(
            t,
            Domain(proc.getPcMap()),
            Implies(
                Equal(Index(proc.getPcMap(), t), Literal(instr)),
                GtE(Index(proc.getNumRegThread(), t), Literal(max_reg_idx + 1)),
            ),
        ),
    )
# Invariant: at a setmaxnreg PC, num_reg_thread[t] must cover the accessed register.
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

with open(f"{module_name}.tla", "w") as f:
    f.write(str(proc))

with open(f"{module_name}.cfg", "w") as f:
    f.write(proc.getConfiguration())

if params.export_dot:
    with open(f"{module_name}.dot", "w") as f:
        sliced.dump_dot(f)

for msg in codegen.log:
    print(msg, file=sys.stderr)
