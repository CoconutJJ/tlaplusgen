import re
import sys
from sass.cfg import build_cfgs, slice_cfg, to_dot
from sass.parser import parse_file
from tla_codegen import SassCFGCodegen
from argparse import ArgumentParser


args = ArgumentParser()

args.add_argument("sassfile")
args.add_argument("--module")
args.add_argument("--keep_control_edges", action="store_true")
args.add_argument("--instr_match", default="WARPSYNC")
args.add_argument("--kernel", default=None)

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

with open("cfg.dot", "w") as f:
    f.write(to_dot(sliced, show_instructions=True))

codegen = SassCFGCodegen()
module_name = params.module or re.sub(r"[^A-Za-z0-9]", "_", params.kernel)[:64]
proc = codegen.generate(sliced, name=module_name)

print(proc)
print(proc.getConfiguration())
for msg in codegen.log:
    print(msg, file=sys.stderr)
