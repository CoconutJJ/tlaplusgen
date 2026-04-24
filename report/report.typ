#set page("a4", margin: 0.75in)
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em)
#show heading.where(level: 1): set text(size: 12pt)
#show heading.where(level: 2): set text(size: 11pt)
#show raw: set text(font: "DejaVu Sans Mono", size: 9pt)

#align(center)[
  #smallcaps[CS 6969: Fast and Verified GPU -- Final Project Report]
  #v(0.3cm)
  #text(size: 16pt, weight: "bold")[
    Verification of GPU Dynamic Register Allocation Semantics
    by Lifting NVIDIA SASS to TLA+
  ]
  #v(0.2cm)
  #grid(columns: 2, gutter: 1.5cm)[
    David Yue\
    `david.yue@utah.edu`
  ][
    Leo Sciortino\
    `leo.sciortino@utah.edu`
  ]
  #v(0.4cm)
]

#show: rest => columns(2, rest)

= Introduction

Modern NVIDIA GPUs reach peak throughput by partitioning a CTA into
specialized warp groups: a small producer warp issues bulk asynchronous
loads through the Tensor Memory Accelerator, while one or more consumer
warps spend most of their cycles inside the tensor cores. Because the
producer needs only a handful of registers and the consumer is
register-hungry, NVIDIA exposes the `USETMAXREG` SASS instruction
(`setmaxnreg` in PTX) to redistribute the per-CTA register pool at run
time. A producer warp deallocates registers it will not use, returning
them to a CTA-wide pool, and a consumer warp then tries to claim those
registers for itself. After the call, the warp group is no longer
permitted to reference registers numbered above the new ceiling.

This redistribution is delicate. The argument must be a multiple of
eight and lie within architectural limits, the increase and decrease
forms must be issued by every thread of a warp group in lockstep, an
allocation request can stall indefinitely if the pool never has enough
free registers to satisfy it, and any subsequent instruction must
respect the new register ceiling. None of these conditions are checked
by the assembler or the hardware, and the public SASS documentation is
sparse, so a kernel that compiles cleanly can still deadlock or
silently violate the contract.

The goal of this project is to verify, at the SASS instruction level,
that real warp-specialized kernels use `USETMAXREG` correctly. We do
this by lifting compiled SASS into a TLA+ specification and running the
TLC model checker against a set of safety invariants and liveness
properties that capture the rules above.

= Background on TLA+

TLA+ is a specification language designed by Leslie Lamport for
describing concurrent and distributed systems above the level of code.
A specification consists of an `Init` predicate that constrains the
initial state, a `Next` predicate that relates each state to its
possible successors, and a set of temporal properties expressed with
the always (`[]`), eventually (`<>`), and leads-to (`~>`) operators.
The TLC model checker then enumerates the reachable state space and
reports any state that violates a stated invariant or any infinite
behavior that violates a stated liveness property. This style fits GPU
register allocation well, since the property we care about, "every
allocation request is eventually granted", is naturally a leads-to
formula, and "no thread ever references an out-of-range register" is
naturally an invariant over the state.

= Pipeline

Our tool, `tlagen`, is a Python pipeline that turns a `.sass` file from
`nvdisasm` into a `.tla` module and a matching `.cfg` configuration for
TLC. The pipeline has five stages. Cleaning strips the raw assembler
output to bare instructions. Parsing uses a `pyparsing` grammar
(adapted from Prof. Sreepathi Pai's `gpucode-analyzer`) to build a
typed AST that recognises general-purpose, uniform, and predicate
registers, special registers, constant banks, descriptors, and branch
targets. A control flow graph is then built per kernel, with
reaching-definition dataflow over basic blocks that handles `BRA`,
indirect `BRX`, `CALL`/`RET`, `EXIT`, and the structured `BSSY`/`BSYNC`
pair. Slicing performs a backward walk over data and control
dependencies starting from every `WARPSYNC` and `USETMAXREG`
instruction; everything that does not influence one of those
instructions is discarded. This last step is critical: real kernels
contain thousands of instructions and would blow up the TLA+ state
space, but the slice that actually controls register allocation is
typically only a few dozen instructions long. Finally, code generation
walks the sliced CFG and emits a `TLASassProcess` whose actions
correspond directly to SASS instruction semantics.

The generator currently models around 65 SASS mnemonic variants,
covering data movement (`MOV`, `S2R`, `CS2R`), integer arithmetic
(`IADD3`, `IMAD` and its many variants, `IABS`, `IMNMX`), address
computation (`LEA`, `LEA.HI.SX32`), shifts and funnel shifts
(`SHF.R.U32.HI`, `SHF.L.U32`), logical operations (`LOP3.LUT`,
`PLOP3.LUT`, `SEL`), integer predicate setting (`ISETP` in all
LT/GT/GE/NE/EQ/LE forms with AND/OR composition), special-register
reads (`P2R`), constant bank loads (`LDC`, `ULDC`, including 64-bit
variants), and the synchronisation and register-allocation primitives
themselves (`WARPSYNC`, `SYNCS.PHASECHK`, `USETMAXREG.TRY_ALLOC` and
`USETMAXREG.DEALLOC`). The few floating-point instructions we
encounter (`I2F`, `MUFU.RCP`, `F2I`) are modelled as opaque data
dependencies rather than with full semantics, since the value they
compute does not influence the register-allocation control flow we
care about. Instructions outside this set are logged as `UNSUPPORTED`
and skipped, which is safe for slicing-target instructions but worth
flagging.

= Encoding into TLA+

Each thread becomes a state variable in the generated module, with a
program counter `pcs[t]` ranging over labelled CFG positions and four
register maps `regs_regular`, `regs_predicate`, `regs_uniform`, and
`regs_uniform_pred` holding per-thread register state. A shared
`BlockToRegPoolCount` array tracks the free-register pool of each
CTA, and `numRegThread[t]` records the current per-thread register
ceiling. Every SASS instruction in the slice is emitted as a guarded
TLA+ action that fires only when `pcs[t]` matches its label, advances
the program counter, updates the registers it writes, and leaves
everything else `UNCHANGED`. Branches are split into two actions
predicated on the same condition; indirect branches become disjunctions
over the resolved target set. The `Next` predicate non-deterministically
selects a thread and lets one of its enabled actions fire, which gives
TLC the interleaving semantics of warp-level concurrency. Weak fairness
on `Next` rules out behaviours in which an enabled thread is starved
forever, which is what makes the liveness properties meaningful.

The two register-allocation instructions are translated with care.
`USETMAXREG.DEALLOC.CTAPOOL n` sets `numRegThread[t]` to `n` and adds
the difference back to the pool. `USETMAXREG.TRY_ALLOC.CTAPOOL UPx, n`
is only enabled when the CTA pool has enough free registers to cover
the increase from the current ceiling to `n`; if so, the pool is
debited and the ceiling raised. If the pool is short, the action is
disabled and the thread is stuck at that PC, which is precisely the
behaviour we want the model checker to expose as a liveness violation.

The whole pipeline is driven by `sass2tla.py`, which takes the SASS
file, the kernel name, the grid and block dimensions, and the maximum
register count per thread, and produces `Test.tla`, `Test.cfg`, and
optionally a Graphviz dump of the sliced CFG.

= Invariants and Properties

On top of the generated module, `sass2tla.py` synthesises a fixed
collection of invariants and temporal properties parameterised by the
labels discovered during code generation. The first three are global
sanity checks. `NoErrorState` says that no thread is ever in the
distinguished `error` PC, which is where we route impossible
transitions. `RegReqCheck` enforces that `numRegThread[t]` always
stays between 24 and 256 and is divisible by 8, mirroring the
architectural constraint on the `USETMAXREG` immediate. `RegInRange_*`
is generated once per PC label and asserts that whenever a thread is
about to execute an instruction whose largest source or destination
register is `Rk`, its current ceiling is at least $k+1$. This is the
key safety property that catches a thread reading a register it has
already given away.

Two more invariants govern the allocation instructions themselves.
`SetmaxnregUniform_*` asserts that within a warp group, if any thread
sits at a `USETMAXREG` PC then every thread of that warp group sits at
the same PC, capturing the SIMT requirement that the call is
all-or-nothing. `IncLTCurr_*` and `DecGTCurr_*` together encode that
an increase is only valid when the current ceiling is below the new
target, and a decrease only when the current ceiling is above it.

The progress condition `UsetmaxregProgress_i` is a leads-to formula:
once a thread is at an allocation PC, it must eventually leave that
PC. Combined with the disabled-action semantics of `TRY_ALLOC`, this
property is what catches the case where a producer warp never releases
enough registers and a consumer warp is left waiting indefinitely. We
prove this property under weak fairness of `Next`, which formalises
the assumption that the GPU scheduler will eventually give every
ready thread a chance to run.

A useful side effect of generating these properties from the discovered
PC labels is that we never have to write per-kernel invariants by
hand. The same `sass2tla.py` invocation produces a self-checking model
for any kernel the parser can ingest.

= Workflow and Results

The end-to-end workflow is exercised by `run_examples.sh`, which walks
every `.sass` file in the `regalloc-dataset` corpus, lists the
contained kernels, runs the codegen for each one, and then invokes TLC
on the produced `.tla`/`.cfg` pair with a configurable timeout. We
ship `tla2tools_i64.jar`, a 64-bit-integer build of the standard TLC
distribution, because GPU address arithmetic regularly exceeds the
default 32-bit integer range and the unmodified jar reports overflow
on otherwise valid models. The driver classifies each run as a clean
pass, a property violation, an integer overflow, a TLC timeout, or a
codegen failure, which gives us a coarse but useful view of how the
tool fares on real kernels.

The corpus contains hand-written and CUTLASS-generated kernels that
exercise warp specialization in different ways: a simple reduction,
fused softmax-linear, a warp-specialized GEMM, persistent GEMM in
both straight-line and graph forms, and a flash-attention kernel. On
the smaller kernels the slice is small enough that TLC explores the
entire state space in seconds and reports no violations, which is the
expected outcome for code that the CUDA team has presumably already
audited. On the larger kernels we see two recurring failure modes:
TLC times out before exhausting the state space, and the slice
contains an instruction whose semantics we have not yet modeled, in
which case the corresponding action becomes a no-op that may suppress
real violations. Neither failure mode indicates a bug in the kernel,
but both limit the strength of the result we can claim.

= Limitations and Future Work

The most obvious limitation is the SASS coverage gap. NVIDIA does not
publish a SASS reference, and most of what is publicly known has been
reverse engineered. We derived the semantics of around sixty-five
mnemonics from a mix of the NVBit reference C in `sass_insns.h`,
register traces collected with NVBit, and Prof. Pai's notes; the
remainder are skipped. Whenever a skipped instruction appears between
a `USETMAXREG` and the next reference to a register the call
restricts, our model is unsound for that path. Closing this gap is
mostly a matter of adding handler entries, but it requires patient
trace-driven derivation for the long tail of variants.

A related limitation is that the parser and slicer are not fully
hardened. SASS dump syntax has no published grammar, and operands
appear in mixed case with a variety of decorators that vary between
architectures and even between disassembler versions. We currently
target Hopper (`sm_90a`) and Blackwell (`sm_100a`) and have run into
edge cases on each. Some of this will be absorbed by switching parts
of the front end to the `gpucode-analyzer` parser, which the codegen
already consumes directly, but the surface remains brittle.

There are also natural extensions on the verification side. A PTX
front end would let us reason at a level the compiler also targets,
side-stepping some SASS coverage gaps and giving access to a richer
type structure. Mutating the input kernel by deleting or perturbing a
`USETMAXREG` is an obvious way to validate that our invariants
actually catch the violations they are meant to catch. And going the
other direction, we would like to use the model as an oracle in an
inference loop: given an unannotated kernel, propose `USETMAXREG`
placements and use TLC to certify that the resulting kernel still
satisfies every invariant. Beyond the class deadline, we plan to keep
working on the project with the goal of submitting to POPL '27.

= Conclusion

This project demonstrates that a relatively small TLA+ encoding,
combined with backward slicing on the SASS CFG, is enough to
mechanically check the safety and liveness rules surrounding NVIDIA's
dynamic register allocation instructions. The infrastructure is
generic over kernels, the invariants are generated automatically from
the slice, and the same driver can be pointed at any kernel in the
`regalloc-dataset` corpus. The work that remains is mostly breadth
(more SASS semantics, a more robust parser) rather than depth, which
makes us optimistic that the tool will scale to the kernels we want to
verify in earnest.

= Division of Work

Parser and slicer integration was led by David, building on the
`gpucode-analyzer` project from Prof. Pai. The TLA+ code generator and
its Python AST were primarily David's work, with feature additions
from Leo as new instruction patterns appeared. Instruction semantics
came mostly from Prof. Pai, with Leo deriving several from NVBit
register traces. The invariants and properties were primarily Leo's
work. In practice, most of the project was done together and the line
between contributions is soft.

= Acknowledgements

We thank Prof. Sreepathi Pai for his advice throughout the semester
and for providing the SASS parser, slicer, and NVBit trace tooling
underlying our pipeline (`github.com/pyxis-roc/gpucode-analyzer`). We
also thank Amir for his guidance on the project.
