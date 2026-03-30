#!/usr/bin/env python3
import re
import json
import argparse
import sys


class Instruction:
    label = None
    opcode = None

    def targets(self):
        if self.is_control():
            raise NotImplementedError

    def is_control(self):
        return isinstance(self, ControlInsn)

    def reads(self):
        raise NotImplementedError

    def writes(self):
        raise NotImplementedError

    def code(self):
        raise NotImplementedError


class ControlInsn(Instruction):
    indirect_targets = None

    def targets(self):
        """Return multiple target pcs"""
        raise NotImplementedError

    def target(self):
        """Returns pc of target"""
        raise NotImplementedError

    def is_conditional(self):
        raise NotImplementedError


class Operand:
    def __init__(self, operand, read=False, write=False, implicit=False):
        self.operand = operand
        self.read = read
        self.write = write
        self.implicit = implicit  # does not appear in the instruction

    def is_read(self):
        return self.read

    def is_write(self):
        return self.write

    def access(self):
        a = ""

        if self.is_read():
            a += "R"

        if self.is_write():
            a += "W"

        return a

    def is_implicit(self):
        return self.implicit


class Register:
    def __init__(self, n):
        self.n = n

    def __str__(self):
        return f"Register({self.n})"

    def is_constant(self):
        return False

    __repr__ = __str__


class Memory:
    pass


class BasicBlock:
    def __init__(self, name, code):
        self.name = name
        self.code = code
        self.successors = {}
        self.predecessors = {}

    def reverse(self):
        succ = {}
        pred = {}

        for s in self.successors.values():
            pred[s.name] = s

        for p in self.predecessors.values():
            succ[p.name] = p

        self.successors = succ
        self.predecessors = pred

    def add_successor(self, label, bb):
        # TODO: prevent duplicate adding?
        assert label not in self.successors, f"Duplicate successor label {label}"
        self.successors[label] = bb

    def add_predecessor(self, bb):
        self.predecessors[bb.name] = bb

    def remove_successor(self, label):
        self.successors[label].remove_predecessor(self)
        del self.successors[label]

    def remove_predecessor(self, bb):
        del self.predecessors[bb.name]

    def _mark_as_predecessor(self):
        # must be called once all successors have been added

        for s in self.successors:
            self.successors[s].add_predecessor(self)

    def __str__(self):
        return f"{self.name}: {self.successors}\n" + "\n".join(
            [f"  {c}" for c in self.code]
        )

    def __repr__(self):
        return f"BasicBlock({self.name}, ...)"

    def copy(self):
        o = BasicBlock(self.name, self.code)
        o.successors = dict(self.successors.items())
        o.predecessors = dict(self.predecessors.items())

        return o

    def target(self):
        if hasattr(self, "_target"):
            return self._target
        elif len(self.code) > 0:
            return self.code[0].label
        elif self.name in ("_start", "_exit"):
            return self.name
        else:
            raise ValueError


class CFG:
    def __init__(self, codefile, fn_name=None):
        self.codefile = codefile
        self.fn_name = fn_name
        self.blocks = [BasicBlock("_start", []), BasicBlock("_exit", [])]
        self.labels_to_blocks = {"_start": self.blocks[0], "_exit": self.blocks[1]}
        self.names_to_blocks = {
            "_start": self.labels_to_blocks["_start"],
            "_exit": self.labels_to_blocks["_exit"],
        }

    def build(self):
        def add_bb(bbcode, ndx, last_bb):
            bb = BasicBlock(f"BB{ndx}", bbcode)
            self.blocks.append(bb)

            self.names_to_blocks[bb.name] = bb
            self.labels_to_blocks[bb.code[0].label] = bb

            if last_bb:
                if len(last_bb.code) and last_bb.code[-1].is_control():
                    # true targets patched up later by other code
                    if last_bb.code[-1].is_conditional():
                        last_bb.add_successor("false", bb)
                else:
                    last_bb.add_successor("next", bb)

            return [], ndx + 1, bb

        starts = set()
        ends = set()
        next_is_start = False
        indirects = set()

        if self.fn_name:
            code = self.codefile.codes[self.fn_name]
        else:
            code = self.codefile.code

        for i in code:
            if next_is_start:
                starts.add(i.label)
                next_is_start = False

            if i.is_control():
                ends.add(i.label)
                for tgt in i.targets():
                    starts.add(tgt)

                if i.is_indirect():
                    indirects.add(i.label)

                next_is_start = True

        bbndx = 0
        last_bb = self.labels_to_blocks["_start"]
        current = []
        for i in code:
            if i.label in starts:
                if len(current):
                    current, bbndx, last_bb = add_bb(current, bbndx, last_bb)
            elif i.label in ends:
                current.append(i)
                current, bbndx, last_bb = add_bb(current, bbndx, last_bb)
                continue

            current.append(i)

        if len(current):
            current, bbndx, last_bb = add_bb(current, bbndx, last_bb)

        # fix up branch targets to bb; could be avoided
        for bb in self.blocks:
            # print(bb)
            if len(bb.code) == 0:
                continue
            last_insn = bb.code[-1]
            if last_insn.is_control():
                targets = last_insn.targets()
                succlabel = "true" if last_insn.is_conditional() else "next"
                for tgt, tgtndx in zip(
                    targets, [""] + [str(x) for x in range(1, len(targets))]
                ):
                    bb.add_successor(f"{succlabel}{tgtndx}", self.labels_to_blocks[tgt])

        # populate predecessors
        for bb in self.blocks:
            bb._mark_as_predecessor()

        if len(indirects):
            self.codefile.resolve_indirects(self, indirects)

        self.check_consistency()

    def update_indirects(self, indirects):
        for label, addr in indirects.items():
            for a in addr:
                if a not in self.labels_to_blocks:
                    print(
                        f"Address {a} not found as a basic block. NotYetImplemented, splitting blocks"
                    )
                    raise NotImplementedError

        changed = False
        for bb in self.blocks:
            if len(bb.code) == 0:
                continue
            last_insn = bb.code[-1]
            if last_insn.is_control() and last_insn.is_indirect():
                if last_insn.label in indirects:
                    cur_targets = set(last_insn.targets())
                    res_targets = set(indirects[last_insn.label])
                    new_targets = res_targets - cur_targets
                    if len(new_targets):
                        changed = True
                        k = len(cur_targets)
                        for r in new_targets:
                            bb.add_successor(f"indirect{k}", self.labels_to_blocks[r])
                            k = k + 1

                        bb._mark_as_predecessor()

                        if last_insn.indirect_targets is None:
                            last_insn.indirect_targets = new_targets
                        else:
                            last_insn.indirect_targets.extend(new_targets)

                        current_targets = set(last_insn.indirect_targets)
                        remove = []
                        for s in bb.successors:
                            if bb.successors[s].target() not in current_targets:
                                remove.append(s)

                        for rs in remove:
                            bb.remove_successor(rs)

    def check_consistency(self):
        for b in self.blocks:
            if (
                len(b.predecessors) == 0 and b.name != "_start" and b.target() != "0000"
            ):  # TODO: fix the 0000
                print(f"WARNING:generic_cfg:no predecessors: {b.target()}")

            if len(b.successors) == 0 and b.name != "_exit":
                print(f"WARNING:generic_cfg: no successors: {b.target()}")

    def dump(self):
        for b in self.blocks:
            print(b)

    def dump_code(self, output):
        for b in self.blocks:
            print(f">>> {b.target()}", file=output)
            for i in b.code:
                print(i.code(), file=output)

    def dump_dot(self, output, code=True, xinsn=lambda i: i.insn, count=False):
        print("digraph {", file=output)
        for b in self.blocks:
            if count:
                count_prefix = str(len(b.code)) + "\\n"
            else:
                count_prefix = ""

            bbcode = (
                f'"{count_prefix}'
                + b.target()
                + "\\n"
                + "\\n".join([str(xinsn(s)) for s in b.code])
                + '"'
            )
            if not code:
                bbcode = f'"{count_prefix}"'

            print(b.name + f" [label={bbcode},shape=rect];", file=output)
            print(
                "\n".join(
                    f'{b.name} -> {succ.name} [label="{lbl if lbl != "next" else ""}"];'
                    for lbl, succ in b.successors.items()
                ),
                file=output,
            )
        print("}", file=output)

    def copy(self):
        x = CFG(self.codefile, self.fn_name)
        x.blocks = list([b.copy() for b in self.blocks])

        l2b = {}
        l2b["_start"] = self.labels_to_blocks["_start"].copy()
        l2b["_exit"] = self.labels_to_blocks["_exit"].copy()

        l2b.update(dict((b.code[0].label, b) for b in x.blocks if len(b.code)))

        # note, doesn't deep copy instructions
        for l in self.labels_to_blocks:
            x.labels_to_blocks[l] = l2b[l]

        # note, doesn't deep copy instructions
        x.names_to_blocks = dict([(b.name, b) for b in x.blocks])

        return x

    def convert(self, converter, func_name=None):
        assert (func_name or self.fn_name) is not None, f"Needs a function name"
        converter.init_cfg(self, func_name or self.fn_name)
        block_order = converter.output_block_order()

        for b in block_order:
            converter.convert_block(self.labels_to_blocks[b])

        converter.finish_cfg()

    def reverse(self):
        for b in self.blocks:
            b.reverse()

    def all_instructions(self):
        for b in self.blocks:
            for i in b.code:
                yield i


class DefUseAnalysis:
    def __init__(self, cfg):
        self.cfg = cfg
        self.defns = {}

    def build_definitions(self):
        defs_r2n = {}
        defs_n2ri = {}
        defs_i2n = {}

        defno = 0
        for b in self.cfg.blocks:
            for i in b.code:
                for r in i.writes():
                    if isinstance(r, Register):
                        if r.n not in defs_r2n:
                            defs_r2n[r.n] = set()

                        if i.label not in defs_i2n:
                            defs_i2n[i.label] = set()

                        defs_n2ri[defno] = (r.n, i.label)
                        defs_r2n[r.n].add(defno)
                        defs_i2n[i.label].add(defno)

                        defno += 1

        self.defns_r2n = defs_r2n
        self.defns_n2ri = defs_n2ri
        self.defns_i2n = defs_i2n

    def reaching_defns(self, quiet=False):
        def get_predecessor_rd(n, b):
            if n == 0:
                # first instruction in block, look at predecessors of block
                out = set()
                for p in b.predecessors.values():
                    if not len(p.code):
                        continue
                    out = out.union(rd[p.code[-1].label])
                return out
            else:
                return rd[b.code[n - 1].label]

        kills = {}
        gens = {}

        rd = {}
        rd_in = {}
        for b in self.cfg.blocks:
            for i in b.code:
                k = set()
                g = self.defns_i2n.get(i.label, set())  # could be empty
                for r in i.writes():
                    if isinstance(r, Register):
                        all_defs = self.defns_r2n[r.n]
                        k = k.union(all_defs)
                        k = k - g

                if i.predicated():
                    kills[i.label] = set()
                else:
                    kills[i.label] = k
                gens[i.label] = g
                rd[i.label] = set()

        changed = True
        while changed:
            changed = False
            for b in self.cfg.blocks:
                for n, i in enumerate(b.code):
                    x = get_predecessor_rd(n, b)
                    if n == 0:
                        rd_in[i.label] = x
                    else:
                        rd_in[i.label] = rd[b.code[n - 1].label]

                    rdnew = gens[i.label].union(x - kills[i.label])
                    if rdnew != rd[i.label]:
                        rd[i.label] = rdnew
                        changed = True

        rdefs = {}
        for b in self.cfg.blocks:
            for i in b.code:
                reaching = rd_in[i.label]

                reads = set(
                    [
                        r.n
                        for r in i.reads()
                        if isinstance(r, Register) and not r.is_constant()
                    ]
                )

                rsub = [self.defns_n2ri[d] for d in reaching]
                rsub = [x for x in rsub if x[0] in reads]

                if not quiet:
                    if len(reads) != len(set([x[0] for x in rsub])):
                        print("*** MISSING DEFNS ***", i.label, reads, rsub)

                rdefs[i.label] = rsub

        self.rdefs = rdefs


class DFA:
    def __init__(self, cfg):
        self.cfg = cfg
        self.IN = {}
        self.OUT = {}

    def initialize(self, block):
        raise NotImplementedError

    def xfer(self, block, flowfact):
        raise NotImplementedError

    def merge(self, flowblocks):
        raise NotImplementedError

    def forward(self):
        blocks = []

        # TODO: change order to "optimal"
        for b in self.cfg.blocks:
            self.initialize(b)
            blocks.append(b)

        changed = True
        while changed:
            changed = False

            for b in blocks:
                n = b.name
                self.IN[n] = self.merge(b.predecessors)
                out = self.xfer(b, self.IN[n])
                changed = changed or (out != self.OUT[n])
                self.OUT[n] = out

    def backward(self):
        raise NotImplementedError


class Dominators(DFA):
    def initialize(self, block):
        self.IN[block.name] = set()
        self.OUT[block.name] = set(self.cfg.names_to_blocks.keys())

    def xfer(self, block, flowfacts):
        return flowfacts.union(set([block.name]))

    def merge(self, predecessors):
        out = None

        for p in predecessors:
            if out is None:
                out = set(self.OUT[p])
            else:
                out = out.intersection(self.OUT[p])

        return out if out is not None else set()

    def compute_dominators(self):
        self.forward()

    def compute_idom(self):
        self.IDOM = {}
        for b in self.cfg.blocks:
            bdom = self.DOM(b)
            max_ob = 0
            max_b = None

            for o in bdom:
                if o == b.name:
                    continue
                ob = self.DOM(self.cfg.names_to_blocks[o])
                if len(ob) > max_ob:
                    max_ob = len(ob)
                    max_b = o

            self.IDOM[b.name] = max_b

    def compute_dominance_frontiers(self):
        self.DF = dict([(b.name, set()) for b in self.cfg.blocks])

        for b in self.cfg.blocks:
            is_join = len(b.predecessors) > 1
            if not is_join:
                continue

            for p in b.predecessors:
                while p != self.IDOM[b.name]:
                    self.DF[p].add(b.name)
                    p = self.IDOM[p]

    def DOM(self, block, as_targets=False):
        dom = self.OUT[block.name]
        if as_targets:
            return set([self.cfg.names_to_blocks[d].target() for d in dom])
        else:
            return dom

    def reverse(self):
        self.cfg = self.cfg.copy()
        self.cfg.reverse()
        self.reversed = True


class Skeletonizer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.da = DefUseAnalysis(self.cfg)
        self.da.build_definitions()
        self.da.reaching_defns()

    def _mark_important(self, important):
        to_process = list(important)

        while len(to_process):
            new_to_process = []
            for il in to_process:
                rdefs = self.da.rdefs[il]
                for _, rdil in rdefs:
                    if rdil not in important:
                        new_to_process.append(rdil)
                        important.add(rdil)

            to_process = new_to_process

        return important

    def build_skeleton(self):
        important = set()

        for b in self.cfg.blocks:
            if len(b.code) == 0:
                continue
            last_insn = b.code[-1]
            if last_insn.is_control():
                if last_insn.label not in important:
                    important.add(last_insn.label)

        self.important = self._mark_important(important)

    def get_skeleton_cfg(self):
        ocfg = self.cfg.copy()
        for b in ocfg.blocks:
            b._target = b.target()
            b.code = [i for i in b.code if i.label in self.important]

        return ocfg


# assemblies extracted from nvuc files are "bare" with no function name and have one function.
# assemblies dumped from cuobjdump usually have function name information and are multiple functions.

SASS_INSN_RE = re.compile(r"^\s*/\*([0-9a-f]+)\*/\s+(.+) ;(\s*/\* 0x([0-9a-f]+) \*/)?$")
SASS_REG_RE = re.compile(
    r"-?(((UR|R|!?P|B|!?UP)\d+)(\.reuse|\.B[123]|\.H0_H0|\.H1_H1)?)|(UPR|UPT|PR|PT|-?RZ|URZ|SRZ|SR_CTAID\.?|SR_TID\.?)"
)
SASS_ADDR_RE = re.compile(
    r"\[(?P<reg1>R[0-9Z]+)(\.(?P<suff>U32|X16))?(\+(?P<reg2>UR[0-9Z]+)|(?P<imm>0x.+))?\]"
)

CX_RE = re.compile(r"-?cx\[(?P<regbase>.+)\]\[(?P<offset>.+)\]")
C_RE = re.compile(r"-?c\[(?P<bank>.+)\]\[(?P<regoffset>R.+)\]")

CONSTANT_REGS = set(
    [
        "RZ",
        "SRZ",
        "URZ",
        "PT",
        "UPT",
        "SR_TID.X",
        "SR_CTAID.X",
        "SR_TID.Y",
        "SR_TID.Z",
        "SR_CTAID.Y",
        "SR_CTAID.Z",
    ]
)

REG_NUMBER = re.compile(r"(?P<prefix>[^0-9]+)(?P<num>\d+|T)$")
PT_NUM = 7
PR_NUM = 8

FUNCTION_BEGIN_RE = re.compile(r"\s+Function : (.*)$")
FUNCTION_END_RE = re.compile(r"\s+\.\.\.\.\.\.\.\.\.\.$")


class SASSRegister(Register):
    def __init__(
        self, n, is_inverted=False, is_negated=False, is_reuse=False, suffix=None
    ):
        super().__init__(n)

        # todo: reuse, !, .cc
        self.is_inverted = is_inverted
        self.is_negated = is_negated
        self.is_reuse = is_reuse
        self.suffix = suffix

    def is_constant(self):
        return self.n in CONSTANT_REGS

    def is_uniform(self):
        return self.n[0] == "U"

    def is_predicate(self):
        return self.n[0] == "P" or self.n.startswith("UP")

    def is_barrier(self):
        return self.n[0] == "B"

    def is_regular(self):
        return self.n[0] == "R" or self.n.startswith("UR")

    def is_sr(self):
        return self.n.startswith("SR")

    def operand(self, reuse=True):
        n = self.n

        if reuse and self.is_reuse:
            n = n + ".reuse"

        if self.suffix:
            n = n + self.suffix

        if self.is_inverted:
            return "!" + n

        if self.is_negated:
            return "-" + n

        return n

    def number(self):
        if self.n == "PR":
            return PR_NUM
        elif self.n == "UPR":
            raise NotImplementedError

        m = REG_NUMBER.search(self.n)
        assert m is not None, self.n
        num = m.group("num")
        if num == "T":
            return PT_NUM
        else:
            return int(num)

    def adjacent(self, adj=1):
        if self.n == "RZ" or self.n == "URZ":
            return [self] * adj

        regno = re.compile(r"(?P<prefix>[^0-9]+)(?P<num>\d+)$")
        m = regno.search(self.n)
        assert m is not None, self.n
        pfx = m.group("prefix")
        r = int(m.group("num"))

        return [SASSRegister(f"{pfx}{n}") for n in range(r + 1, r + adj + 1)]


class SASSAddress(Memory):
    def __init__(self, addr, reg1, suff, reg2, imm):
        self.addr = addr
        self.reg1 = SASSRegister(reg1)
        self.suff = suff
        self.reg2 = SASSRegister(reg2) if reg2 else None
        self.imm = imm

    def __str__(self):
        return self.addr

    def registers(self):
        out = [self.reg1]
        if self.reg2:
            out.append(self.reg2)

        return out


class SASSOperand(Operand):
    pass


class SASSInstruction(Instruction):
    # has multiple explicit write registers
    WRITE_COUNT = {
        "BSYNC": 0,
        ("IADD3", 5): 2,
        ("IADD3", 6): 3,
        ("UIADD3", 5): 2,
        ("LOP3.LUT", 7): 2,
        ("UIADD3", 6): 3,
        "RET.REL.NODEC": 0,
        ("BRA.U", 2): 0,  # for the BRA.U UP1, 0x... form
        "BRX": 0,
        "ELECT": 2,
    }

    # writes to multiple registers implicitly
    MULTI_WRITER = {
        "LDG.E.128.STRONG.GPU": {0: 4},
        "LDG.E.128.CONSTANT": {0: 4},
        "LDG.E.LTC128B.CONSTANT": {0: 4},
        "ULDC.64": {0: 2},
        "LDC.64": {0: 2},
        "HMMA.16816.F32": {0: 4},
        "IMAD.WIDE.U32": {0: 2},
        "UIMAD.WIDE.U32": {0: 2},
        "UIMAD.WIDE": {0: 2},
        "IMAD.WIDE": {0: 2},
        "LDSM.16.MT88.4": {0: 4},
        "LDS.128": {0: 4},
        "LDS.64": {0: 2},
        "LDG.E.128": {0: 4},
        "LDCU.64": {0: 2},
        "LDCU.128": {0: 4},
        "FMUL2.FTZ.RZ": {0: 2},
        "LDTM.x32": {0: 32},
        "FFMA2.FTZ.RZ": {0: 2},
        "LDTM.16dp256bit.x16": {0: 64},
        "LDTM.16dp256bit.x4": {0: 16},
        "LDTM.x128": {0: 128},
        "LDTM.x4": {0: 4},
        "HGMMA.64x256x16.F32": {0: 128},
    }

    READ_WRITE = {"IMAD.HI.U32": {0}}

    def __init__(self, pc, pred, opcode, args, insn):
        self.label = pc
        self.predicate = pred
        self.opcode = opcode
        self.args = args
        self.insn = insn

        out = []
        for a in self.args:
            m = SASS_REG_RE.match(a)
            if m is not None:
                r = None
                is_inverted = False
                is_negated = False
                is_reuse = False

                if " " in a:
                    a = a.split()[0]
                    # a = a[:-len(" 0x0")] # RET.REL.NODEC R4 0x0 ; BRX, etc.

                if a[0] == "!":
                    a = a[1:]
                    is_inverted = True

                if a[0] == "-":
                    a = a[1:]
                    is_negated = True

                suffix_list = []
                reg_suffixes = [
                    ".reuse",
                    ".B1",
                    ".B2",
                    ".B3",
                    ".H0_H0",
                    ".H1_H1",
                    ".H1",
                    ".HI_LO",
                    ".F32",
                    ".F32x2",
                ]
                while True:
                    for rs in reg_suffixes:
                        if a.endswith(rs):
                            a = a[: -len(rs)]
                            suffix_list.append(rs)
                            if rs == ".reuse":
                                is_reuse = True
                            break
                    else:
                        break

                suffix_str = "".join(reversed(suffix_list))
                if len(suffix_str) == 0:
                    suffix_str = None

                r = SASSRegister(
                    a,
                    is_inverted=is_inverted,
                    is_negated=is_negated,
                    is_reuse=is_reuse,
                    suffix=suffix_str,
                )

                out.append(r)
            else:
                m = SASS_ADDR_RE.match(a)
                if m:
                    addr = SASSAddress(
                        a,
                        reg1=m.group("reg1"),
                        suff=m.group("suff"),
                        reg2=m.group("reg2"),
                        imm=m.group("imm"),
                    )
                    out.append(addr)
                else:
                    out.append(a)

        self.args = out

    def __str__(self):
        return f"{self.label}: {self.predicate if self.predicate else ''} {self.opcode} {self.args} {self.reads()} {self.writes()}"

    __repr__ = __str__

    def code(self):
        return f"/*{self.label}*/ {self.insn} ;"

    def predicated(self):
        return self.predicate is not None

    def write_count(self):
        if self.opcode in SASSInstruction.WRITE_COUNT:
            write_args = SASSInstruction.WRITE_COUNT[self.opcode]
        elif (self.opcode, len(self.args)) in SASSInstruction.WRITE_COUNT:
            write_args = SASSInstruction.WRITE_COUNT[(self.opcode, len(self.args))]
        else:
            write_args = 1

        return write_args

    def operands(self):
        write_args = self.write_count()  # explicit
        writes = self.args[:write_args]

        mw = self.MULTI_WRITER[self.opcode] if self.opcode in self.MULTI_WRITER else {}

        for k, a in enumerate(writes):
            read = self.opcode in self.READ_WRITE and k in self.READ_WRITE[self.opcode]

            yield SASSOperand(a, write=True, read=read)
            if k in mw:
                for r in a.adjacent(mw[k] - 1):
                    yield SASSOperand(r, read=read, write=True, implicit=True)

        reads = self.args[write_args:]
        for a in reads:
            yield SASSOperand(a, read=True)

    def _decode_predset_imm(self, regset, uniform=""):
        rs = int(regset, 16)
        assert rs < 256, rs

        out = []
        for i in range(8):
            if rs & 1:
                out.append(SASSRegister(f"{uniform}P{i}"))

            rs >>= 1
            if rs == 0:
                break

        return out

    def reads(self):
        write_args = self.write_count()
        rds = list(
            x
            for x in self.args[write_args:]
            if isinstance(x, Register) and (x.n not in {"PR", "UPR"})
        )
        if self.predicate:
            n = self.predicate
            if n[0] == "!":
                n = n[1:]

            rds.append(SASSRegister(n, is_inverted=self.predicate[0] == "!"))

        # TODO: multi-readers

        # hack, need to fix this.
        for x in self.args[write_args:]:
            if isinstance(x, str):
                cxm = CX_RE.match(x)
                if cxm:
                    rds.append(SASSRegister(cxm.group("regbase")))
                else:
                    cm = C_RE.match(x)
                    if cm:
                        rds.append(SASSRegister(cm.group("regoffset")))

        if self.opcode == "P2R":
            rds.extend(self._decode_predset_imm(self.args[-1]))
        elif self.opcode == "UP2UR":
            assert isinstance(self.args[1], Register) and self.args[1].n == "UPR"
            rds.extend(self._decode_predset_imm(self.args[-1], uniform="U"))

        for x in self.args[write_args:]:
            if isinstance(x, SASSAddress):
                rds.extend(x.registers())

        return rds

    def writes(self):
        def extend(r, n):
            if r.n[0] == "R":
                pfx = "R"
            elif r.n[0] == "U":
                pfx = "UR"

            rno = int(r.n[len(pfx) :])
            return [SASSRegister(f"{pfx}{d}") for d in range(rno, rno + n)]

        write_args = self.write_count()
        writes = list(
            x
            for x in self.args[:write_args]
            if isinstance(x, Register) and not x.is_constant()
        )

        if self.opcode in SASSInstruction.MULTI_WRITER:
            for arg, ext in SASSInstruction.MULTI_WRITER[self.opcode].items():
                writes[arg] = extend(writes[arg], ext)

            o = []
            for x in writes:
                if isinstance(x, list):
                    o.extend(x)
                else:
                    o.append(x)

            return o
        else:
            return writes

    @staticmethod
    def parse(insn):
        if insn[0] == "@":
            predicate, insn = insn.split(" ", 1)
            predicate = predicate[1:]
        else:
            predicate = None

        opcode_args = insn.split(" ", 1)
        if len(opcode_args) == 1:
            opcode = opcode_args[0]
            args = []
        else:
            opcode = opcode_args[0]
            args = opcode_args[1].split(", ")

        return (predicate, opcode, args)


class SASSControlInsn(SASSInstruction, ControlInsn):
    def targets(self):
        # TODO: at some point handle conditional branches as well
        if self.indirect_targets is None:
            return [self.target()]
        else:
            return self.indirect_targets

    def target(self):
        if self.opcode == "EXIT":
            return "_exit"
        elif self.opcode == "RET.REL.NODEC":
            if self.indirect_targets is None:
                return "_exit"  # for now
            else:
                raise ValueError  # must call targets()
        elif self.opcode == "BRX":
            if self.indirect_targets is None:
                # metadata must set this
                raise NotImplementedError
            else:
                raise ValueError  # must call targets()
        else:
            addr_arg = 0
            if self.opcode == "BRA.U":
                # predicated version available
                if isinstance(self.args[0], SASSRegister):
                    addr_arg = 1  # BRA.U !UP0, addr
            elif self.opcode == "BRA":
                if isinstance(self.args[0], SASSRegister):
                    # on CC 10.0: @!P1 BRA !P2, 0xe8a0
                    addr_arg = 1
            elif self.opcode == "BRA.DIV":
                assert isinstance(self.args[0], SASSRegister), self.args[0]
                addr_arg = 1

            assert addr_arg < len(self.args), f"{self.opcode}, {self.args}, {addr_arg}"
            assert isinstance(self.args[addr_arg], str), (
                f"Expecting address: {self.opcode} {self.args[addr_arg]} {addr_arg}"
            )
            if self.args[addr_arg].startswith("0x"):
                tgt = self.args[addr_arg][2:]
                if len(tgt) < 4:
                    tgt = "0" * (4 - len(tgt)) + tgt
                return tgt
            else:
                return self.args[0]

    def is_conditional(self):
        return (
            (self.predicate is not None)
            or (self.opcode == "BRA.U" and isinstance(self.args[0], Register))
            or (self.opcode == "BRA.DIV")
        )

    def is_indirect(self):
        return self.opcode == "RET.REL.NODEC" or self.opcode == "BRX"


class SASSIndirectResolver:
    def __init__(self, cfg, indirects):
        self.cfg = cfg
        self.indirects = indirects
        self.instructions = dict([(i.label, i) for i in self.cfg.all_instructions()])

    def resolve(self):
        self.da = DefUseAnalysis(self.cfg)
        self.da.build_definitions()
        self.da.reaching_defns(quiet=True)

        iaddr = {}
        for i in self.indirects:
            iaddr[i] = self.resolve_indirect(i)
            insn = self.instructions[i]

        return self.cfg.update_indirects(iaddr)

    def resolve_indirect(self, indirect):
        if self.instructions[indirect].opcode == "BRX":
            return self.instructions[indirect].indirect_targets

        chain = [self.instructions[indirect]]
        k = 0
        while k < len(chain):
            for r in self.da.rdefs[chain[k].label]:
                chain.append(self.instructions[r[1]])

            k = k + 1

        addresses = []
        for i in chain:
            assert i.opcode in {"RET.REL.NODEC", "MOV", "IMAD.MOV.U32"}, (
                f"{i.opcode} {chain}"
            )
            if (
                i.opcode == "MOV"
                and isinstance(i.args[1], str)
                and i.args[1].startswith("0x")
            ):
                addr = i.args[1][2:]
                if len(addr) < 4:
                    addr = "0" * (4 - len(addr)) + addr

                assert addr in self.instructions, (
                    f"{i} does not contain a valid address {addr}"
                )
                addresses.append(addr)
            elif i.opcode == "IMAD.MOV.U32":
                # this reads a register but the producer is next
                continue

        return addresses


class SASSFile:
    SASS_CONTROL_INSN = re.compile("EXIT|BRA|CALL.REL.NOINC|RET.REL.NODEC|BRX")

    def __init__(self, sass_string, metadata=None):
        self.code = []
        self.metadata = metadata
        self._parse(sass_string)

    def _parse(self, sass_string):
        state = "out"
        code = []
        codes = {}
        ff = None
        function = None
        for l in sass_string.splitlines():
            if state == "out":
                m = FUNCTION_BEGIN_RE.match(l)
                if m:
                    state = "in"
                    function = m.group(1)
                    ff = ff or function
                    continue

                insn = self._mkinsn(l, function)
                if insn is not None:
                    function = None
                    state = "in"
                    code.append(insn)
                    continue
            elif state == "in":
                insn = self._mkinsn(l, function)
                if insn is None:
                    m = FUNCTION_END_RE.match(l)
                    if m:
                        codes[function] = code
                        function = None
                        code = []
                        state = "out"
                else:
                    code.append(insn)

        if len(code):
            if len(codes) == 0:
                # old view
                self.code = code
            else:
                # missed ending?
                assert function is not None
                codes[function] = code
                function = None
                code = []

        if len(codes):
            self.codes = codes
            self.code = codes[ff]

        if len(self.code) == 0 or len(getattr(self, "codes", {})) == 0:
            print("WARNING:sass: No instructions matched in the provided SASS string")

    def _mkinsn(self, sassinsn, fn_name=None):
        m = SASS_INSN_RE.match(sassinsn)
        if m:
            pc = m.group(1)
            pred, o, a = SASSInstruction.parse(m.group(2))

            if SASSFile.SASS_CONTROL_INSN.match(o):
                i = SASSControlInsn(pc, pred, o, a, m.group(2))
            else:
                i = SASSInstruction(pc, pred, o, a, m.group(2))

            if i.is_control() and i.is_indirect() and i.opcode == "BRX":
                assert self.metadata is not None, (
                    "Code contains BRX and metadata about indirect branches must be provided"
                )
                assert fn_name is not None, (
                    f"Multiple functions present, but current function unknown"
                )
                brx = self.metadata[fn_name].get("EIATTR_INDIRECT_BRANCH_TARGETS", {})
                assert i.label in brx, f"No indirect targets for {i.label} found"
                i.indirect_targets = brx[i.label]

            return i

        return None

    def resolve_indirects(self, cfg, indirects):
        if len(indirects) == 0:
            return

        changed = True
        while changed:
            ir = SASSIndirectResolver(cfg, indirects)
            changed = ir.resolve()

    def dump(self):
        for i in self.code:
            print(i)


class Slicer(Skeletonizer):
    def slice(self, addresses):

        def _get_important_cdeps(block):
            rdf = self.dom.DF[block.name]
            out = set()
            for cdep in rdf:
                b = self.cfg.names_to_blocks[cdep]
                if len(b.code) and b.code[-1].is_control():
                    if b.code[-1].label not in self.important:
                        out.add(b.code[-1].label)

            return out

        self.dom = Dominators(self.cfg)
        self.dom.reverse()
        self.dom.compute_dominators()
        self.dom.compute_idom()
        self.dom.compute_dominance_frontiers()

        self.important = set(addresses)
        change = True
        while change:
            for b in self.cfg.blocks:
                for c in b.code:
                    if c.label in self.important:
                        new_addr = _get_important_cdeps(b)
                        addresses |= new_addr

            change = not (len(addresses) == 0)
            self.important |= self._mark_important(addresses)
            addresses = set()


def get_labels(cfg, labels_or_re):
    res = []
    labels = set()
    for lr in labels_or_re:
        if lr.startswith("re:"):
            rexp = re.compile(lr[3:])
            res.append(rexp)
        else:
            labels.add(lr)

    for i in cfg.all_instructions():
        if i.label in labels:
            continue
        if any(r.match(i.opcode) for r in res):
            labels.add(i.label)

    if len(labels) == 0:
        print("WARNING: no labels matched. Slice will be empty.")

    return labels


def slice_sass(
    sass_code: str, labels_or_re: list, fn_name: str = None, metadata: dict = None
) -> CFG:
    """
    Parses the provided SASS code, builds the CFG, slices it based on the
    specified labels or regular expressions, and returns the skeletonized CFG.
    """
    code_obj = SASSFile(sass_code, metadata=metadata)
    cfg = CFG(code_obj, fn_name)
    cfg.build()

    sk = Slicer(cfg)
    labels = get_labels(cfg, labels_or_re)
    sk.slice(labels)

    return sk.get_skeleton_cfg()
