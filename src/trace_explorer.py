#!/usr/bin/env python3
"""
trace_explorer.py  --  Interactive NVBit trace explorer.

Parses the output of NVBit's record_reg_vals_thread tool, replays per-warp
register state across the instruction stream, and offers a REPL for guessing
the semantics of each instruction.

Usage:
    python trace_explorer.py path/to/trace.txt

The REPL at each instruction shows every captured trace slot alongside the
pre-execution ("before") value of the register the slot refers to.  Type a
Python expression using the captured slots to test a hypothesis:

    > reg1 * ureg0 + reg2            # per-thread arithmetic
    > c32                            # uniform constant-bank read
    > (reg1 >> 1) & 0x7fffffff

Scalar (uniform) and constant values are broadcast across the 32 lanes so
that expressions can freely mix per-thread and per-warp values.  The result
is compared against the selected target (default: reg0, i.e. the first
destination slot) lane-by-lane.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

WARP_SIZE = 32
U32_MASK = 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_KERNEL_RE = re.compile(
    r"^Kernel\s+(\d+):(.+?)\s+-\s+grid size\s+([\d,]+)\s+-\s+block size\s+([\d,]+)"
)
_INSN_HEADER_RE = re.compile(
    r"^CTA\s+(\d+,\d+,\d+)\s+-\s+warp\s+(\d+)\s+-\s+(\d+)\s+-\s+(.+?)\s*:\s*$"
)
_WIDTH_RE = re.compile(r"^\*\s*Width:\s*(\d+)")
_REG_TOKEN_RE = re.compile(r"Reg(\d+)_T(\d+):\s*(0x[0-9a-fA-F]+)")
_UREG_LINE_RE = re.compile(r"^\*\s*UReg(\d+):\s*(0x[0-9a-fA-F]+)")
_C_LINE_RE = re.compile(r"^\*\s*(C\d+):\s*(0x[0-9a-fA-F]+)")
_PRED_LINE_RE = re.compile(r"^\*\s*Pred:\s*(0x[0-9a-fA-F]+)\s+(0x[0-9a-fA-F]+)")


@dataclass
class TraceRecord:
    kernel_idx: int
    cta: tuple[int, int, int]
    warp: int
    seq: int
    disasm: str
    width: int = 1
    reg_values: dict[int, list[int]] = field(default_factory=dict)
    ureg_values: dict[int, int] = field(default_factory=dict)
    c_values: dict[str, int] = field(default_factory=dict)
    pred_values: list[int] = field(default_factory=list)


@dataclass
class Kernel:
    idx: int
    name: str
    grid: tuple
    block: tuple
    records: list[TraceRecord] = field(default_factory=list)


def parse_trace(path: str) -> list[Kernel]:
    kernels: list[Kernel] = []
    current_kernel: Optional[Kernel] = None
    current_record: Optional[TraceRecord] = None

    def commit():
        nonlocal current_record
        if current_record is not None and current_kernel is not None:
            current_kernel.records.append(current_record)
        current_record = None

    with open(path, "r", errors="replace") as f:
        for raw in f:
            line = raw.rstrip()
            if not line.strip():
                continue

            m = _KERNEL_RE.match(line)
            if m:
                commit()
                current_kernel = Kernel(
                    idx=int(m.group(1)),
                    name=m.group(2).strip(),
                    grid=tuple(int(x) for x in m.group(3).split(",")),
                    block=tuple(int(x) for x in m.group(4).split(",")),
                )
                kernels.append(current_kernel)
                continue

            m = _INSN_HEADER_RE.match(line)
            if m and current_kernel is not None:
                commit()
                disasm = m.group(4).strip().rstrip(";").strip()
                current_record = TraceRecord(
                    kernel_idx=current_kernel.idx,
                    cta=tuple(int(x) for x in m.group(1).split(",")),
                    warp=int(m.group(2)),
                    seq=int(m.group(3)),
                    disasm=disasm,
                )
                continue

            if current_record is None:
                continue

            m = _WIDTH_RE.match(line)
            if m:
                current_record.width = int(m.group(1))
                continue

            if "Reg" in line and "_T" in line:
                slots_seen: dict[int, dict[int, int]] = {}
                for tm in _REG_TOKEN_RE.finditer(line):
                    s = int(tm.group(1))
                    t = int(tm.group(2))
                    v = int(tm.group(3), 16)
                    slots_seen.setdefault(s, {})[t] = v
                for slot, per_thread in slots_seen.items():
                    lst = [per_thread.get(t, 0) for t in range(WARP_SIZE)]
                    current_record.reg_values[slot] = lst
                continue

            m = _UREG_LINE_RE.match(line)
            if m:
                current_record.ureg_values[int(m.group(1))] = int(m.group(2), 16)
                continue

            m = _C_LINE_RE.match(line)
            if m:
                current_record.c_values[m.group(1)] = int(m.group(2), 16)
                continue

            m = _PRED_LINE_RE.match(line)
            if m:
                current_record.pred_values = [
                    int(m.group(1), 16),
                    int(m.group(2), 16),
                ]
                continue

    commit()
    return kernels


# ---------------------------------------------------------------------------
# Disassembly → operand list (best-effort, no strict SASS grammar)
# ---------------------------------------------------------------------------


@dataclass
class Operand:
    raw: str
    kind: str      # reg, ureg, pred, spreg, imm, cbank, desc, mem, other
    name: str
    mods: tuple[str, ...] = ()
    width: int = 1      # 1 for plain R, 2 for .64 / .128 etc. when we can tell


_REG_RE = re.compile(r"^(-?)(R)(Z|\d+)(\..+)?$")
_UREG_RE = re.compile(r"^(-?)(UR)(Z|\d+)(\..+)?$")
_PRED_RE = re.compile(r"^(!?)(U?P)(T|\d+)$")
_IMM_RE = re.compile(r"^-?(0x[0-9a-fA-F]+|\d+(?:\.\d*)?|\.\d+)$")


def parse_operands(disasm: str) -> tuple[str, list[Operand], Optional[str]]:
    """Return (mnemonic, operands, predicate_guard_name)."""
    body = disasm.strip()
    pred = None
    pm = re.match(r"^@(!?)(U?P(?:T|\d+))\s+", body)
    if pm:
        pred = pm.group(1) + pm.group(2)
        body = body[pm.end():]

    head = body.split(None, 1)
    mnemonic = head[0]
    rest = head[1] if len(head) > 1 else ""

    pieces: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in rest:
        if ch in "[(":
            depth += 1
            buf.append(ch)
        elif ch in "])":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            pieces.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf and "".join(buf).strip():
        pieces.append("".join(buf).strip())

    operands: list[Operand] = []
    for raw in pieces:
        operands.append(_classify(raw))
    return mnemonic, operands, pred


def _classify(raw: str) -> Operand:
    s = raw.strip().rstrip(";").strip()

    m = _PRED_RE.match(s)
    if m:
        return Operand(raw=s, kind="pred", name=m.group(2) + m.group(3))

    m = _REG_RE.match(s)
    if m:
        name = "R" + ("Z" if m.group(3) == "Z" else m.group(3))
        mods = tuple(m.group(4).lstrip(".").split(".")) if m.group(4) else ()
        width = 1
        for mod in mods:
            if mod == "64":
                width = 2
            elif mod == "128":
                width = 4
            elif mod == "256":
                width = 8
        return Operand(raw=s, kind="reg", name=name, mods=mods, width=width)

    m = _UREG_RE.match(s)
    if m:
        name = "UR" + ("Z" if m.group(3) == "Z" else m.group(3))
        mods = tuple(m.group(4).lstrip(".").split(".")) if m.group(4) else ()
        return Operand(raw=s, kind="ureg", name=name, mods=mods)

    if s.startswith("SR_") or s in ("SRZ",):
        return Operand(raw=s, kind="spreg", name=s)

    if _IMM_RE.match(s):
        return Operand(raw=s, kind="imm", name=s)

    if s.startswith("c[") or s.startswith("cx["):
        return Operand(raw=s, kind="cbank", name=s)

    if s.startswith("desc["):
        return Operand(raw=s, kind="desc", name=s)

    if s.startswith("["):
        return Operand(raw=s, kind="mem", name=s)

    return Operand(raw=s, kind="other", name=s)


# ---------------------------------------------------------------------------
# Destination inference
# ---------------------------------------------------------------------------

_STORE_STEMS = {"STG", "STS", "STL", "RED", "ATOM", "ATOMS", "ATOMG", "STSM", "STGSTS"}
_NO_REG_DST_STEMS = {"BAR", "WARPSYNC", "BRA", "BRX", "EXIT", "RET", "NOP", "DEPBAR", "SYNC"}


@dataclass
class SlotMap:
    """Tells us what each Reg slot corresponds to.

    role:        "dst"  or "src"
    reg_name:    e.g. "R2"
    word_offset: 0 for low 32 bits, 1 for high 32 bits, ...
    """
    role: str
    reg_name: str
    word_offset: int


def infer_slot_map(mnem: str, operands: list[Operand], width: int) -> list[SlotMap]:
    """Best-effort mapping from reg-slot index → (role, reg, word)."""
    stem = mnem.split(".")[0]
    mapping: list[SlotMap] = []

    dst_count = 0
    if stem in _STORE_STEMS or stem in _NO_REG_DST_STEMS:
        dst_count = 0
    elif operands and operands[0].kind == "reg":
        dst_count = width

    if dst_count > 0:
        base = operands[0].name
        base_num = 0 if base == "RZ" else int(base[1:])
        for i in range(dst_count):
            mapping.append(SlotMap("dst", f"R{base_num + i}", i))
        src_iter = operands[1:]
    else:
        src_iter = operands

    for op in src_iter:
        if op.kind == "reg":
            if op.name == "RZ":
                continue
            base_num = int(op.name[1:])
            for i in range(op.width):
                mapping.append(SlotMap("src", f"R{base_num + i}", i))

    return mapping


# ---------------------------------------------------------------------------
# State tracking
# ---------------------------------------------------------------------------


class WarpState:
    def __init__(self):
        self.regs: dict[str, list[Optional[int]]] = {}
        self.uregs: dict[str, Optional[int]] = {}
        self.preds: dict[str, Optional[int]] = {}

    @staticmethod
    def _copy_reg(lst: list[Optional[int]]) -> list[Optional[int]]:
        return list(lst)

    def get_reg(self, name: str) -> list[Optional[int]]:
        if name == "RZ":
            return [0] * WARP_SIZE
        return list(self.regs.get(name, [None] * WARP_SIZE))

    def get_ureg(self, name: str) -> Optional[int]:
        if name == "URZ":
            return 0
        return self.uregs.get(name)

    def get_pred(self, name: str) -> Optional[int]:
        if name in ("PT", "UPT"):
            return U32_MASK
        return self.preds.get(name)

    def set_reg(self, name: str, values: list[int]):
        if name == "RZ":
            return
        self.regs[name] = list(values)

    def set_ureg(self, name: str, value: int):
        if name == "URZ":
            return
        self.uregs[name] = value

    def set_pred(self, name: str, value: int):
        if name in ("PT", "UPT"):
            return
        self.preds[name] = value


# ---------------------------------------------------------------------------
# Replay: produce per-record "before" snapshots keyed by warp
# ---------------------------------------------------------------------------


@dataclass
class ReplayedRecord:
    record: TraceRecord
    mnemonic: str
    operands: list[Operand]
    predicate: Optional[str]
    slots: list[SlotMap]
    before_regs: dict[str, list[Optional[int]]]
    before_uregs: dict[str, Optional[int]]
    before_preds: dict[str, Optional[int]]


def _warp_key(rec: TraceRecord) -> tuple[tuple[int, int, int], int]:
    return (rec.cta, rec.warp)


def replay_kernel(kernel: Kernel) -> list[ReplayedRecord]:
    state_by_warp: dict[tuple[tuple[int, int, int], int], WarpState] = {}
    out: list[ReplayedRecord] = []

    for rec in kernel.records:
        key = _warp_key(rec)
        state = state_by_warp.setdefault(key, WarpState())

        mnemonic, operands, predicate = parse_operands(rec.disasm)
        slots = infer_slot_map(mnemonic, operands, rec.width)

        # Snapshot BEFORE values for anything touched by this insn.
        touched_regs: set[str] = set()
        touched_uregs: set[str] = set()
        touched_preds: set[str] = set()
        for sm in slots:
            touched_regs.add(sm.reg_name)
        for op in operands:
            if op.kind == "reg":
                touched_regs.add(op.name)
            elif op.kind == "ureg":
                touched_uregs.add(op.name)
            elif op.kind == "pred":
                touched_preds.add(op.name)
        if predicate is not None:
            p = predicate.lstrip("!")
            if p.startswith("U"):
                touched_preds.add(p)
            else:
                touched_preds.add(p)

        before_regs = {n: state.get_reg(n) for n in touched_regs}
        before_uregs = {n: state.get_ureg(n) for n in touched_uregs}
        before_preds = {n: state.get_pred(n) for n in touched_preds}

        # Apply writes from trace to state.
        for slot_idx, sm in enumerate(slots):
            if sm.role != "dst":
                continue
            vals = rec.reg_values.get(slot_idx)
            if vals is not None:
                state.set_reg(sm.reg_name, vals)

        # Apply ureg writes.  For pure uniform ops we write the first UReg slot
        # back to the destination operand if it's a ureg.
        if operands and operands[0].kind == "ureg" and rec.ureg_values:
            base = operands[0].name
            if rec.width >= 1 and 0 in rec.ureg_values:
                if base != "URZ":
                    state.set_ureg(base, rec.ureg_values[0])
            if rec.width >= 2:
                base_num = 0 if base == "URZ" else int(base[2:])
                for i in range(rec.width):
                    v = rec.ureg_values.get(i)
                    if v is not None and base != "URZ":
                        state.set_ureg(f"UR{base_num + i}", v)

        # Apply predicate writes (first operand is a predicate dst).
        if operands and operands[0].kind == "pred" and rec.pred_values:
            state.set_pred(operands[0].name, rec.pred_values[0])
            if len(operands) > 1 and operands[1].kind == "pred" and len(rec.pred_values) > 1:
                state.set_pred(operands[1].name, rec.pred_values[1])

        out.append(
            ReplayedRecord(
                record=rec,
                mnemonic=mnemonic,
                operands=operands,
                predicate=predicate,
                slots=slots,
                before_regs=before_regs,
                before_uregs=before_uregs,
                before_preds=before_preds,
            )
        )

    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt_lane_list(vals: list[Optional[int]], width_hex: int = 8) -> str:
    if not vals:
        return "(empty)"
    uniq = {v for v in vals if v is not None}
    if None in vals and len(uniq) == 0:
        return "<unknown — never written>"
    if len(uniq) == 1:
        only = next(iter(uniq))
        return f"{'?' if None in vals else ''}uniform 0x{only:0{width_hex}x}"
    # show as range of per-thread values
    head = ", ".join(
        "----" if v is None else f"0x{v:0{width_hex}x}" for v in vals[:4]
    )
    tail = ", ".join(
        "----" if v is None else f"0x{v:0{width_hex}x}" for v in vals[-2:]
    )
    return f"[T0..T3: {head}  …  T30..T31: {tail}]"


def _fmt_scalar(v: Optional[int]) -> str:
    if v is None:
        return "<unknown>"
    return f"0x{v:08x}"


def _show_record(rr: ReplayedRecord, idx: int, total: int) -> None:
    rec = rr.record
    print()
    print(f"── [{idx+1}/{total}] kernel {rec.kernel_idx} · CTA {rec.cta} · "
          f"warp {rec.warp} · seq {rec.seq} ──")
    guard = f"@{rr.predicate} " if rr.predicate else ""
    print(f"    {guard}{rec.disasm}")
    print(f"    width: {rec.width}")

    # Registers
    if rr.slots:
        print("    operand regs:")
        for i, sm in enumerate(rr.slots):
            before = rr.before_regs.get(sm.reg_name)
            after = rec.reg_values.get(i)
            before_s = _fmt_lane_list(before) if before else "<unknown>"
            after_s = _fmt_lane_list(after) if after else "<unknown>"
            tag = "DST" if sm.role == "dst" else "src"
            print(f"      reg{i} [{tag} {sm.reg_name}]  before: {before_s}")
            print(f"                             after : {after_s}")

    # Uniform regs
    if rec.ureg_values:
        for slot, v in sorted(rec.ureg_values.items()):
            # Source ureg read comes from operand list
            ops_u = [op for op in rr.operands if op.kind == "ureg"]
            name = ops_u[slot].name if slot < len(ops_u) else f"UReg{slot}"
            before = rr.before_uregs.get(name)
            print(f"      ureg{slot} [{name}]  before: {_fmt_scalar(before)}  "
                  f"after: {_fmt_scalar(v)}")

    # Constant bank reads (always sources — no "before")
    for k, v in rec.c_values.items():
        w = 16 if k == "C64" else 8
        print(f"      {k}: 0x{v:0{w}x}")

    # Predicate writes
    if rec.pred_values:
        ops_p = [op for op in rr.operands if op.kind == "pred"]
        for i, v in enumerate(rec.pred_values):
            name = ops_p[i].name if i < len(ops_p) else f"Pred{i}"
            before = rr.before_preds.get(name)
            print(f"      pred{i} [{name}]  before: {_fmt_scalar(before)}  "
                  f"after: {_fmt_scalar(v)}")


# ---------------------------------------------------------------------------
# Guess evaluation
# ---------------------------------------------------------------------------


class Lane:
    """Wraps a 32-lane value list so arithmetic broadcasts scalars."""

    __slots__ = ("vals",)

    def __init__(self, vals):
        if isinstance(vals, Lane):
            self.vals = vals.vals
        elif isinstance(vals, (list, tuple)):
            assert len(vals) == WARP_SIZE
            self.vals = [int(v) & U32_MASK if v is not None else None for v in vals]
        else:
            # scalar
            scalar = int(vals) & U32_MASK if vals is not None else None
            self.vals = [scalar] * WARP_SIZE

    def _bop(self, other, fn):
        other = other if isinstance(other, Lane) else Lane(other)
        out: list[Optional[int]] = []
        for a, b in zip(self.vals, other.vals):
            if a is None or b is None:
                out.append(None)
            else:
                out.append(fn(a, b) & U32_MASK)
        return Lane(out)

    def _cmp(self, other, fn):
        """Comparison → Lane of 0/1 per lane (None preserved)."""
        other = other if isinstance(other, Lane) else Lane(other)
        out: list[Optional[int]] = []
        for a, b in zip(self.vals, other.vals):
            if a is None or b is None:
                out.append(None)
            else:
                out.append(1 if fn(a, b) else 0)
        return Lane(out)

    def _uop(self, fn):
        return Lane([None if v is None else fn(v) & U32_MASK for v in self.vals])

    def __add__(self, o): return self._bop(o, lambda a, b: a + b)
    def __radd__(self, o): return Lane(o)._bop(self, lambda a, b: a + b)
    def __sub__(self, o): return self._bop(o, lambda a, b: a - b)
    def __rsub__(self, o): return Lane(o)._bop(self, lambda a, b: a - b)
    def __mul__(self, o): return self._bop(o, lambda a, b: a * b)
    def __rmul__(self, o): return Lane(o)._bop(self, lambda a, b: a * b)
    def __floordiv__(self, o): return self._bop(o, lambda a, b: a // b if b else 0)
    def __mod__(self, o): return self._bop(o, lambda a, b: a % b if b else 0)
    def __and__(self, o): return self._bop(o, lambda a, b: a & b)
    def __rand__(self, o): return Lane(o)._bop(self, lambda a, b: a & b)
    def __or__(self, o): return self._bop(o, lambda a, b: a | b)
    def __ror__(self, o): return Lane(o)._bop(self, lambda a, b: a | b)
    def __xor__(self, o): return self._bop(o, lambda a, b: a ^ b)
    def __rxor__(self, o): return Lane(o)._bop(self, lambda a, b: a ^ b)
    def __lshift__(self, o): return self._bop(o, lambda a, b: a << b)
    def __rshift__(self, o): return self._bop(o, lambda a, b: a >> b)
    def __neg__(self): return self._uop(lambda v: -v)
    def __invert__(self): return self._uop(lambda v: ~v)

    # Comparisons return per-lane 0/1 Lanes (usable with `where`, `&`, `|`).
    def __lt__(self, o): return self._cmp(o, lambda a, b: a < b)
    def __le__(self, o): return self._cmp(o, lambda a, b: a <= b)
    def __eq__(self, o): return self._cmp(o, lambda a, b: a == b)
    def __ne__(self, o): return self._cmp(o, lambda a, b: a != b)
    def __gt__(self, o): return self._cmp(o, lambda a, b: a > b)
    def __ge__(self, o): return self._cmp(o, lambda a, b: a >= b)

    # Default __hash__ disappears once __eq__ is overridden; Lanes aren't used
    # as dict keys, but we still want the object to be usable in generic
    # containers, so keep identity-based hashing.
    __hash__ = object.__hash__

    def __repr__(self):
        return _fmt_lane_list(self.vals)

    def to_list(self):
        return list(self.vals)


def _sext32(v: int) -> int:
    v &= U32_MASK
    return v - (1 << 32) if v & (1 << 31) else v


def _build_eval_ns(rr: ReplayedRecord) -> dict:
    """Build the variable namespace used when evaluating a guess expression.

    `regN` / `uregN` / `predN` resolve to the *pre-execution* value of the
    register held in slot N (from the replay state), because the natural
    semantic equation reads inputs, not outputs.  The captured post-execution
    values (what the instruction actually produced) are exposed under the
    `after_` prefix and are what target comparisons match against.

    For a slot that is a pure source (and therefore not written by this
    instruction), pre == post, so `regN` and `after_regN` are identical.
    """
    ns: dict = {}
    rec = rr.record

    # ---- Reg slots -------------------------------------------------------
    # Slot → register name mapping (e.g., slot 0 → "R9" for `IMAD R9, …`).
    slot_to_reg: dict[int, str] = {}
    for i, sm in enumerate(rr.slots):
        slot_to_reg[i] = sm.reg_name

    for i, vals in rec.reg_values.items():
        ns[f"after_reg{i}"] = Lane(vals)
        name = slot_to_reg.get(i)
        before = rr.before_regs.get(name) if name else None
        if before is not None:
            ns[f"reg{i}"] = Lane(before)
        else:
            # No mapping or never-written register: fall back to trace value.
            # For pure-source slots that equals the correct pre-value; for an
            # uninitialised dst slot it's "the captured after", which is the
            # best we can offer.
            ns[f"reg{i}"] = Lane(vals)

    # ---- UReg slots ------------------------------------------------------
    ops_u = [op for op in rr.operands if op.kind == "ureg"]
    for i, v in rec.ureg_values.items():
        ns[f"after_ureg{i}"] = Lane(v)
        if i < len(ops_u):
            name = ops_u[i].name
            before = rr.before_uregs.get(name)
            ns[f"ureg{i}"] = Lane(before) if before is not None else Lane(v)
        else:
            ns[f"ureg{i}"] = Lane(v)

    # ---- Predicate slots -------------------------------------------------
    ops_p = [op for op in rr.operands if op.kind == "pred"]
    for i, v in enumerate(rec.pred_values):
        ns[f"after_pred{i}"] = Lane(v)
        if i < len(ops_p):
            name = ops_p[i].name
            before = rr.before_preds.get(name)
            ns[f"pred{i}"] = Lane(before) if before is not None else Lane(v)
        else:
            ns[f"pred{i}"] = Lane(v)

    for k, v in rec.c_values.items():
        ns[k.lower()] = Lane(v)  # c32 / c64 (constant-bank reads — no pre/post)

    # Explicit-by-name before bindings (useful when several slots alias the
    # same register, e.g., IMAD R9, R9, … — `before_r9` is unambiguous).
    for name, vals in rr.before_regs.items():
        ns[f"before_{name.lower()}"] = Lane(vals)
    for name, v in rr.before_uregs.items():
        ns[f"before_{name.lower()}"] = Lane(v)
    for name, v in rr.before_preds.items():
        ns[f"before_{name.lower()}"] = Lane(v)

    # Special-register / coordinate bindings from the record header.
    cta_x, cta_y, cta_z = rec.cta
    ns["ctaid_x"] = Lane(cta_x)
    ns["ctaid_y"] = Lane(cta_y)
    ns["ctaid_z"] = Lane(cta_z)
    ns["warp_id"] = Lane(rec.warp)
    # tid: lane-id within the warp (0..31).  For the full block-level TID_X
    # you usually need to add `32 * warp_id` (or an offset observed in a
    # prior S2R — NVBit can relabel warps, so verify against `reg0` first).
    ns["tid"] = Lane(list(range(WARP_SIZE)))
    ns["tid_x"] = Lane([rec.warp * WARP_SIZE + t for t in range(WARP_SIZE)])

    # Helpers
    ns["sext"] = lambda x: Lane([_sext32(v) if v is not None else None for v in Lane(x).vals])
    ns["clz"] = lambda x: Lane([(32 - v.bit_length()) if v is not None else None for v in Lane(x).vals])
    ns["where"] = _where
    ns["lane_pred"] = _lane_pred
    ns["mask_of"] = _mask_of
    ns["warp_size"] = WARP_SIZE
    return ns


def _where(cond, a, b) -> Lane:
    """Per-lane select: returns `a` on lanes where `cond` is non-zero, else `b`.

    Any argument can be a scalar, a list, or a Lane.
    """
    c = cond if isinstance(cond, Lane) else Lane(cond)
    la = a if isinstance(a, Lane) else Lane(a)
    lb = b if isinstance(b, Lane) else Lane(b)
    out: list[Optional[int]] = []
    for i in range(WARP_SIZE):
        cv = c.vals[i]
        if cv is None:
            out.append(None)
        elif cv != 0:
            out.append(la.vals[i])
        else:
            out.append(lb.vals[i])
    return Lane(out)


def _lane_pred(p) -> Lane:
    """Unpack a 32-bit predicate bitmask into a Lane of 0/1 per lane.

    Accepts a Lane (broadcast scalar) or a raw int.  Bit t of the mask becomes
    the value for lane t.
    """
    lp = p if isinstance(p, Lane) else Lane(p)
    # predN is stored broadcast — every lane holds the same 32-bit mask.
    mask = lp.vals[0]
    if mask is None:
        return Lane([None] * WARP_SIZE)
    return Lane([(mask >> t) & 1 for t in range(WARP_SIZE)])


def _mask_of(lane) -> Lane:
    """Pack a 32-lane "boolean" Lane into a uniform 32-bit bitmask Lane.

    Lane values are treated as truthy iff non-zero.  The resulting broadcast
    Lane has bit t set when lane t's input was truthy.  None lanes contribute
    0 (conservative).  Useful for matching against a `predN` target.
    """
    l = lane if isinstance(lane, Lane) else Lane(lane)
    mask = 0
    for t, v in enumerate(l.vals):
        if v is not None and v != 0:
            mask |= 1 << t
    return Lane(mask)


def _compare(name: str, got: Lane, expect: list[int]) -> tuple[int, list[int]]:
    matches = 0
    mismatches: list[int] = []
    for t in range(WARP_SIZE):
        g = got.vals[t]
        e = expect[t] & U32_MASK
        if g is not None and (g & U32_MASK) == e:
            matches += 1
        else:
            mismatches.append(t)
    return matches, mismatches


# ---------------------------------------------------------------------------
# Guess-expression linting
# ---------------------------------------------------------------------------


_SASS_IDIOM_HINTS = {
    r"\bSR_TID\b": "SR_TID.X is the thread-id special register; use `tid` (0..31 lane id) "
                    "or `tid_x` (32*warp + lane)",
    r"\bSR_CTAID\b": "SR_CTAID.{X,Y,Z} is the block-id special register; use "
                      "`ctaid_x`, `ctaid_y`, `ctaid_z`",
    r"\bR\d+\b": "use `regN` or `before_rN` (lowercase); `R0` etc. are SASS syntax, not variables here",
    r"\bUR\d+\b": "use `uregN` or `before_urN`; `UR4` etc. are SASS syntax, not variables here",
    r"\bP\d+\b": "use `predN` or `before_pN`; `P0` etc. are SASS syntax, not variables here",
}


def _guess_syntax_hint(expr: str) -> Optional[str]:
    """Detect common mistakes and return a diagnostic, or None if the expression looks OK.

    Catches the most common confusions:
      1. Writing a SASS assignment like `R0 = SR_TID.X` (use bare rhs instead).
      2. Using SASS register names like `R0`, `UR4`, `P0` (lowercase `regN` etc.).
    """
    stripped = expr.strip()

    # Look for `=` that isn't part of `==`, `!=`, `<=`, `>=`.
    i = 0
    while i < len(stripped):
        ch = stripped[i]
        if ch == "=":
            prev = stripped[i - 1] if i > 0 else ""
            nxt = stripped[i + 1] if i + 1 < len(stripped) else ""
            if prev not in "=!<>" and nxt != "=":
                return (
                    "write only the right-hand side of the equation — the `g` command\n"
                    "  compares your expression against the currently-selected target\n"
                    "  (set via `t <slot>`). For example, instead of `R0 = SR_TID.X`, type\n"
                    "    g tid"
                )
        i += 1

    for pat, msg in _SASS_IDIOM_HINTS.items():
        if re.search(pat, stripped):
            return f"hint: {msg}\n  (run `vars` to see what's actually bound)"

    return None


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------


HELP = """\
Memory-referencing instructions (LDG/STG/LDS/STS/LDL/STL/ATOM/... or any
instruction with a `[...]` or `desc[...]` operand) are *skipped* during
step navigation and refused at accept time — their semantics can't be
expressed as a pure register→register function.  Use `j <idx>` to land on
one anyway for inspection.

commands:
  <enter> / n      next non-memory instruction
  p                previous non-memory instruction
  j <idx>          jump to record index (1-based; bypasses the skip filter)
  k <idx>          jump to kernel (1-based)
  f <regex>        find next non-memory instruction matching regex
  t <slot>         pick target slot (e.g. "reg0", "pred0"); default = first dst
  g <expr>         guess: evaluate python expr, compare to current target
  vars             list variables available to the current instruction
  raw              dump the raw TraceRecord for the current instruction
  l                list next 10 instructions
  accepted         list guesses that matched all 32 lanes (auto-saved)
  note <text>      annotate the most recently saved guess
  forget <i|all>   remove a saved guess
  export <path>    write saved guesses to file (.json / .md / .txt by suffix)
  q                quit
  ?                this help

Guess expressions are PYTHON, not SASS.  Write only the right-hand side of
the semantic equation — the tool compares it against the currently-selected
target (by default reg0, the first destination slot).  Do NOT write `=`, and
do NOT use SASS register names like `R0` or `SR_TID.X`; use the bound
variables listed below.  Examples:

    # S2R R0, SR_TID.X        →  R0 gets the thread-id special register
    g tid

    # IMAD R9, R9, UR4, R0    →  R9 = R9 * UR4 + R0   (regN reads pre-values)
    g reg1 * ureg0 + reg2

    # ISETP.GE.AND P0, …, R9, UR5, PT    (target auto-picks pred0)
    g mask_of(reg0 >= ureg0)

Vars bound for the current record (run `vars` to see the actual set):
  regN, uregN, predN                 — PRE-execution values of slot N
  after_regN, after_uregN, after_predN — post-execution (trace) values
  before_rN, before_urN, before_pN   — pre-execution values by register NAME
  c32, c64                           — constant-bank reads (no pre/post)
  tid                                — the lane-id constant (0..31)
  tid_x                              — 32*warp_id + lane  (guess at SR_TID.X)
  ctaid_x, ctaid_y, ctaid_z          — block-id from the CTA header
  warp_id                            — recorded-warp index
  sext(x), clz(x)                    — helpers
  where(cond, a, b)                  — per-lane select (a if cond!=0 else b)
  lane_pred(predN)                   — unpack a 32-bit pred mask → Lane of 0/1
  mask_of(lane)                      — pack a 0/1 Lane → uniform 32-bit mask

Target comparison: `t reg0` (or any regN/uregN/predN target) is compared
against the POST value of that slot (`after_reg0`).  Writing `reg1 + reg2`
in your expression therefore naturally reads "source values going in".

Predicates are captured as 32-bit bitmasks (bit t = lane t).  Lift a mask
into per-lane booleans with `lane_pred(predN)`, then condition with `where`.
Comparisons on Lanes (`<`, `<=`, `==`, `!=`, `>`, `>=`) also return a 0/1
Lane usable as a condition.  To compare back to a `predN` target (another
32-bit mask), pack your per-lane boolean with `mask_of(...)`.  Examples:

    # @P0 FADD R0, R1, R2
    #   R0 = R1+R2 on lanes where P0 (pre) is true, else unchanged
    g where(lane_pred(pred0), reg1 + reg2, reg0)

    # ISETP.GE.AND P0, PT, R9, UR5, PT   (target auto-picks pred0)
    g mask_of(reg0 >= ureg0)
"""


def _is_memory_ref(rr: ReplayedRecord) -> bool:
    """True if the instruction touches addressable memory.

    Constant-bank reads (`c[...]`, captured as C32/C64 in the trace) are NOT
    considered memory references — they're guessable through the exposed
    `c32`/`c64` bindings.
    """
    stem = rr.mnemonic.split(".")[0]
    if stem in _STORE_STEMS:
        return True
    if any(op.kind in ("mem", "desc") for op in rr.operands):
        return True
    return False


def _auto_target(rr: ReplayedRecord) -> str:
    """Pick the slot that best represents this instruction's destination."""
    # If the slot map has a dst entry, the first Reg slot is it.
    for i, sm in enumerate(rr.slots):
        if sm.role == "dst":
            return f"reg{i}"
    # No Reg dst — check the first operand for a non-register destination.
    if rr.operands:
        first = rr.operands[0]
        if first.kind == "ureg" and rr.record.ureg_values:
            return "ureg0"
        if first.kind == "pred" and rr.record.pred_values:
            return "pred0"
    # Fallbacks, in order of usefulness
    if rr.record.reg_values:
        return "reg0"
    if rr.record.ureg_values:
        return "ureg0"
    if rr.record.pred_values:
        return "pred0"
    return "reg0"


class Explorer:
    def __init__(self, kernels: list[Kernel]):
        self.kernels = kernels
        self.replayed: list[list[ReplayedRecord]] = [replay_kernel(k) for k in kernels]
        self.k_idx = 0
        self.i_idx = 0
        # None = auto (derived per-record from the instruction's destination).
        # A string overrides that auto-pick until cleared with a bare `t`.
        self.target_override: Optional[str] = None
        # Accepted guesses: list of dicts with fields
        #   mnemonic, stem, disasm, target, expression,
        #   kernel_idx, record_idx, note.
        self.accepted: list[dict] = []
        # If the very first record is a memory reference, step forward until
        # we land on one that isn't — so the user never opens on a skipped insn.
        if self.current() is not None and _is_memory_ref(self.current()):
            self._step_past_memrefs(+1)

    def current(self) -> Optional[ReplayedRecord]:
        if not self.kernels:
            return None
        if not self.replayed[self.k_idx]:
            return None
        return self.replayed[self.k_idx][self.i_idx]

    def effective_target(self) -> str:
        if self.target_override is not None:
            return self.target_override
        rr = self.current()
        if rr is None:
            return "reg0"
        return _auto_target(rr)

    def show(self):
        if not self.kernels:
            print("(no kernels)")
            return
        rr = self.current()
        if rr is None:
            print("(kernel is empty)")
            return
        total = len(self.replayed[self.k_idx])
        kernel = self.kernels[self.k_idx]
        print()
        print(f"=== kernel {kernel.idx}: {kernel.name}")
        print(f"    grid {kernel.grid}  block {kernel.block}")
        _show_record(rr, self.i_idx, total)
        suffix = "" if self.target_override else "  (auto)"
        print(f"    target = {self.effective_target()}{suffix}")

    def _advance(self, direction: int) -> bool:
        """Move k_idx/i_idx by one record in `direction` (+1 or -1).

        Returns False if we ran off the end.
        """
        if direction > 0:
            rs = self.replayed[self.k_idx]
            if self.i_idx + 1 < len(rs):
                self.i_idx += 1
                return True
            if self.k_idx + 1 < len(self.kernels):
                self.k_idx += 1
                self.i_idx = 0
                return True
            return False
        else:
            if self.i_idx > 0:
                self.i_idx -= 1
                return True
            if self.k_idx > 0:
                self.k_idx -= 1
                self.i_idx = len(self.replayed[self.k_idx]) - 1
                return True
            return False

    def _step_past_memrefs(self, direction: int) -> bool:
        """Advance one step, then keep advancing while the record is a memory
        reference.  Returns True if a non-mem record was reached."""
        if not self._advance(direction):
            return False
        while True:
            rr = self.current()
            if rr is None or not _is_memory_ref(rr):
                return True
            if not self._advance(direction):
                return False

    def cmd_next(self):
        if not self._step_past_memrefs(+1):
            print("(end of trace — no more non-memory records)")
            return
        self.show()

    def cmd_prev(self):
        if not self._step_past_memrefs(-1):
            print("(start of trace — no earlier non-memory records)")
            return
        self.show()

    def cmd_jump(self, arg: str):
        try:
            idx = int(arg) - 1
        except ValueError:
            print("usage: j <1-based record index>")
            return
        rs = self.replayed[self.k_idx]
        if 0 <= idx < len(rs):
            self.i_idx = idx
            self.show()
        else:
            print(f"out of range (1..{len(rs)})")

    def cmd_kernel(self, arg: str):
        try:
            idx = int(arg) - 1
        except ValueError:
            print("usage: k <1-based kernel index>")
            return
        if 0 <= idx < len(self.kernels):
            self.k_idx = idx
            self.i_idx = 0
            self.show()
        else:
            print(f"out of range (1..{len(self.kernels)})")

    def cmd_find(self, arg: str):
        try:
            pat = re.compile(arg)
        except re.error as e:
            print(f"bad regex: {e}")
            return
        rs = self.replayed[self.k_idx]
        for i in range(self.i_idx + 1, len(rs)):
            if _is_memory_ref(rs[i]):
                continue
            if pat.search(rs[i].record.disasm):
                self.i_idx = i
                self.show()
                return
        print("(no match after current position)")

    def cmd_list(self, _arg: str):
        rs = self.replayed[self.k_idx]
        lo = self.i_idx
        hi = min(lo + 10, len(rs))
        for i in range(lo, hi):
            mark = ">" if i == self.i_idx else " "
            print(f"  {mark} [{i+1:4d}] {rs[i].record.disasm}")

    def cmd_target(self, arg: str):
        arg = arg.strip()
        if not arg:
            self.target_override = None
            print(f"target = {self.effective_target()}  (auto)")
            return
        self.target_override = arg
        print(f"target = {arg}")

    def cmd_vars(self, _arg: str):
        rr = self.current()
        if rr is None:
            return
        ns = _build_eval_ns(rr)
        for k in sorted(ns):
            if k in ("sext", "clz", "tid", "warp_size"):
                continue
            v = ns[k]
            if isinstance(v, Lane):
                print(f"  {k:20s} = {v}")

    def cmd_raw(self, _arg: str):
        rr = self.current()
        if rr is None:
            return
        rec = rr.record
        print(f"  disasm:      {rec.disasm}")
        print(f"  width:       {rec.width}")
        print(f"  reg_values:  {sorted(rec.reg_values)}")
        for slot in sorted(rec.reg_values):
            print(f"    reg{slot}: {_fmt_lane_list(rec.reg_values[slot])}")
        print(f"  ureg_values: {rec.ureg_values}")
        print(f"  c_values:    {rec.c_values}")
        print(f"  pred_values: [{', '.join(f'0x{v:08x}' for v in rec.pred_values)}]")

    def _record_accepted(self, rr: ReplayedRecord, target: str, expr: str) -> None:
        if _is_memory_ref(rr):
            print("  (not saved — instruction references memory)")
            return
        entry = {
            "mnemonic": rr.mnemonic,
            "stem": rr.mnemonic.split(".")[0],
            "disasm": rr.record.disasm,
            "target": target,
            "expression": expr.strip(),
            "kernel_idx": rr.record.kernel_idx,
            "record_idx": rr.record.seq,
            "note": "",
        }
        # Dedup on (mnemonic, target, expression).
        for prior in self.accepted:
            if (prior["mnemonic"] == entry["mnemonic"]
                    and prior["target"] == entry["target"]
                    and prior["expression"] == entry["expression"]):
                return
        self.accepted.append(entry)
        print(f"  (saved, {len(self.accepted)} total — `accepted` to list, `export <path>` to write)")

    def cmd_accepted(self, _arg: str) -> None:
        if not self.accepted:
            print("(no guesses saved yet — a guess is saved automatically when it matches all 32 lanes)")
            return
        for i, e in enumerate(self.accepted, start=1):
            note = f"  // {e['note']}" if e["note"] else ""
            print(f"  [{i:3d}] {e['mnemonic']:24s} {e['target']} = {e['expression']}{note}")
            print(f"        from kernel {e['kernel_idx']} record {e['record_idx']}: {e['disasm']}")

    def cmd_forget(self, arg: str) -> None:
        if not arg.strip():
            print("usage: forget <1-based index from `accepted`>   or   forget all")
            return
        if arg.strip().lower() == "all":
            n = len(self.accepted)
            self.accepted.clear()
            print(f"forgot {n} entries")
            return
        try:
            idx = int(arg) - 1
        except ValueError:
            print("usage: forget <index> | forget all")
            return
        if 0 <= idx < len(self.accepted):
            gone = self.accepted.pop(idx)
            print(f"forgot [{idx+1}] {gone['mnemonic']}")
        else:
            print(f"out of range (1..{len(self.accepted)})")

    def cmd_note(self, arg: str) -> None:
        """Attach a note to the most recently accepted guess."""
        if not self.accepted:
            print("(nothing to annotate — accept a guess first)")
            return
        self.accepted[-1]["note"] = arg.strip()
        print(f"  note on [{len(self.accepted)}] {self.accepted[-1]['mnemonic']}: {arg.strip()}")

    def cmd_export(self, arg: str) -> None:
        path = arg.strip()
        if not path:
            print("usage: export <path>     (.json or .md suffix selects format)")
            return
        if not self.accepted:
            print("(nothing to export yet)")
            return
        import json, os
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in (".json", ".jsonl"):
                with open(path, "w") as f:
                    json.dump(self.accepted, f, indent=2)
                    f.write("\n")
            elif ext == ".md":
                with open(path, "w") as f:
                    f.write("# Accepted SASS semantic guesses\n\n")
                    f.write("| # | mnemonic | target | expression | example disasm | note |\n")
                    f.write("|---|----------|--------|------------|-----------------|------|\n")
                    for i, e in enumerate(self.accepted, start=1):
                        f.write(f"| {i} | `{e['mnemonic']}` | `{e['target']}` "
                                f"| `{e['expression']}` | `{e['disasm']}` "
                                f"| {e['note']} |\n")
                    f.write("\n")
            else:
                # Plain text fallback.
                with open(path, "w") as f:
                    for e in self.accepted:
                        note = f"  # {e['note']}" if e["note"] else ""
                        f.write(f"{e['mnemonic']}\t{e['target']} = {e['expression']}"
                                f"\t({e['disasm']}){note}\n")
        except OSError as err:
            print(f"write failed: {err}")
            return
        print(f"wrote {len(self.accepted)} entries to {path}")

    def cmd_guess(self, expr: str):
        rr = self.current()
        if rr is None:
            return
        if not expr.strip():
            print("usage: g <expression>")
            return

        hint = _guess_syntax_hint(expr)
        if hint:
            print(hint)
            return

        ns = _build_eval_ns(rr)
        try:
            result = eval(expr, {"__builtins__": {}}, ns)
        except SyntaxError as e:
            print(f"syntax error: {e.msg}")
            print("  guesses are Python expressions, not SASS. Try `?` for examples.")
            return
        except NameError as e:
            print(f"unknown name: {e}")
            print("  run `vars` to see variables bound in this record.")
            return
        except Exception as e:
            print(f"evaluation failed: {type(e).__name__}: {e}")
            return
        if not isinstance(result, Lane):
            result = Lane(result)

        target_name = self.effective_target()
        # Target names like "reg0" identify a destination slot; the thing we
        # compare against is the *post*-execution value of that slot, which
        # lives under the "after_" prefix in the namespace.
        tgt_key = target_name if target_name.startswith("after_") else f"after_{target_name}"
        tgt = ns.get(tgt_key)
        if tgt is None:
            # Fall back to the literal name (e.g. user set target to something
            # like "c32" that has no pre/post distinction).
            tgt = ns.get(target_name)
        if tgt is None:
            print(f"target {target_name!r} is not a variable in this record "
                  f"(try: vars)")
            return
        if not isinstance(tgt, Lane):
            tgt = Lane(tgt)

        matches, mismatches = _compare(target_name, result, tgt.vals)
        print(f"  guess : {result}")
        print(f"  target: {tgt}  ({target_name})")
        if matches == WARP_SIZE:
            print(f"  ✓ matches all 32 lanes")
            self._record_accepted(rr, target_name, expr)
        else:
            print(f"  ✗ matches {matches}/32 lanes; "
                  f"differs on lanes {mismatches[:8]}"
                  f"{'...' if len(mismatches) > 8 else ''}")
            # show first diff in detail
            if mismatches:
                t = mismatches[0]
                gv = result.vals[t]
                tv = tgt.vals[t]
                gs = "<None>" if gv is None else f"0x{gv:08x}"
                ts = "<None>" if tv is None else f"0x{tv:08x}"
                print(f"    e.g. T{t}:  guess={gs}  target={ts}")


def repl(path: str) -> None:
    kernels = parse_trace(path)
    if not kernels:
        print("no kernels parsed")
        return

    explorer = Explorer(kernels)
    print(f"loaded {len(kernels)} kernel(s) from {path}")
    for k in kernels:
        print(f"  kernel {k.idx}: {k.name}  ({len(k.records)} records)")
    explorer.show()
    print("type ? for help")

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not raw:
            explorer.cmd_next()
            continue
        head, _, arg = raw.partition(" ")
        cmd = head.lower()
        if cmd in ("q", "quit", "exit"):
            return
        if cmd in ("?", "h", "help"):
            print(HELP)
            continue
        if cmd == "n":
            explorer.cmd_next()
        elif cmd == "p":
            explorer.cmd_prev()
        elif cmd == "j":
            explorer.cmd_jump(arg)
        elif cmd == "k":
            explorer.cmd_kernel(arg)
        elif cmd == "f":
            explorer.cmd_find(arg)
        elif cmd == "l":
            explorer.cmd_list(arg)
        elif cmd == "t":
            explorer.cmd_target(arg)
        elif cmd == "g":
            explorer.cmd_guess(arg)
        elif cmd == "vars":
            explorer.cmd_vars(arg)
        elif cmd == "raw":
            explorer.cmd_raw(arg)
        elif cmd in ("accepted", "saved"):
            explorer.cmd_accepted(arg)
        elif cmd == "forget":
            explorer.cmd_forget(arg)
        elif cmd == "note":
            explorer.cmd_note(arg)
        elif cmd == "export":
            explorer.cmd_export(arg)
        else:
            print("unknown command; type ? for help")


def main():
    ap = argparse.ArgumentParser(description="Interactive NVBit trace explorer")
    ap.add_argument("trace", help="path to NVBit trace file")
    args = ap.parse_args()
    repl(args.trace)


if __name__ == "__main__":
    main()
