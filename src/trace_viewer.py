#!/usr/bin/env python3
"""
trace_viewer.py — Parse NVBit register traces and display register values
before/after each instruction with derived semantics.

Usage:
    python trace_viewer.py ../traces.txt
    python trace_viewer.py ../traces.txt --mnemonic FADD
    python trace_viewer.py ../traces.txt --kernel fadd
    python trace_viewer.py ../traces.txt --thread 0
"""

import re
import struct
import sys
import math
from argparse import ArgumentParser

# --------------------------------------------------------------------------
# Float helpers
# --------------------------------------------------------------------------

def u32_to_f32(v):
    return struct.unpack('f', struct.pack('I', v & 0xFFFFFFFF))[0]

def f32_to_u32(f):
    return struct.unpack('I', struct.pack('f', f))[0]

def u16_to_f16(v):
    """Rough fp16 -> float conversion."""
    sign = (v >> 15) & 1
    exp = (v >> 10) & 0x1F
    frac = v & 0x3FF
    if exp == 0:
        val = (2**-14) * (frac / 1024.0)
    elif exp == 0x1F:
        val = float('inf') if frac == 0 else float('nan')
    else:
        val = (2 ** (exp - 15)) * (1 + frac / 1024.0)
    return -val if sign else val

def fmt_val(v, as_float=False):
    """Format a 32-bit value as hex, optionally with float interpretation."""
    s = f"0x{v:08x}"
    if as_float:
        f = u32_to_f32(v)
        if not (math.isnan(f) or math.isinf(f)):
            s += f" ({f:g})"
        elif math.isinf(f):
            s += f" ({'+' if f > 0 else '-'}INF)"
        else:
            s += " (NaN)"
    return s

def fmt_fp16x2(v):
    lo = v & 0xFFFF
    hi = (v >> 16) & 0xFFFF
    return f"0x{v:08x} (fp16: lo={u16_to_f16(lo):g}, hi={u16_to_f16(hi):g})"

# --------------------------------------------------------------------------
# Semantics deriver
# --------------------------------------------------------------------------

def derive_semantics(mnemonic, dst_vals, src_vals_list, uregs, consts, pred, width, instr_text=""):
    """Try to derive what the instruction computed for thread 0."""
    stem = mnemonic.split('.')[0]
    mnem = mnemonic.replace('.', '_').upper()
    t = 0  # analyze thread 0

    d0 = dst_vals[0][t] if dst_vals and dst_vals[0] else None
    s = [sv[t] if sv else None for sv in src_vals_list]  # src values for T0
    u = uregs  # uniform regs

    parts = []

    # Integer arithmetic
    if stem in ('IMAD',):
        d1 = dst_vals[1][t] if len(dst_vals) >= 2 and dst_vals[1] else None
        dst64 = ((d1 << 32) | d0) if d0 is not None and d1 is not None else None

        if 'WIDE' in mnemonic and width == 2 and dst64 is not None:
            # IMAD.WIDE: dst64 = src1 * src2_or_imm + src3_64
            #
            # NVBit captures all regs AFTER execution.  The first source
            # reg (Reg<width>) is the multiplicand.  If dst overlaps with
            # src3, the trace shows the NEW dst — we back-compute src3.
            #
            # Trace layout: Reg0/Reg1 = dst64, Reg2 = src1, Reg3(+Reg4) = src3 (may be stale)
            src1 = s[0] if s else None
            found = False

            if src1 is not None:
                # Extract the immediate from the instruction text if present
                imm_match = re.search(r',\s*(0x[0-9a-fA-F]+|\d+)\s*,', instr_text)
                if imm_match:
                    imm = int(imm_match.group(1), 0)
                    product = src1 * imm
                    old_src3 = (dst64 - product) & 0xFFFFFFFFFFFFFFFF
                    parts.append(
                        f"dst64 = src1*0x{imm:x} + src3_64 = {src1}*{imm} + 0x{old_src3:016x} = 0x{dst64:016x}"
                    )
                    found = True
                elif len(s) >= 2 and s[1] is not None:
                    # src2 is a register (s[1])
                    product = src1 * s[1]
                    old_src3 = (dst64 - product) & 0xFFFFFFFFFFFFFFFF
                    parts.append(
                        f"dst64 = src1*src2 + src3_64 = {src1}*{s[1]} + 0x{old_src3:016x} = 0x{dst64:016x}"
                    )
                    found = True

            # Fallback: 3+ src regs, try direct computation
            if not found and len(s) >= 3 and all(x is not None for x in s[:3]):
                wide = (s[0] * s[1] + s[2]) & 0xFFFFFFFFFFFFFFFF
                if dst64 == wide:
                    parts.append(f"dst64 = {s[0]}*{s[1]}+{s[2]} = 0x{wide:016x}")
                    found = True

            if not found:
                parts.append(f"dst64 = 0x{dst64:016x}")

        else:
            # Non-WIDE IMAD: 32-bit result = src1*src2+src3
            found = False

            # 3 register sources
            if len(s) >= 3 and all(x is not None for x in s[:3]):
                computed = (s[0] * s[1] + s[2]) & 0xFFFFFFFF
                if d0 is not None and d0 == computed:
                    parts.append(f"dst = src1*src2+src3 = {s[0]}*{s[1]}+{s[2]} = {computed}")
                    found = True

            # With uniform reg as src2: dst = src1 * UReg + src3
            if not found and len(s) >= 2 and u and all(x is not None for x in s[:2]):
                u0 = list(u.values())[0]
                computed = (s[0] * u0 + s[1]) & 0xFFFFFFFF
                if d0 is not None and d0 == computed:
                    parts.append(f"dst = src1*UReg+src3 = {s[0]}*{u0}+{s[1]} = {computed}")
                    found = True

            # src2 is an immediate: try common values
            if not found and d0 is not None and len(s) >= 2:
                for imm in (0, 1, 2, 4, 8, 16):
                    computed = (s[0] * imm + s[1]) & 0xFFFFFFFF
                    if d0 == computed:
                        parts.append(f"dst = src1*{imm}+src3 = {s[0]}*{imm}+{s[1]} = {computed}")
                        found = True
                        break

    elif stem in ('IADD3',):
        if len(s) >= 3 and all(x is not None for x in s[:3]):
            computed = (s[0] + s[1] + s[2]) & 0xFFFFFFFF
            if d0 == computed:
                parts.append(f"dst = src1+src2+src3 = {s[0]}+{s[1]}+{s[2]} = {computed}")

    # Float arithmetic
    elif stem in ('FADD',):
        if len(s) >= 2 and all(x is not None for x in s[:2]):
            f1, f2 = u32_to_f32(s[0]), u32_to_f32(s[1])
            result = f1 + f2
            if d0 is not None and f32_to_u32(result) == d0:
                parts.append(f"dst = float(src1)+float(src2) = {f1:g}+{f2:g} = {result:g}")

    elif stem in ('FMUL',):
        if len(s) >= 2 and all(x is not None for x in s[:2]):
            f1, f2 = u32_to_f32(s[0]), u32_to_f32(s[1])
            result = f1 * f2
            if d0 is not None and f32_to_u32(result) == d0:
                parts.append(f"dst = float(src1)*float(src2) = {f1:g}*{f2:g} = {result:g}")

    elif stem in ('FFMA',):
        if len(s) >= 3 and all(x is not None for x in s[:3]):
            f1, f2, f3 = u32_to_f32(s[0]), u32_to_f32(s[1]), u32_to_f32(s[2])
            result = math.fma(f1, f2, f3) if hasattr(math, 'fma') else f1 * f2 + f3
            if d0 is not None and f32_to_u32(result) == d0:
                parts.append(f"dst = fma(src1,src2,src3) = fma({f1:g},{f2:g},{f3:g}) = {result:g}")

    elif 'MUFU' in mnemonic:
        if 'EX2' in mnemonic and len(s) >= 1 and s[0] is not None:
            f1 = u32_to_f32(s[0])
            result = 2.0 ** f1
            if d0 is not None:
                parts.append(f"dst = 2^src = 2^{f1:g} = {result:g} (got {u32_to_f32(d0):g})")
        elif 'RCP' in mnemonic and len(s) >= 1 and s[0] is not None:
            f1 = u32_to_f32(s[0])
            if f1 != 0:
                result = 1.0 / f1
                if d0 is not None:
                    parts.append(f"dst = 1/src = 1/{f1:g} = {result:g} (got {u32_to_f32(d0):g})")

    # Float predicates
    elif stem in ('FSETP',):
        if pred is not None:
            pred_bits = pred[1] if len(pred) > 1 else pred[0]
            all_true = pred_bits == 0x00000001 or pred_bits == 0xFFFFFFFF
            all_false = pred_bits == 0x00000000
            parts.append(f"pred result: {'ALL TRUE' if all_true else 'ALL FALSE' if all_false else f'mask=0x{pred_bits:08x}'}")

    # ISETP
    elif stem in ('ISETP', 'UISETP'):
        if pred is not None:
            pred_bits = pred[1] if len(pred) > 1 else pred[0]
            all_true = pred_bits == 0x00000001 or pred_bits == 0xFFFFFFFF
            all_false = pred_bits == 0x00000000
            desc = 'ALL TRUE' if all_true else 'ALL FALSE' if all_false else f'mask=0x{pred_bits:08x}'
            if len(s) >= 2 and s[0] is not None and s[1] is not None:
                parts.append(f"compare({s[0]}, {s[1]}) -> {desc}")
            elif len(s) >= 1 and s[0] is not None and u:
                parts.append(f"compare({s[0]}, UReg={list(u.values())[0]}) -> {desc}")
            else:
                parts.append(f"pred result: {desc}")

    # HFMA2
    elif stem in ('HFMA2',):
        if d0 is not None:
            parts.append(f"dst packed fp16x2 = {fmt_fp16x2(d0)}")

    # F2FP
    elif 'F2FP' in mnemonic:
        if d0 is not None and len(s) >= 2:
            hi_bf16 = (s[0] >> 16) & 0xFFFF if s[0] is not None else 0
            lo_bf16 = (s[1] >> 16) & 0xFFFF if s[1] is not None else 0
            expected = (hi_bf16 << 16) | lo_bf16
            if d0 == expected:
                parts.append(f"dst = pack_bf16(src1,src2) = (0x{hi_bf16:04x}<<16)|0x{lo_bf16:04x} = 0x{expected:08x}")
            else:
                parts.append(f"dst = 0x{d0:08x}")

    # VIMNMX
    elif 'VIMNMX' in mnemonic:
        if len(s) >= 2 and s[0] is not None and s[1] is not None:
            mn = min(s[0], s[1])
            mx = max(s[0], s[1])
            if 'RELU' in mnemonic:
                relu_mn = max(0, mn)
                if d0 == relu_mn:
                    parts.append(f"dst = max(0, min({s[0]},{s[1]})) = max(0,{mn}) = {relu_mn}")
            elif d0 == mn:
                parts.append(f"dst = min({s[0]},{s[1]}) = {mn}")
            elif d0 == mx:
                parts.append(f"dst = max({s[0]},{s[1]}) = {mx}")

    # Memory loads
    elif stem in ('LDC', 'LDCU', 'LDG', 'LDS'):
        if d0 is not None:
            parts.append(f"loaded value")

    # VOTE
    elif stem in ('VOTE', 'VOTEU'):
        if pred is not None:
            pred_bits = pred[1] if len(pred) > 1 else pred[0]
            parts.append(f"warp vote -> mask=0x{pred_bits:08x}")

    if not parts:
        if d0 is not None:
            parts.append(f"dst_T0 = {fmt_val(d0, True)}")

    return "; ".join(parts)


# --------------------------------------------------------------------------
# Trace parser
# --------------------------------------------------------------------------

_HEADER_RE = re.compile(
    r'^CTA\s+(\d+,\d+,\d+)\s+-\s+warp\s+(\d+)\s+-\s+(\d+)\s+-\s+(.+?)\s*;:'
)
_REG_RE = re.compile(r'Reg(\d+)_T(\d+):\s+(0x[0-9a-fA-F]+)')
_UREG_RE = re.compile(r'UReg(\d+):\s+(0x[0-9a-fA-F]+)')
_CONST_RE = re.compile(r'C(\d+):\s+(0x[0-9a-fA-F]+)')
_PRED_RE = re.compile(r'Pred:\s+(0x[0-9a-fA-F]+)\s+(0x[0-9a-fA-F]+)')
_WIDTH_RE = re.compile(r'\*\s*Width:\s*(\d+)')
_KERNEL_RE = re.compile(r'^(\d+)\s+Kernel\s+\d+:(\S+)\(')


def parse_trace(path):
    entries = []
    current = None
    current_kernel = ""

    with open(path, 'r') as f:
        for line in f:
            line = line.rstrip()
            if not line:
                if current:
                    entries.append(current)
                    current = None
                continue

            km = _KERNEL_RE.match(line)
            if km:
                if current:
                    entries.append(current)
                    current = None
                current_kernel = km.group(2)
                continue

            hm = _HEADER_RE.match(line)
            if hm:
                if current:
                    entries.append(current)
                current = {
                    'kernel': current_kernel,
                    'cta': hm.group(1),
                    'warp': int(hm.group(2)),
                    'seq': int(hm.group(3)),
                    'instr': hm.group(4).strip(),
                    'width': 1,
                    'regs': {},   # reg_idx -> {thread_id -> value}
                    'uregs': {},  # ureg_idx -> value
                    'consts': {}, # bit_width -> value
                    'pred': None,
                }
                continue

            if current is None:
                continue

            wm = _WIDTH_RE.match(line)
            if wm:
                current['width'] = int(wm.group(1))
                continue

            for m in _REG_RE.finditer(line):
                ridx, tid, val = int(m.group(1)), int(m.group(2)), int(m.group(3), 16)
                current['regs'].setdefault(ridx, {})[tid] = val

            for m in _UREG_RE.finditer(line):
                uidx, val = int(m.group(1)), int(m.group(2), 16)
                current['uregs'][uidx] = val

            for m in _CONST_RE.finditer(line):
                bits, val = int(m.group(1)), int(m.group(2), 16)
                current['consts'][bits] = val

            pm = _PRED_RE.search(line)
            if pm:
                current['pred'] = (int(pm.group(1), 16), int(pm.group(2), 16))

    if current:
        entries.append(current)

    return entries


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------

def display_entry(e, thread=0, show_all_threads=False):
    instr = e['instr']
    mnemonic = instr.split()[0] if instr else '?'
    # strip predicate prefix like @!P0
    if mnemonic.startswith('@'):
        parts = instr.split()
        mnemonic = parts[1] if len(parts) > 1 else mnemonic

    width = e['width']
    regs = e['regs']

    # Separate dst regs (0..width-1) and src regs (width..)
    dst_indices = sorted([i for i in regs if i < width])
    src_indices = sorted([i for i in regs if i >= width])

    dst_vals = [regs[i] for i in dst_indices]
    src_vals = [regs[i] for i in src_indices]

    is_float = any(k in mnemonic for k in ('FADD', 'FMUL', 'FFMA', 'FSETP', 'MUFU', 'F2FP', 'HFMA'))

    print(f"  {'─' * 70}")
    print(f"  {e['kernel']}  CTA {e['cta']}  warp {e['warp']}  seq {e['seq']}")
    print(f"  {instr}")
    print()

    # Source registers
    if src_vals:
        for i, si in enumerate(src_indices):
            vals = regs[si]
            if show_all_threads:
                for tid in sorted(vals):
                    print(f"    src{i} (Reg{si}) T{tid}: {fmt_val(vals[tid], is_float)}")
            else:
                v = vals.get(thread, next(iter(vals.values())))
                print(f"    src{i} (Reg{si}) T{thread}: {fmt_val(v, is_float)}")

    # Uniform registers
    for ui, uv in sorted(e['uregs'].items()):
        print(f"    UReg{ui}: {fmt_val(uv, is_float)}")

    # Constants
    for bits, cv in sorted(e['consts'].items()):
        print(f"    C{bits}: 0x{cv:0{bits // 4}x}")

    # Arrow
    if dst_vals or e['pred'] is not None:
        print(f"    {'→':>6}")

    # Destination registers
    if dst_vals:
        for i, di in enumerate(dst_indices):
            vals = regs[di]
            if show_all_threads:
                for tid in sorted(vals):
                    print(f"    dst{i} (Reg{di}) T{tid}: {fmt_val(vals[tid], is_float)}")
            else:
                v = vals.get(thread, next(iter(vals.values())))
                print(f"    dst{i} (Reg{di}) T{thread}: {fmt_val(v, is_float)}")

    # Predicate output
    if e['pred'] is not None:
        p0, p1 = e['pred']
        print(f"    Pred: ctrl=0x{p0:08x} mask=0x{p1:08x}")

    # Derived semantics
    semantics = derive_semantics(
        mnemonic, dst_vals, src_vals, e['uregs'], e['consts'],
        e['pred'], width, instr_text=instr
    )
    if semantics:
        print(f"    ≡ {semantics}")
    print()


def main():
    ap = ArgumentParser(description="NVBit SASS trace viewer")
    ap.add_argument("tracefile", help="Path to traces.txt")
    ap.add_argument("--mnemonic", "-m", help="Filter by mnemonic (substring match)")
    ap.add_argument("--kernel", "-k", help="Filter by kernel name (substring match)")
    ap.add_argument("--thread", "-t", type=int, default=0, help="Thread to show (default: 0)")
    ap.add_argument("--all-threads", "-a", action="store_true", help="Show all 32 threads")
    ap.add_argument("--limit", "-n", type=int, default=0, help="Max entries to show (0=all)")
    args = ap.parse_args()

    entries = parse_trace(args.tracefile)
    print(f"Parsed {len(entries)} trace entries\n")

    shown = 0
    for e in entries:
        instr = e['instr']
        mnemonic = instr.split()[0] if instr else ''
        if mnemonic.startswith('@'):
            parts = instr.split()
            mnemonic = parts[1] if len(parts) > 1 else mnemonic

        if args.mnemonic and args.mnemonic.upper() not in mnemonic.upper():
            continue
        if args.kernel and args.kernel.lower() not in e['kernel'].lower():
            continue

        display_entry(e, thread=args.thread, show_all_threads=args.all_threads)
        shown += 1
        if args.limit and shown >= args.limit:
            break

    print(f"Showed {shown} entries")


if __name__ == '__main__':
    main()
