#!/usr/bin/env python3
"""
trace_viewer.py — Parse NVBit register traces and show before/after
register values for each instruction.

Tracks architectural register state across instructions so that
destination registers show their old and new values.

Usage:
    python trace_viewer.py ../docs/traces.txt
    python trace_viewer.py ../docs/traces.txt --mnemonic FADD
    python trace_viewer.py ../docs/traces.txt --kernel fadd
    python trace_viewer.py ../docs/traces.txt --thread 0
"""

import re
import sys
from argparse import ArgumentParser

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
_KERNEL_RE = re.compile(r'^(?:\d+\s+)?Kernel\s+\d+:(\S+)\(')

# Instruction stems that do NOT write a GPR destination
_NO_GPR_DST = {
    'STG', 'STS', 'SST', 'ST', 'STL',          # stores
    'EXIT', 'BRA', 'BSSY', 'BSYNC', 'BAR',     # control flow / sync
    'NOP', 'DEPBAR', 'WARPSYNC', 'YIELD',       # misc
    'NANOSLEEP',
    'ISETP', 'FSETP', 'UISETP', 'DSETP',        # predicate-only destination
    'HSETP2', 'PLOP3', 'VOTE', 'VOTEU',
}

# Instruction stems that write a uniform register destination
_UREG_DST = {'S2UR', 'LDCU', 'UIMAD', 'UIADD3', 'ULDC', 'UMOV', 'USEL', 'USHF', 'UPRMT'}


def parse_trace(path):
    """Parse an NVBit trace file into a list of instruction entries."""
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
                current_kernel = km.group(1)
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
                    'regs': {},     # reg_idx -> {thread_id -> value}
                    'uregs': {},    # ureg_idx -> value
                    'consts': {},   # bit_width -> value
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
# Instruction register analysis
# --------------------------------------------------------------------------

def get_mnemonic(instr):
    """Extract the mnemonic from an instruction, skipping predicate guards."""
    text = instr
    if text.startswith('@'):
        parts = text.split(None, 1)
        text = parts[1] if len(parts) > 1 else text
    return text.split()[0] if text else '?'


def get_stem(mnemonic):
    return mnemonic.split('.')[0]


def analyze_regs(instr, width):
    """Analyze instruction to determine dst/src GPR and UReg names.

    Returns (dst_gprs, src_gprs, all_uregs) where each is a list of
    architectural register name strings like 'R9', 'UR4'.
    """
    text = instr
    if text.startswith('@'):
        text = text.split(None, 1)[1] if ' ' in text else text

    parts = text.split(None, 1)
    mnemonic = parts[0]
    operands = parts[1] if len(parts) > 1 else ''
    stem = get_stem(mnemonic)

    # Extract all Rn references (with duplicates, in order)
    gpr_refs = [f'R{m.group(1)}' for m in re.finditer(r'\bR(\d+)\b', operands)]
    # Deduplicate preserving order
    seen = set()
    unique_gprs = []
    for r in gpr_refs:
        if r not in seen:
            seen.add(r)
            unique_gprs.append(r)

    # Extract all URn references
    ur_refs = list(dict.fromkeys(
        f'UR{m.group(1)}' for m in re.finditer(r'\bUR(\d+)\b', operands)
    ))

    # Determine dst vs src GPRs
    if stem in _NO_GPR_DST or stem in _UREG_DST:
        dst_gprs = []
        src_gprs = unique_gprs
    elif unique_gprs:
        dst_gprs = [unique_gprs[0]]
        if width >= 2:
            base = int(unique_gprs[0][1:])
            hi = f'R{base + 1}'
            dst_gprs.append(hi)
        src_gprs = [r for r in unique_gprs[1:] if r not in dst_gprs]
    else:
        dst_gprs = []
        src_gprs = []

    return dst_gprs, src_gprs, ur_refs


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------

def fmt(v):
    return f"0x{v:08x}" if v is not None else "--------"


def display_entry(e, gpr_state, ureg_state, thread):
    """Display before/after register values and update state."""
    instr = e['instr']
    width = e['width']
    regs = e['regs']
    t = thread

    dst_gprs, src_gprs, all_uregs = analyze_regs(instr, width)

    # Collect all GPR names we'll display (dst first, then src, plus implicit hi)
    all_gprs = list(dst_gprs)
    for r in src_gprs:
        if r not in all_gprs:
            all_gprs.append(r)

    # --- Capture "before" values from tracked state ---
    before = {r: gpr_state.get(r, {}).get(t) for r in all_gprs}
    before_u = {r: ureg_state.get(r) for r in all_uregs}

    # --- Initialize source register state from trace (for regs not in dst) ---
    # Source regs in the trace are Reg{width}.. and reflect pre-execution values
    # (since the instruction didn't modify them). Map them by position.
    src_gpr_ordered = [f'R{m.group(1)}' for m in re.finditer(r'\bR(\d+)\b',
        instr.split(None, 2)[-1] if len(instr.split(None, 1)) > 1 else '')]
    # Remove the first GPR ref (destination) for non-store/non-pred instructions
    mnemonic = get_mnemonic(instr)
    stem = get_stem(mnemonic)
    if stem not in _NO_GPR_DST and stem not in _UREG_DST and src_gpr_ordered:
        src_gpr_ordered = src_gpr_ordered[1:]

    for i, name in enumerate(src_gpr_ordered):
        reg_idx = width + i if stem not in _NO_GPR_DST and stem not in _UREG_DST else i
        if name not in dst_gprs and reg_idx in regs:
            vals = regs[reg_idx]
            if t in vals:
                if before[name] is None:
                    before[name] = vals[t]
                gpr_state.setdefault(name, {})[t] = vals[t]

    # For UReg sources: map UReg trace indices to names in order
    for i, name in enumerate(all_uregs):
        if i in e['uregs']:
            if before_u.get(name) is None:
                before_u[name] = e['uregs'][i]

    # --- Update destination state from trace ---
    for i, name in enumerate(dst_gprs):
        if i in regs and t in regs[i]:
            gpr_state.setdefault(name, {})[t] = regs[i][t]

    # For UReg-destination instructions, update ureg state
    if stem in _UREG_DST:
        instr_text = instr
        if instr_text.startswith('@'):
            instr_text = instr_text.split(None, 1)[1]
        ur_match = re.search(r'\bUR(\d+)\b', instr_text.split(None, 1)[1] if ' ' in instr_text else '')
        if ur_match:
            dst_ur = f'UR{ur_match.group(1)}'
            if 0 in e['uregs']:
                ureg_state[dst_ur] = e['uregs'][0]

    # --- Capture "after" values ---
    after = {r: gpr_state.get(r, {}).get(t) for r in all_gprs}
    after_u = {r: ureg_state.get(r) for r in all_uregs}

    # --- Print ---
    print(f"[{e['seq']:>3}] {instr}")
    print(f"      {e['kernel']}  CTA {e['cta']}  warp {e['warp']}")

    for r in all_gprs:
        b, a = before.get(r), after.get(r)
        tag = " *" if b != a else ""
        label = "dst" if r in dst_gprs else "src"
        print(f"    {r:>4} ({label}): {fmt(b)} -> {fmt(a)}{tag}")

    for r in all_uregs:
        b, a = before_u.get(r), after_u.get(r)
        tag = " *" if b != a else ""
        print(f"    {r:>4}:       {fmt(b)} -> {fmt(a)}{tag}")

    if e['pred'] is not None:
        p0, p1 = e['pred']
        print(f"    Pred:       0x{p0:08x}  0x{p1:08x}")

    for bits, cv in sorted(e['consts'].items()):
        print(f"    C{bits}:        0x{cv:0{bits // 4}x}")

    print()


def main():
    ap = ArgumentParser(description="NVBit SASS trace viewer — before/after register values")
    ap.add_argument("tracefile", help="Path to traces.txt")
    ap.add_argument("--mnemonic", "-m", help="Filter by mnemonic (substring match)")
    ap.add_argument("--kernel", "-k", help="Filter by kernel name (substring match)")
    ap.add_argument("--thread", "-t", type=int, default=0, help="Thread to show (default: 0)")
    ap.add_argument("--limit", "-n", type=int, default=0, help="Max entries to show (0=all)")
    args = ap.parse_args()

    entries = parse_trace(args.tracefile)
    print(f"Parsed {len(entries)} trace entries\n")

    # Per-(kernel, cta, warp) register file state
    gpr_states = {}   # key -> {reg_name: {tid: value}}
    ureg_states = {}  # key -> {reg_name: value}

    shown = 0
    for e in entries:
        key = (e['kernel'], e['cta'], e['warp'])
        if key not in gpr_states:
            gpr_states[key] = {}
            ureg_states[key] = {}

        mnemonic = get_mnemonic(e['instr'])
        do_show = True
        if args.mnemonic and args.mnemonic.upper() not in mnemonic.upper():
            do_show = False
        if args.kernel and args.kernel.lower() not in e['kernel'].lower():
            do_show = False

        if do_show:
            display_entry(e, gpr_states[key], ureg_states[key], args.thread)
            shown += 1
            if args.limit and shown >= args.limit:
                break
        else:
            update_state(e, gpr_states[key], ureg_states[key], args.thread)

    print(f"Showed {shown}/{len(entries)} entries")


def update_state(e, gpr_state, ureg_state, thread):
    """Update register state without printing (for filtered-out entries)."""
    instr = e['instr']
    width = e['width']
    regs = e['regs']
    t = thread
    mnemonic = get_mnemonic(instr)
    stem = get_stem(mnemonic)

    dst_gprs, src_gprs, all_uregs = analyze_regs(instr, width)

    # Update source registers from trace
    src_gpr_ordered = [f'R{m.group(1)}' for m in re.finditer(r'\bR(\d+)\b',
        instr.split(None, 2)[-1] if len(instr.split(None, 1)) > 1 else '')]
    if stem not in _NO_GPR_DST and stem not in _UREG_DST and src_gpr_ordered:
        src_gpr_ordered = src_gpr_ordered[1:]

    for i, name in enumerate(src_gpr_ordered):
        reg_idx = width + i if stem not in _NO_GPR_DST and stem not in _UREG_DST else i
        if name not in dst_gprs and reg_idx in regs:
            vals = regs[reg_idx]
            if t in vals:
                gpr_state.setdefault(name, {})[t] = vals[t]

    # Update destination registers from trace
    for i, name in enumerate(dst_gprs):
        if i in regs and t in regs[i]:
            gpr_state.setdefault(name, {})[t] = regs[i][t]

    # Update UReg state
    if stem in _UREG_DST:
        instr_text = instr
        if instr_text.startswith('@'):
            instr_text = instr_text.split(None, 1)[1]
        parts = instr_text.split(None, 1)
        operands = parts[1] if len(parts) > 1 else ''
        ur_match = re.search(r'\bUR(\d+)\b', operands)
        if ur_match:
            dst_ur = f'UR{ur_match.group(1)}'
            if 0 in e['uregs']:
                ureg_state[dst_ur] = e['uregs'][0]


if __name__ == '__main__':
    main()
