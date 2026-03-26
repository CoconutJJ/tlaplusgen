#!/usr/bin/env python3
"""
clean_sass.py  –  Strip NVIDIA SASS cuobjdump/nvdisasm dumps down to
                  bare instructions.

Removes:
  • Assembler directives  (.section, .align, .byte, .word, …)
  • ELF / nv.info metadata blocks
  • The per-instruction register-liveness comment tables
  • Section-divider banner comments
  • Label-only lines
  • The trailing BRA-trap + legend block

Usage:
    python clean_sass.py  input.sass              # writes to stdout
    python clean_sass.py  input.sass  out.sass    # writes to file
    cat input.sass | python clean_sass.py         # stdin → stdout

Options (env-vars for simplicity):
    KEEP_ADDR=0   strip the /*addr*/ prefix        (default: keep)
    KEEP_PRED=0   strip predicate (@P0 / @!P1)     (default: keep)
"""

import re
import sys
import os

# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------

# Matches the /*addr*/ address token that appears on every instruction line
# AND on metadata lines.  We use it as an anchor, then validate the rest.
_ADDR = r'/\*(?P<addr>[0-9a-fA-F]+)\*/'

# Optional predicate:  @P3   @!P5   @PT   (PT is "always-true" pred)
_PRED = r'(?P<pred>@[!]?P(?:[T]|\d+)\s+)?'

# Mnemonic: uppercase, may contain dots (F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C)
# Starts with an ASCII uppercase letter — this alone excludes .byte/.short lines.
_MNEM = r'(?P<mnem>[A-Z][A-Z0-9]*(?:\.[A-Z0-9_]+)*)'

# Everything up to (and including) the mandatory semicolon terminator.
# [^;]* (zero-or-more) handles zero-operand instructions like EXIT ; or BRA ;
_BODY = r'(?P<body>[^;]*;)'

_INSTR_RE = re.compile(
    r'\s*' + _ADDR + r'\s+' + _PRED + _MNEM + r'\s+' + _BODY
)

_BRX_TARGETS = re.compile(r'\(\*"BRANCH_TARGETS (?P<brx_targets>\.(.*),?)+\*\)')
# A second, simpler form for zero-operand instructions like EXIT / BRA target
# (they still have a semicolon, so the above handles them — kept for clarity)

# Lines we want to skip entirely before even trying the regex, to avoid
# accidentally matching the /*addr*/ tokens inside .nv.info data blocks.
# Branch-target labels we want to KEEP: .L_foo:  .L_x_0:  etc.
_LABEL_RE = re.compile(r'^\s*(?P<label>\.[A-Za-z]\w*):')

_SKIP_RE = re.compile(
    r'^\s*(?:'
    r'\.'                          # any assembler directive (.section .byte …)
    r'|//-+'                       # section-divider banner  //----…
    r'|//\s*\|'                    # register-liveness table //  |  1   ^ …
    r'|//\s*\+'                    # table border            // +---…
    r'|//\s*Legend'                # legend header
    r'|//\s*[#^v:x ]'             # legend symbol lines
    r'|//\s*$'                     # empty comment
    r'|kernel_\w+:$'              # bare kernel entry label
    r'|\.text\.\S+:$'             # .text.kernel_…: section label
    r')'
)


def clean(text: str, keep_addr: bool = True, keep_pred: bool = True) -> str:
    out = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        # Keep branch-target labels (.L_foo:) — checked BEFORE _SKIP_RE
        # because the skip rule matches any line starting with '.'
        lm = _LABEL_RE.match(line)
        if lm:
            out.append(f'{lm["label"]}:')
            continue

        # Fast-path: blank or should-skip
        if not line.strip():
            continue
        if _SKIP_RE.match(line):
            continue

        m = _INSTR_RE.match(line)
        if not m:
            continue

        brx_targets = _BRX_TARGETS.search(line)
        if brx_targets is not None:
            print(brx_targets)
        parts = []
        if keep_addr:
            parts.append(f'/*{m["addr"]}*/')
        if keep_pred and m["pred"]:
            parts.append(m["pred"].strip())
        parts.append(m["mnem"])
        parts.append(m["body"].strip())

        out.append("  ".join(parts))

    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    keep_addr = os.environ.get("KEEP_ADDR", "1") != "0"
    keep_pred = os.environ.get("KEEP_PRED", "1") != "0"

    args = [a for a in sys.argv[1:] if not a.startswith("-")]

    if len(args) >= 1:
        with open(args[0], "r", errors="replace") as fh:
            text = fh.read()
    else:
        text = sys.stdin.read()

    result = clean(text, keep_addr=keep_addr, keep_pred=keep_pred)

    if len(args) >= 2:
        with open(args[1], "w") as fh:
            fh.write(result + "\n")
        print(f"Wrote {len(result.splitlines())} instructions → {args[1]}", file=sys.stderr)
    else:
        print(result)


if __name__ == "__main__":
    main()