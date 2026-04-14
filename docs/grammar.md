# NVIDIA SASS Grammar

BNF grammar for the cleaned SASS assembly format parsed by `src/sass/parser.py`.
Derived from analysis of 14 example kernels (243 unique mnemonics) targeting
sm_90a (Hopper) and sm_100a (Blackwell).

## Line Types

```
program        ::= line*
line           ::= instruction | label | function_def | EMPTY

label          ::= "." IDENT ":"
function_def   ::= ".function" FUNC_NAME

instruction    ::= "/*" HEX_ADDR "*/" predicate? mnemonic operand_list? ";"
```

## Instruction Components

```
predicate      ::= "@" "!"? pred_name
pred_name      ::= "UP" DIGITS          /* UP0..UP6 */
                 | "UPT"                 /* uniform always-true */
                 | "PT"                  /* always-true */
                 | "P" DIGITS            /* P0..P6 */
                 | "B" DIGITS            /* barrier register */

mnemonic       ::= UPPER_LETTER UPPER_ALNUM* ("." UPPER_ALNUM+)*
```

Mnemonics may have multiple dot-separated modifiers (e.g.,
`IMAD.HI.U32`, `F2FP.SATFINITE.BF16.S2_6.UNPACK_B`,
`UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN`).

## Operand List

Operands are usually comma-separated but may also be space-separated
(e.g., BRX `R5 -0x50`). Commas and bare dots (from float modifier
suffixes like `4.789e-41.H0`) are treated as optional separators.

```
operand_list   ::= (operand | "," | ".")*
```

Alternatives are tried in the order listed (most specific first):

```
operand        ::= label_ref
                 | annotation
                 | desc_op
                 | const_bank
                 | mem_addr
                 | abs_reg
                 | neg_pred
                 | special_imm
                 | hex_imm
                 | float_imm
                 | int_imm
                 | neg_reg
                 | register
                 | mnem_word
```

## Operand Details

### Label Reference

Branch target wrapped in backtick-parens.

```
label_ref      ::= "`(" LABEL_NAME ")"
LABEL_NAME     ::= [^\)]+               /* .L_x_0, $__internal_..., kernel_... */
```

All observed labels use the `.L_x_N` format. The grammar also accepts
`$`-prefixed and plain identifiers (seen in `CALL.REL.NOINC` and
`RET.REL.NODEC` operands).

### Annotation

nvdisasm metadata blobs. Only `BRANCH_TARGETS` annotations produce AST
nodes; all others are silently discarded.

```
annotation     ::= '(*"' TEXT '"*)'
```

### Descriptor

TMA/TMEM descriptor-based memory references. One or two bracketed indices.

```
desc_op        ::= desc_kind bracket_idx+
desc_kind      ::= "desc" | "gdesc" | "tmem" | "idesc"
bracket_idx    ::= "[" INNER "]"
```

Observed forms:
- `desc[UR4][R4.64+0x8]` — two indices (base descriptor + offset)
- `gdesc[UR4]` — one index (global descriptor)
- `tmem[UR27]` — one index (tensor memory)
- `idesc[UR29]` — one index (indirect descriptor)

### Constant Bank

Constant memory reference with bank selector and offset.

```
const_bank     ::= "c" "[" HEX_NUM "]" "[" OFFSET "]"
```

Observed banks: `0x0`, `0x2`. Offset may be a hex immediate (`0x37c`),
a register (`R2`), or register+immediate (`R4+0xc`).

### Memory Address

Bracketed address expression with up to 3 `+`-separated parts.

```
mem_addr       ::= "[" PARTS "]"
PARTS          ::= PART ("+" PART)*
PART           ::= register | hex_imm | int_imm
```

Observed forms:
- `[R4]` — base only
- `[UR4+0x8]` — base + offset
- `[R43+URZ+0x70]` — base + index + offset

### Absolute Value Register

```
abs_reg        ::= "|" register "|"
```

Observed: `|R0|`, `|R74|`.

### Negated Predicate (Operand Position)

```
neg_pred       ::= "!" pred_name
```

Observed: `!PT`, `!P0`, `!UP0`, `!UP1`.

### Negated Register

Arithmetic negation on a register (e.g., `-RZ` in HFMA2).

```
neg_reg        ::= "-" register
```

### Immediates

```
special_imm    ::= SIGN? ("INF" | "+INF" | "-INF" | "QNAN" | "-QNAN" | "NaN")
hex_imm        ::= "-"? "0x" HEX_DIGITS+
float_imm      ::= SIGN? (DIGITS "." DIGITS EXP? | "." DIGITS EXP?)
int_imm        ::= "-"? DIGITS

EXP            ::= ("e" | "E") SIGN? DIGITS
SIGN           ::= "+" | "-"
```

Float immediates may carry a trailing modifier suffix (e.g.,
`4.789e-41.H0`) which is split off by the operand list's dot separator.

### Register

```
register       ::= reg_base ("." MODIFIER)*
reg_base       ::= sr_reg
                 | "UR" DIGITS            /* UR0..UR63 */
                 | "URZ"                  /* uniform zero */
                 | "R" DIGITS             /* R0..R255 */
                 | "RZ"                   /* zero register */
                 | "UP" DIGITS            /* UP0..UP6 */
                 | "UPT"                  /* uniform always-true pred */
                 | "PT"                   /* always-true pred */
                 | "P" DIGITS             /* P0..P6 */
                 | "B" DIGITS             /* barrier */
                 | "SRZ"                  /* special zero */

sr_reg         ::= "SR_" SR_NAME
SR_NAME        ::= ALNUM+ ("." LETTER)?   /* mixed case: SR_CgaCtaId, SR_TID.X */
```

Observed special registers: `SR_CTAID.X`, `SR_CTAID.Y`,
`SR_CTAID.Z`, `SR_CgaCtaId`, `SR_LANEID`, `SR_SWINHI`, `SR_TID.X`,
`SR_TID.Y`.

Observed modifiers: `reuse`, `64`, `F32`, `F32x2`, `H1`, `H0_H0`,
`H1_H1`, `HI_LO`, `abs` (synthetic), `neg` (synthetic).

### Mnemonic Word

Uppercase identifier in operand position (fallback for tokens not
matching any other pattern). Parsed as `ImmediateOp(raw=..., value=None)`.

```
mnem_word      ::= UPPER_LETTER UPPER_ALNUM* ("." UPPER_ALNUM+)*
```

Observed: `ALL`, `C`, `I`, `H0`, `H1`, `R`, `UP`.
