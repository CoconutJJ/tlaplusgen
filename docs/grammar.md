# NVIDIA SASS Grammar

BNF grammar for the cleaned SASS assembly format parsed by `src/sass/parser.py`.

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
pred_name      ::= "UP" DIGITS | "UPT" | "PT" | "P" DIGITS | "B" DIGITS

mnemonic       ::= UPPER_LETTER UPPER_ALNUM* ("." UPPER_ALNUM+)*
```

## Operand List

Operands are usually comma-separated but may also be space-separated
(e.g., BRX `R5 -0x50`). Commas are treated as optional separators.

```
operand_list   ::= (operand | ",")*
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

```
label_ref      ::= "`(" "." IDENT ")"
annotation     ::= '(*"' TEXT '"*)'
desc_op        ::= desc_kind ("[" INNER "]")+
desc_kind      ::= "desc" | "gdesc" | "tmem" | "idesc"
const_bank     ::= "c" "[" HEX_NUM "]" "[" INNER "]"
mem_addr       ::= "[" INNER "]"
abs_reg        ::= "|" register "|"
neg_pred       ::= "!" pred_name
neg_reg        ::= "-" register
special_imm    ::= SIGN? ("INF" | "+INF" | "-INF" | "QNAN" | "-QNAN" | "NaN")
hex_imm        ::= "-"? "0x" HEX_DIGITS+
float_imm      ::= SIGN? (DIGITS "." DIGITS EXP? | "." DIGITS EXP?)
int_imm        ::= "-"? DIGITS
register       ::= reg_base ("." MODIFIER)*
reg_base       ::= "SR_" SR_NAME | "U"? "R" "Z"? DIGITS?
                 | "UP" DIGITS | "UPT" | "PT" | "P" DIGITS
                 | "B" DIGITS | "SRZ"
mnem_word      ::= UPPER_LETTER UPPER_ALNUM* ("." UPPER_ALNUM+)*

EXP            ::= ("e" | "E") SIGN? DIGITS
SIGN           ::= "+" | "-"
SR_NAME        ::= UPPER_ALNUM+ ("." UPPER_LETTER)?
```
