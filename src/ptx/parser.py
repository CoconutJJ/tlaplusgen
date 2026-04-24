"""PTX parser.

A pyparsing grammar for NVIDIA's Parallel Thread Execution (PTX) ISA,
derived from the CUDA PTX ISA manual:

    https://docs.nvidia.com/cuda/parallel-thread-execution/

The grammar accepts a full PTX source module -- module-level directives,
global variable declarations, and kernel/device function definitions --
and produces a :class:`ptx.ast.Module` tree directly via
``set_parse_action`` wiring. Instruction modifiers are captured as an
explicit enumeration (see :data:`_SINGLE_WORD_MODIFIERS` /
:data:`_QUALIFIED_MODIFIERS`) so unknown suffixes fail fast.

Public entry point:
    parse(text) -> :class:`ptx.ast.Module`
"""

from functools import reduce
from struct import pack, unpack

from pyparsing import (
    Char,
    Combine,
    Forward,
    Group,
    Keyword,
    Literal,
    MatchFirst,
    OneOrMore,
    Optional,
    ParserElement,
    QuotedString,
    Regex,
    StringEnd,
    Suppress,
    Word,
    ZeroOrMore,
    alphanums,
    alphas,
    c_style_comment,
    cpp_style_comment,
    delimited_list,
    hexnums,
    infix_notation,
    nested_expr,
    nums,
    opAssoc,
)

from . import ast as _ast

# Packrat memoisation keeps the mutually-recursive Expr / Statement rules fast.
ParserElement.enable_packrat()


# ===========================================================================
# Small helpers
# ===========================================================================


def _alt(words):
    """Ordered alternation of Keywords, longest-first.

    Sorting longest-first prevents a short keyword (e.g. ``add``) from
    short-matching a longer one (``addc``) under MatchFirst semantics.
    """
    return MatchFirst(
        [Keyword(w) for w in sorted(set(words), key=lambda w: (-len(w), w))]
    )


def Integers(start: int, end: int):
    """Match any decimal literal in the inclusive range [start, end].

    Retained from the original grammar; callers working with a single
    digit range should prefer :class:`pyparsing.Char` directly.
    """
    return reduce(
        lambda accum, x: accum | x,
        [Literal(str(c)) for c in range(start, end + 1)],
    )


# ===========================================================================
# Comments and punctuation
# ===========================================================================

Comment = cpp_style_comment | c_style_comment

LBRACE, RBRACE = Suppress("{"), Suppress("}")
LBRACK, RBRACK = Suppress("["), Suppress("]")
LPAREN, RPAREN = Suppress("("), Suppress(")")
COMMA, COLON, SEMI = Suppress(","), Suppress(":"), Suppress(";")


# ===========================================================================
# Identifiers and string literals (PTX ISA section 4.2)
# ===========================================================================

# followsym  := [a-zA-Z0-9_$]
# identifier := [a-zA-Z] followsym*  |  [_$%] followsym+
_FOLLOWSYM = alphanums + "_$"
Identifier = Combine(
    Word(alphas, _FOLLOWSYM) | (Char("_$%") + Word(_FOLLOWSYM))
)

# Parameterised variable declarator such as ``%r<10>`` which declares
# ``%r0`` through ``%r9`` in a single statement.
ParameterizedName = Combine(
    Identifier + Literal("<") + Word(nums) + Literal(">")
)

StringLiteral = QuotedString('"', esc_char="\\", multiline=False)


# ===========================================================================
# Numeric constants (PTX ISA section 4.3)
# ===========================================================================

_U = Suppress(Optional(Char("Uu")))

HexConstant = Combine(Literal("0") + Char("xX") + Word(hexnums)) + _U
HexConstant.set_parse_action(lambda t: int(t[0], 16))

BinaryConstant = Combine(Literal("0") + Char("bB") + Word("01")) + _U
BinaryConstant.set_parse_action(lambda t: int(t[0][2:], 2))

# Octal: leading ``0`` followed by one or more octal digits. Ordered after the
# hex and binary forms so a leading ``0x`` / ``0b`` isn't consumed here first.
OctalConstant = Combine(Literal("0") + Word("01234567")) + _U
OctalConstant.set_parse_action(lambda t: int(t[0], 8))

DecimalConstant = Combine(Char("123456789") + Optional(Word(nums))) + _U
DecimalConstant.set_parse_action(lambda t: int(t[0]))

# Bare ``0`` - not followed by ``x``, ``b``, or an octal digit.
ZeroConstant = Literal("0") + _U
ZeroConstant.set_parse_action(lambda t: 0)

IntegerConstant = (
    HexConstant
    | BinaryConstant
    | OctalConstant
    | DecimalConstant
    | ZeroConstant
)


def _hex_to_f32(hex_str: str) -> float:
    return unpack("<f", pack("<I", int(hex_str, 16)))[0]


def _hex_to_f64(hex_str: str) -> float:
    return unpack("<d", pack("<Q", int(hex_str, 16)))[0]


SingleFloatConstant = Combine(
    Suppress(Literal("0") + Char("fF")) + Word(hexnums, exact=8)
)
SingleFloatConstant.set_parse_action(lambda t: _hex_to_f32(t[0]))

DoubleFloatConstant = Combine(
    Suppress(Literal("0") + Char("dD")) + Word(hexnums, exact=16)
)
DoubleFloatConstant.set_parse_action(lambda t: _hex_to_f64(t[0]))

FloatingPointConstant = DoubleFloatConstant | SingleFloatConstant


# ===========================================================================
# Opcodes, directives, state spaces, types (PTX ISA chapters 5-9)
# ===========================================================================

INSTRUCTION_OPCODES = [
    "abs", "activemask", "add", "addc", "alloca", "and", "applypriority",
    "atom",
    "bar", "barrier", "bfe", "bfi", "bfind", "bmsk", "bra", "brev", "brkpt",
    "brx",
    "call", "clusterlaunchcontrol", "clz", "cnot", "copysign", "cos", "cp",
    "createpolicy", "cvt", "cvta",
    "discard", "div", "dp2a", "dp4a",
    "elect", "ex2", "exit",
    "fence", "fma", "fns",
    "getctarank", "griddepcontrol",
    "isspacep", "istypep",
    "ld", "ldmatrix", "ldu", "lg2", "lop3",
    "mad", "mad24", "madc", "mapa", "match", "max", "mbarrier", "membar",
    "min", "mma", "mov", "movmatrix", "mul", "mul24", "multimem",
    "nanosleep", "neg", "not",
    "or",
    "pmevent", "popc", "prefetch", "prefetchu", "prmt",
    "rcp", "red", "redux", "rem", "ret", "rsqrt",
    "sad", "selp", "set", "setmaxnreg", "setp", "shf", "shfl", "shl", "shr",
    "sin", "slct", "sqrt", "st", "stackrestore", "stacksave", "stmatrix",
    "sub", "subc", "suld", "suq", "sured", "sust", "szext",
    "tanh", "tcgen05", "tensormap", "testp", "tex", "tld4", "trap", "txq",
    "vabsdiff", "vabsdiff2", "vabsdiff4",
    "vadd", "vadd2", "vadd4",
    "vavrg2", "vavrg4",
    "vmad",
    "vmax", "vmax2", "vmax4",
    "vmin", "vmin2", "vmin4",
    "vote",
    "vset", "vset2", "vset4",
    "vshl", "vshr",
    "vsub", "vsub2", "vsub4",
    "wgmma", "wmma",
    "xor",
]

InstructionKeyword = _alt(INSTRUCTION_OPCODES)


DIRECTIVE_NAMES = [
    "address_size", "alias", "align",
    "branchtargets",
    "callprototype", "calltargets", "common", "const",
    "entry", "explicitcluster", "extern",
    "file", "func",
    "global",
    "loc", "local",
    "maxclusterrank", "maxnctapersm", "maxnreg", "maxntid", "minnctapersm",
    "noreturn",
    "param", "pragma",
    "reg", "reqnctapercluster", "reqntid",
    "section", "shared", "sreg",
    "target", "tex",
    "version", "visible",
    "weak",
]

DirectiveKeyword = Combine(Literal(".") + _alt(DIRECTIVE_NAMES))


STATE_SPACE_NAMES = [
    "reg", "sreg", "const", "global", "local", "param", "shared", "tex",
]
StateSpaceKeyword = Combine(Literal(".") + _alt(STATE_SPACE_NAMES))


TYPE_NAMES = [
    "s8", "s16", "s32", "s64",
    "u8", "u16", "u32", "u64",
    "f16", "f16x2", "f32", "f64",
    "b8", "b16", "b32", "b64", "b128",
    "pred",
]
TypeKeyword = Combine(Literal(".") + _alt(TYPE_NAMES))

VectorKeyword = Combine(
    Literal(".") + (Keyword("v8") | Keyword("v4") | Keyword("v2"))
)


# ===========================================================================
# Constant expressions (PTX ISA section 4.4)
# ===========================================================================

Expr = Forward()

Atom = (
    FloatingPointConstant
    | IntegerConstant
    | Identifier
    | Group(LPAREN + Expr + RPAREN)
)

Expr <<= infix_notation(
    Atom,
    [
        (Char("+-!~"), 1, opAssoc.RIGHT),
        (Char("*/%"), 2, opAssoc.LEFT),
        (Char("+-"), 2, opAssoc.LEFT),
        (Literal("<<") | Literal(">>"), 2, opAssoc.LEFT),
        (
            Literal("<=") | Literal(">=") | Literal("<") | Literal(">"),
            2,
            opAssoc.LEFT,
        ),
        (Literal("==") | Literal("!="), 2, opAssoc.LEFT),
        (Literal("&"), 2, opAssoc.LEFT),
        (Literal("^"), 2, opAssoc.LEFT),
        (Literal("|"), 2, opAssoc.LEFT),
        (Literal("&&"), 2, opAssoc.LEFT),
        (Literal("||"), 2, opAssoc.LEFT),
        ((Literal("?"), Literal(":")), 3, opAssoc.RIGHT),
    ],
)


# ===========================================================================
# Operands (PTX ISA section 6)
# ===========================================================================

# An operand-position name allows the ``.x`` / ``.y`` / ``.z`` / ``.w``
# vector-element suffix used to index vector registers (including the
# built-in special registers ``%tid``, ``%ntid``, ``%ctaid``, ...).
_VectorElementSuffix = Combine(Literal(".") + Char("xyzw"))
OperandName = Combine(Identifier + Optional(_VectorElementSuffix))
OperandName.set_parse_action(_ast.make_name)

# Braced vector / initialiser, e.g. ``{ %r1, %r2, %r3, %r4 }``.
BracedList = LBRACE + Optional(delimited_list(Expr | OperandName)) + RBRACE
BracedList.set_parse_action(_ast.make_braced_list)

# Address expression used inside ``[ ... ]`` for loads, stores, and atomics.
# Forms: ``[%rd1]``, ``[%rd1 + 4]``, ``[array + %rd2 - 8]``, ``[0x1000]``.
AddressExpr = (OperandName | IntegerConstant) + ZeroOrMore(
    (Literal("+") | Literal("-")) + (IntegerConstant | OperandName)
)
AddressOperand = LBRACK + AddressExpr + RBRACK
AddressOperand.set_parse_action(_ast.make_address)

# Parenthesised tuple operand, used for the return slot and argument
# list in ``call`` instructions.
TupleOperand = (
    LPAREN
    + Optional(delimited_list(OperandName | IntegerConstant))
    + RPAREN
)
TupleOperand.set_parse_action(_ast.make_tuple)

# Optionally signed numeric immediate. Kept separate from the general
# expression rule so the common case doesn't pay for operator parsing.
SignedImmediate = (
    Optional(Char("+-")) + (FloatingPointConstant | IntegerConstant)
)
SignedImmediate.set_parse_action(_ast.make_signed_immediate)

Operand = (
    BracedList
    | AddressOperand
    | TupleOperand
    | SignedImmediate
    | OperandName
)

OperandList = delimited_list(Operand)


# ===========================================================================
# Instruction modifiers (PTX ISA chapter 9)
#
# Every dotted suffix that may follow an opcode is enumerated here. The names
# are grouped by role (types, rounding, cache hints, ...) so they're easy to
# maintain against future ISA revisions. Compound ``::``-qualified modifiers
# are kept in a dedicated list and tried first under MatchFirst so the longer
# form is chosen over any prefix that happens to overlap.
# ===========================================================================

# -- Scalar / vector / packed types (PTX ISA section 4.5) ------------------
_MOD_TYPES = [
    "s8", "s16", "s32", "s64",
    "u8", "u16", "u32", "u64",
    "f16", "f16x2", "f32", "f64",
    "bf16", "bf16x2", "tf32",
    "e4m3", "e4m3x2", "e5m2", "e5m2x2", "e2m1", "e2m1x2", "e3m2", "e3m2x2",
    "b8", "b16", "b32", "b64", "b128",
    "pred",
]

# -- Vector widths, state spaces ------------------------------------------
_MOD_VECTOR = ["v2", "v4", "v8"]
_MOD_STATE_SPACES = [
    "reg", "sreg", "const", "global", "local", "param", "shared", "tex",
    "generic",
]

# -- Comparison operators (setp / set / selp / slct / testp / vset) --------
_MOD_COMPARE = [
    "eq", "ne", "lt", "le", "gt", "ge",
    "lo", "ls", "hi", "hs",
    "equ", "neu", "ltu", "leu", "gtu", "geu",
    "num", "nan",
    "finite", "infinite", "number", "notanumber", "normal", "subnormal",
]

# -- Boolean combinator on a predicate operand (setp) ---------------------
_MOD_BOOL = ["and", "or", "xor"]

# -- Rounding modes -------------------------------------------------------
_MOD_ROUNDING = [
    "rn", "rz", "rm", "rp", "rna",
    "rni", "rzi", "rmi", "rpi",
]

# -- Numeric flags attached to arithmetic instructions --------------------
_MOD_NUMERIC_FLAGS = [
    "sat", "ftz", "approx", "full", "wide",
    "cc",
    "abs", "relu", "NaN", "xorsign",
    "lo", "hi",  # mul/mad/dp4a/dp2a result half selectors
    "shiftamt",  # bfind
]

# -- Cache operators (ld/st/red/atom/prefetch) ----------------------------
_MOD_CACHE_OPS = [
    "ca", "cg", "cs", "lu", "cv",  # loads
    "wb", "wt",                     # stores
    "L1", "L2",                     # prefetch levels
]

# -- Memory ordering semantics and scopes (PTX ISA section 8.3) -----------
_MOD_MEMORY_SEMANTICS = [
    "weak", "relaxed", "acquire", "release", "acq_rel", "sc",
    "volatile", "mmio", "nc",
]
_MOD_SCOPES = ["cta", "cluster", "gpu", "sys"]

# -- Atomic and reduction operations --------------------------------------
_MOD_ATOMIC_OPS = ["exch", "inc", "dec", "cas"]
# .add/.and/.or/.xor/.min/.max are shared with boolean/arith names.

# -- Branch / call hints --------------------------------------------------
_MOD_BRANCH = ["uni", "to"]

# -- Warp-synchronous primitives (shfl/vote/match/redux) ------------------
_MOD_WARP = [
    "sync", "up", "down", "bfly", "idx",
    "all", "any", "ballot", "ball", "popc",
    "arrive", "red",
]

# -- Texture / surface geometry -------------------------------------------
_MOD_TEXTURE_GEOM = ["1d", "2d", "3d", "a1d", "a2d", "cube", "acube"]

# -- prmt modes -----------------------------------------------------------
_MOD_PRMT_MODES = ["f4e", "b4e", "rc8", "rc16", "ecl", "ecr"]

# -- Shift / mask behaviour (shf/szext/bmsk/bfe/bfi) ----------------------
_MOD_SHIFT = ["clamp", "wrap", "l", "r"]

# -- cp / cp.async / cp.bulk and related async primitives -----------------
_MOD_ASYNC = [
    "async", "bulk",
    "commit_group", "wait_group", "wait_all",
    "tile", "im2col", "im2col_w", "im2col_w_128",
    "tensor", "scatter", "gather",
    "multicast", "bulk_group",
    "fence", "commit", "reduce",
    "proxy",
]

# -- mbarrier (PTX ISA section 9.7.17) ------------------------------------
_MOD_MBARRIER = [
    "init", "inval", "arrive_drop",
    "test_wait", "try_wait",
    "pending_count", "expect_tx", "complete_tx",
]

# -- Grid / cluster launch dependency control -----------------------------
_MOD_GRID_CONTROL = [
    "wait", "launch_dependents",
    "try_cancel", "query_cancel",
]

# -- Matrix instructions (wmma/mma/wgmma/tcgen05) --------------------------
_MOD_MATRIX_LAYOUT = [
    "aligned", "trans", "row", "col",
    "load", "store", "mma", "mma_async",
    "a", "b", "c", "d",  # operand-role selectors on wmma/mma variants
]
_MOD_MATRIX_SHAPES = [
    "m8n8", "m16n8", "m16n16", "m32n8", "m8n32",
    "m8n8k4", "m8n8k16",
    "m8n16k8", "m8n16k16",
    "m16n8k8", "m16n8k16", "m16n8k32", "m16n8k64",
    "m16n16k8", "m16n16k16", "m16n16k32",
    "m32n8k16", "m8n32k16", "m32n8k8",
    "m64n8k4", "m64n16k4", "m64n32k4", "m64n64k4",
    "m64n8k8", "m64n16k8", "m64n32k8", "m64n64k8",
    "m64n8k16", "m64n16k16", "m64n32k16", "m64n64k16",
    "m64n128k16", "m64n256k16",
    "m64n8k32", "m64n16k32", "m64n32k32", "m64n64k32",
    "m64n128k32", "m64n256k32",
]

# -- tensormap / clusterlaunchcontrol / prefetch/applypriority helpers ----
_MOD_TENSORMAP = ["replace", "cp_fenceproxy"]

# -- ldmatrix / stmatrix / movmatrix ---------------------------------------
_MOD_MATRIX_MISC = ["x1", "x2", "x4", "num"]

# -- Packed / byte-selection modifiers on v-instructions -------------------
_MOD_PACKED_SELECTORS = [
    "h0", "h1",
    "b0", "b1", "b2", "b3",
    "po", "shr7", "shr15",
    "scale",
    "secop", "sec_half",
]

# -- Misc / small singletons that don't fit above but are valid per ISA ---
_MOD_MISC = [
    "bias",
    "pack",
    "shift",
    "alloc", "dealloc",
    "mode",
    "channel",
    "stochastic",
    "entry", "func",
]

# Single-word modifiers: every flat (no ``::``) token we accept.
_SINGLE_WORD_MODIFIERS = sorted(set(
    _MOD_TYPES
    + _MOD_VECTOR
    + _MOD_STATE_SPACES
    + _MOD_COMPARE
    + _MOD_BOOL
    + _MOD_ROUNDING
    + _MOD_NUMERIC_FLAGS
    + _MOD_CACHE_OPS
    + _MOD_MEMORY_SEMANTICS
    + _MOD_SCOPES
    + _MOD_ATOMIC_OPS
    + _MOD_BRANCH
    + _MOD_WARP
    + _MOD_TEXTURE_GEOM
    + _MOD_PRMT_MODES
    + _MOD_SHIFT
    + _MOD_ASYNC
    + _MOD_MBARRIER
    + _MOD_GRID_CONTROL
    + _MOD_MATRIX_LAYOUT
    + _MOD_MATRIX_SHAPES
    + _MOD_TENSORMAP
    + _MOD_MATRIX_MISC
    + _MOD_PACKED_SELECTORS
    + _MOD_MISC
    # Opcode-like names that also show up as atomic/reduction op modifiers.
    + ["add", "sub", "mul", "div", "min", "max", "shl", "shr"]
))

# ``::``-qualified compound modifiers. The list is explicit so unknown forms
# fail fast rather than silently accepting a malformed suffix. Longest-first
# sorting (done inside ``_alt``) picks ``A::B::C`` over ``A::B`` when both
# would match the same prefix.
_QUALIFIED_MODIFIERS = [
    # Cache-level qualifiers with eviction / hint refinements.
    "L1::no_allocate",
    "L2::64B", "L2::128B", "L2::256B",
    "L2::cache_hint", "L2::no_allocate",
    "L2::evict_normal", "L2::evict_last",
    "L2::evict_first", "L2::evict_unchanged",
    # mbarrier operation qualifiers.
    "mbarrier::arrive",
    "mbarrier::arrive::one",
    "mbarrier::arrive::expect_tx",
    "mbarrier::arrive_drop::expect_tx",
    "mbarrier::complete_tx",
    "mbarrier::complete_tx::bytes",
    "mbarrier::proxy",
    "mbarrier::init",
    # Cluster-scope qualifiers.
    "multicast::cluster",
    "multicast::cluster::all",
    # cta_group selector (wgmma/tcgen05).
    "cta_group::1", "cta_group::2",
    # Fence / proxy scopes.
    "proxy::async",
    "proxy::async::global",
    "proxy::async::shared::cta",
    "proxy::async::shared::cluster",
    # Async bulk copy qualifiers.
    "async::bulk",
    "async::bulk_group",
    "async::bulk::tensor",
    "async::bulk::tensor::tile",
    "async::bulk::tensor::im2col",
    # State-space refinements.
    "shared::cta", "shared::cluster",
    "param::entry", "param::func",
    # Semantic / level qualifiers used on ld/st/prefetch/applypriority.
    "sem::release", "sem::acquire", "sem::relaxed", "sem::acq_rel",
    "level::eviction_priority",
    "level::cache_hint",
    # tcgen05 packing selectors.
    "pack::16b", "pack::8b",
    # cvt rs selector (stochastic rounding).
    "rs::stochastic",
    # Warp-synchronous restriction variants (PTX 8.x).
    "sync_restrict::shared::cta",
    "sync_restrict::shared::cluster",
    # cp / cp.async / cp.reduce refinements.
    "cp_async::bulk",
    "cp_async::bulk::tensor",
]

Modifier = Combine(
    Literal(".")
    + (_alt(_QUALIFIED_MODIFIERS) | _alt(_SINGLE_WORD_MODIFIERS))
)


# ===========================================================================
# Instructions (PTX ISA section 9)
# ===========================================================================

# Predicate guard: ``@p`` or ``@!p``.
PredicateGuard = Combine(Literal("@") + Optional(Literal("!")) + Identifier)
PredicateGuard.set_parse_action(_ast.make_predicate_guard)

Instruction = (
    Optional(PredicateGuard)
    + InstructionKeyword
    + ZeroOrMore(Modifier)
    + Optional(OperandList)
    + SEMI
)
Instruction.set_parse_action(_ast.make_instruction)


# ===========================================================================
# Variable declarations (PTX ISA section 5)
# ===========================================================================

Linking = Combine(
    Literal(".")
    + (
        Keyword("visible")
        | Keyword("extern")
        | Keyword("weak")
        | Keyword("common")
    )
)

Alignment = (
    Suppress(Combine(Literal(".") + Keyword("align"))) + IntegerConstant
)

# Array suffix: ``buf[32]``, ``buf[]`` (inferred size), or multi-dim ``buf[4][8]``.
ArraySuffix = OneOrMore(LBRACK + Optional(IntegerConstant) + RBRACK)

# Variable initialisers may be nested braced lists, constant expressions,
# or string literals (the latter is used by some toolchains for debug data).
Initializer = Forward()
_BracedInitializer = LBRACE + Optional(delimited_list(Initializer)) + RBRACE
_BracedInitializer.set_parse_action(_ast.make_braced_initializer)
Initializer <<= _BracedInitializer | Expr | StringLiteral

Declarator = (
    (ParameterizedName | Identifier)
    + Optional(ArraySuffix)
    + Optional(Suppress("=") + Initializer)
)
Declarator.set_parse_action(_ast.make_declarator)

VariableDeclaration = (
    Optional(Linking)
    + StateSpaceKeyword
    + Optional(Alignment)
    + Optional(VectorKeyword)
    + TypeKeyword
    + delimited_list(Declarator)
    + SEMI
)
VariableDeclaration.set_parse_action(_ast.make_variable)


# ===========================================================================
# Function declarations and definitions (PTX ISA section 11)
# ===========================================================================

# A single parameter slot in a function signature. PTX allows any state
# space here (kernels use ``.param``; device-function returns use ``.reg``).
ParameterDeclaration = (
    StateSpaceKeyword
    + Optional(Alignment)
    + Optional(VectorKeyword)
    + TypeKeyword
    + (ParameterizedName | Identifier)
    + Optional(ArraySuffix)
)
ParameterDeclaration.set_parse_action(_ast.make_parameter)

# The ``(args)`` list following a function name. A separate rule instance
# is used for return slots (below) so the two can carry distinct parse
# actions -- the function factory then knows which role each list plays.
ParameterList = LPAREN + Optional(delimited_list(ParameterDeclaration)) + RPAREN
ParameterList.set_parse_action(_ast.make_parameter_list)

ReturnParameterList = LPAREN + delimited_list(ParameterDeclaration) + RPAREN
ReturnParameterList.set_parse_action(_ast.make_return_parameter_list)

# Kernels (.entry) never return values. Device functions (.func) optionally
# return a parenthesised list of register slots before the function name.
EntryKind = Combine(Literal(".") + Keyword("entry"))
FuncKind = Combine(Literal(".") + Keyword("func"))

# Performance-tuning and attribute decorators attached to a function
# signature (launch bounds, cluster shape, register limits, pragmas).
_DECORATOR_NAMES = [
    "maxntid", "reqntid", "minnctapersm", "maxnctapersm", "maxnreg",
    "explicitcluster", "maxclusterrank", "reqnctapercluster",
    "noreturn", "pragma",
]
Decorator = (
    Combine(Literal(".") + _alt(_DECORATOR_NAMES))
    + Optional(delimited_list(Expr | StringLiteral))
)
Decorator.set_parse_action(_ast.make_decorator)

# -- Statements permitted inside a function body ---------------------------

Statement = Forward()

StatementBlock = LBRACE + ZeroOrMore(Statement) + RBRACE
StatementBlock.set_parse_action(_ast.make_statement_block)

Label = Identifier + COLON
Label.set_parse_action(_ast.make_label)

LocDirective = (
    Combine(Literal(".") + Keyword("loc"))
    + IntegerConstant  # file index
    + IntegerConstant  # line
    + IntegerConstant  # column
    + ZeroOrMore(
        COMMA
        + Word(alphas, _FOLLOWSYM)
        + OneOrMore(IntegerConstant | StringLiteral | Identifier)
    )
)
LocDirective.set_parse_action(_ast.make_ignored_directive)

PragmaDirective = (
    Combine(Literal(".") + Keyword("pragma"))
    + delimited_list(StringLiteral)
    + SEMI
)
PragmaDirective.set_parse_action(_ast.make_ignored_directive)

CallTargetsDirective = (
    Combine(Literal(".") + Keyword("calltargets"))
    + delimited_list(Identifier)
    + SEMI
)
CallTargetsDirective.set_parse_action(_ast.make_ignored_directive)

BranchTargetsDirective = (
    Combine(Literal(".") + Keyword("branchtargets"))
    + delimited_list(Identifier)
    + SEMI
)
BranchTargetsDirective.set_parse_action(_ast.make_ignored_directive)

# ``proto_name : .callprototype (ret) _ (params);``
CallPrototypeDirective = (
    Identifier
    + COLON
    + Combine(Literal(".") + Keyword("callprototype"))
    + Optional(ParameterList)
    + Optional(Suppress(Literal("_")))
    + Optional(ParameterList)
    + SEMI
)
CallPrototypeDirective.set_parse_action(_ast.make_ignored_directive)

Statement <<= (
    StatementBlock
    | LocDirective
    | PragmaDirective
    | CallTargetsDirective
    | BranchTargetsDirective
    | CallPrototypeDirective
    | VariableDeclaration
    | Label
    | Instruction
)

# -- Function signatures and bodies ----------------------------------------

# .entry and .func have slightly different shapes, so keep them separate.
_EntrySignature = (
    Optional(Linking)
    + EntryKind
    + Identifier
    + Optional(ParameterList)
    + ZeroOrMore(Decorator)
)

_FuncSignature = (
    Optional(Linking)
    + FuncKind
    + Optional(ReturnParameterList)
    + Identifier
    + Optional(ParameterList)
    + ZeroOrMore(Decorator)
)

FunctionSignature = _EntrySignature | _FuncSignature

FunctionDefinition = FunctionSignature + StatementBlock
FunctionDefinition.set_parse_action(_ast.make_function_definition)

FunctionDeclaration = FunctionSignature + SEMI
FunctionDeclaration.set_parse_action(_ast.make_function_declaration)

Function = FunctionDefinition | FunctionDeclaration


# ===========================================================================
# Module-level directives (PTX ISA section 3)
# ===========================================================================

VersionDirective = (
    Combine(Literal(".") + Keyword("version")) + Regex(r"\d+\.\d+")
)
VersionDirective.set_parse_action(_ast.make_version)

TargetDirective = (
    Combine(Literal(".") + Keyword("target"))
    + delimited_list(Word(alphanums + "_"))
)
TargetDirective.set_parse_action(_ast.make_target)

AddressSizeDirective = (
    Combine(Literal(".") + Keyword("address_size")) + IntegerConstant
)
AddressSizeDirective.set_parse_action(_ast.make_address_size)

FileDirective = (
    Combine(Literal(".") + Keyword("file"))
    + IntegerConstant
    + StringLiteral
    + Optional(COMMA + IntegerConstant + COMMA + IntegerConstant)
)
FileDirective.set_parse_action(_ast.make_file)

AliasDirective = (
    Combine(Literal(".") + Keyword("alias"))
    + Identifier
    + COMMA
    + Identifier
    + SEMI
)
AliasDirective.set_parse_action(_ast.make_ignored_directive)

# ``.section`` is reserved for DWARF debug info. Its body contains byte
# literals and nested type descriptors that are uninteresting to the rest
# of this pipeline, so we consume the whole balanced block opaquely.
SectionDirective = (
    Combine(Literal(".") + Keyword("section"))
    + Regex(r"\.?[A-Za-z_][A-Za-z0-9_]*")
    + nested_expr("{", "}")
)
SectionDirective.set_parse_action(_ast.make_ignored_directive)

ModuleDirective = (
    VersionDirective
    | TargetDirective
    | AddressSizeDirective
    | FileDirective
    | AliasDirective
    | SectionDirective
    | PragmaDirective
    | LocDirective
)


# ===========================================================================
# Top-level module
# ===========================================================================

TopLevelItem = ModuleDirective | Function | VariableDeclaration

ModuleGrammar = ZeroOrMore(TopLevelItem) + StringEnd()
ModuleGrammar.set_parse_action(_ast.make_module)
ModuleGrammar.ignore(Comment)


# ===========================================================================
# Public API
# ===========================================================================


def parse(text: str):
    """Parse a PTX source module and return a :class:`ptx.ast.Module`."""
    return ModuleGrammar.parse_string(text, parse_all=True)[0]
