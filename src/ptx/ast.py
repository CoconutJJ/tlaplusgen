"""PTX abstract syntax tree.

Defines the classes that represent a parsed PTX module and the factory
helpers that :mod:`ptx.parser` binds to grammar rules via
``set_parse_action``. Every grammar rule that corresponds to a tree node
has a ``make_*`` function here; the parser attaches those functions so
the ``ParseResults`` tree is transformed into :class:`Module` bottom-up,
while parsing.

Usage::

    from ptx import ast
    module = ast.parse(source_text)
    for fn in module.functions:
        for block in fn.blocks:
            for inst in block.instructions:
                ...

The top-level types -- :class:`Module`, :class:`Function`,
:class:`BasicBlock`, :class:`Instruction`, :class:`Operand` and its
subclasses, :class:`Variable`, :class:`Parameter`, :class:`Decorator` --
form the public API. A small number of ``_PrivateMarker`` dataclasses
exist only so one parse action can hand a structured result to the next;
they are implementation details and should not appear in any public
output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union


# ===========================================================================
# Operands
# ===========================================================================


@dataclass
class Operand:
    """Abstract operand base class."""


@dataclass
class Name(Operand):
    """Identifier-shaped operand: register, label, or function reference.

    ``element`` captures the ``.x`` / ``.y`` / ``.z`` / ``.w`` vector-element
    suffix used on operands like ``%tid.x``.
    """

    name: str
    element: Optional[str] = None

    def __str__(self) -> str:
        return self.name + (self.element or "")


@dataclass
class Immediate(Operand):
    """A numeric literal operand."""

    value: Union[int, float]
    negative: bool = False

    def __str__(self) -> str:
        return repr(-self.value if self.negative else self.value)


@dataclass
class AddressTerm:
    """One ``[+-] value`` component of an address expression."""

    sign: str            # '+' or '-'
    value: Any           # identifier string or numeric immediate


@dataclass
class AddressOperand(Operand):
    """Memory address operand: contents of ``[ ... ]``."""

    terms: List[AddressTerm] = field(default_factory=list)

    def __str__(self) -> str:
        parts: List[str] = []
        for i, t in enumerate(self.terms):
            if i == 0:
                parts.append(f"-{t.value}" if t.sign == "-" else str(t.value))
            else:
                parts.append(f"{t.sign}{t.value}")
        return "[" + "".join(parts) + "]"


@dataclass
class BracedListOperand(Operand):
    """Vector / initialiser list operand: ``{ a, b, c, d }``."""

    elements: List[Operand] = field(default_factory=list)


@dataclass
class TupleOperand(Operand):
    """Parenthesised operand tuple, used by ``call`` for return / arg lists."""

    elements: List[Operand] = field(default_factory=list)


# ===========================================================================
# Instructions
# ===========================================================================


@dataclass
class PredicateGuard:
    """``@p`` or ``@!p`` guard preceding an instruction."""

    name: str
    negated: bool = False

    def __str__(self) -> str:
        return "@" + ("!" if self.negated else "") + self.name


#: Opcodes that unconditionally transfer control out of their basic block.
TERMINATOR_OPCODES = frozenset({"bra", "brx", "ret", "exit", "trap"})


@dataclass
class Instruction:
    opcode: str
    modifiers: List[str] = field(default_factory=list)
    operands: List[Operand] = field(default_factory=list)
    predicate: Optional[PredicateGuard] = None

    @property
    def is_terminator(self) -> bool:
        return self.opcode in TERMINATOR_OPCODES

    def __str__(self) -> str:
        parts: List[str] = []
        if self.predicate is not None:
            parts.append(str(self.predicate))
        parts.append(self.opcode + "".join(self.modifiers))
        if self.operands:
            parts.append(", ".join(str(o) for o in self.operands))
        return " ".join(parts) + ";"


# ===========================================================================
# Declarations (variables and parameters)
# ===========================================================================


@dataclass
class Declarator:
    """One name + optional shape + optional initialiser in a declaration."""

    name: str
    count: Optional[int] = None                        # ``%r<10>`` -> 10
    array_shape: List[Optional[int]] = field(default_factory=list)
    initializer: Any = None


@dataclass
class Variable:
    state_space: str
    type: str
    declarators: List[Declarator] = field(default_factory=list)
    linking: Optional[str] = None
    alignment: Optional[int] = None
    vector: Optional[str] = None  # ``.v2`` / ``.v4`` / ``.v8``


@dataclass
class Parameter(Variable):
    """Function parameter slot. Same shape as :class:`Variable`; kept as a
    separate class so passes can filter by role."""


# ===========================================================================
# Functions and basic blocks
# ===========================================================================


@dataclass
class BasicBlock:
    label: Optional[str] = None
    instructions: List[Instruction] = field(default_factory=list)

    @property
    def terminator(self) -> Optional[Instruction]:
        if self.instructions and self.instructions[-1].is_terminator:
            return self.instructions[-1]
        return None


@dataclass
class Decorator:
    name: str                   # includes the leading ``.``
    args: List[Any] = field(default_factory=list)


@dataclass
class Function:
    name: str
    kind: str                                           # "entry" or "func"
    parameters: List[Parameter] = field(default_factory=list)
    return_parameters: List[Parameter] = field(default_factory=list)
    decorators: List[Decorator] = field(default_factory=list)
    linking: Optional[str] = None
    is_definition: bool = True
    locals: List[Variable] = field(default_factory=list)
    blocks: List[BasicBlock] = field(default_factory=list)

    def block_by_label(self, label: str) -> Optional[BasicBlock]:
        for bb in self.blocks:
            if bb.label == label:
                return bb
        return None


# ===========================================================================
# Module
# ===========================================================================


@dataclass
class Module:
    version: Optional[str] = None
    target: List[str] = field(default_factory=list)
    address_size: Optional[int] = None
    globals: List[Variable] = field(default_factory=list)
    functions: List[Function] = field(default_factory=list)
    file_entries: List[Tuple[int, str]] = field(default_factory=list)

    def get_function(self, name: str) -> Optional[Function]:
        for fn in self.functions:
            if fn.name == name:
                return fn
        return None


# ===========================================================================
# Internal markers used to pipeline intermediate results between rules
# ===========================================================================
#
# These carry information up from one parse action to the next. They exist
# so that, for instance, the ParameterList parse action can tell the
# Function parse action "these are the args, not the return slots" without
# relying on positional guesswork. None of these types should leak out
# through the final :class:`Module`.


@dataclass
class _LabelMarker:
    """A label statement, consumed by the statement-block builder."""

    name: str


@dataclass
class _StatementBlock:
    """Function body: local declarations plus instruction basic blocks."""

    locals: List[Variable] = field(default_factory=list)
    blocks: List[BasicBlock] = field(default_factory=list)


@dataclass
class _ParameterList:
    """Argument list: the ``(.param ...)`` group that follows a function name."""

    params: List[Parameter] = field(default_factory=list)


@dataclass
class _ReturnParameterList:
    """Return slot list: the ``(.reg ...)`` group preceding a ``.func`` name."""

    params: List[Parameter] = field(default_factory=list)


@dataclass
class _VersionInfo:
    value: str


@dataclass
class _TargetInfo:
    targets: List[str] = field(default_factory=list)


@dataclass
class _AddressSizeInfo:
    bits: int


@dataclass
class _FileInfo:
    index: int
    name: str


@dataclass
class _IgnoredDirective:
    """Statement-level directive we don't currently materialise (``.loc``,
    ``.pragma``, ``.calltargets``, ``.branchtargets``, ``.callprototype``)."""


# ===========================================================================
# Factory helpers invoked via ``set_parse_action`` in parser.py
# ===========================================================================


_LINKING = frozenset({".visible", ".extern", ".weak", ".common"})
_STATE_SPACES = frozenset({
    ".reg", ".sreg", ".const", ".global", ".local",
    ".param", ".shared", ".tex",
})
_VECTORS = frozenset({".v2", ".v4", ".v8"})


def _split_parameterised(name: str) -> Tuple[str, Optional[int]]:
    """``%r<10>`` -> ``("%r", 10)``; plain names are returned unchanged."""
    if "<" in name and name.endswith(">"):
        base, _, n = name[:-1].partition("<")
        try:
            return base, int(n)
        except ValueError:
            pass
    return name, None


def _split_vector_element(raw: str) -> Tuple[str, Optional[str]]:
    """``%tid.x`` -> ``("%tid", ".x")``; non-vector names pass through."""
    if not raw.startswith(".") and "." in raw:
        base, _, suffix = raw.rpartition(".")
        if suffix in ("x", "y", "z", "w") and base:
            return base, f".{suffix}"
    return raw, None


# ---------------------------------------------------------------------------
# Operand factories
# ---------------------------------------------------------------------------


def make_name(tokens) -> Name:
    """Build a :class:`Name` from the combined identifier string."""
    base, element = _split_vector_element(tokens[0])
    return Name(name=base, element=element)


def make_predicate_guard(tokens) -> PredicateGuard:
    raw = tokens[0]
    negated = raw.startswith("@!")
    return PredicateGuard(name=raw[2:] if negated else raw[1:], negated=negated)


def make_signed_immediate(tokens) -> Immediate:
    items = list(tokens)
    if len(items) == 1:
        return Immediate(value=items[0])
    return Immediate(value=items[1], negative=(items[0] == "-"))


def _coerce_operand(tok: Any) -> Any:
    """Wrap raw numeric / identifier tokens as Operand AST nodes."""
    if isinstance(tok, Operand):
        return tok
    if isinstance(tok, bool):
        return Immediate(value=int(tok))
    if isinstance(tok, (int, float)):
        return Immediate(value=tok)
    if isinstance(tok, str):
        base, element = _split_vector_element(tok)
        return Name(name=base, element=element)
    return tok  # Expr subtree or similar: pass through.


def make_braced_list(tokens) -> BracedListOperand:
    return BracedListOperand(elements=[_coerce_operand(t) for t in tokens])


def make_tuple(tokens) -> TupleOperand:
    return TupleOperand(elements=[_coerce_operand(t) for t in tokens])


def _address_operand_value(v: Any) -> Any:
    """Flatten a :class:`Name` to its source form for storage in an
    :class:`AddressTerm` (addresses don't need the nested operand object)."""
    if isinstance(v, Name):
        return v.name + (v.element or "")
    return v


def make_address(tokens) -> AddressOperand:
    items = list(tokens)
    if not items:
        return AddressOperand()
    terms = [AddressTerm(sign="+", value=_address_operand_value(items[0]))]
    i = 1
    while i + 1 < len(items):
        terms.append(
            AddressTerm(
                sign=str(items[i]),
                value=_address_operand_value(items[i + 1]),
            )
        )
        i += 2
    return AddressOperand(terms=terms)


# ---------------------------------------------------------------------------
# Declarator / variable factories
# ---------------------------------------------------------------------------


def make_declarator(tokens) -> Declarator:
    items = list(tokens)
    base, count = _split_parameterised(items[0])
    shape: List[Optional[int]] = []
    initializer: Any = None
    for tok in items[1:]:
        if tok is None or isinstance(tok, int):
            shape.append(tok)
        else:
            initializer = tok
    return Declarator(
        name=base, count=count, array_shape=shape, initializer=initializer
    )


def make_braced_initializer(tokens) -> list:
    """Produce a Python list for ``= { v1, v2, ... }`` initialisers.

    Wrapped in an outer single-item list so pyparsing treats the inner
    Python ``list`` as one token rather than splaying its contents.
    """
    return [list(tokens)]


def _classify_decl_prefix(group) -> Tuple[Optional[str], str, Optional[int], Optional[str], str, List[Any]]:
    """Scan a (Variable|Parameter)Declaration token stream into its prefix
    parts (linking, state space, alignment, vector, type) and the trailing
    tokens that describe its declarator(s)."""
    linking: Optional[str] = None
    state_space = ""
    alignment: Optional[int] = None
    vector: Optional[str] = None
    type_name = ""
    trailing: List[Any] = []
    for tok in group:
        if isinstance(tok, str) and tok in _LINKING:
            linking = tok
        elif isinstance(tok, str) and tok in _STATE_SPACES:
            state_space = tok
        elif isinstance(tok, str) and tok in _VECTORS:
            vector = tok
        elif isinstance(tok, str) and tok.startswith(".") and not type_name:
            type_name = tok
        elif isinstance(tok, int) and not type_name:
            alignment = tok
        else:
            trailing.append(tok)
    return linking, state_space, alignment, vector, type_name, trailing


def make_variable(tokens) -> Variable:
    """Factory for ``VariableDeclaration`` (trailing is a list of declarator
    objects built by :func:`make_declarator`)."""
    linking, state_space, alignment, vector, type_name, trailing = _classify_decl_prefix(tokens)
    declarators = [t for t in trailing if isinstance(t, Declarator)]
    return Variable(
        state_space=state_space,
        type=type_name,
        declarators=declarators,
        linking=linking,
        alignment=alignment,
        vector=vector,
    )


def make_parameter(tokens) -> Parameter:
    """Factory for ``ParameterDeclaration`` (trailing is a bare name followed
    by any array-shape integers from the suffix)."""
    linking, state_space, alignment, vector, type_name, trailing = _classify_decl_prefix(tokens)
    name: Optional[str] = None
    shape: List[Optional[int]] = []
    for tok in trailing:
        if isinstance(tok, str) and name is None:
            name = tok
        elif tok is None or isinstance(tok, int):
            shape.append(tok)
    declarators: List[Declarator] = []
    if name is not None:
        base, count = _split_parameterised(name)
        declarators.append(
            Declarator(name=base, count=count, array_shape=shape)
        )
    return Parameter(
        state_space=state_space,
        type=type_name,
        declarators=declarators,
        linking=linking,
        alignment=alignment,
        vector=vector,
    )


# ---------------------------------------------------------------------------
# Statement / function / module factories
# ---------------------------------------------------------------------------


def make_decorator(tokens) -> Decorator:
    items = list(tokens)
    return Decorator(name=items[0], args=list(items[1:]))


def make_label(tokens) -> _LabelMarker:
    return _LabelMarker(name=tokens[0])


def make_instruction(tokens) -> Instruction:
    items = list(tokens)
    idx = 0
    predicate: Optional[PredicateGuard] = None
    if items and isinstance(items[0], PredicateGuard):
        predicate = items[0]
        idx = 1
    if idx >= len(items):
        raise ValueError("instruction missing opcode")
    opcode = items[idx]
    idx += 1

    modifiers: List[str] = []
    while idx < len(items) and isinstance(items[idx], str) and items[idx].startswith("."):
        modifiers.append(items[idx])
        idx += 1

    operands: List[Operand] = []
    for tok in items[idx:]:
        if isinstance(tok, Operand):
            operands.append(tok)
        elif isinstance(tok, bool):
            operands.append(Immediate(value=int(tok)))
        elif isinstance(tok, (int, float)):
            operands.append(Immediate(value=tok))
        elif isinstance(tok, str):
            base, element = _split_vector_element(tok)
            operands.append(Name(name=base, element=element))
    return Instruction(
        opcode=opcode,
        modifiers=modifiers,
        operands=operands,
        predicate=predicate,
    )


def make_parameter_list(tokens) -> _ParameterList:
    return _ParameterList(params=[t for t in tokens if isinstance(t, Parameter)])


def make_return_parameter_list(tokens) -> _ReturnParameterList:
    return _ReturnParameterList(
        params=[t for t in tokens if isinstance(t, Parameter)]
    )


def make_ignored_directive(tokens) -> _IgnoredDirective:
    return _IgnoredDirective()


def make_statement_block(tokens) -> _StatementBlock:
    """Flatten a brace-delimited statement list into locals + basic blocks."""
    locals_: List[Variable] = []
    blocks: List[BasicBlock] = []
    current = BasicBlock()

    def flush() -> None:
        nonlocal current
        if current.instructions or current.label is not None:
            blocks.append(current)
        current = BasicBlock()

    for item in tokens:
        if isinstance(item, _LabelMarker):
            flush()
            current.label = item.name
        elif isinstance(item, Instruction):
            current.instructions.append(item)
            if item.is_terminator:
                flush()
        elif isinstance(item, Parameter):
            # Shouldn't occur -- parameters never appear in bodies.
            continue
        elif isinstance(item, Variable):
            locals_.append(item)
        elif isinstance(item, _StatementBlock):
            locals_.extend(item.locals)
            blocks.extend(item.blocks)
            flush()
        # _IgnoredDirective and any unrecognised tokens are dropped.

    if current.instructions or current.label is not None:
        blocks.append(current)
    return _StatementBlock(locals=locals_, blocks=blocks)


def _build_function(tokens, is_definition: bool) -> Function:
    items = list(tokens)
    idx = 0

    linking: Optional[str] = None
    if idx < len(items) and isinstance(items[idx], str) and items[idx] in _LINKING:
        linking = items[idx]
        idx += 1

    if idx >= len(items) or items[idx] not in (".entry", ".func"):
        raise ValueError("function group missing .entry / .func marker")
    kind = items[idx][1:]  # drop the leading '.'
    idx += 1

    return_params: List[Parameter] = []
    if idx < len(items) and isinstance(items[idx], _ReturnParameterList):
        return_params = items[idx].params
        idx += 1

    if idx >= len(items) or not isinstance(items[idx], str):
        raise ValueError("function group missing name")
    name = items[idx]
    idx += 1

    parameters: List[Parameter] = []
    if idx < len(items) and isinstance(items[idx], _ParameterList):
        parameters = items[idx].params
        idx += 1

    decorators: List[Decorator] = []
    body: Optional[_StatementBlock] = None
    while idx < len(items):
        it = items[idx]
        if isinstance(it, Decorator):
            decorators.append(it)
        elif isinstance(it, _StatementBlock):
            body = it
        idx += 1

    fn = Function(
        name=name,
        kind=kind,
        parameters=parameters,
        return_parameters=return_params,
        decorators=decorators,
        linking=linking,
        is_definition=is_definition and body is not None,
    )
    if body is not None:
        fn.locals = body.locals
        fn.blocks = body.blocks
    return fn


def make_function_definition(tokens) -> Function:
    return _build_function(tokens, is_definition=True)


def make_function_declaration(tokens) -> Function:
    return _build_function(tokens, is_definition=False)


def make_version(tokens) -> _VersionInfo:
    return _VersionInfo(value=tokens[1])


def make_target(tokens) -> _TargetInfo:
    return _TargetInfo(targets=[t for t in tokens[1:] if isinstance(t, str)])


def make_address_size(tokens) -> _AddressSizeInfo:
    return _AddressSizeInfo(bits=tokens[1])


def make_file(tokens) -> _FileInfo:
    return _FileInfo(index=tokens[1], name=tokens[2])


def make_module(tokens) -> Module:
    module = Module()
    for item in tokens:
        if isinstance(item, _VersionInfo):
            module.version = item.value
        elif isinstance(item, _TargetInfo):
            module.target = item.targets
        elif isinstance(item, _AddressSizeInfo):
            module.address_size = item.bits
        elif isinstance(item, _FileInfo):
            module.file_entries.append((item.index, item.name))
        elif isinstance(item, Function):
            module.functions.append(item)
        elif isinstance(item, Parameter):
            continue  # parameters never float to module scope
        elif isinstance(item, Variable):
            module.globals.append(item)
        # _IgnoredDirective and other markers are intentionally dropped.
    return module


# ===========================================================================
# Convenience wrapper
# ===========================================================================


def parse(text: str) -> Module:
    """Parse PTX source text into a :class:`Module`."""
    from . import parser  # deferred -- parser imports this module at load.

    return parser.parse(text)
