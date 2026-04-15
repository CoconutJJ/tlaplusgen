from functools import reduce
from abc import ABC
from types import UnionType
from typing import Any

type MappingIndex = str | int
type MappingValue = Expr


class TLAType:
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    @staticmethod
    def fromNative(value: object):

        if isinstance(value, Literal):
            value = value.value

        if isinstance(value, int):
            return TLAInt()
        elif isinstance(value, str):
            return TLAStr()
        elif isinstance(value, bool):
            return TLABool()

        raise ValueError("Invalid value type")

    def __str__(self) -> str:
        return self.name


class TLAInt(TLAType):
    def __init__(self) -> None:
        super().__init__("Int")


class TLABool(TLAType):
    def __init__(self) -> None:
        super().__init__("Bool")


class TLAStr(TLAType):
    def __init__(self) -> None:
        super().__init__("Str")


class TLAMap(TLAType):
    def __init__(self, from_type: TLAType, to_type: TLAType) -> None:
        super().__init__(f"{from_type} -> {to_type}")


class Expr(ABC):
    def __init__(self) -> None:
        pass

    def asDefinition(self) -> "Definition":
        return NotImplemented

    def __eq__(self, other) -> "Expr":  # type: ignore
        return Equal(self, other)

    def __ne__(self, other) -> "Expr":  # type: ignore
        return NotEqual(self, other)

    def __and__(self, other) -> "Expr":
        return And(self, other)

    def __or__(self, value) -> "Expr":
        return Or(self, value)

    def __lt__(self, other) -> "Expr":
        return Lt(self, other)

    def __gt__(self, other):
        return Gt(self, other)

    def __le__(self, other) -> "Expr":
        return LtE(self, other)

    def __ge__(self, other):
        return GtE(self, other)

    def __add__(self, other):
        assert isinstance(other, Expr)

        return Add(self, other)

    def __mul__(self, other):
        assert isinstance(other, Expr)

        return Mul(self, other)

    def __truediv__(self, other):
        assert isinstance(other, Expr)

        return Div(self, other)

    def __lshift__(self, other):
        return Shl(self, other)

    def __rshift__(self, other):
        return Shr(self, other)

    def __pow__(self, other):
        return Pow(self, other)

    def __sub__(self, other: "Expr"):

        assert isinstance(other, Expr)

        return Sub(self, other)


class Paren(Expr):
    def __init__(self, value) -> None:
        super().__init__()
        self.value = value

    def __str__(self) -> str:

        if (
            isinstance(self.value, Literal)
            or isinstance(self.value, Variable)
            or isinstance(self.value, Next)
            or isinstance(self.value, Index)
            or isinstance(self.value, Constant)
            or isinstance(self.value, Mapping)
            or isinstance(self.value, MappingUpdate)
            or isinstance(self.value, Definition)
            or isinstance(self.value, Parameter)
            or isinstance(self.value, SetComprehension)
            or isinstance(self.value, MapComprehension)
        ):
            return str(self.value)

        return f"({str(self.value)})"


class Literal(Expr):
    def __init__(self, value: int | str | bool) -> None:
        super().__init__()
        self.value = value

    def __str__(self) -> str:

        if isinstance(self.value, str):
            return f'"{self.value}"'
        elif isinstance(self.value, bool):
            return str(self.value).upper()
        elif isinstance(self.value, int) and self.value > 2147483647:
            # TLC only handles integer literals up to 2^31-1.
            # Reinterpret large unsigned 32-bit values as signed 32-bit.
            return str(self.value - 4294967296)
        else:
            return str(self.value)


class Variable(Expr):
    def __init__(self, name: str, tla_type: TLAType | None = None) -> None:
        super().__init__()
        self.name = name
        self.tla_type = tla_type

    def __str__(self):
        return self.name

    def __getitem__(self, key):

        if isinstance(key, int) or isinstance(key, str) or isinstance(key, bool):
            return Index(self, Literal(key))

        assert isinstance(key, Expr)

        return Index(self, key)

    def typeAnnotation(self):
        assert self.tla_type is not None
        return f"\\* @type: {self.tla_type};"

    def next(self):
        return Next(self)


class Next(Expr):
    def __init__(self, v: Variable) -> None:
        super().__init__()
        self.v = v

    def __str__(self):
        return str(self.v) + "'"

    def __getitem__(self, key):
        return Index(self, key)


class Index(Expr):
    def __init__(self, value: Expr, index: Expr) -> None:
        super().__init__()
        self.value = value
        self.index = index

    def __eq__(self, other: Expr) -> Expr:  # type: ignore

        assert isinstance(self.value, Variable)

        return self.value.next() == MappingUpdate(self.value, [(self.index, other)])

    def __getitem__(self, key):

        if isinstance(key, int) or isinstance(key, str) or isinstance(key, bool):
            return Index(self, Literal(key))

        assert isinstance(key, Expr)

        return Index(self, key)

    def __str__(self) -> str:
        return str(self.value) + f"[{str(self.index)}]"


class Constant(Expr):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __str__(self):
        return self.name


class Mapping(Expr):
    def __init__(
        self, indicies: list[MappingIndex], values: list[MappingValue]
    ) -> None:
        super().__init__()
        self.indicies = indicies
        self.values = values
        assert len(indicies) == len(values)

    def __str__(self):
        return (
            "["
            + ", ".join([f"{i} |-> {v}" for i, v in zip(self.indicies, self.values)])
            + "]"
        )


class MappingRange(Expr):
    def __init__(
        self, start: int, end: int, parameter: "Parameter", value: MappingValue
    ) -> None:
        super().__init__()
        self.start = start
        self.end = end
        self.value = value
        self.parameter = parameter

    def __str__(self):
        return f"[{self.parameter} \\in {self.start}..{self.end} |-> {self.value}]"


class SetComprehension(Expr):
    def __init__(
        self, start: int, end: int, parameter: "Parameter", expr: Expr
    ) -> None:
        super().__init__()
        self.start = start
        self.end = end
        self.expr = expr
        self.parameter = parameter

    def __str__(self) -> str:
        return (
            "{" + f"{self.expr} : {self.parameter} \\in {self.start}..{self.end}" + "}"
        )


class MapComprehension(Expr):
    def __init__(
        self,
        parameter: "Parameter",
        domainSet: "SetComprehension | Domain",
        value: MappingValue,
    ) -> None:
        super().__init__()

        self.parameter = parameter
        self.domain = domainSet
        self.value = value

    def __str__(self) -> str:
        return f"[n \\in {self.domain} |-> {self.value}]"


class Tuple:
    def __init__(self, *args) -> None:
        self.args = args

    def __str__(self) -> str:
        return "<<" + ", ".join(str(v) for v in self.args) + ">>"


class Unchanged(Expr):
    def __init__(self, variables: list[Variable]) -> None:
        super().__init__()
        self.variables = variables

    def __str__(self) -> str:
        if len(self.variables) == 1:
            return f"UNCHANGED {self.variables[0]}"
        return f"UNCHANGED {Tuple(*self.variables)}"


class IfThenElse(Expr):
    def __init__(self, condition: Expr, if_body: Expr, else_body: Expr) -> None:
        super().__init__()
        self.condition = condition
        self.if_body = if_body
        self.else_body = else_body

    def __str__(self) -> str:

        if isinstance(self.condition, Literal):
            if isinstance(self.condition.value, bool):
                return (
                    str(self.if_body) if self.condition.value else str(self.else_body)
                )

        return f"IF {str(Paren(self.condition))} THEN ({str(Paren(self.if_body))}) ELSE ({str(Paren(self.else_body))})"


class BinOp(Expr):
    def __init__(self, op: str, lhs: Expr, rhs: Expr) -> None:
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs
        self.op = op

    def __str__(self):
        return str(Paren(self.lhs)) + " " + self.op + " " + str(Paren(self.rhs))

    def __call__(self) -> int | str | bool:
        return NotImplemented


class AssociativeOp(Expr):
    def __init__(self, op: str, *args) -> None:
        super().__init__()
        self.args = args
        self.op = op

    def __str__(self) -> str:
        return f" {self.op} ".join([str(Paren(s)) for s in self.args])

    def __call__(self, *args: Any) -> Any:
        raise NotImplementedError

    def identity(self):
        raise NotImplementedError

    def simplify(self, *args):

        constants = []
        rem = []
        for r in args:
            if isinstance(r, Literal):
                constants.append(r.value)
            else:
                rem.append(r)

        if len(constants) == 0:
            return rem

        result = self(*constants)

        if result == self.identity():
            return rem if len(rem) > 0 else [Literal(result)]

        return [Literal(result)] + rem

    @classmethod
    def expandArgs(cls, *args):

        expanded_args = []
        for r in args:
            if isinstance(r, cls):
                expanded_args.extend(cls.expandArgs(*r.args))
            else:
                expanded_args.append(r)

        return expanded_args


class Add(AssociativeOp):
    def __init__(self, *args) -> None:
        super().__init__("+", *args)
        args = Add.expandArgs(*args)
        self.args = self.simplify(*args)

    def __call__(self, *args: bool) -> Any:
        return reduce(lambda accum, x: accum + x, args, 0)

    def identity(self):
        return 0


class Mod(BinOp):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        super().__init__("%", lhs, rhs)

    def __call__(self) -> int | str | bool:
        if isinstance(self.lhs, Literal) and isinstance(self.rhs, Literal):
            assert isinstance(self.lhs.value, int) and isinstance(self.rhs.value, int)

            return self.lhs.value % self.rhs.value


class Sub(BinOp):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        super().__init__("-", lhs, rhs)

    def __call__(self) -> int | str | bool:
        if isinstance(self.lhs, Literal) and isinstance(self.rhs, Literal):
            assert isinstance(self.lhs.value, int) and isinstance(self.rhs.value, int)

            return self.lhs.value - self.rhs.value


class Mul(AssociativeOp):
    def __init__(self, *args) -> None:
        super().__init__("*", *args)
        args = Mul.expandArgs(*args)
        self.args = self.simplify(*args)

    def __call__(self, *args: bool) -> Any:
        return reduce(lambda accum, x: accum * x, args, 1)

    def identity(self):
        return 1


class Div(BinOp):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        super().__init__("\\div", lhs, rhs)


class Pow(BinOp):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        super().__init__("^", lhs, rhs)


class Concat(BinOp):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        super().__init__("\\o", lhs, rhs)


class Shl(Expr):
    def __init__(self, target: Expr, shift: Expr) -> None:
        super().__init__()

        if isinstance(target, Shl):
            shift = shift + target.shift
            target = target.target

        self.target = target
        self.shift = shift

    def __str__(self) -> str:

        return str(self.target * (Literal(2) ** self.shift))


class Shr(Expr):
    def __init__(self, target: Expr, shift: Expr) -> None:
        super().__init__()

        if isinstance(target, Shr):
            shift = shift + target.shift
            target = target.target

        self.target = target
        self.shift = shift

    def __str__(self) -> str:

        return str(self.target / (Literal(2) ** self.shift))


class FunnelShr(Expr):
    def __init__(self, hi: Expr, lo: Expr, shift: Expr) -> None:
        super().__init__()
        self.hi = hi
        self.lo = lo
        self.shift = shift

    def __str__(self) -> str:

        return str(((self.hi << Literal(32)) + self.lo) >> (self.shift + Literal(32)))


class And(AssociativeOp):
    def __init__(self, *args) -> None:
        super().__init__("/\\", *args)
        args = And.expandArgs(*args)
        self.args = self.simplify(*args)

    def __call__(self, *args: bool) -> Any:
        return reduce(lambda accum, x: accum and x, args, True)

    def identity(self):
        return True


class Or(AssociativeOp):
    def __init__(self, *args) -> None:
        super().__init__("\\/", *args)
        args = Or.expandArgs(*args)
        self.args = self.simplify(*args)

    def __call__(self, *args: bool) -> Any:
        return reduce(lambda accum, x: accum or x, args, False)

    def identity(self):
        return False


class Equal(BinOp):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        super().__init__("=", lhs, rhs)


class NotEqual(BinOp):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        super().__init__("/=", lhs, rhs)

    def __call__(self, *args: bool) -> Any:

        pass


class Gt(BinOp):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        super().__init__(">", lhs, rhs)


class Lt(BinOp):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        super().__init__("<", lhs, rhs)


class GtE(BinOp):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        super().__init__(">=", lhs, rhs)


class LtE(BinOp):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        super().__init__("<=", lhs, rhs)


class Implies(BinOp):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        super().__init__("=>", lhs, rhs)


class Max(Expr):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return str(IfThenElse(self.lhs < self.rhs, self.rhs, self.lhs))


class Min(Expr):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return str(IfThenElse(self.lhs < self.rhs, self.lhs, self.rhs))


class Parameter(Expr):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __str__(self):
        return self.name


class DefinitionInvoke(Expr):
    def __init__(self, name: str, arguments: list[Parameter] = []) -> None:
        super().__init__()
        self.name = name
        self.arguments = arguments

    def __str__(self) -> str:
        argumentList = "(" + ", ".join([str(p) for p in self.arguments]) + ")"
        return self.name + argumentList


class Definition(Expr):
    def __init__(self, name: str, value: Expr, params: list[Parameter] = []) -> None:
        super().__init__()
        self.name = name
        self.value = value
        self.params = params

    def __str__(self):
        return self.name

    def __call__(self, *args) -> "DefinitionInvoke":
        assert len(args) == len(self.params)

        return DefinitionInvoke(self.name, list(args))

    def toDefString(self):
        argumentList = ""
        if len(self.params) > 0:
            argumentList = "(" + ", ".join([str(p) for p in self.params]) + ")"

        return str(self) + argumentList + " == " + str(self.value)

    @staticmethod
    def createParameter(name: str) -> Parameter:
        return Parameter(name)


class MappingUpdate(Expr):
    def __init__(self, mapping: Expr, updates: list[tuple[Expr, Expr]]) -> None:
        super().__init__()
        self.mapping = mapping
        self.updates = updates

    def __str__(self):
        return (
            f"[{str(self.mapping)} EXCEPT "
            + ", ".join([f"![{str(i)}] = {str(v)}" for i, v in self.updates])
            + "]"
        )

class MappingUpdateBuilder:
    def __init__(self, mapping: Expr) -> None:
        self.mapping = mapping
        self.killed = set()
        self.updates = set()

    def update(self, key: Expr, value: Expr) -> bool:

        if not isinstance(key, Literal):
            return False

        assert isinstance(key, Literal)

        if key.value in self.killed:
            return False

        self.killed.add(key.value)

        self.updates.add((key, value))

        return True

    def build(self):

        return MappingUpdate(self.mapping, list(self.updates))


class UnrOp(Expr):
    def __init__(self, op: str, expr: Expr) -> None:
        super().__init__()
        self.op = op
        self.expr = expr

    def __str__(self) -> str:
        return f"{self.op} ({self.expr})"


class Not(UnrOp):
    def __init__(self, expr: Expr) -> None:
        super().__init__("~", expr)


class Eventually(UnrOp):
    def __init__(self, expr: Expr) -> None:
        super().__init__("<>", expr)


class Always(UnrOp):
    def __init__(self, expr: Expr) -> None:
        super().__init__("[]", expr)


class LeadsTo(BinOp):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        super().__init__("~>", lhs, rhs)


class Enabled(UnrOp):
    def __init__(self, expr: Expr) -> None:
        super().__init__("ENABLED", expr)


class ToString(UnrOp):
    def __init__(self, expr: Expr) -> None:
        super().__init__("ToString", expr)


class ForAll(Expr):
    def __init__(self, var: Parameter, domain: Expr, body: Expr) -> None:
        super().__init__()
        self.var = var
        self.domain = domain
        self.body = body

    def __str__(self) -> str:
        return f"\\A {self.var} \\in {self.domain} : {Paren(self.body)}"


class Exists(Expr):
    def __init__(self, var: Parameter, domain: Expr, body: Expr) -> None:
        super().__init__()
        self.var = var
        self.domain = domain
        self.body = body

    def __str__(self) -> str:
        return f"\\E {self.var} \\in {self.domain} : {Paren(self.body)}"


class Domain(Expr):
    def __init__(self, mapping: Expr) -> None:
        super().__init__()
        self.mapping = mapping

    def __str__(self) -> str:
        return f"DOMAIN {Paren(self.mapping)}"


class TLAModule:
    def __init__(self, name: str, apalache_compatible=False) -> None:
        self.name = name
        self.variables: list[Variable] = []
        self.constants: list[Constant] = []
        self.definitions: list[Definition] = []
        self.initialState: Definition | None = None
        self.nextState: Definition | None = None
        self.invariants: list[Expr] = []
        self.checkDeadlock: bool = True
        self.constantDefs: list[tuple[Constant, Expr]] = []
        self.properties: list[Expr] = []
        self.enableWeakFairness: bool = False

        self.apalache_compatible = apalache_compatible

    def createVariable(self, name: str, tla_type: TLAType | None = None) -> Variable:
        v = Variable(name, tla_type=tla_type)
        self.variables.append(v)
        return v

    def createConstant(self, name: str):
        c = Constant(name)
        self.constants.append(c)
        return c

    def createDefinition(
        self, name: str, expr: Expr, params: list[Parameter] | None = None
    ):
        d = Definition(name, expr, params=params or [])
        self.definitions.append(d)
        return d

    def createInvariant(self, name: str, expr: Expr):
        d = self.createDefinition(name, expr)
        self.invariants.append(d)
        return d

    def createProperty(self, name: str, expr: Expr):
        d = self.createDefinition(name, expr)
        self.properties.append(d)
        return d

    def setInitialState(self, expr: Expr):
        self.initialState = Definition("Init", expr)

    def setNextState(self, expr: Expr):
        self.nextState = Definition("Next", expr)

    def addInvariant(self, expr: Expr):
        self.invariants.append(expr)

    def allowDeadlock(self):
        self.checkDeadlock = False

    def getConfiguration(self):

        lines = []

        if len(self.properties) > 0:
            lines.append("SPECIFICATION Spec")
        else:
            lines.append("INIT Init")
            lines.append("NEXT Next")

        lines.append(f"CHECK_DEADLOCK {str(self.checkDeadlock).upper()}")

        for c, exp in self.constantDefs:
            lines.append(f"CONSTANT {c} = {exp}")

        for inv in self.invariants:
            lines.append(f"INVARIANT {inv}")

        for prop in self.properties:
            lines.append(f"PROPERTY {prop}")

        return "\n".join(lines)

    def __str__(self):

        lines = []

        assert self.initialState is not None
        assert self.nextState is not None

        moduleHeader = "-" * 10 + " MODULE " + self.name + " " + 10 * "-"

        lines.append(moduleHeader)
        lines.append("EXTENDS Integers")

        if len(self.variables) > 0:
            if self.apalache_compatible:
                lines.append("VARIABLES")
                lines.append(
                    ",\n".join(
                        [v.typeAnnotation() + "\n" + str(v) for v in self.variables]
                    )
                )
            else:
                lines.append(f"VARIABLES {', '.join([str(v) for v in self.variables])}")

        if len(self.constants) > 0:
            lines.append(f"CONSTANTS {', '.join([str(v) for v in self.constants])}")

        lines.append(self.initialState.toDefString())

        if len(self.properties) > 0:
            spec = Definition("Spec", self.initialState)
            lines.append(spec.toDefString())

        for d in self.definitions:
            lines.append(d.toDefString())

        lines.append(self.nextState.toDefString())

        lines.append("=" * len(moduleHeader))

        return "\n".join(lines)


if __name__ == "__main__":
    module = TLAModule("Hello")

    a = module.createVariable("a")
    b = module.createVariable("b")
    c = module.createVariable("c")

    module.setInitialState(a & b)
    module.setNextState(a | b)

    module.allowDeadlock()

    print(module)
    print(module.getConfiguration())
