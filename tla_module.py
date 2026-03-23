from functools import reduce
from abc import ABC

type MappingIndex = str | int
type MappingValue = Expr


class Expr(ABC):
    def __init__(self) -> None:
        pass

    def asDefinition(self) -> "Definition":
        return NotImplemented


class Literal(Expr):
    def __init__(self, value: int | str) -> None:
        super().__init__()
        self.value = value

    def __str__(self) -> str:

        if isinstance(self.value, str):
            return f'"{self.value}"'
        else:
            return str(self.value)


class Variable(Expr):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __str__(self):
        return self.name

    def next(self):
        return Next(self)


class Next(Expr):
    def __init__(self, v: Variable) -> None:
        super().__init__()
        self.v = v

    def __str__(self):
        return str(self.v) + "'"


class Index(Expr):
    def __init__(self, value: Expr, index: Expr) -> None:
        super().__init__()
        self.value = value
        self.index = index

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


class Unchanged(Expr):
    def __init__(self, variables: list[Variable]) -> None:
        super().__init__()
        self.variables = variables

    def __str__(self) -> str:
        if len(self.variables) == 1:
            return f"UNCHANGED {self.variables[0]}"
        inner = ", ".join(str(v) for v in self.variables)
        return f"UNCHANGED <<{inner}>>"


class IfThenElse(Expr):
    def __init__(self, condition: Expr, if_body: Expr, else_body: Expr) -> None:
        super().__init__()
        self.condition = condition
        self.if_body = if_body
        self.else_body = else_body

    def __str__(self) -> str:
        return f"IF ({str(self.condition)}) THEN ({str(self.if_body)}) ELSE ({str(self.else_body)})"


class BinOp(Expr):
    def __init__(self, op: str, lhs: Expr, rhs: Expr) -> None:
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs
        self.op = op

    def __str__(self):
        return "(" + str(self.lhs) + " " + self.op + " " + str(self.rhs) + ")"


class AssociativeOp(Expr):
    def __init__(self, op: str, *args) -> None:
        super().__init__()
        self.args = args
        self.op = op

    def __str__(self) -> str:
        return f" {self.op} ".join([str(s) for s in self.args])


class Add(AssociativeOp):
    def __init__(self, *args) -> None:
        super().__init__("+", *args)


class Sub(BinOp):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        super().__init__("-", lhs, rhs)


class Mul(BinOp):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        super().__init__("*", lhs, rhs)


class Shl(Expr):
    def __init__(self, target: Expr, shift: Expr) -> None:
        super().__init__()
        self.target = target
        self.shift = shift

    def __str__(self) -> str:

        return f"({str(self.target)} * (2 ^ {str(self.shift)}))"


class Shr(Expr):
    def __init__(self, target: Expr, shift: Expr) -> None:
        super().__init__()
        self.target = target
        self.shift = shift

    def __str__(self) -> str:

        return f"({str(self.target)} \\div (2 ^ {str(self.shift)}))"


class FunnelShr(Expr):
    def __init__(self, hi: Expr, lo: Expr, shift: Expr) -> None:
        super().__init__()
        self.hi = hi
        self.lo = lo
        self.shift = shift

    def __str__(self) -> str:

        return str(
            Shr(Shr(Add(Shl(self.hi, Literal(32)), self.lo), self.shift), Literal(32))
        )


class And(AssociativeOp):
    def __init__(self, *args) -> None:
        super().__init__("/\\", *args)


class Or(AssociativeOp):
    def __init__(self, *args) -> None:
        super().__init__("\\/", *args)


class Equal(BinOp):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        super().__init__("=", lhs, rhs)


class NotEqual(BinOp):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        super().__init__("/=", lhs, rhs)


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


class Max(Expr):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return str(IfThenElse(Lt(self.lhs, self.rhs), self.rhs, self.lhs))


class Min(Expr):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        return str(IfThenElse(Lt(self.lhs, self.rhs), self.lhs, self.rhs))


class DefinitionParameter(Expr):
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self):
        return self.name


class DefinitionInvoke(Expr):
    def __init__(self, name: str, arguments: list[DefinitionParameter] = []) -> None:
        super().__init__()
        self.name = name
        self.arguments = arguments

    def __str__(self) -> str:
        argumentList = "(" + ", ".join([str(p) for p in self.arguments]) + ")"
        return self.name + argumentList


class Definition(Expr):
    def __init__(
        self, name: str, value: Expr, params: list[DefinitionParameter] = []
    ) -> None:
        super().__init__()
        self.name = name
        self.value = value
        self.params = params

    def __str__(self):
        assert len(self.params) == 0

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
    def createParameter(name: str) -> DefinitionParameter:
        return DefinitionParameter(name)


class MappingUpdate(Expr):
    def __init__(self, mapping: Variable, updates: list[tuple[Literal, Expr]]) -> None:
        super().__init__()
        self.mapping = mapping
        self.updates = updates

    def __str__(self):
        return (
            f"[{str(self.mapping)} EXCEPT "
            + ", ".join([f"![{str(i)}] = {str(v)}" for i, v in self.updates])
            + "]"
        )


class TLAModule:
    def __init__(self, name: str) -> None:
        self.name = name
        self.variables: list[Variable] = []
        self.constants: list[Constant] = []
        self.definitions: list[Definition] = []
        self.initialState: Definition | None = None
        self.nextState: Definition | None = None
        self.invariants: list[Expr] = []
        self.checkDeadlock: bool = True
        self.constantDefs: list[tuple[Constant, Expr]] = []

    def createVariable(self, name: str) -> Variable:
        v = Variable(name)
        self.variables.append(v)
        return v

    def createConstant(self, name: str):
        c = Constant(name)
        self.constants.append(c)
        return c

    def createDefinition(self, name: str, expr: Expr):
        d = Definition(name, expr)
        self.definitions.append(d)
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

        lines.append("INIT Init")
        lines.append("NEXT Next")
        lines.append(f"CHECK_DEADLOCK {str(self.checkDeadlock).upper()}")

        for c, exp in self.constantDefs:
            lines.append(f"CONSTANT {c} = {exp}")

        for inv in self.invariants:
            lines.append(f"INVARIANT {inv}")

        return "\n".join(lines)

    def __str__(self):

        lines = []

        assert self.initialState is not None
        assert self.nextState is not None

        moduleHeader = "-" * 10 + " MODULE " + self.name + " " + 10 * "-"

        lines.append(moduleHeader)

        if len(self.variables) > 0:
            lines.append(f"VARIABLES {', '.join([str(v) for v in self.variables])}")

        if len(self.constants) > 0:
            lines.append(f"CONSTANTS {', '.join([str(v) for v in self.constants])}")

        lines.append(self.initialState.toDefString())

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

    module.setInitialState(And(a, b))
    module.setNextState(Or(a, b))

    module.allowDeadlock()

    print(module)
    print(module.getConfiguration())
