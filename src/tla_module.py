from functools import reduce
from abc import ABC
from typing import Any

type MappingIndex = str | int
type MappingValue = Expr


class Expr(ABC):
    def __init__(self) -> None:
        pass

    def asDefinition(self) -> "Definition":
        return NotImplemented


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

class MappingRange(Expr):
    def __init__(
        self, start: int, end: int, value: MappingValue
    ) -> None:
        super().__init__()
        self.start = start
        self.end = end
        self.value = value

    def __str__(self):
        return f"[n \\in {self.start} |-> {self.value}]"


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
        return f"IF {str(Paren(self.condition))} THEN ({str(Paren(self.if_body))}) ELSE ({str(Paren(self.else_body))})"


class BinOp(Expr):
    def __init__(self, op: str, lhs: Expr, rhs: Expr) -> None:
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs
        self.op = op

    def __str__(self):
        return str(Paren(self.lhs)) + " " + self.op + " " + str(Paren(self.rhs))


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


class Sub(BinOp):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        super().__init__("-", lhs, rhs)


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


class Shl(Expr):
    def __init__(self, target: Expr, shift: Expr) -> None:
        super().__init__()

        if isinstance(target, Shl):
            shift = Add(shift, target.shift)
            target = target.target

        self.target = target
        self.shift = shift

    def __str__(self) -> str:

        return str(Mul(self.target, Pow(Literal(2), self.shift)))


class Shr(Expr):
    def __init__(self, target: Expr, shift: Expr) -> None:
        super().__init__()

        if isinstance(target, Shr):
            shift = Add(shift, target.shift)
            target = target.target

        self.target = target
        self.shift = shift

    def __str__(self) -> str:

        return str(Div(self.target, Pow(Literal(2), self.shift)))


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


class UnrOp(Expr):
    def __init__(self, op: str, expr: Expr) -> None:
        super().__init__()
        self.op = op
        self.expr = expr

    def __str__(self) -> str:
        return f"{self.op} ({self.expr})"


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
        self.properties: list[Expr] = []
        self.enableWeakFairness: bool = False

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

    def createInvariant(self, name: str, expr: Expr):
        d = self.createDefinition(name, expr)
        self.invariants.append(d)
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

        if len(self.variables) > 0:
            lines.append(f"VARIABLES {', '.join([str(v) for v in self.variables])}")

        if len(self.constants) > 0:
            lines.append(f"CONSTANTS {', '.join([str(v) for v in self.constants])}")

        if len(self.properties) > 0:
            self.createDefinition("Spec", self.initialState)
        else:
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
