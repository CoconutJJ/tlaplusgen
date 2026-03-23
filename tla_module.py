type MappingIndex = str | int
type MappingValue = Expr


class Expr:
    def __init__(self) -> None:
        pass


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


class Add(BinOp):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        super().__init__("+", lhs, rhs)


class Sub(BinOp):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        super().__init__("-", lhs, rhs)


class And(BinOp):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        super().__init__("/\\", lhs, rhs)


class Or(BinOp):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        super().__init__("\\/", lhs, rhs)


class Equal(BinOp):
    def __init__(self, lhs: Expr, rhs: Expr) -> None:
        super().__init__("=", lhs, rhs)


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


class Definition(Expr):
    def __init__(self, name: str, value: Expr) -> None:
        super().__init__()
        self.name = name
        self.value = value

    def __str__(self):
        return self.name

    def toDefString(self):
        return str(self) + " == " + str(self.value)


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
    print(module)
