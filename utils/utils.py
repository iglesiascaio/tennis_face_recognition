import click


class TupleParamType(click.ParamType):
    name = "tuple"

    def convert(self, value, param, ctx):
        try:
            return tuple(map(int, value.split(",")))
        except ValueError:
            self.fail(f"{value} is not a valid tuple", param, ctx)


TUPLE = TupleParamType()
