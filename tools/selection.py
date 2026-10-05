"""Event selections: a cut expression evaluated over per-event arrays.

A selection is one expression in the variables a caller supplies, e.g.

    event_calo_edep_vis > 10 and primary_start_z > 5400

evaluated column-wise with numpy, so it costs one pass over each array. The
expression comes from an agent, so it is never handed to `eval`: it is parsed
with `ast` and only a whitelist of node types is walked — variable names,
numbers, arithmetic, comparisons, boolean logic and a few math functions.
Anything else (attribute access, subscripts, lambdas, ...) is refused with a
message naming it.

Both spellings of the boolean operators work: Python's `and` / `or` / `not`,
and ROOT/C's `&&` / `||` / `!`, as a TTree::Draw cut would write them. A
comparison against a missing value (NaN) is false, so an event without the
quantity a cut asks about fails that cut.
"""

from __future__ import annotations

import ast
import re
from typing import Mapping

import numpy as np


class SelectionError(ValueError):
    """A selection that cannot be parsed or evaluated, worded for the caller."""


_FUNCTIONS = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "hypot": np.hypot,
    "min": np.minimum,
    "max": np.maximum,
}

_BINARY = {
    ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
    ast.Div: np.true_divide, ast.Pow: np.power, ast.Mod: np.mod,
    ast.BitAnd: np.logical_and, ast.BitOr: np.logical_or,
}

_COMPARE = {
    ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater,
    ast.GtE: np.greater_equal, ast.Eq: np.equal, ast.NotEq: np.not_equal,
}


def _pythonize(expression: str) -> str:
    """ROOT/C boolean spellings to Python's: && -> and, || -> or, ! -> not."""
    text = expression.replace("&&", " and ").replace("||", " or ")
    return re.sub(r"!(?!=)", " not ", text)


def _check_names(tree: ast.AST, variables: Mapping[str, np.ndarray]) -> None:
    unknown = sorted({
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id not in variables
        and node.id not in _FUNCTIONS
    })
    if unknown:
        raise SelectionError(
            f"unknown variable(s) {', '.join(unknown)}; known: "
            f"{', '.join(sorted(variables))}"
        )


def _eval(node: ast.AST, variables: Mapping[str, np.ndarray]):
    if isinstance(node, ast.Expression):
        return _eval(node.body, variables)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        raise SelectionError(f"'{node.id}' is a function, call it with ()")
    if isinstance(node, ast.BoolOp):
        combine = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        result = _eval(node.values[0], variables)
        for value in node.values[1:]:
            result = combine(result, _eval(value, variables))
        return result
    if isinstance(node, ast.UnaryOp):
        operand = _eval(node.operand, variables)
        if isinstance(node.op, (ast.Not, ast.Invert)):
            return np.logical_not(operand)
        if isinstance(node.op, ast.USub):
            return np.negative(operand)
        if isinstance(node.op, ast.UAdd):
            return operand
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](_eval(node.left, variables),
                                      _eval(node.right, variables))
    if isinstance(node, ast.Compare):
        # a < b < c means a < b and b < c, as in Python
        result = True
        left = _eval(node.left, variables)
        for op, comparator in zip(node.ops, node.comparators):
            if type(op) not in _COMPARE:
                break
            right = _eval(comparator, variables)
            result = np.logical_and(result, _COMPARE[type(op)](left, right))
            left = right
        else:
            return result
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS and not node.keywords):
        return _FUNCTIONS[node.func.id](*(_eval(arg, variables) for arg in node.args))
    raise SelectionError(
        f"'{ast.unparse(node)}' is not allowed in a selection: use variables, "
        "numbers, arithmetic, comparisons, and/or/not, and "
        f"{', '.join(_FUNCTIONS)}"
    )


def apply_selection(expression: str, variables: Mapping[str, np.ndarray],
                    nevents: int) -> np.ndarray:
    """The boolean mask of the nevents events passing `expression`.

    An empty expression selects everything. Raises SelectionError, worded for
    the caller, for anything that does not parse, names an unknown variable,
    or does not come out as one true/false per event.
    """
    if not expression.strip():
        return np.ones(nevents, dtype=bool)
    try:
        tree = ast.parse(_pythonize(expression).strip(), mode="eval")
    except SyntaxError as exc:
        raise SelectionError(f"selection {expression!r} does not parse: {exc.msg}")
    _check_names(tree, variables)
    with np.errstate(all="ignore"):
        result = np.asarray(_eval(tree, variables))
    if result.dtype != bool:
        raise SelectionError(
            f"selection {expression!r} is not a true/false condition "
            "(did you leave out a comparison?)"
        )
    if result.ndim == 0:
        return np.full(nevents, bool(result))
    if result.shape != (nevents,):
        raise SelectionError(f"selection {expression!r} did not give one value per event")
    return result
