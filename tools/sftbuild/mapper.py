"""Map of top-level functions and classes, from ast, for a file read up to a given line."""
import ast
import re


def ast_map(path: str, upto: int):
    """'name start-end' per top-level def/class that starts within the first `upto` lines;
    one that ends after `upto` is open: 'name start-'. None on a syntax error."""
    try:
        tree = ast.parse(open(path).read())
    except SyntaxError:
        return None
    parts = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.lineno <= upto:
            parts.append("%s %d-%s" % (n.name, n.lineno, n.end_lineno if n.end_lineno <= upto else ""))
    return " -> ".join(parts)


def map_end_accuracy(model_map: str, path: str) -> float:
    """Share of closed `end` values in model_map equal to the ast end line of that name."""
    tree = ast.parse(open(path).read())
    real = {n.name: n.end_lineno for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    closed = re.findall(r"(\w+) \d+-(\d+)", model_map)
    if not closed:
        return 0.0
    return sum(1 for name, end in closed if real.get(name) == int(end)) / len(closed)
