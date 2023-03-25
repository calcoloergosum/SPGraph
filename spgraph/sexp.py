"""S-expression representation of SPGraph"""
import sexpdata
from .spgraph import SPGraph, SPException


def from_sexp(s: str) -> SPGraph:
    return _from_sexp(sexpdata.loads(s))


def _from_sexp(sexp: sexpdata.Sequence) -> SPGraph:
    if len(sexp) == 1:
        return SPGraph.make_node(str(sexp[0]))
    op = str(sexp[0])
    if op == 's':
        return SPGraph.series(*list(map(_from_sexp, sexp[1:])))
    if op == 'p':
        return SPGraph.parallel(*list(map(_from_sexp, sexp[1:])))
    raise SPException(f"Unknown operation {sexp[0]}")


SPGraph.from_sexp = from_sexp
