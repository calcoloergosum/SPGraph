"""S-expression representation of SPGraph"""
from typing import Any

import sexpdata

from .spgraph import (ExistsNode, ForallNode, LeafNode, SeriesNode,
                      SPException, SPGraph, X, Y)


def from_sexpr(sexpr: str) -> SPGraph[X, Y]:
    """Convert s-expression into Series-parallel graph.
    >>> print(from_sexpr('(a _ _)').draw(pretty=True))
    ╺┯━┯╸
     │ │ 
    ╺┷━┷╸
    >>> print(from_sexpr('(e _ _)').draw(pretty=True))
    ╺┯━┯╸
     │ │ 
    ╺┷━┷╸
    >>> print(from_sexpr('(s _ _)').draw(pretty=True))
    ┯
    │
    ┿
    │
    ┷
    """
    return _from_sexpr(sexpdata.loads(sexpr))


def _from_sexpr(sexp: sexpdata.Sequence) -> SPGraph[X, Y]:
    if len(sexp) == 1:
        return SPGraph.make_node(str(sexp[0]))
    op = str(sexp[0])
    if op == 's':
        return SPGraph.series(*list(map(_from_sexpr, sexp[1:])))
    if op == 'a':
        return SPGraph.forall(*list(map(_from_sexpr, sexp[1:])))
    if op == 'e':
        return SPGraph.exists(*list(map(_from_sexpr, sexp[1:])))
    raise SPException(f"Unknown operation {sexp[0]}")


def to_sexpr(self: SPGraph[Any, Any]) -> str:
    """Convert spgraph into S-expression"""
    if isinstance(self, LeafNode):
        return "_"
    if isinstance(self, SeriesNode):
        return f"(s {' '.join(list(map(to_sexpr, self.inner)))})"
    if isinstance(self, ForallNode):
        return f"(a {' '.join(list(map(to_sexpr, self.inner)))})"
    if isinstance(self, ExistsNode):
        return f"(e {' '.join(list(map(to_sexpr, self.inner)))})"
    raise NotImplementedError


SPGraph.from_sexpr = from_sexpr
SPGraph.to_sexpr = to_sexpr
