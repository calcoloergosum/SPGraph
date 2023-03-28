from spgraph import SPGraph

import pytest

TESTCASES = [
    "_",
    "(s _ _)",
    "(s (p _ _) _)",
    "(p (s _ _) (s _ _))",
]


@pytest.mark.parametrize("sexp", TESTCASES)
def test_load(sexp):
    assert sexp == SPGraph.from_sexpr(sexp).to_sexpr()
