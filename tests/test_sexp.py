from spgraph import SPGraph

import pytest

TESTCASES = [
    "_",
    "(s _ _)",
    "(s (a _ _) _)",
    "(a (s _ _) (s _ _))",
    "(a (e _ _) (s _ _))",
]


@pytest.mark.parametrize("sexp", TESTCASES)
def test_load(sexp):
    assert sexp == SPGraph.from_sexpr(sexp).to_sexpr()
