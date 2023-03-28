from spgraph import SPGraph
from spgraph.testutil import make_random


# For each testcase we are asserting twice for caching behavior check
def test_automorphism_simplest():
    g = SPGraph.make_node(None)
    sym = g.get_symmetry()
    assert sym.vertical
    assert sym.horizontal
    sym = g.get_symmetry()
    assert sym.vertical
    assert sym.horizontal

def test_automorphism_simple_series():
    g = SPGraph.from_sexpr('(s _ _)')
    sym = g.get_symmetry()
    assert sym.vertical
    assert sym.horizontal
    sym = g.get_symmetry()
    assert sym.vertical
    assert sym.horizontal

def test_automorphism_simple_parallel():
    g = SPGraph.from_sexpr('(p _ _)')
    sym = g.get_symmetry()
    assert sym.vertical
    assert sym.horizontal
    sym = g.get_symmetry()
    assert sym.vertical
    assert sym.horizontal

def test_automorphism_parallel_series():
    g = SPGraph.from_sexpr('(p _ (s _ _))')
    sym = g.get_symmetry()
    assert not sym.vertical
    assert sym.horizontal
    sym = g.get_symmetry()
    assert not sym.vertical
    assert sym.horizontal

def test_automorphism_series_parallel():
    g = SPGraph.from_sexpr('(s _ (p _ _))')
    sym = g.get_symmetry()
    assert sym.vertical
    assert not sym.horizontal
    sym = g.get_symmetry()
    assert sym.vertical
    assert not sym.horizontal

def test_automorphism_random_smoke():
    g = make_random(10, 10, 10)
    g.get_symmetry()
    g.get_symmetry()
