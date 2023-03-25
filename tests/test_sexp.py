from spgraph import SPGraph

import pprint

def test_load():
    sexp = "(s (p _ _) _)"
    pprint.pprint(SPGraph.from_sexp(sexp).asdict())
