from spgraph.testutil import make_random

def test_graphml():
    spg = make_random(3, 5, 5)
    g = spg.to_graphml()
    # load in graphml viewer, and see if series-parallel graph
    from pygraphml import GraphMLParser
    import tempfile
    parser = GraphMLParser()
    parser.write(g, tempfile.gettempdir() + '/asd.graphml')
