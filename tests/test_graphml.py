from spgraph.testutil import make_random
from pygraphml import GraphMLParser
import tempfile

def test_graphml():
    spg = make_random(3, 5)
    g = spg.to_graphml()
    parser = GraphMLParser()
    parser.write(g, tempfile.gettempdir() + '/asd.graphml')
    # load in graphml viewer, and see if series-parallel graph
