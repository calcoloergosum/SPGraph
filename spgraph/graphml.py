"""Dump to GraphML
It is useful as you can load it from viewer e.g. yEd Graph Editor
"""
import pygraphml
from .spgraph import SPGraph, SeriesNode, ParallelNode, LeafNode
from typing import Optional


def to_graphml(self: SPGraph, g: Optional[pygraphml.Graph] = None) -> pygraphml.Graph:
    if g is None:
        g = pygraphml.Graph()

    s, t = g.add_node('s'), g.add_node('t')
    _add_edge(self, g, s, t)
    return g


def _add_edge(self: SPGraph, g: pygraphml.Graph, s: pygraphml.Node, t: pygraphml.Node) -> None:
    if isinstance(self, LeafNode):
        g.add_edge(s, t, True)
        return

    if isinstance(self, SeriesNode):
        # prepare intermediate nodes
        nodes = [s]
        for _ in self.inner[:-1]:
            nodes += [g.add_node(len(g.nodes()))]
        nodes += [t]

        # add edges
        for c, s, t in zip(self.inner, nodes, nodes[1:]):
            _add_edge(c, g, s, t)
        return
    
    if isinstance(self, ParallelNode):
        for c in self.inner:
            _add_edge(c, g, s, t)
        return

SPGraph.to_graphml = to_graphml
