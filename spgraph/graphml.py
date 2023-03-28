"""Dump to GraphML
It is useful as you can load it from viewer e.g. yEd Graph Editor
"""
from typing import Optional, Any

import pygraphml

from .spgraph import LeafNode, ParallelNode, SeriesNode, SPGraph


def to_graphml(self: SPGraph[Any, Any], g: Optional[pygraphml.Graph] = None) -> pygraphml.Graph:
    """Convert into pygraphml format"""
    if g is None:
        g = pygraphml.Graph()

    s, t = g.add_node('s'), g.add_node('t')
    s['x'], s['y'] = getattr(self, 'xy_src', (0, 0))
    t['x'], t['y'] = getattr(self, 'xy_dst', (0, 0))
    _add_edge(self, g, s, t)
    return g


def _add_edge(self: SPGraph[Any, Any], g: pygraphml.Graph, s: pygraphml.Node, t: pygraphml.Node) -> None:
    if isinstance(self, LeafNode):
        g.add_edge(s, t, True)
        return

    if isinstance(self, SeriesNode):
        # prepare intermediate nodes
        nodes = [s]
        for c in self.inner[:-1]:
            n = g.add_node(len(g.nodes()))
            n['x'], n['y'] = getattr(c, 'xy_src', (0, 0))
            nodes += [n]
        nodes += [t]

        # add edges
        for c, _s, _t in zip(self.inner, nodes, nodes[1:]):
            _add_edge(c, g, _s, _t)
        return

    if isinstance(self, ParallelNode):
        for c in self.inner:
            _add_edge(c, g, s, t)
        return

SPGraph.to_graphml = to_graphml
