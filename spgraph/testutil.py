from .spgraph import SPGraph
from random import random


def make_random(max_depth: int, max_children_series: int, max_children_parallel: int, p_leaf: float = 0.3) -> SPGraph:
    if random() < 0.5:
        makefunc = make_random_series_or_leaf
    else:
        makefunc = make_random_parallel_or_leaf
    return makefunc(max_depth, max_children_series, max_children_parallel, p_leaf)


def make_random_series_or_leaf(max_depth: int, max_children_series: int, max_children_parallel: int, p_leaf: float = 0.3) -> SPGraph:
    if max_depth == 0:
        return SPGraph.make_node(None)

    n_children = int((max_children_series - 1) * random() + 1)
    if n_children == 1:
        return make_random_series_or_leaf(max_depth - 1, max_children_series, max_children_parallel, p_leaf)

    children = []
    
    for _ in range(n_children):
        depth = int(max_depth * random())
        children += [make_random_parallel_or_leaf(depth - 1, max_children_series, max_children_parallel, p_leaf)]
    return SPGraph.series(*children)


def make_random_parallel_or_leaf(max_depth: int, max_children_series: int, max_children_parallel: int, p_leaf: float = 0.3) -> SPGraph:
    if max_depth < 0:
        return SPGraph.make_node(None)

    n_children = int((max_children_parallel - 1) * random() + 1)
    if n_children == 1:
        return make_random_parallel_or_leaf(max_depth - 1, max_children_series, max_children_parallel, p_leaf)

    children = []
    for _ in range(n_children):
        depth = int(max_depth * random())
        children += [make_random_series_or_leaf(depth - 1, max_children_series, max_children_parallel, p_leaf)]
    return SPGraph.parallel(*children)
