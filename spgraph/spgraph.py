"""Ordered, Directed Series Parallel Graph
Drawn in canonical representation
"""
from __future__ import annotations

from enum import IntEnum
from typing import Tuple, Any, Union, Optional


class SPException(Exception):
    pass


class NodeType(IntEnum):
    Series = 0
    Parallel = 1
    Leaf = 2


class SPGraph:
    def __init__(self, **kwargs) -> None:
        for k, v in kwargs:
            setattr(self, k, v)

    def type(self) -> NodeType:
        raise NotImplementedError

    @classmethod
    def parallel(cls, *args) -> ParallelNode:
        return ParallelNode(tuple(args))

    @classmethod
    def series(cls, *args) -> SeriesNode:
        return SeriesNode(tuple(args))

    @classmethod
    def make_node(cls, obj) -> SPGraph:
        if isinstance(obj, cls):
            return obj
        return LeafNode(obj)

    # Patched properties
    def to_graphml(self: SPGraph, g: Optional["pygraphml.Graph"] = None) -> "pygraphml.Graph":
        raise SPException("Load .graphml module")
    
    def get_symmetry(self: SPGraph) -> "OrientedPlanarSymmetry":
        raise SPException("Load .layout module")
    
    def from_sexp(s: str) -> SPGraph:
        raise SPException("Load .sexp module")


class InternalNode(SPGraph):
    """Compositions"""
    def __init__(self, inner: Tuple[Union[ParallelNode, LeafNode]], **kwargs) -> None:
        super().__init__(**kwargs)
        if len(inner) < 2:
            raise SPException("Not canonical representation")
        for c in inner:
            if isinstance(c, self.__class__):
                import traceback
                traceback.print_stack()
                raise SPException("Not canonical representation")
            if not isinstance(c, SPGraph):
                raise SPException(f"Child {c} is not a node")
        self.inner = inner
    
    def asdict(self) -> str:
        return {"type": self.type().name, "children": [c.asdict() for c in self.inner]}


class SeriesNode(InternalNode):
    """Series composition"""
    def type(self) -> NodeType:
        return NodeType.Series


class ParallelNode(InternalNode):
    """Parallel composition"""
    def type(self) -> NodeType:
        return NodeType.Parallel

    def asdict(self) -> str:
        return {"type": self.type().name, "children": [c.asdict() for c in self.inner]}


class LeafNode(SPGraph):
    """External node"""
    def __init__(self, inner: Any, **kwargs) -> None:
        super().__init__(**kwargs)
        self.inner = inner

    def type(self) -> NodeType:
        return NodeType.Leaf

    def asdict(self) -> str:
        return {"type": self.type().name, "inner": self.inner}


SPGraph.__doc__ == __doc__
