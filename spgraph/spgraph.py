"""Ordered, Directed Series Parallel Graph
Drawn in canonical representation
"""
from __future__ import annotations

import functools
from enum import IntEnum
from typing import (TYPE_CHECKING, Any, Callable, Dict, Generic, Optional, List,
                    Tuple, TypeVar, Union, cast)

from typing_extensions import Self

if TYPE_CHECKING:
    import pygraphml

    from .stream import ThreadedSPGraph
    from .symmetry import OrientedPlanarSymmetry

V = TypeVar("V")
W = TypeVar("W")
X = TypeVar("X")
Y = TypeVar("Y")
Z = TypeVar("Z")


class SPException(Exception):
    """General Exception that is recognized by spgraph library"""


class RepresentationError(SPException):
    """Some unacceptable representation; e.g. nested series representation"""


class ShouldNotBeReachable(SPException):
    """Code should not be reachable; such as at the end of complete pattern matching"""


class NodeType(IntEnum):
    """This class is only sometimes useful, e.g. serialization.
    If possible, use the class directly rather than using this.
    """
    SERIES = 0
    PARALLEL = 1
    LEAF = 2


class SPGraph(Generic[X, Y]):
    """__doc__ will be overriden at the end of module"""
    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def type(self) -> NodeType:
        """get type enum"""
        raise NotImplementedError

    def asdict(self) -> str:
        """serialize"""
        raise NotImplementedError

    # compositions
    @classmethod
    def parallel(cls, *args: List[SeriesNode[X, Y] | LeafNode[X, Y]]) -> ParallelNode[X, Y]:
        """Ordered parallel composition of Series-Parallel Graph"""
        return ParallelNode(tuple(args))

    @classmethod
    def series(cls, *args) -> SeriesNode[X, Y]:
        """Sequential (series) composition of Series-Parallel Graph"""
        return SeriesNode(tuple(args))

    def and_(self, sibling: SPGraph[X, Z] | Callable[[X], Z], **kwargs) -> SPGraph[X, Tuple[Y, Z]]:
        """Ordered parallel composition of Series-Parallel Graph. Self becomes left."""
        if not isinstance(sibling, SPGraph):
            sibling = LeafNode.from_python(sibling, **kwargs)
        args: List[SeriesNode[X, Z] | LeafNode[X, Z]] = []
        args += self.inner if isinstance(self, ParallelNode) else [self]
        args += sibling.inner if isinstance(sibling, ParallelNode) else [sibling]
        return self.__class__.parallel(*args)

    def then(self, sibling: SPGraph[Y, Z] | Callable[[Y], Z], **kwargs) -> SPGraph[X, Z]:
        """Sequential (series) composition of Series-Parallel Graph. Self becomes parent."""
        if not isinstance(sibling, SPGraph):
            sibling = LeafNode.from_python(sibling, **kwargs)
        args = []
        args += self.inner if isinstance(self, SeriesNode) else [self]
        args += sibling.inner if isinstance(sibling, SeriesNode) else [sibling]
        return self.__class__.series(*args)

    before = then

    def after(self, sibling: SPGraph[W, X] | Callable[[W], X], **kwargs) -> SPGraph[W, Y]:
        """f.after(g)(x) <=> f(g)(x) <=> (f.g)(x)"""
        if not isinstance(sibling, SPGraph):
            sibling = LeafNode.from_python(sibling, **kwargs)
        if isinstance(sibling, SeriesNode):
            return self.__class__.series(*sibling.inner, self)
        return self.__class__.series(sibling, self)

    @classmethod
    def make_node(cls, obj: Any) -> LeafNode:
        """Return LeafNode. If already LeafNode, do nothing"""
        if isinstance(obj, cls):
            return obj
        return LeafNode(obj)

    # Patched properties
    def to_graphml(self, g: Optional["pygraphml.Graph"] = None) -> "pygraphml.Graph":
        """Placeholder for patch"""
        raise SPException("Load .graphml module")

    def get_symmetry(self) -> "OrientedPlanarSymmetry":
        """Placeholder for patch"""
        raise SPException("Load .symmetry module")

    def draw(self, pretty: bool = False) -> str:
        """Placeholder for patch"""
        raise SPException("Load .layout module")

    @classmethod
    def from_sexpr(cls, s: str) -> Self:
        """Placeholder for patch"""
        raise SPException("Load .sexp module")

    def to_sexpr(self) -> str:
        """Placeholder for patch"""
        raise SPException("Load .sexp module")

    def build_stream(self) -> "ThreadedSPGraph[X, Y]":
        """Placeholder for patch"""
        raise SPException("Load .stream module")


class InternalNode(SPGraph[X, Y]):
    """Compositions"""
    def __init__(self, inner: Tuple[SeriesNode[X, Y] | LeafNode[X, Y] | ParallelNode[Any, Any], ...], **kwargs) -> None:
        super().__init__(**kwargs)
        if len(inner) < 2:
            raise RepresentationError("Not canonical")
        for c in inner:
            if isinstance(c, self.__class__):
                raise RepresentationError("Not canonical")
            if not isinstance(c, SPGraph):
                raise RepresentationError(f"Child {c} is not a node." +
                    "Consider building using `LeafNode.from_python`")
        self.inner = inner

    def type(self) -> NodeType:
        raise NotImplementedError

    def asdict(self) -> Dict[str, str]:
        return {"type": self.type().name, "children": [c.asdict() for c in self.inner]}


class SeriesNode(InternalNode[X, Y]):
    """Series composition"""
    def type(self) -> NodeType:
        return NodeType.SERIES

    def __call__(self, *args, **kwargs):
        ret = self.inner[0](*args, **kwargs)
        for f in self.inner[1:]:
            ret = f(ret)
        return ret


class ParallelNode(InternalNode[X, Y]):
    """Parallel composition"""
    def type(self) -> NodeType:
        return NodeType.PARALLEL

    def asdict(self) -> Dict[str, str]:
        return {"type": self.type().name, "children": [c.asdict() for c in self.inner]}

    def __call__(self, x: X) -> Y:
        return cast(Y, tuple(f(x) for f in self.inner))


class LeafNode(SPGraph[X, Y]):
    """External node"""
    def __init__(self, inner: Any, batchsize: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.inner = inner
        self.batchsize = batchsize

    def type(self) -> NodeType:
        return NodeType.LEAF

    def asdict(self) -> Dict[str, str]:
        return {"type": self.type().name, **vars(self)}

    def __call__(self, x: X) -> Y:
        if not callable(self.inner):
            raise RepresentationError(f"This leaf node is not a python function ({self.inner})")
        if getattr(self, 'batchsize', 0) > 0:
            return cast(Y, self.inner([x])[0])
        return self.inner(x)

    @classmethod
    def from_python(cls, inner: Optional[Callable[[X], Y]], batchsize: int = 0
    ) -> Union[
        Callable[[Callable[[X], Y], int], SPGraph[X, Y]],
        SPGraph[X, Y],
    ]:
        """Build the smallest Series-Parallel Graph from python function """
        if inner is None:
            return functools.partial(cls.from_python, batchsize=batchsize)
        assert callable(inner)
        return LeafNode(inner, batchsize=batchsize)

SPGraph.__doc__ == __doc__  # pylint: disable=pointless-statement
function = LeafNode.from_python
