"""Ordered, Directed Series Parallel Graph
Drawn in canonical representation
"""
from __future__ import annotations

import functools
from enum import IntEnum
from typing import (TYPE_CHECKING, Any, Callable, Dict, Generic, List,
                    Optional, Tuple, Type, TypeVar, Union, cast)

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
    PARALLEL_FORALL = 1
    PARALLEL_EXISTS = 2
    LEAF = 3


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
    def forall(cls, *args: List[SeriesNode[X, Y] | LeafNode[X, Y]]) -> ForallNode[X, Y]:
        """Ordered parallel composition of Series-Parallel Graph"""
        return ForallNode(tuple(args))

    @classmethod
    def exists(cls, *args: List[SeriesNode[X, Y] | LeafNode[X, Y]]) -> ExistsNode[X, Y]:
        """Ordered parallel composition of Series-Parallel Graph"""
        return ExistsNode(tuple(args))

    @classmethod
    def series(cls, *args) -> SeriesNode[X, Y]:
        """Sequential (series) composition of Series-Parallel Graph"""
        return SeriesNode(tuple(args))

    @classmethod
    def merge(cls, klass: Type[InternalNode], *args, **kwargs) -> SPGraph[Any, Any]:
        """Merge. Preserves canonoical representation"""
        _args = []
        for arg in args:
            if isinstance(arg, klass):
                _args += arg.inner
            else:
                _args += [arg]
        return klass(tuple(_args), **kwargs)

    def and_(self, sibling: SPGraph[X, Z] | Callable[[X], Z], **kwargs) -> SPGraph[X, Tuple[Y, Z]]:
        """Ordered parallel composition of Series-Parallel Graph. Self becomes left."""
        if not isinstance(sibling, SPGraph):
            sibling = LeafNode.from_python(sibling, **kwargs)
        return self.merge(ForallNode, self, sibling)

    def or_(self, sibling: SPGraph[X, Z] | Callable[[X], Z], **kwargs) -> SPGraph[X, Tuple[Y, Z]]:
        """Ordered parallel composition of Series-Parallel Graph. Self becomes left."""
        if not isinstance(sibling, SPGraph):
            sibling = LeafNode.from_python(sibling, **kwargs)
        return self.merge(ExistsNode, self, sibling)

    def then(self, sibling: SPGraph[Y, Z] | Callable[[Y], Z], **kwargs) -> SPGraph[X, Z]:
        """Sequential (series) composition of Series-Parallel Graph. Self becomes parent."""
        if not isinstance(sibling, SPGraph):
            sibling = LeafNode.from_python(sibling, **kwargs)
        return self.merge(SeriesNode, self, sibling)

    before = then

    def after(self, sibling: SPGraph[W, X] | Callable[[W], X], **kwargs) -> SPGraph[W, Y]:
        """f.after(g)(x) <=> f(g)(x) <=> (f.g)(x)"""
        if not isinstance(sibling, SPGraph):
            sibling = LeafNode.from_python(sibling, **kwargs)
        return sibling.then(self)

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
        raise SPException("Load .sexpr module")

    def build_stream(self) -> "ThreadedSPGraph[X, Y]":
        """Placeholder for patch"""
        raise SPException("Load .stream module")


class InternalNode(SPGraph[X, Y]):
    """Compositions"""
    def __init__(self, inner: Tuple[SeriesNode[X, Y] | LeafNode[X, Y] | ForallNode[Any, Any], ...], **kwargs) -> None:
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

    def asdict(self) -> Dict[str, str]:
        return {"type": self.type().name, "children": [c.asdict() for c in self.inner]}


class ForallNode(ParallelNode[X, Y]):
    """Broadcasting behavior. Default is index getter"""
    def type(self) -> NodeType:
        return NodeType.PARALLEL_FORALL

    def __call__(self, x: X) -> Y:
        assert len(x) == len(self.inner), (x, self.inner)
        return cast(Y, tuple(f(_x) for _x, f in zip(x, self.inner)))
    

class ExistsNode(ParallelNode[X, Y]):
    """Load balancing behavior. Default is round-robin"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._index = 0

    def type(self) -> NodeType:
        return NodeType.PARALLEL_EXISTS

    def __call__(self, x: X) -> Y:
        ret = self.inner[self._index % len(self.inner)](x)
        self._index += 1
        return cast(Y, ret)


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
