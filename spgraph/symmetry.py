"""
[1] S.-H. Hong, P. Eades, and S.-H. Lee, “Drawing series parallel digraphs symmetrically 6,” Computational Geometry, 2000.
TODO: rotational symmetry
"""
from dataclasses import dataclass
from typing import Any, Tuple

from .misc import array2indexmap
from .spgraph import InternalNode, LeafNode, ParallelNode, SeriesNode, SPGraph

Label = Tuple[int, ...]
Code = int


def vertical_check(self: SPGraph) -> bool:
    """Ever so slightly modified from the paper to retain the symmetry found."""
    if isinstance(self, LeafNode):
        return True

    if isinstance(self, SeriesNode):
        return all(vertical_check(v) for v in self.inner)

    assert isinstance(self, ParallelNode)
    # classify isomorphisms
    _assign_code_vertical(self)
    code2idxs = array2indexmap(c.code_vertical for c in self.inner)
    # classify isomorphisms done

    # load retained symmetry info if exists
    if (
        (asyms := getattr(self, 'asymparts_vertical', None)) is not None and
        asyms == []
    ):
        return True

    # retain symmetry info
    syms, asyms = [], []
    for idxs in code2idxs.values():
        if len(idxs) % 2 == 0:
            syms.append(idxs)
            continue
        asyms.append(idxs)

    if len(asyms) == 1:
        if vertical_check(self.inner[asyms[0][0]]):
            syms += [asyms[0]]
            asyms = []
    setattr(self, 'symparts_vertical', syms)
    setattr(self, 'asymparts_vertical', asyms)
    return asyms == []


def _assign_label_vertical(u) -> Label:
    if (label := getattr(u, 'label_vertical', None)) is not None:
        return label

    if isinstance(u, LeafNode):
        label = (0,)
        setattr(u, 'label_vertical', label)
    else:
        assert isinstance(u, InternalNode)
        _assign_code_vertical(u)

        codes = [c.code_vertical for c in u.inner]
        label = codes
        if isinstance(u, ParallelNode):
            label = sorted(label)
        label = tuple(label)
        setattr(u, 'label_vertical', label)
    return label


def _assign_code_vertical(u: SPGraph[Any, Any]) -> None:
    # Leaf
    if isinstance(u, LeafNode):
        setattr(u, 'code_vertical', 1)
        return

    assert isinstance(u, InternalNode)
    # Already done
    if (code := getattr(u.inner[0], 'code_vertical', None)) is not None:
        return

    labels = []
    for i, c in enumerate(u.inner):
        _label = _assign_label_vertical(c)
        labels += [_label]

    label2idxs = array2indexmap(labels)
    label2codes = {l: i for i, l in enumerate(label2idxs.keys())}

    for l in labels:
        code = label2codes[l]
        for i in label2idxs[l]:
            setattr(u.inner[i], 'code_vertical', code)


def horizontal_check(self: SPGraph[Any, Any]) -> bool:
    """Ever so slightly modified from the paper to retain the symmetry found."""
    if isinstance(self, LeafNode):
        return True

    if isinstance(self, ParallelNode):
        return all(horizontal_check(v) for v in self.inner)

    assert isinstance(self, SeriesNode)

    # load retained symmetry info if exists
    if (
        (asyms := getattr(self, 'asymparts_horizontal', None)) is not None and
        asyms == []
    ):
        return True

    # retain symmetry info
    tup = _assign_label_horizontal(self, reversed=False)

    syms, asyms = [], []
    for i, (l, r) in enumerate(zip(tup[:len(tup)//2], tup[:-(len(tup)+1)//2:-1])):
        idxs = (i, len(tup) - 1 - i)
        if l == r:
            syms.append(idxs)
        if l != r:
            asyms.append(idxs)

    if len(asyms) % 2 == 1:
        i = len(tup) // 2  # middle
        (syms if horizontal_check(self.inner[i]) else asyms).append([i])

    setattr(self, 'symparts_horizontal', syms)
    setattr(self, 'asymparts_horizontal', asyms)
    return asyms == []


def _assign_label_horizontal(u: SPGraph, reversed: bool) -> Label:
    if (label := getattr(u, 'label_horizontal', None)) is not None:
        return label

    if isinstance(u, LeafNode):
        label = (0,)
        setattr(u, 'label_horizontal', label)
    else:
        assert isinstance(u, InternalNode)
        _assign_code_horizontal(u)
        codes = [c.code_horizontal for c in u.inner]
        label = tuple(codes[::-1] if reversed else codes)
        if isinstance(u, ParallelNode):
            label = sorted(label)
        label = tuple(label)
        setattr(u, 'label_horizontal', label)
    return label


def _assign_code_horizontal(u: SPGraph[Any, Any]) -> None:
    # Leaf
    if isinstance(u, LeafNode):
        setattr(u, 'code_horizontal', 1)
        return

    assert isinstance(u, InternalNode)
    # Already done
    if (code := getattr(u.inner[0], 'code_horizontal', None)) is not None:
        return

    labels = []
    for i, c in enumerate(u.inner):
        _label = _assign_label_horizontal(c, reversed=i < len(u.inner) // 2)
        labels += [_label]

    label2idxs = array2indexmap(labels)
    label2codes = {l: i for i, l in enumerate(label2idxs.keys())}

    for l in labels:
        code = label2codes[l]
        for i in label2idxs[l]:
            setattr(u.inner[i], 'code_horizontal', code)


@dataclass
class OrientedPlanarSymmetry:
    """Planar automorphism group"""
    vertical: bool
    horizontal: bool
    # rotational: bool


def get_symmetry(self: SPGraph[Any, Any]) -> OrientedPlanarSymmetry:
    """Get all planar symmetries that given series parallel graph has,
    assuming that the order of edges are irrelevant.
    """
    return OrientedPlanarSymmetry(
        vertical_check(self),
        horizontal_check(self),
        # rotational_check(self),
    )


SPGraph.get_symmetry = get_symmetry
