from .symmetry import get_symmetry
from .spgraph import SPGraph, InternalNode, LeafNode, SeriesNode, ParallelNode
from typing import List


def set_xy(self: SPGraph) -> None:
    if getattr(self, 'xy_src', None) is not None:
        return

    # Fit everything in [-height, height] x [-width, width]
    if isinstance(self, LeafNode):
        setattr(self, 'xy_src', [0, -1])
        setattr(self, 'xy_dst', [0,  1])
        setattr(self, 'width', 1)
        setattr(self, 'height', 1)
        return
    assert isinstance(self, InternalNode)

    for c in self.inner:
        set_xy(c)

    width = 0
    height = 0
    if isinstance(self, SeriesNode):
        for c in self.inner:
            translate(c, 0, height)
            width = max(width, c.width)
            height += c.height

    if isinstance(self, ParallelNode):
        for c in self.inner:
            translate(c, 0, width)
            width += c.width
            height = max(height, c.height)

    setattr(self, 'xy_src', [0, -height])
    setattr(self, 'xy_dst', [0, height])
    setattr(self, 'width', width)
    setattr(self, 'height', height)

    _ = self.get_symmetry()


def translate(self: SPGraph, x: float, y: float) -> None:
    self.xy_src[0] -= x
    self.xy_src[1] -= y
    self.xy_dst[0] -= x
    self.xy_dst[1] -= y
    if isinstance(self, LeafNode):
        return
    for c in self.inner:
        translate(c, x, y)


def draw(self: SPGraph, pretty: bool = False) -> List[str]:
    set_xy(self)
    ret = _draw(self, True, True)
    if not pretty:
        return ret

    return [r.replace("-", "━") for r in ret]

def _draw(self: SPGraph, draw_s: bool, draw_t: bool) -> List[str]:
    set_xy(self)

    mat = []
    if isinstance(self, LeafNode):
        border = "-"
        mat = ["|"]
    elif isinstance(self, SeriesNode):
        children = [_draw(c, True, True) for c in self.inner]
        max_width = max(len(c[0]) for c in children)
        border = "-" * max_width
        children = [extend_box_width(c, max_width) for c in children]
        for c in children:
            if len(mat) == 0:
                mat += c
                continue
            l = max([mat[-1], c[0]], key=lambda l: sum([c != ' ' for c in l]))
            mat = mat[:-1] + [l] + c[1:]
        mat = mat[1:-1]  # ignore first and last sink
    else:
        assert isinstance(self, ParallelNode)
        children = [_draw(c, False, False) for c in self.inner]
        max_height = max(len(c) for c in children)
        children = [extend_box_height(c, max_height) for c in children]
        for lines in zip(*children):
            mat += [" " + " ".join(lines) + " "]
        border = '-' * len(mat[-1])

    if draw_s:
        mat = [border] + mat
    if draw_t:
        mat = mat + [border]
    return mat


def extend_box_height(c, height: int) -> List[str]:
    w, h = len(c[0]), len(c)
    if h > height:
        raise ValueError(f"{h} > {height}")
    diff = height - h
    if diff == 0:
        return c

    line = " " * ((w - 1) // 2) + "|" + " " * (w // 2)
    c = (diff // 2) * [line] + c + ((diff + 1) // 2) * [line]
    return c


def extend_box_width(c, width: int) -> List[str]:
    w, h = len(c[0]), len(c)
    if w > width:
        raise ValueError(f"{w} > {width}")
    diff = width - w
    ret = []
    for l in c:
        ret += [" " * (diff // 2) + l + " " * ((diff + 1) // 2)]
    return ret

SPGraph.draw = draw