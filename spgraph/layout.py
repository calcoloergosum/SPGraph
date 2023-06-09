"""In charge of calculating position of each node"""
# NOTE: using symmetry is not implemented yet
# from .symmetry import get_symmetry
from typing import Any, List

from .spgraph import ExistsNode, ForallNode, LeafNode, SeriesNode, SPGraph

# def set_xy(self: SPGraph) -> None:
#     if getattr(self, 'xy_src', None) is not None:
#         return

#     # Fit everything in [-height, height] x [-width, width]
#     if isinstance(self, LeafNode):
#         setattr(self, 'xy_src', [0, -1])
#         setattr(self, 'xy_dst', [0,  1])
#         setattr(self, 'width', 1)
#         setattr(self, 'height', 1)
#         return
#     assert isinstance(self, InternalNode)

#     for c in self.inner:
#         set_xy(c)

#     width = 0
#     height = 0
#     if isinstance(self, SeriesNode):
#         for c in self.inner:
#             translate(c, 0, height)
#             width = max(width, c.width)
#             height += c.height

#     if isinstance(self, ParallelNode):
#         for c in self.inner:
#             translate(c, 0, width)
#             width += c.width
#             height = max(height, c.height)

#     setattr(self, 'xy_src', [0, -height])
#     setattr(self, 'xy_dst', [0, height])
#     setattr(self, 'width', width)
#     setattr(self, 'height', height)

#     _ = self.get_symmetry()


# def translate(self: SPGraph, x: float, y: float) -> None:
#     self.xy_src[0] -= x
#     self.xy_src[1] -= y
#     self.xy_dst[0] -= x
#     self.xy_dst[1] -= y
#     if isinstance(self, LeafNode):
#         return
#     for c in self.inner:
#         translate(c, x, y)


def draw(self: SPGraph[Any, Any], pretty: bool = False) -> str:
    """Draw series-parallel graph.
    >>> import spgraph
    >>> print(spgraph.function(lambda x: x + 1).draw(pretty=True))
    ┯
    │
    ┷
    """
    # pylint: disable=invalid-unary-operand-type,import-outside-toplevel
    import numpy as np

    # set_xy(self)
    val = _draw(self, True, True)
    if not pretty:
        return '\n'.join(val)

    # pretty print
    arr = np.array([list(l) for l in val], dtype='<U1')
    t_empty = np.vstack(([[True] * arr.shape[1]], arr[:-1, :] == ' '))
    b_empty = np.vstack((arr[1:, :] == ' ', [[True] * arr.shape[1]]))
    l_empty = np.hstack(([[True]] * arr.shape[0], arr[:, :-1] == ' '))
    r_empty = np.hstack((arr[:, 1:] == ' ', [[True]] * arr.shape[0]))
    sym_hor = arr == '-'
    sym_ver = arr == '|'
    arr[sym_ver] = '│'

    # 4 empty
    arr[sym_hor * l_empty * r_empty * t_empty * b_empty] = "+"

    # 3 empty
    arr[sym_hor * l_empty * r_empty * t_empty * ~b_empty] = '┯'
    arr[sym_hor * l_empty * r_empty * ~t_empty * b_empty] = '┷'
    arr[sym_hor * l_empty * ~r_empty * t_empty * b_empty] = '╺'
    arr[sym_hor * ~l_empty * r_empty * t_empty * b_empty] = '╸'

    # 2 empty
    arr[sym_hor * l_empty * r_empty * ~t_empty * ~b_empty] = '┿'
    arr[sym_hor * l_empty * ~r_empty * t_empty * ~b_empty] = '┍'
    arr[sym_hor * ~l_empty * r_empty * t_empty * ~b_empty] = '┑'
    arr[sym_hor * l_empty * ~r_empty * ~t_empty * b_empty] = '┕'
    arr[sym_hor * ~l_empty * r_empty * ~t_empty * b_empty] = '┙'
    arr[sym_hor * ~l_empty * ~r_empty * t_empty * b_empty] = '━'

    # 1 empty
    arr[sym_hor * l_empty * ~r_empty * ~t_empty * ~b_empty] = '┝'
    arr[sym_hor * ~l_empty * r_empty * ~t_empty * ~b_empty] = '┥'
    arr[sym_hor * ~l_empty * ~r_empty * t_empty * ~b_empty] = '┯'
    arr[sym_hor * ~l_empty * ~r_empty * ~t_empty * b_empty] = '┷'

    # 0 empty
    arr[sym_hor * ~l_empty * ~r_empty * ~t_empty * ~b_empty] = '┿'

    ret = '\n'.join([''.join(l) for l in arr.tolist()])
    assert '-' not in ret
    assert '|' not in ret

    return ret


def _draw(self: SPGraph[Any, Any], draw_s: bool, draw_t: bool) -> List[str]:
    # set_xy(self)
    mat = []
    if isinstance(self, LeafNode):
        border = "-"
        mat = ["|"]
    elif isinstance(self, SeriesNode):
        children = [_draw(c, True, True) for c in self.inner]
        max_width = max(len(c[0]) for c in children)
        border = "-" * max_width
        children = [_extend_box_width(c, max_width) for c in children]
        for c in children:
            if len(mat) == 0:
                mat += c
                continue
            l = max([mat[-1], c[0]], key=lambda l: sum(c != ' ' for c in l))
            mat = mat[:-1] + [l] + c[1:]
        mat = mat[1:-1]  # ignore first and last sink
    else:
        assert isinstance(self, (ForallNode, ExistsNode))
        children = [_draw(c, False, False) for c in self.inner]
        max_height = max(len(c) for c in children)
        children = [_extend_box_height(c, max_height) for c in children]
        for lines in zip(*children):
            mat += [" " + " ".join(lines) + " "]
        border = '-' * len(mat[-1])

    if draw_s:
        mat = [border] + mat
    if draw_t:
        mat = mat + [border]
    return mat


def _extend_box_height(c: List[str], height: int) -> List[str]:
    """Simple function that makes it long
             |
    |        |
    -   =>   -
    |        |
             |
    """
    h = len(c)
    if h > height:
        raise ValueError(f"{h} > {height}")
    diff = height - h
    if diff == 0:
        return c

    c = (diff // 2) * [c[0]] + c + ((diff + 1) // 2) * [c[-1]]
    return c


def _extend_box_width(c: List[str], width: int) -> List[str]:
    """Simple function that makes it long
    " | "     "  |  "
    " - "  => "  -  "
    " | "     "  |  "
    """
    w = len(c[0])
    if w > width:
        raise ValueError(f"{w} > {width}")
    diff = width - w
    ret = []
    for l in c:
        ret += [" " * (diff // 2) + l + " " * ((diff + 1) // 2)]
    return ret

SPGraph.draw = draw
