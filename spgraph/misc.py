"""Things that I don't know where to put"""
from typing import Dict, Sequence, TypeVar, Iterable
from collections import defaultdict


T = TypeVar("T")
def array2indexmap(arr: Iterable[T]) -> Dict[T, Sequence[int]]:
    """
    >>> array2indexmap(['a', 'b', 'c', 'a'])
    {'a': [0, 3], 'b': [1], 'c': [2]}
    """
    ret = defaultdict(list)
    for i, v in enumerate(arr):
        ret[v].append(i)
    return dict(ret)
