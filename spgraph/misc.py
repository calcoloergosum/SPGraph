"""Things that I don't know where to put"""
from typing import Dict, Sequence, TypeVar, Iterable
from collections import defaultdict


T = TypeVar("T")
def array2indexmap(arr: Iterable[T]) -> Dict[T, Sequence[int]]:
    ret = defaultdict(list)
    for i, v in enumerate(arr):
        ret[v].append(i)
    return dict(ret)
