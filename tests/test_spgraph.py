import spgraph
import pytest


@spgraph.function
def add_one(x):
    return x + 1


@spgraph.function
def mul_two(x):
    return 2 * x


# Series composition
def test_series_composition_after():
    # step by step:
    assert add_one(2) == 3
    assert mul_two(3) == 6
    assert mul_two.after(add_one)(2) == 6

    # step by step:
    assert add_one(2) == 3
    assert add_one(3) == 4
    assert add_one.after(add_one)(2) == 4
    # g(f(x))
    # = g after f


def test_series_composition_before():
    assert mul_two(2) == 4
    assert add_one(4) == 5
    assert mul_two.then(add_one)(2) == mul_two.before(add_one)(2) == 5
    # g(f(x))
    # = f then g
    # = f before g


def test_series_composition_before_nested():
    with pytest.raises(spgraph.RepresentationError):
        spgraph.SeriesNode([add_one.then(add_one), add_one.then(add_one)])

    assert (add_one.then(add_one)).then(mul_two.then(mul_two))(2) == 16


# Parallel composition
def test_parallel_composition_and():
    assert add_one(2) == 3
    assert add_one(3) == 4
    assert mul_two(2) == 4
    assert mul_two(3) == 6
    assert add_one.and_(mul_two)((2, 3)) == (3, 6)
    assert mul_two.and_(add_one)((2, 3)) == (4, 4)


def test_parallel_composition_or():
    assert add_one(2) == 3
    assert mul_two(2) == 4
    func = add_one.or_(mul_two)
    assert func(2) == 3
    assert func(2) == 4
    func = mul_two.or_(add_one)
    assert func(2) == 4
    assert func(2) == 3


def test_build_composition_and_nested():
    with pytest.raises(spgraph.RepresentationError):
        spgraph.ForallNode([add_one.and_(add_one), add_one.and_(add_one)])

    assert (add_one.and_(add_one)).and_(mul_two.and_(mul_two))((2, 2, 2, 2)) == (3, 3, 4, 4)
