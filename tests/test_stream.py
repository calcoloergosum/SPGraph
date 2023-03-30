import spgraph
import pytest
import time
import contextlib

@spgraph.function
def add_one(x):
    return x + 1


def mul_two(x):
    return 2 * x


def test_build_series():
    processor = add_one.then(mul_two).build_stream()
    processor.start()
    try:
        assert list(processor([1, 2, 3, 4, 5])) == [4, 6, 8, 10, 12]
    finally:
        processor.stop()


def test_build_forall():
    processor = add_one.and_(mul_two).build_stream()
    processor.start()
    assert list(processor([(1, 1), (2, 2), (3, 3)])) == [(2, 2), (3, 4), (4, 6)]
    processor.stop()


def test_build_exists():
    waitsec = 0.5
    @spgraph.function
    def add_one(x):
        time.sleep(waitsec)
        return x + 1
    
    def mul_two(x):
        return 2 * x

    duration = None
    @contextlib.contextmanager
    def timer():
        nonlocal duration
        start = time.time()
        yield
        duration = time.time() - start

    processor = add_one.or_(mul_two).build_stream()
    processor.start()

    try:
        with timer():
            assert list(processor([1, 2])) == [2, 4]
        assert waitsec < duration < 2*waitsec, duration

        with timer():
            assert list(processor([1, 2, 3])) == [2, 4, 4]
        assert 2*waitsec < duration < 3*waitsec, duration
    finally:
        processor.stop()


# def test_build_parallel_or():
#     processor = add_one.or_(mul_two).build_stream()
#     processor.start()
#     assert list(processor([1, 2, 3])) == [(2, 2), (3, 4), (4, 6)]
#     processor.stop()


def test_cannot_run_when_not_started():
    processor = add_one.then(mul_two).build_stream()
    with pytest.raises(spgraph.StateError):
        assert list(processor([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])) == [4, 6, 8, 10, 12]


def test_threads_can_only_be_started_once():
    processor = add_one.then(mul_two).build_stream()
    processor.start()
    processor.stop()
    with pytest.raises(RuntimeError):
        processor.start()


# Batching test
def mean_normalize(x):
    # e.g. gpu accelerated processing
    avg = sum(x) / len(x)
    return [_x - avg for _x in x]


def test_batch_naive_call():
    assert spgraph.function(mean_normalize, batchsize=3)(10) == 0


def test_batch_basic():
    processor = (
        add_one
        .then(mul_two)
        .then(mean_normalize, batchsize=3)  # if not present, [2, 4, 6]
        .build_stream()
    )
    processor.start()
    assert list(processor(range(3))) == [-2, 0, 2]
    assert list(processor(range(4))) == [-2, 0, 2, 0]
    assert list(processor(range(5))) == [-2, 0, 2, -1, 1]
    processor.stop()


def shift_by_length(x):
    return [_x + len(x) for _x in x]


def test_batch_heterogeneous():
    processor = (
        add_one
        .then(mul_two)
        .then(mean_normalize, batchsize=3)
        .then(shift_by_length, batchsize=4)
    ).build_stream()
    processor.start()
    assert list(processor(range(3))) == [1, 3, 5]
    assert list(processor(range(4))) == [2, 4, 6, 4]
    assert list(processor(range(5))) == [2, 4, 6, 3, 2]
    processor.stop()


def test_report_smoke():
    processor = add_one.then(mul_two).build_stream()
    processor.report()
