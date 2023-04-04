import spgraph
import pytest
import time
import contextlib

@spgraph.function
def add_one(x):
    assert isinstance(x, (int, float))
    return x + 1


def mul_two(x):
    assert isinstance(x, (int, float))
    return 2 * x


def test_build_series():
    processor = add_one.then(mul_two).build_stream()
    processor.start()
    try:
        assert list(processor([1, 2, 3, 4, 5])) == [4, 6, 8, 10, 12]
    finally:
        processor.stop_and_join()


def test_build_series_long():
    processor = add_one.then(add_one).then(add_one).then(add_one).build_stream()
    processor.start()
    try:
        assert list(processor([1, 2, 3, 4, 5])) == [5, 6, 7, 8, 9]
    finally:
        processor.stop_and_join()


def test_build_forall():
    processor = add_one.and_(mul_two).build_stream()
    processor.start()
    processor.put((1, 1))
    assert processor.get(min_timeout=0.1, retry_on_timeout=True) == (2, 2)  #, (3, 4), (4, 6)]
    processor.stop_and_join()


def test_timeout():
    @spgraph.function
    def timeconsuming_job(x):
        time.sleep(1)
        return x
    processor = timeconsuming_job.build_stream()
    processor.start()

    # interface 1. Run by calling it
    assert list(processor([1])) == [1]

    # interface 2. Run by manually feeding it
    processor.put(1)
    assert processor.get(min_timeout=0.1, retry_on_timeout=True) == 1
    processor.stop_and_join(stopped_ok=True, no_raise=True)


class SomeError(Exception):
    pass

@spgraph.function
def raise_(_):
    raise SomeError


def test_raise():
    processor = raise_.build_stream()
    processor.start()

    # simplest case
    with pytest.raises(SomeError):
        assert list(processor([1, 2, 3, 4, 5])) == [5, 6, 7, 8, 9]
    processor.stop_and_join(stopped_ok=True, no_raise=True)


def test_raise_series():
    # series
    processor = raise_.then(mul_two).build_stream()
    processor.start()

    with pytest.raises((SomeError, spgraph.MultipleErrors)):
        processor.put(1)
        _ = processor.get(min_timeout=0.1, retry_on_timeout=True)
    processor.stop_and_join(stopped_ok=True, no_raise=True)


def test_raise_forall():
    processor = raise_.and_(mul_two).build_stream()
    processor.start()

    with pytest.raises((SomeError, spgraph.MultipleErrors)):
        processor.put((1, 2))
        try:
            _ = processor.get(min_timeout=0.1, retry_on_timeout=True)
        finally:
            processor.stop_and_join(stopped_ok=True, no_raise=True)


def test_raise_exists():
    processor = raise_.or_(mul_two).build_stream()
    processor.start()

    with pytest.raises(SomeError):
        processor.put(1)
        try:
            _ = processor.get(min_timeout=0.1, retry_on_timeout=True)
        finally:
            processor.stop_and_join(stopped_ok=True, no_raise=True)


def test_raise_mixed_simpler():
    func = add_one.and_(mul_two).then(raise_)
    processor = func.build_stream()
    processor.start()

    with pytest.raises(SomeError):
        processor.put((1, 2))
        _ = processor.get(min_timeout=0.1, retry_on_timeout=True)
    processor.stop_and_join(stopped_ok=True, no_raise=True)
    assert not processor.inner[0].inner[0].is_alive()
    assert not processor.inner[0].inner[1].is_alive()
    assert not processor.inner[1].is_alive()


def test_raise_mixed_2():
    func = (add_one.then(raise_)).and_(add_one.then(mul_two))
    processor = func.build_stream()
    processor.start()

    with pytest.raises((SomeError, spgraph.MultipleErrors)):
        processor.put((1, 2))
        _ = processor.get(min_timeout=0.1, retry_on_timeout=True)
    processor.stop_and_join(stopped_ok=True, no_raise=True)


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
        processor.stop_and_join()


def test_cannot_run_when_not_started():
    processor = add_one.then(mul_two).build_stream()
    with pytest.raises(spgraph.StateError):
        assert list(processor([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])) == [4, 6, 8, 10, 12]


def test_threads_can_only_be_started_once():
    processor = add_one.then(mul_two).build_stream()
    processor.start()
    processor.stop_and_join()
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
    processor.stop_and_join()


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
    processor.stop_and_join()


def test_report_smoke():
    processor = add_one.then(mul_two).build_stream()
    processor.report()


def test_complex_stream():
    @spgraph.function
    def shot2camshots(shot):
        return tuple(shot)

    @spgraph.function
    def camshot2cleinput_double(camshot):
        camera_name, pixels = camshot
        piles = [1, 2, 3, 4, 5]
        images = [11, 12, 13, 14, 15]
        return [
            (piles[:3], images[:3], camera_name),
            (piles[3:], images[3:], camera_name),
        ]

    @spgraph.function
    def cleinput2im_piles_segs_list(args):
        pd_piles, im_piles, camera_name = args
        segs_list = [i + 100 for i in im_piles]
        return camera_name, im_piles, pd_piles, segs_list

    @spgraph.function
    def segs_list2piles_list(args):
        camera_name, im_piles, pd_piles, segs_list = args
        piles_list = [segs_list]
        return camera_name, im_piles, piles_list

    @spgraph.function
    def piles_list2cropped_lines(args):
        camera_name, im_piles, piles_list = args
        cropped_lines = [9, 8, 7]
        return piles_list, cropped_lines, camera_name

    @spgraph.function
    def cropped_lines2color_recognized_piles(args):
        piles_list, cropped_lines, camera_name = args
        return piles_list, camera_name

    @spgraph.function
    def color_recognized_piles2piles(args):
        piles_list, camera_name = args
        piles = [pile for piles in piles_list for pile in piles]
        return piles

    import itertools
    @spgraph.function
    def piles2merged(piles_list):
        # check all inputs are available
        piles = list(itertools.chain(*itertools.chain(*piles_list)))
        merged_piles = [999, 998, 997]
        return merged_piles

    cleinput2piles = (
        cleinput2im_piles_segs_list
        .then(segs_list2piles_list)
        .then(piles_list2cropped_lines)
        .then(cropped_lines2color_recognized_piles)
        .then(color_recognized_piles2piles)
    )
    camshot2piles = (
        camshot2cleinput_double.then(
            cleinput2piles
            .and_(cleinput2piles)
        )
    )
    shot2merged_piles = (
        shot2camshots
        .then(
            camshot2piles
            .and_(camshot2piles)
            .and_(camshot2piles)
            .and_(camshot2piles)
        )
        .then(piles2merged)
    )
    assert shot2merged_piles((('ll', 1), ('lr', 2), ('rl', 3), ('rr', 4))) == [999, 998, 997]

    xs = shot2merged_piles.build_stream()
    xs.start()
    xs.put((('ll', 1), ('lr', 2), ('rl', 3), ('rr', 4)))
    xs.get(min_timeout=1, retry_on_timeout=False)
    xs.stop_and_join()
