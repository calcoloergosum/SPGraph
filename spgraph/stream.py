"""Streaming interface for Serial-Parallel Graph.
The intended for high-level interface of multithreading.
"""
from __future__ import annotations

import ctypes
import inspect
import itertools
import queue
import threading
from typing import (Any, Callable, Generic, Iterable, Iterator, List,
                    Tuple, Type, cast)

from typing_extensions import Self

from .spgraph import (LeafNode, ParallelNode, SeriesNode, ShouldNotBeReachable,
                      SPException, SPGraph, W, X, Y, Z)


def _async_raise(tid: int, exctype: Type[Exception]) -> None:
    '''Raises an exception in the threads with id tid'''
    if not inspect.isclass(exctype):
        raise TypeError("Only types can be raised (not instances)")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid),
                                                     ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    if res != 1:
        # "if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


class ThreadWithExc(threading.Thread):
    '''A thread class that supports raising an exception in the thread from
       another thread.

       credit: https://stackoverflow.com/questions/323972/is-there-any-way-to-kill-a-thread
    '''
    # pylint: disable=attribute-defined-outside-init, access-member-before-definition, protected-access
    def _get_my_tid(self) -> int:
        """determines this (self's) thread id

        CAREFUL: this function is executed in the context of the caller
        thread, to get the identity of the thread represented by this
        instance.
        """
        if not self.is_alive():
            raise threading.ThreadError("the thread is not active")

        # do we have it cached?
        if hasattr(self, "_thread_id"):
            return self._thread_id  # type: ignore

        # no, look for it in the _active dict
        for tid, tobj in threading._active.items():  # type: ignore
            assert isinstance(tid, int)
            if tobj is self:
                self._thread_id = tid
                return tid

        raise AssertionError("could not determine the thread's id")

    def raise_exc(self, exctype: Type[Exception]) -> None:
        """Raises the given exception type in the context of this thread.

        If the thread is busy in a system call (time.sleep(),
        socket.accept(), ...), the exception is simply ignored.

        If you are sure that your exception should terminate the thread,
        one way to ensure that it works is:

            t.raise_exc( SomeException )
            while t.is_alive():
                time.sleep( 0.1 )
                t.raise_exc( SomeException )

        If the exception is to be caught by the thread, you need a way to
        check that your thread has caught it.

        CAREFUL: this function is executed in the context of the
        caller thread, to raise an exception in the context of the
        thread represented by this instance.
        """
        _async_raise( self._get_my_tid(), exctype )


class StreamingError(SPException):
    """General exception class"""


class StateError(StreamingError):
    """Unhandled state error; e.g. starting a thread twice"""


QueueUnit = queue.Queue


class ControlException(StreamingError):
    """Exceptions that is used for controlling thread behavior"""


class Stop(ControlException):
    """Stop signal for threads"""
    @classmethod
    def wrap(cls, func: Callable[[X], None]) -> Callable[[X], None]:
        """On `Stop`, the function stops"""
        def _func(*args: Any, **kwargs: Any) -> None:
            try:
                func(*args, **kwargs)
            except Stop:
                pass
        return _func


class StreamProcessorUnit(ThreadWithExc, Generic[X, Y]):
    """Threaded wrapper for a function, with queue interface"""
    def __init__(self, func: Callable[[X], Y],
            in_queue: QueueUnit[X | EOBToken] | Junction[Any, X | EOBToken],
            out_queue: QueueUnit[Y | EOBToken] | Junction[Y | EOBToken, Any],
    ) -> None:
        self.func = func
        self.in_queue = in_queue
        self.out_queue = out_queue
        super().__init__(target=self._run)

    @Stop.wrap
    def _run(self) -> None:
        while True:
            try:
                x = self.in_queue.get(timeout=1)
            except queue.Empty:
                continue

            y: Y | EOBToken
            if isinstance(x, EOBToken):
                y = x
            else:
                y = self.func(x)
            self.out_queue.put(y)


class Junction(ThreadWithExc, Generic[X, Y]):
    """
    An object that handle multiple queues to merge in a synchronized manner.
    It is called `meet` and `join` in poset."""
    def __init__(self,
        in_queues: List[QueueUnit[X] | Junction[Any, X]],
        out_queues: List[QueueUnit[Y] | Junction[Y, Any]],
    ) -> None:
        super().__init__()
        self.in_queues = in_queues
        self.out_queues = out_queues

    @classmethod
    def new(cls, n_in: int, n_out: int) -> QueueUnit[Any] | Junction[Any, Any]:
        """Constructor"""
        assert n_in > 0 or n_out > 0
        # start node
        if n_in == 0 and n_out == 0:
            raise ValueError
        if n_in == 0 and n_out == 1:
            return QueueUnit()
        if n_in == 1 and n_out == 0:
            return QueueUnit()
        if n_in == 1 and n_out == 1:
            return QueueUnit()
        return cls(
            in_queues=[QueueUnit() for _ in range(max(n_in, 1))],
            out_queues=[QueueUnit() for _ in range(max(n_out, 1))],
        )

    @Stop.wrap
    def run(self) -> None:
        while True:
            self._put(self.out_queues, self._get(self.in_queues))  # type: ignore

    @staticmethod
    def _put(qs: List[QueueUnit[Y]], item: Y) -> None:
        for q in qs:
            q.put(item)

    def put(self, item: Y) -> None:
        """Goosetyping queue.Queue.put"""
        return self._put(self.out_queues, item)

    @staticmethod
    def _get(qs: List[QueueUnit[Any]]) -> Any | Tuple[Any, ...]:
        items = []
        for q in qs:
            while True:
                try:
                    item = q.get(timeout=1)
                    break
                except queue.Empty:
                    continue
            items.append(item)

        if len(qs) == 1:
            return items[0]
        return tuple(items)

    def get(self, timeout: int) -> Any:  # pylint: disable=unused-argument
        """Goosetyping queue.Queue.get"""
        return self._get(self.out_queues)


class ThreadedSPGraph(Generic[X, Y]):
    """Threaded Series-Parallel Graph, which handles streamed data"""

    @classmethod
    def build(cls, spgraph: SPGraph[X, Y]) -> Self:
        """Build from given Series-Parallel Graph"""
        j_in, j_out = Junction.new(1, cls.find_n(spgraph, True)), Junction.new(cls.find_n(spgraph, False), 1)
        juncts, procs = cls._build(j_in, j_out, spgraph)
        return cls([j_in] + juncts + [j_out], procs, spgraph)

    @classmethod
    def find_n(cls, spgraph: SPGraph, in_out: bool) -> int:
        """find required number of junction parameters.
        `in_out` == True then `in`, otherwise `out`
        """
        if isinstance(spgraph, SeriesNode):
            return cls.find_n(spgraph.inner[0 if in_out else -1], in_out)
        if isinstance(spgraph, ParallelNode):
            return len(spgraph.inner)
        if isinstance(spgraph, LeafNode):
            return 1
        raise ShouldNotBeReachable

    @classmethod
    def _build(cls, j_in: Junction[W, X], j_out: Junction[Y, Z], spgraph: SPGraph[X, Y]) -> Tuple[
        List[Junction[Any, Any]],  # junctions created in the call
        List[StreamProcessorUnit[Any, Any] | StreamPreBuiltProcessor[Any, Any]],  # processors created in the call
    ]:
        """Returns junctions and procs sorted in depth first order"""
        if isinstance(spgraph, LeafNode):
            assert isinstance(j_in, QueueUnit)
            assert isinstance(j_out, QueueUnit)

            proc: StreamPreBuiltProcessor[Any, Any] | StreamProcessorUnit[Any, Any]
            if (batchsize := getattr(spgraph, 'batchsize', 0)) > 0:
                assert isinstance(spgraph, LeafNode)
                proc = StreamPreBuiltProcessor(spgraph.inner,
                    in_queue=j_in, out_queue=j_out,
                    batchsize=batchsize
                )
            else:
                proc = StreamProcessorUnit(
                func=spgraph.inner,
                in_queue=j_in,
                out_queue=j_out,
            )
            return [], [proc]

        if isinstance(spgraph, SeriesNode):
            ns_io = []
            for g in spgraph.inner:
                if isinstance(g, ParallelNode):
                    n = len(g.inner)
                elif isinstance(g, LeafNode):
                    n = 1
                else:
                    raise SPException(f"Unknown type {type(g)}")
                ns_io.append(n)
            juncts_between = [Junction.new(n1, n2) for n1, n2 in zip(ns_io, ns_io[1:])]
            assert len(juncts_between) == len(spgraph.inner) - 1

            juncts = []
            procs = []
            for _j_in, _j_out, g in zip(
                [j_in] + juncts_between,
                juncts_between + [j_out],
                spgraph.inner
            ):
                _juncts, _procs = cls._build(_j_in, _j_out, g)
                juncts.extend(_juncts)
                juncts.append(_j_out)
                procs.extend(_procs)
            juncts = juncts[:-1]
            return juncts, procs

        if isinstance(spgraph, ParallelNode):
            assert len(j_out.in_queues) == len(j_in.out_queues) == len(spgraph.inner)
            juncts = []
            procs = []
            for i, g in enumerate(spgraph.inner):
                _juncts, _procs = cls._build(j_in.out_queues[i], j_out.in_queues[i], g)
                juncts.extend(_juncts)
                procs.extend(_procs)
            return juncts, procs

        raise SPException(f"Unknown type {type(spgraph)}")

    def __init__(self,
                 junctions: List[Junction[Any, Any] | QueueUnit[Any]],
                 procs: List[StreamPreBuiltProcessor[Any, Any] | StreamProcessorUnit[Any, Any]],
                 inner: SPGraph[X, Y],
    ) -> None:
        self.junctions = junctions
        self.procs = procs
        self.inner = inner
        self.is_running = False

    @property
    def in_queue(self) -> QueueUnit[X] | Junction[X, Any]:
        """Get the queue which is the input node of this graph"""
        q = self.junctions[0]
        assert isinstance(q, QueueUnit | Junction), \
            "Something went wrong when building. (First node is not a queue or a junction)"
        return q

    @property
    def out_queue(self) -> QueueUnit[Y] | Junction[Any, Y]:
        """Get the queue which is the output node of this graph"""
        q = self.junctions[-1]
        assert isinstance(q, QueueUnit | Junction), \
            "Something went wrong when building. (Last node is not a queue or a junction)"
        return q

    def start(self) -> None:
        """Start all the child threads. Raises `StateError` if already running"""
        if self.is_running:
            raise StateError("Already running")
        for obj in (self.junctions + self.procs):
            if isinstance(obj, QueueUnit):
                continue
            obj.start()
        self.is_running = True

    def stop(self) -> None:
        """Stop all the child threads. Raises `StateError` if not running"""
        if not self.is_running:
            raise StateError("It is not running")
        for obj in (self.junctions + self.procs):
            if isinstance(obj, QueueUnit):
                continue
            obj.raise_exc(Stop)
        self.is_running = False

    def __call__(self, values: Iterable[X]) -> Iterator[Y]:
        if not self.is_running:
            raise StateError("Processors have not started")
        token = EOBToken()

        n = 0
        for val in itertools.chain(values, [token]):
            self.in_queue.put(cast(X, val))
            n += 1

        for _ in range(n - 1):
            while True:
                try:
                    yield self.out_queue.get(timeout=1)
                    break
                except queue.Empty:
                    continue

        # discard token
        _ = self.out_queue.get(timeout=1)

    def report_queues(self) -> None:
        """Report current queue sizes for each queue
        e.g.
        add_one => mul_two        ( 100  )
        mul_two => mean_normalize (  0   )
        """
        return
        # for obj in self.junctions:
        #     if isinstance(obj, queue.Queue):
        #         xstr = self._format_queue(next(self.inner.inner.predecessors(nid), None))
        #         ystr = self._format_queue(next(self.inner.inner.successors(nid), None))
        #         print(f"{xstr: ^20} => {ystr: ^20} ({obj.qsize(): ^10})")
        #         continue
        #     if isinstance(obj, Junction):
        #         print(f"Junction[{nid}]")
        #         print(' '*4 + ' '.join([f"{q.qsize(): >6}" for q in obj.in_queues]))
        #         print(' '*4 + ' '.join([f"{q.qsize(): >6}" for q in obj.out_queues]))
        #         continue

    def _format_queue(self) -> str:
        return ''
        # if nid is None:
        #     return '()'
        # xproc = self.nid2obj[nid]
        # assert isinstance(xproc, StreamProcessorUnit)
        # return str(xproc.func.__name__)

    def draw(self) -> None:
        """Draw current graph"""


def build_stream(self: SPGraph[X, Y]) -> ThreadedSPGraph[X, Y]:
    """Build streamed processor from graph"""
    return ThreadedSPGraph.build(self)


# Batch control
class EOBToken:  # pylint: disable=too-few-public-methods
    """End of batch token; force to split batching"""


class StreamPreBuiltProcessor(ThreadWithExc, Generic[X, Y]):
    """Wrapper for functions that are streaming-ready;
    i.e. functions that have prototype of [X] -> [Y].
    In that case, "batchsize" is required.
    """
    def __init__(self,
            func: Callable[[Iterable[X]], Iterable[Y]],
            in_queue: QueueUnit[X | EOBToken] | Junction[Any, X | EOBToken],
            out_queue: QueueUnit[Y | EOBToken] | Junction[Y | EOBToken, Any],
            batchsize: int,
    ) -> None:
        self.func = func
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.batchsize = batchsize
        super().__init__(target=self._run)

    @Stop.wrap
    def _run(self) -> None:
        while True:
            batch = []
            eof = None
            for _ in range(self.batchsize):
                while True:
                    try:
                        x = self.in_queue.get(timeout=1)
                        break
                    except queue.Empty:
                        continue
                if isinstance(x, EOBToken):
                    eof = x
                    break
                batch += [x]

            if len(batch) > 0:
                for y in self.func(batch):
                    self.out_queue.put(y)

            if eof is not None:
                self.out_queue.put(eof)


SPGraph.build_stream = build_stream
