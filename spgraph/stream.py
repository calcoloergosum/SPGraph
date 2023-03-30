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

from .spgraph import (LeafNode, ForallNode, SeriesNode, ShouldNotBeReachable, ExistsNode,
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


class ThreadedSPGraph:
    """Threaded Series-Parallel Graph, which handles streamed data"""
    def put(self, x: X) -> None:
        raise NotImplementedError
    
    def get(self, timeout: float) -> Y:
        raise NotImplementedError

    def report(self) -> None:
        raise NotImplementedError

    @classmethod
    def build(cls, spgraph: SPGraph[X, Y]) -> Self:
        """Build from given Series-Parallel Graph"""
        q_in, q_out = QueueUnit(), QueueUnit()
        return cls._build(spgraph, q_in, q_out)
    
    @classmethod
    def _build(cls, spgraph: SPGraph[X, Y], q_in: QueueUnit, q_out: QueueUnit) -> Self:
        if isinstance(spgraph, SeriesNode):
            qs = [q_in] + [QueueUnit() for _ in spgraph.inner[:-1]] + [q_out]
            children = [
                cls._build(g, i, o)
                for g, i, o in zip(spgraph.inner, qs[:-1], qs[1:])
            ]
            return ThreadedSeriesNode(children)
        if isinstance(spgraph, ForallNode):
            _q_ins = [QueueUnit() for _ in spgraph.inner]
            _q_outs = [QueueUnit() for _ in spgraph.inner]
            children = [cls._build(g, i, o) for g, i, o in zip(spgraph.inner, _q_ins, _q_outs)]
            return ThreadedForallNode(children)
        if isinstance(spgraph, ExistsNode):
            _q_ins = [QueueUnit() for _ in spgraph.inner]
            _q_outs = [QueueUnit() for _ in spgraph.inner]
            children = [cls._build(g, i, o) for g, i, o in zip(spgraph.inner, _q_ins, _q_outs)]
            return ThreadedExistsNode(children)
        if isinstance(spgraph, LeafNode):
            if (batchsize := getattr(spgraph, 'batchsize', 0)) > 0:
                assert isinstance(spgraph, LeafNode)
                return ThreadedBatchLeafNode(
                    spgraph.inner,
                    in_queue=q_in, out_queue=q_out,
                    batchsize=batchsize
                )
            else:
                return ThreadedLeafNode(spgraph.inner, q_in, q_out)
        raise NotImplementedError(type(spgraph))


class ThreadedInternalNode(ThreadedSPGraph):
    def __init__(self, inner: ThreadedSPGraph) -> None:
        self.inner = inner
        self.is_running = False

    def start(self) -> None:
        """Start all the child threads. Raises `StateError` if already running"""
        if self.is_running:
            raise StateError("Already running")
        for obj in self.inner:
            obj.start()
        self.is_running = True

    def stop(self) -> None:
        """Stop all the child threads. Raises `StateError` if not running"""
        if not self.is_running:
            raise StateError("It is not running")
        for obj in self.inner:
            obj.raise_exc(Stop)
        self.is_running = False

    def __call__(self, values: Iterable[X]) -> Iterator[Y]:
        if not self.is_running:
            raise StateError("Processors have not started")
        token = FlushToken()

        n = 0
        for val in itertools.chain(values, [token]):
            self.put(cast(X, val))
            n += 1

        for _ in range(n - 1):
            while True:
                try:
                    yield self.get(timeout=1)
                    break
                except queue.Empty:
                    continue

        # discard eob token
        _token = self.get(timeout=1)
        assert token == _token, (token, _token)

    def report(self) -> None:
        for c in self.inner:
            c.report()


class ThreadedSeriesNode(ThreadedInternalNode):
    def put(self, x: X) -> None:
        self.inner[0].put(x)

    def get(self, timeout: float) -> Y:
        return self.inner[-1].get(timeout=timeout)


class ThreadedForallNode(ThreadedInternalNode):
    def put(self, x: X) -> None:
        if isinstance(x, FlushToken):
            for c in self.inner:
                c.put(x)
            return

        assert len(x) == len(self.inner)
        for _x, c in zip(x, self.inner):
            c.put(_x)

    def get(self, timeout: float) -> Y:
        ys = tuple(c.get(timeout=timeout) for c in self.inner)

        # eob token check
        is_token = False
        for y in ys:
            if isinstance(y, FlushToken):
                is_token = True
                continue
            if is_token:
                raise RuntimeError("Mixed EOBToken and non-EOBToken. Something is wrong")
        # eob token check done

        if is_token:
            return ys[0]
        return ys


class ThreadedExistsNode(ThreadedInternalNode):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._put_counter = 0
        self._get_counter = 0

    def put(self, x: X) -> None:
        if isinstance(x, FlushToken):
            for c in self.inner:
                c.put(x)
            return
        self.inner[self._put_counter % len(self.inner)].put(x)
        self._put_counter += 1

    def get(self, timeout: float) -> Y:
        i = self._get_counter % len(self.inner)
        ret = self.inner[i].get(timeout=timeout)

        if isinstance(ret, FlushToken):
            for _i, c in enumerate(self.inner):
                if i == _i:
                    continue
                assert isinstance(c.get(timeout=timeout), FlushToken)
            return ret

        self._get_counter += 1
        return ret


class ThreadedLeafNode(ThreadWithExc, ThreadedSPGraph, Generic[X, Y]):
    """Threaded wrapper for a function, with queue interface"""
    def __init__(self,
        inner: Callable[[X], Y],
        in_queue: QueueUnit[X | FlushToken],
        out_queue: QueueUnit[Y | FlushToken],
    ) -> None:
        self.inner = inner
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

            y: Y | FlushToken
            if isinstance(x, FlushToken):
                y = x
            else:
                y = self.inner(x)
            self.out_queue.put(y)

    def put(self, x: X) -> None:
        self.in_queue.put(x)

    def get(self, timeout: float) -> Y:
        return self.out_queue.get(timeout)

    def report(self):
        print(f"{self.in_queue.qsize(): >6} => {self.inner.__name__: ^20} => {self.out_queue.qsize(): <6}")


def build_stream(self: SPGraph[X, Y]) -> ThreadedSPGraph[X, Y]:
    """Build streamed processor from graph"""
    return ThreadedSPGraph.build(self)


# Batch control
class FlushToken:  # pylint: disable=too-few-public-methods
    """End of batch token; force to split batching"""


class ThreadedBatchLeafNode(ThreadWithExc, ThreadedSPGraph, Generic[X, Y]):
    """Wrapper for functions that are streaming-ready;
    i.e. functions that have prototype of [X] -> [Y].
    In that case, "batchsize" is required.
    """
    def __init__(self,
            inner: Callable[[Iterable[X]], Iterable[Y]],
            in_queue: QueueUnit[X | FlushToken],
            out_queue: QueueUnit[Y | FlushToken],
            batchsize: int,
    ) -> None:
        self.inner = inner
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
                if isinstance(x, FlushToken):
                    eof = x
                    break
                batch += [x]

            if len(batch) > 0:
                for y in self.inner(batch):
                    self.out_queue.put(y)

            if eof is not None:
                self.out_queue.put(eof)

    def put(self, x: X) -> None:
        self.in_queue.put(x)

    def get(self, timeout: float) -> Y:
        return self.out_queue.get(timeout)

    def report(self):
        print(f"{self.in_queue.qsize(): >6} => {self.inner.__name__: ^20} => {self.out_queue.qsize(): <6}")


SPGraph.build_stream = build_stream
