import time
from spgraph.stream import ThreadWithExc

class KillException(Exception):
    pass


def test_thread_with_exc():
    def wait_forever():
        try:
            while True:
                time.sleep(1)
        except KillException:
            return

    thread = ThreadWithExc(target=wait_forever)
    thread.start()
    thread.raise_exc(KillException)

    for _ in range(50):  # wait for maximum 5 seconds
        if not thread.is_alive():
            break
        time.sleep(0.1)
    else:
        raise AssertionError("Thread is not killed in 5 seconds")
