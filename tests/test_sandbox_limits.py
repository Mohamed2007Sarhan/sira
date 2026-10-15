import os
import threading
import unittest

from sira.sandbox import _count_user_procs


@unittest.skipUnless(os.path.isdir("/proc"), "needs /proc")
class TaskCount(unittest.TestCase):
    def test_threads_are_counted(self):
        before = _count_user_procs()
        stop = threading.Event()
        ts = [threading.Thread(target=stop.wait, daemon=True) for _ in range(40)]
        for t in ts:
            t.start()
        try:
            self.assertGreaterEqual(_count_user_procs() - before, 30)   # RLIMIT_NPROC counts threads, so must we
        finally:
            stop.set()
            for t in ts:
                t.join()


if __name__ == "__main__":
    unittest.main()
