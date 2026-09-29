"""Imported by tpu.online's forkserver as it starts, so it dies with the trainer (#53).

The loader's workers come from the forkserver, and `tpu.run.init_worker` ties each to its
parent, which is the forkserver, not the trainer. Left to itself the forkserver ends only
once every copy of a pipe's write end is closed, and each worker it forks holds one. So a
trainer killed from outside -- the OOM killer, SIGKILL, an abort -- left the forkserver
waiting on its workers and the workers on the forkserver, for good, both holding the
trainer's stdout: `python -m tpu.online | tee` never ended, and neither did the script.
Measured before this: still there 11 minutes after the trainer.

With it the kernel kills the forkserver when the trainer goes, and the workers with it.
The trainer names itself in MORTAL_FORKSERVER_PARENT; in any other process this does nothing.
"""
import ctypes
import os
import signal

_parent = os.environ.pop('MORTAL_FORKSERVER_PARENT', None)
if _parent is not None:
    ctypes.CDLL(None).prctl(1, signal.SIGKILL)       # PR_SET_PDEATHSIG
    if os.getppid() != int(_parent):                 # the trainer died before that took hold
        os._exit(1)
