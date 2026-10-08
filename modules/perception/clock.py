"""
modules.perception.clock
========================

Person 2's analytics (SpeedEstimator, CongestionAnalyzer, BottleneckAnalyzer)
measure durations with ``time.time()``.  That is exactly right for a live
camera, but wrong when the same code is replayed over a recorded file: a
30-second clip decoded in 6 seconds would report a 6-second congestion
window instead of 30.

Rather than edit Person 2's files, the adapter swaps the ``time`` object
inside the *loaded module namespace* for this shim, which returns
**video time**.  Their code is untouched on disk and keeps working
unchanged for a live camera (where the shim simply mirrors wall clock).

Set ``perception.vehicle.use_video_clock: false`` in the config to switch the
patch off and use the original wall-clock behaviour.
"""

from __future__ import annotations

import time as _real_time
from typing import Any, List


class VideoClock:
    """A clock the pipeline advances once per frame."""

    def __init__(self, live: bool = False):
        self.live = live
        self._t = 0.0
        self._origin = _real_time.time()

    def set_video_time(self, seconds: float) -> None:
        self._t = float(seconds)

    def time(self) -> float:
        if self.live:
            return _real_time.time()
        return self._origin + self._t

    # The vendored modules only ever use time.time / time.strftime /
    # time.localtime, so mirroring those three is enough for a drop-in swap.
    def strftime(self, fmt: str, struct_time=None) -> str:
        if struct_time is None:
            struct_time = self.localtime(self.time())
        return _real_time.strftime(fmt, struct_time)

    def localtime(self, seconds: float | None = None):
        return _real_time.localtime(self.time() if seconds is None else seconds)

    def sleep(self, seconds: float) -> None:                     # pragma: no cover
        _real_time.sleep(seconds)

    def __getattr__(self, item: str) -> Any:                     # pragma: no cover
        return getattr(_real_time, item)


def patch_module_clock(modules: List[Any], clock: VideoClock) -> None:
    """Point each loaded vendor module's ``time`` name at the shim.

    This changes the in-memory module object only -- the file in ``vendor/``
    is never modified.
    """
    for module in modules:
        if module is not None and hasattr(module, "time"):
            module.time = clock
