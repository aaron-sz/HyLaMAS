"""Verbose debug tracing for LatentMAS++ (opt-in via --debug).
"""

import os
import sys
import threading
import time

import theory

_ENABLED = False
_COLOR = False
_TTY = False
_T0 = time.time()
_CONTEXT = ""

# ANSI codes
_RESET = "\033[0m"
_DIM = "\033[2m"
_RED = "\033[31m"       # phase failed
_CYAN = "\033[36m"      # phase start
_GREEN = "\033[32m"     # phase done
_BGREEN = "\033[92m"    # decode finished
_YELLOW = "\033[33m"    # heartbeat / waiting
_BLUE = "\033[94m"      # live tok/s
_MAGENTA = "\033[95m"   # live latent steps


def _c(code: str, text: str) -> str:
    return f"{code}{text}{_RESET}" if _COLOR else text


def _fmt_dur(seconds: float) -> str:
    """Compact duration: '12.3s' under a minute, '2m03s' up to an hour, '1h04m' above."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds + 0.5), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def enable_windows_ansi(std_handle: int = -12) -> bool:
    """Turn on ANSI escape processing for the classic Windows console.
    Turn on ANSI escape processing for the classic Windows console.

    Windows Terminal handles ANSI natively; legacy conhost needs the console
    mode bit set. std_handle: -11 for stdout, -12 for stderr.
    """
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(std_handle)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        return bool(kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING))
    except Exception:
        return False


def enable() -> None:
    global _ENABLED, _COLOR, _TTY, _T0
    _ENABLED = True
    _T0 = time.time()
    # Make unicode marks safe on any console (Windows included).
    for stream in (sys.stderr, sys.stdout):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    # Live redraws only for an interactive terminal; colors additionally
    # honor NO_COLOR and require ANSI to actually work on this console.
    try:
        _TTY = sys.stderr.isatty()
    except Exception:
        _TTY = False
    _COLOR = _TTY and os.environ.get("NO_COLOR") is None
    if _COLOR:
        _COLOR = enable_windows_ansi(-12)


def enabled() -> bool:
    return _ENABLED


def _elapsed() -> float:
    return time.time() - _T0


def _stamp() -> str:
    return _c(_DIM, f"[dbg +{_elapsed():7.1f}s]")


def _tag(color: str, glyph: str, label: str) -> str:
    return _c(color, f"[{glyph} {label}]")


def set_context(label: str = "") -> None:
    """Name the active agent, so meters can label their output with it.
    Name the active agent, so meters can label their output with it.

    Lets `models.py` print "latent Refiner" without threading an agent name
    through the generation APIs. Call with no argument to clear.
    """
    global _CONTEXT
    _CONTEXT = label or ""


def _with_context(label: str) -> str:
    return f"{label} {_CONTEXT}" if _CONTEXT else label


def _cr() -> str:
    """Prefix that wipes any live heartbeat/meter remnant off the line.
    Prefix that wipes any live heartbeat/meter remnant off the line.

    \\r alone leaves stale characters when the new line is shorter, so clear
    to end-of-line too when ANSI is available. When stderr is not a terminal
    there is no remnant to wipe, emit nothing so logs stay clean.
    """
    if not _TTY:
        return ""
    return "\r\x1b[K" if _COLOR else "\r"


def dbg(msg: str) -> None:
    """Print a one-off line, clearing any live heartbeat remnant first."""
    if not _ENABLED:
        return
    print(f"{_cr()}{_stamp()} {msg}", file=sys.stderr, flush=True)


class phase:
    """Context manager: announce a phase, show a live elapsed heartbeat, report duration.
    Context manager: announce a phase, show a live elapsed heartbeat, report duration.

    Use for silent ops with no token stream (model load, latent rollout).
    """

    def __init__(self, label: str, heartbeat: bool = True, interval: float = 0.5) -> None:
        self.label = label
        self.heartbeat = heartbeat
        self.interval = interval
        self._stop = None
        self._thread = None
        self._t = None

    def __enter__(self):
        if not _ENABLED:
            return self
        self._t = time.time()
        print(f"{_cr()}{_stamp()} {_tag(_CYAN, '▶', self.label)} ...", file=sys.stderr, flush=True)
        # A live redraw is meaningless in a log file; the start/done lines
        # already bracket the phase there.
        if self.heartbeat and _TTY:
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._beat, daemon=True)
            self._thread.start()
        return self

    def _beat(self) -> None:
        while not self._stop.wait(self.interval):
            el = time.time() - self._t
            print(
                f"{_cr()}{_stamp()} {_tag(_YELLOW, '⏳', self.label)} {el:5.1f}s elapsed ",
                end="", file=sys.stderr, flush=True,
            )

    def __exit__(self, exc_type, exc, tb):
        if not _ENABLED:
            return False
        if self._stop is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)
        dt = time.time() - self._t
        if exc_type is None:
            print(f"{_cr()}{_stamp()} {_tag(_GREEN, '✓', self.label)} {dt:.2f}s",
                  file=sys.stderr, flush=True)
        else:
            print(f"{_cr()}{_stamp()} {_tag(_RED, '✗', self.label)} "
                  f"failed after {dt:.2f}s ({exc_type.__name__})",
                  file=sys.stderr, flush=True)
        return False  # never suppress exceptions


class TokMeter:
    """A streamer object for HF `generate`: live tok/s per sequence and aggregate.
    """

    def __init__(self, label: str = "decode", batch_size: int = 1,
                 cap: int = 0) -> None:
        self.label = _with_context(label)
        self.bs = max(1, int(batch_size))
        self.cap = max(0, int(cap))  # max_new_tokens; 0 = unknown, no ETA
        self.n = 0
        self.t0 = time.time()
        self._last = self.t0  # so the first redraw waits a beat (no bogus spike)
        self._done = False
        # Running least-squares accumulators for dt(i) = a + b·i over the
        # per-token decode times (prefill token excluded, see class docstring).
        self._prev = None
        self._rn = 0
        self._sx = self._sy = self._sxx = self._sxy = 0.0

    def _rates(self, elapsed: float):
        per_seq = self.n / elapsed if elapsed > 0 else 0.0
        return per_seq, per_seq * self.bs

    def _eta(self) -> float:
        """Seconds to reach the cap, modelling the per-token slowdown. -1 if unknown."""
        remaining = self.cap - self.n
        if self.cap <= 0 or remaining <= 0 or self._rn < 2:
            return -1.0
        denom = self._rn * self._sxx - self._sx * self._sx
        if denom > 0:
            b = (self._rn * self._sxy - self._sx * self._sy) / denom
            a = (self._sy - b * self._sx) / self._rn
        else:
            b, a = 0.0, self._sy / self._rn
        b = max(0.0, b)  # the cache only grows the cost; ignore noise that says otherwise
        # Σ_{i=n+1}^{cap} (a + b·i) = a·remaining + b·Σi
        i_sum = (self.cap * (self.cap + 1) - self.n * (self.n + 1)) / 2.0
        eta = a * remaining + b * i_sum
        if eta <= 0:  # a can go negative on a short, noisy fit, fall back to flat
            eta = (self._sy / self._rn) * remaining
        return max(0.0, eta)

    def put(self, value) -> None:
        if not _ENABLED:
            return
        ndim = value.dim() if hasattr(value, "dim") else 1
        if ndim >= 2:
            return  # initial prompt echo, not a generated token
        self.n += 1
        if not _TTY:
            return  # no live redraws into a log file; end() prints the summary
        now = time.time()
        # Accumulate the decay model every token (cheap); the first generated
        # token has no _prev, so its prefill-laden time never enters the fit.
        if self._prev is not None:
            dt = now - self._prev
            x = float(self.n)
            self._rn += 1
            self._sx += x
            self._sy += dt
            self._sxx += x * x
            self._sxy += x * dt
        self._prev = now
        if now - self._last >= 0.25:  # throttle redraws
            self._last = now
            per_seq, total = self._rates(now - self.t0)
            eta = self._eta()
            eta_str = f" · eta≲{_fmt_dur(eta)}" if eta >= 0 else ""
            print(
                f"{_cr()}{_stamp()} {_tag(_BLUE, '⚡', self.label)} "
                f"{self.n:4d} tok · {per_seq:6.1f} tok/s/seq · ×{self.bs} = {total:7.1f} tok/s"
                f"{eta_str} ",
                end="", file=sys.stderr, flush=True,
            )

    def end(self) -> None:
        # HF generate() calls streamer.end() itself and the caller calls it
        # again defensively, print the summary only once.
        if not _ENABLED or self._done:
            return
        self._done = True
        el = time.time() - self.t0
        per_seq, total = self._rates(el)
        print(
            f"{_cr()}{_stamp()} {_tag(_BGREEN, '⚡', self.label)} "
            f"{self.n} tok in {el:.2f}s · {per_seq:.1f} tok/s/seq · ×{self.bs} = {total:.1f} tok/s",
            file=sys.stderr, flush=True,
        )


class StepMeter:
    """Live counter for a fixed-length loop of silent steps (latent rollout).

        [dbg +  48.4s] [◆ latent Refiner] step   34/80 ·   12.4 step/s · eta   3.7s
        [dbg +  48.5s] [◆ latent Refiner] 80 steps in 6.42s · 12.5 step/s (bs=20)

    When the Theorem 3.1 ratio is configured (see theory.py), the lines also
    carry the text-token lower bound those latent thoughts stand in for:
    """

    def __init__(self, label: str, total: int, batch_size: int = 1,
                 interval: float = 0.25) -> None:
        self.label = _with_context(label)
        self.total = max(0, int(total))
        self.bs = max(1, int(batch_size))
        self.interval = interval
        self.n = 0
        self.t0 = time.time()
        self._last = self.t0  # so the first redraw waits a beat (no bogus spike)
        # --latent_steps 0 is a valid config whose loop never runs; stay
        # silent rather than emit an empty start/summary pair per agent.
        self._done = self.total == 0
        if _ENABLED and not self._done:
            print(
                f"{_cr()}{_stamp()} {_tag(_MAGENTA, '◆', self.label)} "
                f"{self.total} steps (bs={self.bs}) ...",
                file=sys.stderr, flush=True,
            )

    def step(self) -> None:
        """Call once per completed iteration."""
        if not _ENABLED:
            return
        self.n += 1
        if not _TTY:
            return  # no live redraws into a log file; end() prints the summary
        now = time.time()
        if now - self._last >= self.interval:  # throttle redraws
            self._last = now
            el = now - self.t0
            rate = self.n / el if el > 0 else 0.0
            eta = (self.total - self.n) / rate if rate > 0 else 0.0
            print(
                f"{_cr()}{_stamp()} {_tag(_MAGENTA, '◆', self.label)} "
                f"step {self.n:>4}/{self.total} · {rate:6.1f} step/s · eta {eta:5.1f}s"
                f"{self._equiv(self.n)} ",
                end="", file=sys.stderr, flush=True,
            )

    def _equiv(self, steps: int) -> str:
        """Theorem 3.1 text-token lower bound for `steps` latent thoughts.
        Theorem 3.1 text-token lower bound for `steps` latent thoughts.

        Per sequence, not multiplied by batch size, it describes the
        reasoning one sample would have had to write out in text.
        """
        tokens = theory.text_equivalent(steps)
        if tokens is None:
            return ""
        return f" · ≳{theory.compact(tokens)} text-tok"

    def end(self) -> None:
        if not _ENABLED or self._done:
            return
        self._done = True
        el = time.time() - self.t0
        rate = self.n / el if el > 0 else 0.0
        ratio = theory.ratio()
        equiv = f"{self._equiv(self.n)} (×{ratio:.1f})" if ratio else ""
        print(
            f"{_cr()}{_stamp()} {_tag(_GREEN, '◆', self.label)} "
            f"{self.n} steps in {el:.2f}s · {rate:.1f} step/s (bs={self.bs}){equiv}",
            file=sys.stderr, flush=True,
        )
