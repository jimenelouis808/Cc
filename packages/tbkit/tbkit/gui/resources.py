"""What the machine has and what each calculation uses: CPU, RAM, GPU.

``psutil`` gives it on every platform (part of the ``gui`` extra); without it, Linux's
``/proc`` gives the same numbers, and elsewhere only the core count is known. The GPU comes
from ``nvidia-smi`` when it exists (NVIDIA only; asked at most every few seconds, it takes
tens of milliseconds). Nothing here imports Qt.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

try:
    import psutil
except ImportError:                          # the /proc fallback below
    psutil = None


def cores() -> int:
    """Processors this process may use (its affinity mask, not the machine's total)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# --------------------------------------------------------------------------- /proc fallback

def _proc_cpu_times():
    with open("/proc/stat", encoding="ascii") as handle:
        fields = [float(x) for x in handle.readline().split()[1:]]
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0.0)
    return sum(fields), idle


def _proc_memory():
    info = {}
    with open("/proc/meminfo", encoding="ascii") as handle:
        for line in handle:
            key, value = line.split(":", 1)
            info[key] = float(value.split()[0]) * 1024
    total = info["MemTotal"]
    available = info.get("MemAvailable", info.get("MemFree", 0.0))
    return total, available


def _proc_rss(pid: int) -> float | None:
    try:
        with open(f"/proc/{pid}/statm", encoding="ascii") as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def _proc_cpu_seconds(pid: int) -> float | None:
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as handle:
            fields = handle.read().rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def _proc_children(pid: int) -> list[int]:
    out = []
    for task in Path(f"/proc/{pid}/task").glob("*/children"):
        try:
            out += [int(x) for x in task.read_text().split()]
        except OSError:
            continue
    return out


# --------------------------------------------------------------------------- monitor

class Monitor:
    """Machine and per-process usage. Call :meth:`machine` and :meth:`process` from a timer;
    CPU percentages are measured between consecutive calls."""

    GPU_EVERY = 5.0

    def __init__(self):
        self._cpu_last = None
        self._proc_last: dict[int, tuple[float, float]] = {}
        self._gpu: list[dict] = []
        self._gpu_time = 0.0
        self.has_gpu_tool = shutil.which("nvidia-smi") is not None
        if psutil is not None:
            psutil.cpu_percent(None)

    def machine(self) -> dict:
        """``cpu_percent`` (all cores), ``ram_used_gb``, ``ram_total_gb``,
        ``ram_available_gb``, ``cores``, ``gpus`` (list of dicts) and ``source``."""
        out = {"cores": os.cpu_count() or 1, "cpu_percent": None, "ram_used_gb": None,
               "ram_total_gb": None, "ram_available_gb": None, "source": "ninguna"}
        if psutil is not None:
            memory = psutil.virtual_memory()
            out.update(cpu_percent=psutil.cpu_percent(None),
                       ram_total_gb=memory.total / 1e9,
                       ram_available_gb=memory.available / 1e9,
                       ram_used_gb=(memory.total - memory.available) / 1e9, source="psutil")
        elif Path("/proc/stat").exists():
            total, idle = _proc_cpu_times()
            if self._cpu_last is not None:
                d_total, d_idle = total - self._cpu_last[0], idle - self._cpu_last[1]
                out["cpu_percent"] = 100.0 * (1 - d_idle / d_total) if d_total > 0 else 0.0
            self._cpu_last = (total, idle)
            memory_total, available = _proc_memory()
            out.update(ram_total_gb=memory_total / 1e9, ram_available_gb=available / 1e9,
                       ram_used_gb=(memory_total - available) / 1e9, source="/proc")
        out["gpus"] = self.gpus()
        return out

    def gpus(self) -> list[dict]:
        """``[{"name", "percent", "memory_used_gb", "memory_total_gb"}]`` from nvidia-smi."""
        if not self.has_gpu_tool:
            return []
        now = time.monotonic()
        if now - self._gpu_time >= self.GPU_EVERY:
            self._gpu_time = now
            try:
                text = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True,
                    timeout=3, check=False).stdout
                self._gpu = []
                for line in text.strip().splitlines():
                    name, util, used, total = (x.strip() for x in line.split(","))
                    self._gpu.append({"name": name, "percent": float(util),
                                      "memory_used_gb": float(used) / 1024,
                                      "memory_total_gb": float(total) / 1024})
            except (OSError, subprocess.SubprocessError, ValueError):
                self._gpu = []
        return self._gpu

    def process(self, pid: int, children: bool = True) -> dict | None:
        """``{"rss_gb", "cpu_percent", "alive"}`` of a process (with its children: the
        parts of a recipe, mpiexec's ranks); None if it is gone."""
        pids = [pid] + (descendants(pid) if children else [])
        rss, cpu_seconds, alive = 0.0, 0.0, False
        for p in pids:
            if psutil is not None:
                try:
                    process = psutil.Process(p)
                    with process.oneshot():
                        rss += process.memory_info().rss
                        times = process.cpu_times()
                        cpu_seconds += times.user + times.system
                    alive = alive or p == pid
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            else:
                r, c = _proc_rss(p), _proc_cpu_seconds(p)
                if r is None:
                    continue
                rss += r
                cpu_seconds += c or 0.0
                alive = alive or p == pid
        if not alive:
            self._proc_last.pop(pid, None)
            return None
        now = time.monotonic()
        last = self._proc_last.get(pid)
        self._proc_last[pid] = (now, cpu_seconds)
        percent = None
        if last is not None and now > last[0]:
            percent = 100.0 * (cpu_seconds - last[1]) / (now - last[0])
        return {"rss_gb": rss / 1e9, "cpu_percent": percent, "alive": True}


def descendants(pid: int) -> list[int]:
    """Every process started under ``pid`` (children, grandchildren…)."""
    if psutil is not None:
        try:
            return [c.pid for c in psutil.Process(pid).children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return []
    out, todo = [], [pid]
    while todo:
        kids = _proc_children(todo.pop())
        out += kids
        todo += kids
    return out


def alive(pid: int) -> bool:
    if psutil is not None:
        try:
            return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def terminate(pid: int, tree: bool = True) -> list[int]:
    """Stop a process and everything under it (a recipe's parts, mpiexec's ranks): SIGTERM,
    then SIGKILL two seconds later for whatever is left. Returns the pids signalled."""
    import signal

    pids = (descendants(pid) if tree else []) + [pid]
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and any(alive(p) for p in pids):
        time.sleep(0.1)
    for p in pids:
        if alive(p):
            try:
                os.kill(p, getattr(signal, "SIGKILL", signal.SIGTERM))
            except OSError:
                pass
    return pids


def gb(value: float | None) -> str:
    return "—" if value is None else (f"{value:.1f} GB" if value >= 1 else f"{value * 1e3:.0f} MB")
