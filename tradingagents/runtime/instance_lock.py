from __future__ import annotations

import json
import os
from pathlib import Path


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information, False, pid
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class SingleInstanceLock:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._remove_if_stale():
                    continue
                raise RuntimeError(f"Another trading bot instance owns {self.path}")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"pid": os.getpid()}, handle)
            self.acquired = True
            return
        raise RuntimeError(f"Could not acquire trading bot lock {self.path}")

    def release(self) -> None:
        if self.acquired:
            try:
                self.path.unlink(missing_ok=True)
            finally:
                self.acquired = False

    def _remove_if_stale(self) -> bool:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            pid = int(data["pid"])
            if _pid_is_running(pid):
                return False
            self.path.unlink(missing_ok=True)
            return True
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            self.path.unlink(missing_ok=True)
            return True

    def __enter__(self) -> "SingleInstanceLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()
