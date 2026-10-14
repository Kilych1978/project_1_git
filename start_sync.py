#!/usr/bin/env python3
"""Single entry point for CMD and Python. No processing on import."""
import sys
import os
import tempfile
from pathlib import Path


def main():
    if sys.version_info < (3, 11):
        print("Требуется Python 3.11 или новее. Рекомендуется Python 3.12.", file=sys.stderr)
        return 1
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "wavesync-matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "wavesync-cache"))
    try:
        from wavesync.cli import main as run
        return run()
    except ModuleNotFoundError as exc:
        print(f"Не установлен пакет {exc.name}. Запустите SETUP_WINDOWS.cmd или установите requirements.txt.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")
        return 130
    except Exception as exc:
        print("\nОшибка: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
