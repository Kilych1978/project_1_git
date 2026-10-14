from __future__ import annotations

import sys
from launcher import main


if __name__ == "__main__":
    try:
        if '--check' in sys.argv:
            from pcm_core.audio import find_tools
            import numpy, scipy, matplotlib
            print('Python:', sys.version.split()[0])
            print('NumPy:', numpy.__version__, 'SciPy:', scipy.__version__, 'Matplotlib:', matplotlib.__version__)
            print('FFmpeg / FFprobe:', find_tools())
        elif len(sys.argv) == 1:
            from pcm_core.audio import find_tools
            find_tools()
            main()
        else:
            raise ValueError('Для обычного запуска: start_sync.py. Проверка установки: start_sync.py --check.')
    except KeyboardInterrupt:
        print()
        print()
        print("=" * 78)
        print("ПРОГРАММА ОСТАНОВЛЕНА ПОЛЬЗОВАТЕЛЕМ")
        print("=" * 78)
        sys.exit(1)
    except Exception as exc:
        print()
        print()
        print("=" * 78)
        print("CRITICAL ERROR")
        print("=" * 78)
        print()
        print(str(exc))
        print()
        sys.exit(1)
