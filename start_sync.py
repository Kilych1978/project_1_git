from __future__ import annotations

import sys
from launcher import main


if __name__ == "__main__":
    try:
        if sys.argv[1:] == ['--check']:
            from pcm_core.audio import find_tools
            import numpy, scipy, matplotlib
            print('Python:', sys.version.split()[0])
            print('NumPy:', numpy.__version__, 'SciPy:', scipy.__version__, 'Matplotlib:', matplotlib.__version__)
            print('FFmpeg / FFprobe:', find_tools())
        elif sys.argv[1:] == ['--micro-only']:
            from pcm_core.audio import find_tools
            from micro_launcher import main as micro_main
            find_tools()
            micro_main()
        elif len(sys.argv) == 1:
            from pcm_core.audio import find_tools
            find_tools()
            main()
        else:
            raise ValueError('Запуск: start_sync.py; микроуточнение: --micro-only; проверка: --check.')
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
