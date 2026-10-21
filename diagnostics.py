"""Current-run logging and complete, size-limited diagnostic ZIPs.

No prior result directories are read. Full audio and intermediate PCM are not
included; only the short mono excerpts explicitly exported by pcm_refiner are.
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
import time
import traceback
import zipfile

_SESSION = ContextVar('waveform_diagnostic_session', default=None)
VOLUME_LIMIT = 25 * 1024 * 1024
_SUFFIXES = {'.json', '.csv', '.txt', '.log', '.md', '.jpg', '.jpeg', '.png'}


def _hash(path, offset=0, count=None):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        source.seek(offset)
        while count is None or count > 0:
            block = source.read(min(1024 * 1024, count) if count is not None else 1024 * 1024)
            if not block:
                break
            digest.update(block)
            if count is not None:
                count -= len(block)
    return digest.hexdigest()


def build_bundle(directory, limit=VOLUME_LIMIT):
    """Return independent ZIP volumes, each strictly within the upload limit.

    All reports are collected AFTER the final map and translated reports exist.
    Very large individual files are split losslessly, with offsets and hashes
    in the manifest, rather than silently omitted or truncated.
    """
    directory = Path(directory).resolve()
    if limit < 64 * 1024:
        raise ValueError('Слишком малый размер диагностического тома.')
    files = []
    for path in sorted(directory.rglob('*')):
        relative = path.relative_to(directory)
        if (path.is_symlink() or not path.is_file()
                or any(part.startswith('.diagnostic-') or part in {'.git', '.venv', '__pycache__'}
                       for part in relative.parts)
                or any(parent.is_symlink() for parent in path.parents if parent != directory.parent)):
            continue
        if path.suffix.lower() in _SUFFIXES or relative.as_posix() == 'DIAGNOSTIC_CLIPS.zip':
            files.append(path)
    originals = [{'path': p.relative_to(directory).as_posix(), 'size': p.stat().st_size,
                  'sha256': _hash(p)} for p in files]
    entries = []
    with tempfile.TemporaryDirectory(prefix='.diagnostic-', dir=directory) as temporary:
        staging = Path(temporary)

        def stage(path, name, offset, size):
            destination = staging / f'{len(entries):06d}.zip'
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=6) as archive:
                with archive.open(info, 'w') as member, path.open('rb') as source:
                    source.seek(offset)
                    remaining = size
                    while remaining:
                        block = source.read(min(1024 * 1024, remaining))
                        if not block:
                            raise OSError('Диагностический файл изменился во время упаковки: ' + name)
                        member.write(block)
                        remaining -= len(block)
            return destination.stat().st_size

        # Leave room for central directory, manifest, and the instructions.
        reserve = max(16 * 1024, limit // 50)
        for path, original in zip(files, originals):
            name, size = original['path'], original['size']
            packed_size = stage(path, name, 0, size)
            if packed_size + reserve <= limit:
                entries.append({'path': name, 'source': path, 'offset': 0, 'size': size,
                                'packed_size': packed_size})
            else:
                chunk_size = limit - reserve * 2
                for part, offset in enumerate(range(0, size, chunk_size), 1):
                    length = min(chunk_size, size-offset)
                    member = f'_split/{name}.part{part:04d}'
                    packed_size = stage(path, member, offset, length)
                    entries.append({'path': member, 'source': path, 'offset': offset,
                                    'size': length, 'packed_size': packed_size,
                                    'original': name})
        groups = [[]]
        used = reserve
        for entry in entries:
            if used + entry['packed_size'] > limit:
                groups.append([])
                used = reserve
            groups[-1].append(entry)
            used += entry['packed_size']
        names = (['REPORT_BUNDLE.zip'] if len(groups) == 1 else
                 [f'REPORT_BUNDLE_{i:02d}.zip' for i in range(1, len(groups)+1)])
        manifest = {'schema': 1, 'volumes': names, 'volume_limit_bytes': limit,
                    'scope': 'current run; full audio and intermediate PCM excluded',
                    'originals': originals,
                    'members': [{'path': e['path'], 'volume': name, 'offset': e['offset'],
                                 'size': e['size'], 'original': e.get('original', e['path'])}
                                for name, group in zip(names, groups) for e in group]}
        manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
        instructions = ('ДИАГНОСТИКА ТЕКУЩЕГО ЗАПУСКА\n\n'
                        'Приложите к сообщению все перечисленные архивы:\n' + '\n'.join(names) +
                        '\n\nКаждый том — самостоятельный ZIP размером не более 25 МиБ.\n'
                        'Включены все отчёты, карты, таблицы, графики, журнал и короткие '
                        'монофрагменты. Полные дорожки и промежуточные PCM не включены.\n'
                        'MANIFEST.json содержит перечень файлов и SHA-256. Если очень большой '
                        'файл разделён, его байтовые части находятся в _split; объединять '
                        'в порядке offset. Никакие данные автоматически не отправляются.\n')
        results = []
        for name, group in zip(names, groups):
            destination = staging / name
            with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=6) as archive:
                archive.writestr('MANIFEST.json', manifest_bytes)
                archive.writestr('READ_ME_RU.txt', instructions.encode('utf-8-sig'))
                for entry in group:
                    with archive.open(entry['path'], 'w') as member, entry['source'].open('rb') as source:
                        source.seek(entry['offset'])
                        remaining = entry['size']
                        while remaining:
                            block = source.read(min(1024 * 1024, remaining))
                            if not block:
                                raise OSError('Диагностический файл изменился во время упаковки.')
                            member.write(block)
                            remaining -= len(block)
            if destination.stat().st_size > limit:
                raise RuntimeError('Диагностический том превысил предел; исходные отчёты сохранены.')
            results.append(destination)
        # Publish only after EVERY volume was successfully written.
        for previous in directory.glob('REPORT_BUNDLE*.zip'):
            previous.unlink()
        published = []
        for path in results:
            result = directory / path.name
            path.replace(result)
            published.append(result)
        return published


class _Capture:
    """Keep messages and occasional progress snapshots, not every screen redraw."""
    def __init__(self):
        self.file = tempfile.SpooledTemporaryFile(mode='w+', encoding='utf-8', max_size=1024*1024)
        self.pending = ''
        self.progress = False
        self.last_progress = -float('inf')

    def _line(self):
        value = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', self.pending).rstrip()
        if not self.progress or time.monotonic()-self.last_progress >= 10 or re.match(r'\[100[.,]00%\]', value):
            self.file.write(value+'\n')
            if self.progress:
                self.last_progress = time.monotonic()
        self.pending = ''

    def write(self, value):
        for token in re.split('([\r\n])', value):
            if token == '\r':
                if self.pending:
                    self._line()
                self.progress = True
            elif token == '\n':
                self._line()
                self.progress = False
            else:
                self.pending += token

    def bind(self, directory):
        self.file.seek(0)
        destination = (directory / 'RUN_LOG.txt').open('w+', encoding='utf-8-sig')
        shutil.copyfileobj(self.file, destination)
        self.file.close()
        self.file = destination

    def flush(self):
        self.file.flush()


class _Tee:
    def __init__(self, stream, capture):
        self.stream, self.capture = stream, capture

    def write(self, value):
        count = self.stream.write(value)
        self.capture.write(value)
        return count

    def flush(self):
        self.stream.flush()
        self.capture.flush()

    def __getattr__(self, key):
        return getattr(self.stream, key)


def register_output(directory):
    session = _SESSION.get()
    if session is not None:
        session['output'] = Path(directory)
        session['capture'].bind(Path(directory))


def diagnostic_session(function):
    """Wrap either launcher, including failures and user interruption."""
    @wraps(function)
    def run(*args, **kwargs):
        if _SESSION.get() is not None:
            return function(*args, **kwargs)
        capture = _Capture()
        state = {'capture': capture, 'output': None}
        token = _SESSION.set(state)
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(stdout, capture), _Tee(stderr, capture)
        status = 'COMPLETED'
        try:
            return function(*args, **kwargs)
        except BaseException as exc:
            status = 'INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'ERROR'
            capture.write('\n' + ''.join(traceback.format_exception(exc)))
            raise
        finally:
            sys.stdout, sys.stderr = stdout, stderr
            _SESSION.reset(token)
            output = state['output']
            try:
                if capture.pending:
                    capture._line()
                capture.flush()
                capture.file.close()
                if output is not None:
                    version = Path(__file__).with_name('VERSION.txt').read_text(encoding='utf-8').strip()
                    info = {'status': status, 'version': version, 'python': sys.version,
                            'platform': platform.platform(), 'mode': function.__module__}
                    (output / 'RUN_INFO.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
                    paths = build_bundle(output)
                    print('\nДИАГНОСТИКА: приложите к сообщению ' + ('этот архив:' if len(paths) == 1 else 'все эти архивы:'))
                    for path in paths:
                        print(f'{path} ({path.stat().st_size/1024/1024:.1f} МиБ)')
            except Exception as exc:
                print(f'Не удалось собрать диагностический архив: {exc}. Отчёты сохранены в {output}.', file=stderr)
    return run
