"""Create a reproducible source bundle with hashes; excludes audio and caches."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'Universal_Audio_Waveform_Sync_modular_v6.9.8'


def main():
    allowed = {'.py', '.md', '.txt', '.cmd', '.json', '.csv', '.jpg', '.sha256'}
    files = []
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if any(part in {'.git', '.venv', '__pycache__', 'releases', 'dist', '.pytest_cache', '.mpl-cache'}
               or '_WAVEFORM_SYNC' in part for part in relative.parts):
            continue
        if path.is_file() and (path.suffix.lower() in allowed or path.name == '.gitignore'):
            if path.name != 'MANIFEST_SHA256.json':
                files.append(path)
    files.sort()
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest_path = ROOT / 'MANIFEST_SHA256.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    files.append(manifest_path)
    directory = ROOT / 'releases'
    directory.mkdir(exist_ok=True)
    destination = directory / (NAME + '.zip')
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            item = zipfile.ZipInfo(NAME + '/' + path.relative_to(ROOT).as_posix(), (2026, 10, 22, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o100644 << 16
            archive.writestr(item, path.read_bytes())
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        for path, digest in manifest.items():
            assert hashlib.sha256(archive.read(NAME + '/' + path)).hexdigest() == digest
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix('.zip.sha256').write_text(digest + '  ' + destination.name + '\n')
    print(destination.name, destination.stat().st_size, 'bytes', digest)


if __name__ == '__main__':
    main()
