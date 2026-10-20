"""PCM format selection and sample-count checks for newly rendered WAV files."""
import struct


def output_codec(info):
    # Keep integer PCM precision; converting S16 to float doubled movie files.
    codecs = {'pcm_s16le', 'pcm_s24le', 'pcm_s32le', 'pcm_f32le', 'pcm_f64le'}
    return info.codec if info.codec in codecs else 'pcm_s24le'


def finalize_wave(path, frames, sample_rate, channels):
    """Check our encoder's header and write the actual count into fact/ds64.

    FFmpeg writes 0xFFFFFFFF into RF64's fact chunk even for fewer than 2**32
    frames. Readers using fact then display 24:51:18 at 48 kHz. The frame count
    is known from this render; source files are never opened for writing.
    """
    size = path.stat().st_size
    fact_position = ds64_position = None
    data_size64 = None
    fmt = None
    with path.open('r+b') as file:
        header = file.read(12)
        if len(header) != 12 or header[:4] not in (b'RIFF', b'RF64') or header[8:] != b'WAVE':
            raise ValueError('Некорректный заголовок выходного WAV.')
        rf64 = header[:4] == b'RF64'
        if not rf64 and struct.unpack('<I', header[4:8])[0] != size - 8:
            raise ValueError('Размер RIFF не совпадает с размером выходного файла.')
        while file.tell() + 8 <= size:
            name, length = struct.unpack('<4sI', file.read(8))
            position = file.tell()
            if name == b'data':
                if fmt is None or (rf64 and ds64_position is None):
                    raise ValueError('В выходном WAV отсутствуют fmt/ds64.')
                _, count, rate, byte_rate, block, bits = struct.unpack('<HHIIHH', fmt[:16])
                expected = frames * block
                actual_length = data_size64 if rf64 and length == 0xFFFFFFFF else length
                if (count != channels or rate != sample_rate or block != channels * bits // 8
                        or byte_rate != sample_rate * block or actual_length != expected
                        or position + expected > size):
                    raise ValueError('Данные и параметры выходного WAV не совпадают с планом записи.')
                break
            if length > size - position:
                raise ValueError('Повреждён размер блока выходного WAV.')
            if name == b'fmt ':
                fmt = file.read(min(length, 40))
            elif name == b'fact' and length >= 4:
                fact_position = position
            elif name == b'ds64' and length >= 28:
                riff_size, data_size64, _ = struct.unpack('<QQQ', file.read(24))
                if riff_size != size - 8:
                    raise ValueError('Размер RF64 не совпадает с размером выходного файла.')
                ds64_position = position
            file.seek(position + length + (length & 1))
        else:
            raise ValueError('В выходном WAV отсутствуют аудиоданные.')
        if fact_position is not None:
            file.seek(fact_position)
            file.write(struct.pack('<I', min(frames, 0xFFFFFFFF)))
        if ds64_position is not None:
            file.seek(ds64_position + 16)
            file.write(struct.pack('<Q', frames))
    return {'container': 'RF64' if rf64 else 'RIFF', 'sample_frames': frames,
            'sample_rate': sample_rate, 'channels': channels, 'bits_per_sample': bits,
            'duration_sec': frames / sample_rate}
