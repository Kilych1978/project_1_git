from array import array
from pathlib import Path
import tempfile
import unittest
import wave

from tools.collect_audio_points import extract_first_channel,find_ffmpeg,RATE


class PointCollectorTests(unittest.TestCase):
    def test_exact_first_channel_sample_positions_and_overlapping_excerpts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root/'source.wav'
            original = array('h',(i%30000-15000 for i in range(12*RATE)))
            interleaved = array('h',(value for x in original for value in (x,123)))
            with wave.open(str(source),'wb') as file:
                file.setnchannels(2);file.setsampwidth(2);file.setframerate(RATE)
                file.writeframes(interleaved.tobytes())
            destinations = [(1.125,8.,root/'first.wav'),(2.333,8.,root/'overlap.wav')]
            extract_first_channel(source,destinations,find_ffmpeg(root),root/'ffmpeg.log',False)
            for start,duration,path in destinations:
                with wave.open(str(path),'rb') as file:
                    self.assertEqual((file.getnchannels(),file.getsampwidth(),file.getframerate(),file.getnframes()),(1,2,RATE,round(duration*RATE)))
                    self.assertEqual(file.readframes(file.getnframes()),original[round(start*RATE):round((start+duration)*RATE)].tobytes())


if __name__ == '__main__':
    unittest.main()
