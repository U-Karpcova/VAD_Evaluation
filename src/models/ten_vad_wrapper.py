from src.models.base_vad import VAD_Model
import numpy as np
from ten_vad import TenVad

class TEN_VAD_MODEL(VAD_Model):
  def __init__(self) -> None:
     self.model = TenVad(hop_size=256, threshold=0.5)
     self.name = "ten_vad"
     self.reqired_sr = 16000

  def detect_segments(self, audio_path, frames_mode=True):
    hop_size = self.model.hop_size
    speech_timestamps = []
    is_speaking = False
    start_time = 0.0
    sr=16000
    audio_int16 = self.load_audio_int16(audio_path)

    for i in range(0, len(audio_int16), hop_size):
        chunk = audio_int16[i : i + hop_size]

        if len(chunk) < hop_size:
            chunk = np.pad(chunk, (0, hop_size - len(chunk)), 'constant')

        probability, flag = self.model.process(chunk)
        current_time = i / sr

        if flag == 1 and not is_speaking:
            is_speaking = True
            start_time = current_time

        elif flag == 0 and is_speaking:
            is_speaking = False
            speech_timestamps.append({'start': start_time, 'end': current_time})

    if is_speaking:
        speech_timestamps.append({'start': start_time, 'end': len(audio_int16) / sr})

    if frames_mode:
      speech_timestamps = self.seconds_to_frames(speech_timestamps)

    return speech_timestamps

# model = TEN_VAD_MODEL()
# speech_timestamps = model.detect_segments(test_audio_path, frames_mode=True)
# print(speech_timestamps)