from src.models.base_vad import VAD_Model
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

class Silero_Model(VAD_Model):
  def __init__(self) -> None:
      model_path = 'src/models/pretrained/silero_vad.jit'
      self.model = load_silero_vad(model_path)
      self.name = "silero_vad"
      self.reqired_sr = 16000

  def detect_segments(self, audio_path, frames_mode=True):
    audio = read_audio(audio_path)
    return_seconds = False if frames_mode else True
    speech_timestamps = get_speech_timestamps(audio, self.model, return_seconds=return_seconds)
    speech_timestamps = [(item['start'], item['end']) for item in speech_timestamps]
    return speech_timestamps

# model = Silero_Model()
# speech_timestamps = model.detect_segments(test_audio_path, frames_mode=True)
# print(speech_timestamps)