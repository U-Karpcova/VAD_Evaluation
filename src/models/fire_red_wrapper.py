from src.models.base_vad import VAD_Model
from fireredvad import FireRedVad, FireRedVadConfig

class FireRed_VAD_MODEL(VAD_Model):
  def __init__(self) -> None:
    vad_config = FireRedVadConfig(
      use_gpu=False,
      smooth_window_size=5,
      speech_threshold=0.4,
      min_speech_frame=20,
      max_speech_frame=2000,
      min_silence_frame=20,
      merge_silence_frame=0,
      extend_speech_frame=0,
      chunk_max_frame=30000)
    self.model = FireRedVad.from_pretrained("pretrained_models/FireRedVAD/VAD", vad_config)
    self.name = "fire_red_vad"
    self.reqired_sr = 16000

  def detect_segments(self, audio_path, frames_mode=True):
    audio_int16 = self.load_audio_int16(audio_path)
    result, probs = self.model.detect(audio_int16)
    speech_timestamps = result['timestamps']
    if frames_mode:
      speech_timestamps = self.seconds_to_frames(speech_timestamps)
    return speech_timestamps

# model = FireRed_VAD_MODEL()
# speech_timestamps = model.detect_segments(test_audio_path, frames_mode=True)
# print(speech_timestamps)