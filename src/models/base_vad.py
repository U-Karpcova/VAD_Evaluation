import librosa
import soundfile as sf
import numpy as np


class VAD_Model:
  def __init__(self) -> None:
      self.model = None
      self.name = "vad_model"
      self.reqired_sr = 16000

  def __load_audio(self, input_path):
        """
        Зчитує аудіофайл, ресемплить до 16 кГц та конвертує в моно.
        :input_path: Шлях до вхідного .wav файлу
        """
        audio, sr = librosa.load(input_path, sr=self.reqired_sr, mono=True)
        return audio

  def load_audio_int16(self, input_path):
        """Зчитує аудіо та приводить до int16 і 16кГц."""

        info = sf.info(input_path)
        if info.samplerate == self.reqired_sr and info.channels == 1:
            # напряму в int16, якщо параметри збігаються
            audio, _ = sf.read(input_path, dtype='int16')
            audio_int16 = audio
        else:
            audio = self.__load_audio(input_path)
            audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16) # int16

        return audio_int16

  def seconds_to_frames(self, segments, hop_length=None):
    """
    функція для перетворення таймкодів із секунд у фрейми
    """
    converted = []
    sr=self.reqired_sr

    if isinstance(segments[0], dict):
      for s in segments:
          if hop_length:
              start = int(s["start"] * sr / hop_length)
              end = int(s["end"] * sr / hop_length)
          else:
              start = int(s["start"] * sr)
              end = int(s["end"] * sr)

          new_s = (start, end)
          converted.append(new_s)

    elif isinstance(segments[0], tuple) or isinstance(segments[0], list):
      for s in segments:
          if hop_length:
              start = int(s[0] * sr / hop_length)
              end = int(s[1] * sr / hop_length)
          else:
              start = int(s[0] * sr)
              end = int(s[1] * sr)

          new_s = (start, end)
          converted.append(new_s)

    return converted

  def detect_segments(self, audio_path, frames_mode=True):
    raise NotImplementedError("Метод process_file має бути перевизначений")
