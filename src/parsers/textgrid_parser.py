from praatio import textgrid

# True labels from TextGrid
class TextGrid_Parser():
  def __init__(self) -> None:
      pass

  def extract_speech_segments(self, textgrid_path) -> list:
    """
    Витягує із файлу TextGrid сегменти мовлення кожного мовця
    start, end, speaker, text, emotion
    вигляд елемента: {'start': 10.5449, 'end': 13.3828, 'speaker': 'speaker_1', 'text': 'У нас знову...', 'emotion': 'an2'}
    """
    tg = textgrid.openTextgrid(textgrid_path, includeEmptyIntervals=True)

    segments = []
    tier_pairs = [
        (0, 1, "speaker_1"),
        (2, 3, "speaker_2")
    ]

    for text_idx, emo_idx, speaker in tier_pairs:
      text_tier = tg.tiers[text_idx]
      emo_tier = tg.tiers[emo_idx]

      for (start, end, text), (_, _, emotion) in zip(
            text_tier.entries,
            emo_tier.entries
        ):
            if text.strip():
                segments.append({
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                    "text": text,
                    "emotion": emotion
                  })

    return segments


  def build_vad_segments(self, textgrid_path, frames_mode=True, sr=16000, max_gap=0.05) -> list:
    """
    segments: список сегментів із мовленням на записі
    вигляд елемента: [(79835, 107185),
    """
    segments = self.extract_speech_segments(textgrid_path)

    intervals = [(s["start"], s["end"]) for s in segments] # тільки таймкоди
    intervals.sort()

    vad_segments = []

    for start, end in intervals:
        if not vad_segments:
            vad_segments.append([start, end])
            continue

        last_start, last_end = vad_segments[-1]

        # якщо overlap або дуже близько - об’єднуємо
        if start <= last_end + max_gap:
            vad_segments[-1][1] = max(last_end, end)
        else:
            vad_segments.append([start, end])

    if frames_mode:
      converted_vad_segments = []
      for s in vad_segments:
          start = int(s[0] * sr)
          end = int(s[1] * sr)
          new_s = (start, end)
          converted_vad_segments.append(new_s)
      return converted_vad_segments
    else:
      return vad_segments


  def build_vad_segments_detailed(self, textgrid_path, sr=16000) -> list:
    """
    Додає метадані про спікерів, текст та емоції. Потрібно для аналізу.
    start, end, speakers, emotions
    {'start': 79835, 'end': 107185, 'speakers': 'speaker_2', 'emotions': 'j2'},
    {'start': 222783, 'end': 263141, 'speakers': 'sp_overlap', 'emotions': 'em_overlap'}
    """
    segments = self.extract_speech_segments(textgrid_path)
    for s in segments:
      s['start'] = int(s['start'] * sr)
      s['end'] = int(s['end'] * sr)
    vad_segments = self.build_vad_segments(textgrid_path)

    detailed_segments = []
    for seg in vad_segments:
      segm_info = {
          'start': seg[0],
          'end': seg[1],
          'speakers': [],
          # 'texts': [],
          'emotions': []
      }
      for s in segments:
        if s['start'] >= seg[0] and s['end'] <= seg[1]:
          segm_info['speakers'].append(s['speaker'])
          # segm_info['texts'].append(s['text'])
          segm_info['emotions'].append(s['emotion'])
      detailed_segments.append(segm_info)

    # нормалізація
    for item in detailed_segments:
      item['speakers'] = 'sp_overlap' if len(item['speakers']) > 1 else item['speakers'][0]
      item['emotions'] = 'em_overlap' if len(item['emotions']) > 1 else item['emotions'][0]

    return detailed_segments


# parser = TextGrid_Parser()
# detailed_segments = parser.build_vad_segments_detailed(test_textgrid_path)
# for seg in detailed_segments:
#   print(seg)