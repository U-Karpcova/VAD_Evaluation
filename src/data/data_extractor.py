import os
import pandas as pd
import json

class Data_Extractor():
  def __init__(self, root_dir) -> None:
     self.root_dir = root_dir

  def collect_vad_dataset(self, speaker_ids=[], excluded_ids=[]):
    """
    Збирає пари [wav, TextGrid] для заданих мовців

    Args:
        root_dir (str): шлях до кореневої папки
        speaker_ids (list): список папок мовців

    Returns:
        list of tuples: [(wav_path, textgrid_path), ...]
    """
    excluded = [os.path.join(self.root_dir, x) for x in excluded_ids] + [os.path.join(self.root_dir, 'Group_Pipelines'), ]

    dataset = []
    info = []

    if speaker_ids:
      for speaker in speaker_ids:
         speaker_path = os.path.join(self.root_dir, speaker)

         if not os.path.exists(speaker_path):
            print(f"[WARNING] Speaker folder not found: {speaker_path}")
            continue
         data, stats = self.__extract_dlgs(speaker_path)
         dataset.append(data)
         info.append(stats)

    if not speaker_ids:
      for speaker in os.listdir(self.root_dir):
        speaker_path = os.path.join(self.root_dir, speaker)
        if os.path.isdir(speaker_path) and speaker_path not in excluded:
          data, stats = self.__extract_dlgs(speaker_path)
          dataset.append(data)
          info.append(stats)

    info_df = pd.DataFrame(data=info, columns=['name', 'metadata', 'n_pairs', 'n_missing', 'missing'])
    dataset_df = pd.DataFrame(data=dataset, columns=['name', 'metadata', 'pairs'])
    return dataset_df, info_df


  def __read_metadata_json(self, json_path):
   try:
     with open(f'{json_path}', 'r', encoding='utf-8') as file:
         metadata = json.load(file)
     if "speaker_a" in metadata.keys() or "speaker_b" in metadata.keys():
       if metadata["speaker_a"]["role"] == "author":
         speaker_metadata = metadata["speaker_a"]
       else:
         speaker_metadata = metadata["speaker_b"]
       speaker_metadata.pop("role")
     else:
       return "Incorrect Metadata Format"
     return speaker_metadata
   except Exception as e:
    print(f'An Exception occured: {e}, file: {json_path}')
    return "Incorrect Metadata Format"


  def __extract_dlgs(self, speaker_path):
    '''
    перебирає всі діалоги в папці спікера
    '''
    spk_name = speaker_path.split('/')[-1]
    metadata = []
    pairs = []
    missing_textgrid = []

    for dlg in os.listdir(speaker_path):
            dlg_path = os.path.join(speaker_path, dlg)

            if not os.path.isdir(dlg_path):
                continue

            # metadata
            metadata_path = os.path.join(dlg_path, "metadata.json")
            if os.path.exists(metadata_path) and not metadata:
              metadata = self.__read_metadata_json(metadata_path)

            mic_path = os.path.join(dlg_path, "audio", "MIC3")

            if not os.path.exists(mic_path):
                continue

            # шукаємо всі wav
            wav_files = [f for f in os.listdir(mic_path) if f.endswith(".wav")]

            for wav_file in wav_files:
                wav_path = os.path.join(mic_path, wav_file)

                base_name = os.path.splitext(wav_file)[0]
                tg_file = base_name + ".TextGrid"
                tg_path = os.path.join(mic_path, tg_file)

                if os.path.exists(tg_path):
                    pairs.append((wav_path, tg_path))
                else:
                    missing_textgrid.append(wav_path)

    stats = [spk_name, metadata, len(pairs), len(missing_textgrid), missing_textgrid]

    return [spk_name, metadata, pairs], stats